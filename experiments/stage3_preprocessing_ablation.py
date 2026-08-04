"""
Stage 3: preprocessing ablation.

Takes the SAME frozen Stage 1 (binary, patient-stratified) model and the
SAME noisy conditions from Stage 2, and asks: how much of the accuracy
lost to noise does each preprocessing technique recover?

Motivated directly by the Stage 2 finding: powerline interference (PL)
was by far the most damaging noise type, collapsing sensitivity to 0.00
and AUC below 0.5 at 0 dB SNR. This script's central question is whether
a standard 0.5-40 Hz bandpass (which should attenuate a 60 Hz tone almost
entirely) and/or a targeted 60 Hz notch recovers that collapse.

Ablation conditions tested (see preprocessing/ecg_preprocessing.py):
    none            no preprocessing (matches Stage 2's raw numbers)
    bandpass_only   0.5-40 Hz Butterworth
    baseline_only   per-beat linear detrend
    normalize_only  re-apply z-score normalization after noise
    notch_only      60 Hz notch (added specifically for the PL finding)
    full_pipeline   detrend -> bandpass -> notch -> normalize

Focuses on PL and ALL (the two noise types Stage 2 found damaging) plus
MA as a comparison case (Stage 2 found it comparatively mild) across all
5 SNR levels, to keep runtime reasonable while still directly answering
the "does preprocessing recover the worst-case collapse" question. Add
'BW' / 'EM' to FOCUS_NOISE_TYPES below if you want the full 5x5 grid.

Prerequisites: Stage 1 binary + Stage 2 noise sweep must have already run
(same checkpoint, same test set, same noise files).

Outputs:
    results/stage3_preprocessing_results.txt
    results/stage3_preprocessing_results.csv
    results/figures/binary_fig6_preprocessing_recovery.png
"""

import os, sys, csv
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cnn1d import ShallowCNN1D
from noise.nstdb_noise import (
    add_noise_at_snr, random_noise_segment, synthetic_powerline,
    combine_noise_segments, SNR_LEVELS_DB
)
from preprocessing.ecg_preprocessing import PIPELINES

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2
FS          = 360.0
SEED        = 42
POWERLINE_FREQ_HZ = 60.0

# Focused on the two extremes Stage 2 revealed. Add 'BW', 'EM' for the
# full grid (costs 2x more runtime, all inference so still fast).
FOCUS_NOISE_TYPES = ['PL', 'ALL', 'MA']

PIPELINE_COLORS = {
    'none': '#999999', 'bandpass_only': '#4A90D9', 'baseline_only': '#52B788',
    'normalize_only': '#F4A261', 'notch_only': '#9B5DE5', 'full_pipeline': '#E05C5C',
}
PIPELINE_NAMES = {
    'none': 'No preprocessing', 'bandpass_only': 'Bandpass only',
    'baseline_only': 'Detrend only', 'normalize_only': 'Renormalize only',
    'notch_only': '60Hz notch only', 'full_pipeline': 'Full pipeline',
}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD FROZEN MODEL + DATA  (identical to Stage 2)
# ══════════════════════════════════════════════════════════════════════════════
def load_frozen_model():
    model_path = os.path.join(MODEL_DIR, 'stage1_binary_patient_split.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run stage1_train_binary.py first.")
    model = ShallowCNN1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def load_test_set():
    idx_path = os.path.join(RESULTS_DIR, 'stage1_binary_test_idx.npy')
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"{idx_path} not found. Run stage1_train_binary.py first.")
    test_idx = np.load(idx_path)
    beats  = np.load(os.path.join(DATA_DIR, 'mitbih_beats.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'mitbih_labels_binary.npy'))
    return beats[test_idx], labels[test_idx]


def load_noise_tracks():
    tracks = {}
    for tag in ['bw', 'ma', 'em']:
        path = os.path.join(DATA_DIR, f'nstdb_{tag}.npy')
        if os.path.exists(path):
            tracks[tag.upper()] = np.load(path)
        else:
            print(f"  [!] {path} not found — run data/download_nstdb.py")
    return tracks


def make_noisy_batch(clean_beats, noise_key, noise_tracks, snr_db, rng):
    """Identical logic to Stage 2 — kept in sync so 'none' pipeline here
    reproduces Stage 2's raw numbers exactly, as a consistency check."""
    n_beats, beat_len = clean_beats.shape
    noisy = np.empty_like(clean_beats)
    for i in range(n_beats):
        clean = clean_beats[i]
        if noise_key == 'PL':
            raw_noise = synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=rng)
        elif noise_key == 'ALL':
            segs = [random_noise_segment(noise_tracks[s], beat_len, rng=rng) for s in ('BW', 'MA', 'EM')]
            segs.append(synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=rng))
            raw_noise = combine_noise_segments(segs)
        else:
            raw_noise = random_noise_segment(noise_tracks[noise_key], beat_len, rng=rng)
        noisy[i] = add_noise_at_snr(clean, raw_noise, snr_db)
    return noisy


def evaluate(model, X, y):
    # scipy's sosfiltfilt/filtfilt (used by the bandpass and notch filters)
    # can return arrays with negative strides internally (from reversing
    # the signal for the backward filter pass). torch.tensor() can't
    # convert those directly, so force a contiguous copy first.
    X = np.ascontiguousarray(X)
    X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
    acc  = accuracy_score(y, preds)
    sens = recall_score(y, preds, pos_label=1, zero_division=0)
    tn = np.sum((y == 0) & (preds == 0)); fp = np.sum((y == 0) & (preds == 1))
    spec = tn / (tn + fp + 1e-8)
    f1   = f1_score(y, preds, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y, probs[:, 1])
    except ValueError:
        auc = float('nan')
    return dict(accuracy=acc, sensitivity=sens, specificity=spec, f1=f1, auc=auc)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ABLATION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    model = load_frozen_model()
    X_clean, y = load_test_set()
    noise_tracks = load_noise_tracks()
    have_all_three = all(k in noise_tracks for k in ('BW', 'MA', 'EM'))

    focus = [n for n in FOCUS_NOISE_TYPES if n in ('PL',) or n in noise_tracks or (n == 'ALL' and have_all_three)]
    print(f"Testing noise types: {focus}  x  pipelines: {list(PIPELINES.keys())}  x  SNR: {SNR_LEVELS_DB}")

    rng = np.random.default_rng(SEED)
    results = []

    for noise_key in focus:
        for snr_db in SNR_LEVELS_DB:
            # Same noisy batch reused across all pipelines at this (noise, SNR)
            # so differences are attributable to preprocessing, not different noise draws.
            X_noisy = make_noisy_batch(X_clean, noise_key, noise_tracks, snr_db, rng)

            print(f"\n{noise_key} @ {snr_db} dB:")
            for pipe_name, pipe_fn in PIPELINES.items():
                X_processed = pipe_fn(X_noisy, FS)
                m = evaluate(model, X_processed, y)
                print(f"  {pipe_name:<16} acc {m['accuracy']:.4f} | sens {m['sensitivity']:.4f} | "
                      f"spec {m['specificity']:.4f} | F1 {m['f1']:.4f} | AUC {m['auc']:.4f}")
                results.append(dict(noise=noise_key, snr_db=snr_db, pipeline=pipe_name, **m))

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'stage3_preprocessing_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['noise', 'snr_db', 'pipeline', 'accuracy',
                                                'sensitivity', 'specificity', 'f1', 'auc'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nCSV saved -> {csv_path}")

    # ── Recovery summary: how much of the sensitivity lost to noise does
    #    each pipeline recover, at the worst-case SNR (0 dB) for each noise ──
    txt_path = os.path.join(RESULTS_DIR, 'stage3_preprocessing_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("STAGE 3 — PREPROCESSING ABLATION\n")
        f.write("Same frozen Stage 1 model + Stage 2 noisy conditions, with each\n")
        f.write("preprocessing technique applied before the model sees the signal.\n\n")
        for noise_key in focus:
            f.write(f"\n{'='*60}\n  {noise_key} — recovery at 0 dB (worst case)\n{'='*60}\n")
            none_row = next(r for r in results if r['noise']==noise_key and r['snr_db']==0 and r['pipeline']=='none')
            f.write(f"  {'Pipeline':<18} {'Sens':>8} {'AUC':>8}   Sens recovered vs 'none'\n")
            for pipe_name in PIPELINES:
                row = next(r for r in results if r['noise']==noise_key and r['snr_db']==0 and r['pipeline']==pipe_name)
                delta = row['sensitivity'] - none_row['sensitivity']
                f.write(f"  {pipe_name:<18} {row['sensitivity']:>8.4f} {row['auc']:>8.4f}   {delta:+.4f}\n")
    print(f"Text report saved -> {txt_path}")

    # ── Figure: sensitivity vs SNR, one panel per focus noise type,
    #    one line per pipeline ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(focus), figsize=(6.5*len(focus), 5), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Binary Fig 6 — Does Preprocessing Recover Noise-Driven Sensitivity Loss?',
                 fontsize=13, fontweight='bold', y=1.03)

    for ax, noise_key in zip(axes, focus):
        for pipe_name in PIPELINES:
            xs = [r['snr_db'] for r in results if r['noise']==noise_key and r['pipeline']==pipe_name]
            ys = [r['sensitivity'] for r in results if r['noise']==noise_key and r['pipeline']==pipe_name]
            lw = 3 if pipe_name == 'full_pipeline' else 1.5
            ax.plot(xs, ys, marker='o', lw=lw, ms=4,
                    color=PIPELINE_COLORS[pipe_name], label=PIPELINE_NAMES[pipe_name])
        ax.set_title(f'{noise_key}', fontsize=11, fontweight='bold')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Sensitivity')
        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#FFFFFF')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'binary_fig6_preprocessing_recovery.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Figure saved -> {fig_path}")

    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
