"""
Stage 2: Noise robustness sweep.

Loads the FROZEN Stage 1 (binary, patient-stratified) model and evaluates
it — no retraining — on the same held-out test beats, corrupted by each of
5 noise conditions at 5 SNR levels (24, 18, 12, 6, 0 dB): 25 conditions
total, plus 1 clean reference condition.

Noise conditions:
    BW  - baseline wander alone        (data/nstdb_bw.npy)
    MA  - muscle artifact alone        (data/nstdb_ma.npy)
    EM  - electrode motion alone       (data/nstdb_em.npy)
    PL  - powerline interference alone (synthetic 60 Hz, generated on the fly)
    ALL - all four combined at once, mixed BEFORE scaling to the target SNR
          (see combine_noise_segments() in noise/nstdb_noise.py) so "6 dB"
          means the same total noise power whether it comes from 1 source
          or 4 — this is the more realistic "real recording" condition,
          since a real ECG rarely has just one noise source active.

Why 60 Hz and not 50 Hz: MIT-BIH was recorded at Beth Israel Hospital in
Boston (1975-1979), i.e. on US mains power. 50 Hz is European/Asian mains
and never applied to this recording equipment, so testing it here would
add an unrealistic condition.

Why evaluate on the SAME frozen model rather than retraining per condition:
this isolates "how much does noise alone hurt a model trained on clean
data" — that's the Stage 2 question. Stage 4 (noise-augmented training)
is a separate, later experiment that asks whether training WITH noise
changes this picture.

Prerequisites:
    1. Run experiments/stage1_train_binary.py first (produces the frozen
       model checkpoint + test-set indices this script loads).
    2. Run data/download_nstdb.py to get real BW/MA/EM noise. If you skip
       this, the script still runs using only synthetic powerline noise
       (BW/MA/EM and ALL will be skipped with a warning).

Outputs:
    results/stage2_noise_results.txt
    results/stage2_noise_results.csv
    results/figures/binary_fig5_noise_vs_snr.png
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

# MIT-BIH was recorded at Beth Israel Hospital, Boston (1975-1979) -> US
# mains power. Use 60 Hz, not 50 Hz (European/Asian mains), for the
# synthetic powerline interference condition.
POWERLINE_FREQ_HZ = 60.0

NOISE_COLORS = {'BW': '#4A90D9', 'MA': '#E05C5C', 'EM': '#F4A261',
                 'PL': '#9B5DE5', 'ALL': '#2D2D2D'}
NOISE_NAMES  = {'BW': 'Baseline Wander', 'MA': 'Muscle Artifact',
                'EM': 'Electrode Motion', 'PL': 'Powerline (60 Hz)',
                'ALL': 'All 4 Combined'}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD FROZEN MODEL + DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_frozen_model():
    model_path = os.path.join(MODEL_DIR, 'stage1_binary_patient_split.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found.\n"
            f"Run experiments/stage1_train_binary.py first — it saves this "
            f"checkpoint after training the patient-stratified model."
        )
    model = ShallowCNN1D(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def load_test_set():
    idx_path = os.path.join(RESULTS_DIR, 'stage1_binary_test_idx.npy')
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"{idx_path} not found. Run experiments/stage1_train_binary.py first."
        )
    test_idx = np.load(idx_path)
    beats  = np.load(os.path.join(DATA_DIR, 'mitbih_beats.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'mitbih_labels_binary.npy'))
    return beats[test_idx], labels[test_idx]


def load_noise_tracks():
    """Load whichever NSTDB noise .npy files exist. PL is always available (synthetic)."""
    tracks = {}
    for tag in ['bw', 'ma', 'em']:
        path = os.path.join(DATA_DIR, f'nstdb_{tag}.npy')
        if os.path.exists(path):
            tracks[tag.upper()] = np.load(path)
        else:
            print(f"  [!] {path} not found — skipping {tag.upper()} "
                  f"(run data/download_nstdb.py to add it)")
    return tracks


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def make_noisy_batch(clean_beats, noise_key, noise_tracks, snr_db, rng):
    """
    clean_beats  : (N, 216) array of clean, already-normalised beats.
    noise_key    : 'BW' | 'MA' | 'EM' | 'PL' | 'ALL'
    noise_tracks : dict of noise_key -> 1D noise track (PL/ALL don't need
                   a track entry; PL is synthesized, ALL is built from the
                   other three plus a synthesized PL segment).
    Returns (N, 216) noisy beats at the requested SNR.
    """
    n_beats, beat_len = clean_beats.shape
    noisy = np.empty_like(clean_beats)
    for i in range(n_beats):
        clean = clean_beats[i]

        if noise_key == 'PL':
            raw_noise = synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=rng)

        elif noise_key == 'ALL':
            segs = []
            for src in ('BW', 'MA', 'EM'):
                segs.append(random_noise_segment(noise_tracks[src], beat_len, rng=rng))
            segs.append(synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=rng))
            raw_noise = combine_noise_segments(segs)   # sum FIRST, scale once below

        else:
            raw_noise = random_noise_segment(noise_tracks[noise_key], beat_len, rng=rng)

        noisy[i] = add_noise_at_snr(clean, raw_noise, snr_db)
    return noisy


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X, y):
    X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

    acc  = accuracy_score(y, preds)
    sens = recall_score(y, preds, pos_label=1, zero_division=0)     # sensitivity (Abnormal)
    tn = np.sum((y == 0) & (preds == 0))
    fp = np.sum((y == 0) & (preds == 1))
    spec = tn / (tn + fp + 1e-8)                                    # specificity (Normal)
    f1   = f1_score(y, preds, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y, probs[:, 1])
    except ValueError:
        auc = float('nan')

    return dict(accuracy=acc, sensitivity=sens, specificity=spec, f1=f1, auc=auc)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    model = load_frozen_model()
    X_clean, y = load_test_set()
    print(f"Loaded frozen model + {len(X_clean):,} held-out test beats "
          f"(same patients excluded from training as Stage 1).")

    noise_tracks = load_noise_tracks()
    have_all_three = all(k in noise_tracks for k in ('BW', 'MA', 'EM'))

    conditions = ['PL']  # synthetic, always available
    conditions += [k for k in ('BW', 'MA', 'EM') if k in noise_tracks]
    if have_all_three:
        conditions.append('ALL')
    else:
        print("\n  [!] Real NSTDB noise incomplete — 'ALL' combined condition "
              "skipped.\n      Run data/download_nstdb.py to get BW/MA/EM and enable it.\n")

    rng = np.random.default_rng(SEED)

    # ── Clean reference (should match Stage 1 patient-split numbers) ─────────
    clean_metrics = evaluate(model, X_clean, y)
    print(f"\nClean reference (no noise added):")
    for k, v in clean_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    results = []  # rows of dicts for CSV/plot
    for noise_key in conditions:
        print(f"\n{'='*55}\n  Noise type: {NOISE_NAMES.get(noise_key, noise_key)} ({noise_key})\n{'='*55}")
        for snr_db in SNR_LEVELS_DB:
            X_noisy = make_noisy_batch(X_clean, noise_key, noise_tracks, snr_db, rng)
            m = evaluate(model, X_noisy, y)
            print(f"  SNR {snr_db:>3} dB | acc {m['accuracy']:.4f} | "
                  f"sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f} | "
                  f"F1 {m['f1']:.4f} | AUC {m['auc']:.4f}")
            results.append(dict(noise=noise_key, snr_db=snr_db, **m))

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'stage2_noise_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['noise', 'snr_db', 'accuracy',
                                                'sensitivity', 'specificity', 'f1', 'auc'])
        writer.writeheader()
        writer.writerow(dict(noise='CLEAN', snr_db='', **clean_metrics))
        for r in results:
            writer.writerow(r)
    print(f"\nCSV saved -> {csv_path}")

    # ── Save text report ──────────────────────────────────────────────────
    txt_path = os.path.join(RESULTS_DIR, 'stage2_noise_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("STAGE 2 — NOISE ROBUSTNESS SWEEP\n")
        f.write("Frozen Stage 1 (binary, patient-stratified) model, evaluated\n")
        f.write("under individual + combined noise types x 5 SNR levels. No retraining.\n\n")
        f.write(f"Clean reference: " +
                ", ".join(f"{k}={v:.4f}" for k, v in clean_metrics.items()) + "\n\n")
        for noise_key in conditions:
            f.write(f"\n{NOISE_NAMES.get(noise_key, noise_key)} ({noise_key})\n")
            f.write(f"{'SNR (dB)':>10} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}\n")
            for r in [r for r in results if r['noise'] == noise_key]:
                f.write(f"{r['snr_db']:>10} {r['accuracy']:>8.4f} {r['sensitivity']:>8.4f} "
                        f"{r['specificity']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f}\n")
    print(f"Text report saved -> {txt_path}")

    # ── Performance-vs-SNR figure ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Binary Fig 5 — Noise Robustness: Performance vs SNR',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, metric, title in zip(axes, ['accuracy', 'sensitivity'],
                                  ['Accuracy vs SNR', 'Sensitivity (Abnormal detection) vs SNR']):
        for noise_key in conditions:
            xs = [r['snr_db'] for r in results if r['noise'] == noise_key]
            ys = [r[metric]   for r in results if r['noise'] == noise_key]
            lw = 3 if noise_key == 'ALL' else 2
            ax.plot(xs, ys, marker='o', lw=lw,
                    color=NOISE_COLORS.get(noise_key, '#333'),
                    label=NOISE_NAMES.get(noise_key, noise_key))
        ax.axhline(clean_metrics[metric], color='#999', linestyle=':', lw=1.5,
                   label='Clean (no noise)')
        ax.set_xlabel('SNR (dB) - lower = noisier')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.invert_xaxis()   # so the plot reads left(clean-ish)->right(noisy), matching intuition
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#FFFFFF')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'binary_fig5_noise_vs_snr.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Figure saved -> {fig_path}")

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
