"""
Stage 4: noise-augmented training.

Retrains the CNN on the SAME patient-stratified train/test split as
Stage 1's "correct" model, but with noise injected into a fraction of
training beats EVERY epoch (fresh random noise type + SNR each time a
beat is augmented, so the model sees many different noise realizations
over 20 epochs rather than one fixed corrupted dataset).

Central question (motivated by Stage 3): preprocessing rescued powerline
noise almost completely but could not help with muscle artifact, because
MA overlaps the QRS complex's own frequency band and can't be filtered
out without also damaging the real signal. Does training the model WITH
noise present succeed where filtering structurally cannot?

After training, this script evaluates the noise-augmented model on the
exact same PL / ALL / MA x 5-SNR grid used in Stage 2/3, and builds a
three-way comparison against results already saved to disk:
    (a) clean-trained, no preprocessing        <- results/stage2_noise_results.csv
    (b) clean-trained, best preprocessing       <- results/stage3_preprocessing_results.csv
    (c) noise-augmented trained, no preprocessing  <- this script

Prerequisites: Stage 1 binary, Stage 2, and Stage 3 must have already run
(this script reads their saved CSVs for the comparison).

Outputs:
    models/stage4_noise_augmented_patient_split.pt
    results/stage4_noise_augmented_results.txt
    results/stage4_noise_augmented_results.csv
    results/figures/binary_fig7_training_curves_augmented.png
    results/figures/binary_fig8_three_way_comparison.png
"""

import os, sys, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
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

BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 1e-3
NUM_CLASSES  = 2
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS           = 360.0
POWERLINE_FREQ_HZ = 60.0
SEED         = 42
AUGMENT_PROB = 0.6   # fraction of training beats corrupted each epoch

CLASS_NAMES_S = ['N', 'A']
FOCUS_NOISE_TYPES = ['PL', 'ALL', 'MA']   # same as Stage 3, for direct comparison


# ══════════════════════════════════════════════════════════════════════════════
#  DATA + SPLIT (must match Stage 1 exactly)
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    beats  = np.load(os.path.join(DATA_DIR, 'mitbih_beats.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'mitbih_labels_binary.npy'))
    pids   = np.load(os.path.join(DATA_DIR, 'mitbih_patient_ids.npy'))
    beats  = beats[:, np.newaxis, :]
    return beats, labels, pids


def patient_stratified_split(labels, pids):
    """Identical to stage1_train_binary.py's split function — GroupKFold
    is deterministic given the same inputs, so this reproduces the exact
    same train/test patient partition Stage 1 used."""
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(labels)), labels, groups=pids))
    return splits[0]


def load_noise_tracks():
    tracks = {}
    for tag in ['bw', 'ma', 'em']:
        path = os.path.join(DATA_DIR, f'nstdb_{tag}.npy')
        if os.path.exists(path):
            tracks[tag.upper()] = np.load(path)
        else:
            print(f"  [!] {path} not found — run data/download_nstdb.py")
    return tracks


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def build_noise_pool(noise_tracks):
    pool = ['PL'] + [k for k in ('BW', 'MA', 'EM') if k in noise_tracks]
    if all(k in noise_tracks for k in ('BW', 'MA', 'EM')):
        pool.append('ALL')
    return pool


def augment_batch(clean_beats_2d, noise_tracks, noise_pool, rng, augment_prob=AUGMENT_PROB):
    """
    clean_beats_2d : (N, beat_len) array. Each beat independently has
    `augment_prob` chance of being corrupted with a randomly chosen noise
    type at a randomly chosen SNR level from SNR_LEVELS_DB. Called fresh
    every epoch so the model sees varied noise, not one static corrupted
    copy of the training set.
    """
    n_beats, beat_len = clean_beats_2d.shape
    out = clean_beats_2d.copy()
    for i in range(n_beats):
        if rng.random() >= augment_prob:
            continue
        noise_key = noise_pool[rng.integers(0, len(noise_pool))]
        snr_db = SNR_LEVELS_DB[rng.integers(0, len(SNR_LEVELS_DB))]
        clean = clean_beats_2d[i]
        if noise_key == 'PL':
            raw = synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=rng)
        elif noise_key == 'ALL':
            segs = [random_noise_segment(noise_tracks[s], beat_len, rng=rng) for s in ('BW', 'MA', 'EM')]
            segs.append(synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=rng))
            raw = combine_noise_segments(segs)
        else:
            raw = random_noise_segment(noise_tracks[noise_key], beat_len, rng=rng)
        out[i] = add_noise_at_snr(clean, raw, snr_db)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL (same conventions as Stage 1)
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def eval_model(model, loader):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def evaluate_metrics(model, X_2d, y):
    """X_2d: (N, beat_len) numpy array (no channel dim yet)."""
    X_t = torch.tensor(np.ascontiguousarray(X_2d[:, np.newaxis, :]), dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    sens = recall_score(y, preds, pos_label=1, zero_division=0)
    tn = np.sum((y == 0) & (preds == 0)); fp = np.sum((y == 0) & (preds == 1))
    spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y, preds, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y, probs[:, 1])
    except ValueError:
        auc = float('nan')
    return dict(accuracy=acc, sensitivity=sens, specificity=spec, f1=f1, auc=auc)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    beats, labels, pids = load_data()
    train_idx, test_idx = patient_stratified_split(labels, pids)

    # Sanity check: this should match Stage 1's saved test indices exactly
    saved_test_idx_path = os.path.join(RESULTS_DIR, 'stage1_binary_test_idx.npy')
    if os.path.exists(saved_test_idx_path):
        saved_test_idx = np.load(saved_test_idx_path)
        if np.array_equal(np.sort(saved_test_idx), np.sort(test_idx)):
            print("Split check: test set matches Stage 1's exactly — comparisons are apples-to-apples.")
        else:
            print("  [!] WARNING: test set does NOT match Stage 1's saved indices. "
                  "Results won't be directly comparable.")

    noise_tracks = load_noise_tracks()
    noise_pool = build_noise_pool(noise_tracks)
    print(f"Noise pool for augmentation: {noise_pool}  (augment_prob={AUGMENT_PROB})")

    X_train_clean = beats[train_idx, 0, :]   # (N, 216), drop channel dim for augmentation
    y_train = labels[train_idx]
    X_test_clean = beats[test_idx, 0, :]
    y_test = labels[test_idx]

    model = ShallowCNN1D(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    weights = torch.tensor(1.0 / (counts + 1e-8), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_clean[:, np.newaxis, :], dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False)

    rng = np.random.default_rng(SEED)
    history = {'train_loss': [], 'train_acc': [], 'test_acc_clean': []}

    print(f"\nTraining {EPOCHS} epochs with fresh noise augmentation each epoch "
          f"({len(X_train_clean):,} train beats)...")
    for epoch in range(1, EPOCHS + 1):
        X_train_aug = augment_batch(X_train_clean, noise_tracks, noise_pool, rng)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_aug[:, np.newaxis, :], dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=BATCH_SIZE, shuffle=True)

        loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        tgt, prd, _ = eval_model(model, test_loader)
        te_acc = accuracy_score(tgt, prd)

        history['train_loss'].append(loss)
        history['train_acc'].append(tr_acc)
        history['test_acc_clean'].append(te_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | "
                  f"Train acc (noisy): {tr_acc:.3f} | Test acc (clean): {te_acc:.3f}")

    model_path = os.path.join(MODEL_DIR, 'stage4_noise_augmented_patient_split.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel checkpoint saved -> {model_path}")

    # ── Clean-test performance (check for accuracy trade-off vs Stage 1) ─────
    clean_metrics = evaluate_metrics(model, X_test_clean, y_test)
    print(f"\nClean test performance (noise-augmented model):")
    for k, v in clean_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    # ── Training curve figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Binary Fig 7 — Stage 4 Training Curves (Noise-Augmented)',
                 fontsize=13, fontweight='bold', y=1.01)
    epochs_range = range(1, EPOCHS + 1)
    for ax, title, key in zip(axes,
                               ['Training Loss', 'Training Accuracy (on noisy batches)',
                                'Test Accuracy (on CLEAN held-out set)'],
                               ['train_loss', 'train_acc', 'test_acc_clean']):
        ax.plot(epochs_range, history[key], color='#2D2D2D', lw=2, marker='o', ms=3)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#FFFFFF')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    curve_path = os.path.join(FIG_DIR, 'binary_fig7_training_curves_augmented.png')
    fig.savefig(curve_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training curves saved -> {curve_path}")

    # ── Evaluate on the same PL/ALL/MA x SNR grid as Stage 2/3 ────────────
    print(f"\nEvaluating noise-augmented model on Stage 2/3's noise grid...")
    have_all_three = all(k in noise_tracks for k in ('BW', 'MA', 'EM'))
    focus = [n for n in FOCUS_NOISE_TYPES if n == 'PL' or n in noise_tracks or (n == 'ALL' and have_all_three)]

    eval_rng = np.random.default_rng(SEED + 1)   # different stream from training augmentation
    stage4_results = []
    for noise_key in focus:
        for snr_db in SNR_LEVELS_DB:
            n_beats, beat_len = X_test_clean.shape
            X_noisy = np.empty_like(X_test_clean)
            for i in range(n_beats):
                clean = X_test_clean[i]
                if noise_key == 'PL':
                    raw = synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=eval_rng)
                elif noise_key == 'ALL':
                    segs = [random_noise_segment(noise_tracks[s], beat_len, rng=eval_rng) for s in ('BW', 'MA', 'EM')]
                    segs.append(synthetic_powerline(beat_len, fs=FS, freq=POWERLINE_FREQ_HZ, rng=eval_rng))
                    raw = combine_noise_segments(segs)
                else:
                    raw = random_noise_segment(noise_tracks[noise_key], beat_len, rng=eval_rng)
                X_noisy[i] = add_noise_at_snr(clean, raw, snr_db)
            m = evaluate_metrics(model, X_noisy, y_test)
            print(f"  {noise_key} @ {snr_db:>3} dB | acc {m['accuracy']:.4f} | "
                  f"sens {m['sensitivity']:.4f} | AUC {m['auc']:.4f}")
            stage4_results.append(dict(noise=noise_key, snr_db=snr_db, **m))

    csv_path = os.path.join(RESULTS_DIR, 'stage4_noise_augmented_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['noise', 'snr_db', 'accuracy', 'sensitivity',
                                                'specificity', 'f1', 'auc'])
        writer.writeheader()
        for r in stage4_results:
            writer.writerow(r)
    print(f"\nCSV saved -> {csv_path}")

    # ── Three-way comparison against Stage 2 (none) and Stage 3 (best preprocessing) ──
    stage2_rows = _read_csv(os.path.join(RESULTS_DIR, 'stage2_noise_results.csv'))
    stage3_rows = _read_csv(os.path.join(RESULTS_DIR, 'stage3_preprocessing_results.csv'))

    txt_path = os.path.join(RESULTS_DIR, 'stage4_noise_augmented_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("STAGE 4 — NOISE-AUGMENTED TRAINING vs PREPROCESSING vs NOTHING\n")
        f.write(f"Clean-test check: accuracy={clean_metrics['accuracy']:.4f}, "
                f"sensitivity={clean_metrics['sensitivity']:.4f} "
                f"(compare to Stage 1 clean-trained clean-test numbers for trade-off check)\n\n")
        for noise_key in focus:
            f.write(f"\n{'='*70}\n  {noise_key} at 0 dB (worst case) — three-way comparison\n{'='*70}\n")
            s2_none = _find_row(stage2_rows, noise=noise_key, snr_db='0')
            s3_best = _best_stage3_row(stage3_rows, noise_key, snr_db='0')
            s4_row = next(r for r in stage4_results if r['noise'] == noise_key and r['snr_db'] == 0)
            f.write(f"  {'Condition':<42} {'Sens':>8} {'AUC':>8}\n")
            if s2_none:
                f.write(f"  {'(a) Clean-trained, no preprocessing':<42} "
                        f"{float(s2_none['sensitivity']):>8.4f} {float(s2_none['auc']):>8.4f}\n")
            if s3_best:
                f.write(f"  {'(b) Clean-trained, best preprocessing ('+s3_best['pipeline']+')':<42} "
                        f"{float(s3_best['sensitivity']):>8.4f} {float(s3_best['auc']):>8.4f}\n")
            f.write(f"  {'(c) Noise-augmented trained, no preprocessing':<42} "
                    f"{s4_row['sensitivity']:>8.4f} {s4_row['auc']:>8.4f}\n")
    print(f"Text report saved -> {txt_path}")

    # ── Three-way comparison figure ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(focus), figsize=(6.5*len(focus), 5), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Binary Fig 8 — Clean-Trained vs Preprocessing vs Noise-Augmented Training',
                 fontsize=13, fontweight='bold', y=1.03)

    for ax, noise_key in zip(axes, focus):
        xs = SNR_LEVELS_DB
        # (a) clean-trained, no preprocessing
        ys_a = [float(_find_row(stage2_rows, noise=noise_key, snr_db=str(s))['sensitivity']) for s in xs]
        ax.plot(xs, ys_a, marker='o', lw=2, color='#999999', label='Clean-trained, no preprocessing')
        # (b) clean-trained, best preprocessing (full_pipeline, for consistency across SNR)
        ys_b = [float(_find_row(stage3_rows, noise=noise_key, snr_db=str(s), pipeline='full_pipeline')['sensitivity']) for s in xs]
        ax.plot(xs, ys_b, marker='s', lw=2, color='#4A90D9', label='Clean-trained + full preprocessing')
        # (c) noise-augmented
        ys_c = [next(r for r in stage4_results if r['noise']==noise_key and r['snr_db']==s)['sensitivity'] for s in xs]
        ax.plot(xs, ys_c, marker='^', lw=3, color='#E05C5C', label='Noise-augmented trained')

        ax.set_title(f'{noise_key}', fontsize=11, fontweight='bold')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Sensitivity')
        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#FFFFFF')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'binary_fig8_three_way_comparison.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Comparison figure saved -> {fig_path}")

    print("\nStage 4 complete. Project's core technical arc (Stages 1-4) is done.")


def _read_csv(path):
    if not os.path.exists(path):
        print(f"  [!] {path} not found — comparison will be incomplete")
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _find_row(rows, **kwargs):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kwargs.items()):
            return r
    return None


def _best_stage3_row(rows, noise_key, snr_db):
    """Pick the pipeline with the highest sensitivity for this noise/SNR."""
    candidates = [r for r in rows if r['noise'] == noise_key and r['snr_db'] == snr_db and r['pipeline'] != 'none']
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r['sensitivity']))


if __name__ == "__main__":
    main()
