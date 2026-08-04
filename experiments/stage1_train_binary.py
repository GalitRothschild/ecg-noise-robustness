"""
Stage 1 (Binary): Reproduce Rasti et al. CNN on MIT-BIH, Normal vs Abnormal.

This is the binary counterpart of stage1_train.py (which used the 5-class
AAMI scheme). Binary Normal/Abnormal matches Rasti et al.'s actual task
definition exactly, and avoids the near-empty minority classes (S, F) that
make per-class metrics unstable once noise is added in Stage 2.

Compares beat-level split (leaky) vs patient-stratified split (correct),
same as Stage 1. The patient-stratified model is CHECKPOINTED to disk
(models/stage1_binary_patient_split.pt) so Stage 2's noise sweep can load
this exact frozen model and reuse the exact same test-set beats
(indices saved to results/stage1_binary_test_idx.npy) — this is important:
Stage 2 must evaluate noise robustness on the same model and same held-out
patients as Stage 1's "honest" baseline, not a freshly retrained one.

Outputs:
    results/stage1_binary_results.txt
    results/figures/binary_fig*.png
    models/stage1_binary_patient_split.pt   (frozen model for Stage 2)
    results/stage1_binary_test_idx.npy      (test-set indices for Stage 2)
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cnn1d import ShallowCNN1D

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-3
NUM_CLASSES = 2
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES   = ['Normal', 'Abnormal']
CLASS_NAMES_S = ['N', 'A']

C_LEAKY   = '#E05C5C'
C_CORRECT = '#4A90D9'


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    beats  = np.load(os.path.join(DATA_DIR, 'mitbih_beats.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'mitbih_labels_binary.npy'))
    pids   = np.load(os.path.join(DATA_DIR, 'mitbih_patient_ids.npy'))
    beats  = beats[:, np.newaxis, :]   # (N,216) -> (N,1,216)
    return beats, labels, pids


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


def eval_model(model, loader):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X      = X.to(DEVICE)
            logits = model(X)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def specificity_binary(targets, preds):
    """True negative rate, class 0 (Normal) = negative."""
    tn = np.sum((targets == 0) & (preds == 0))
    fp = np.sum((targets == 0) & (preds == 1))
    return tn / (tn + fp + 1e-8)


def compute_metrics(targets, preds, probs, tag):
    acc   = accuracy_score(targets, preds)
    # Positive class = Abnormal (1) — the clinically important one
    prec  = precision_score(targets, preds, pos_label=1, zero_division=0)
    rec   = recall_score(targets, preds, pos_label=1, zero_division=0)      # sensitivity
    spec  = specificity_binary(targets, preds)
    f1    = f1_score(targets, preds, pos_label=1, zero_division=0)
    mcc   = matthews_corrcoef(targets, preds)
    kappa = cohen_kappa_score(targets, preds)
    auc   = roc_auc_score(targets, probs[:, 1])

    # Majority-baseline accuracy for context (always predict Normal)
    baseline_acc = (targets == 0).mean()

    cm     = confusion_matrix(targets, preds)
    report = classification_report(targets, preds,
                                   target_names=CLASS_NAMES_S, digits=4)

    return dict(
        tag=tag,
        accuracy=acc, precision=prec, recall=rec, specificity=spec,
        f1=f1, auc=auc, mcc=mcc, kappa=kappa, baseline_acc=baseline_acc,
        conf_matrix=cm, report=report,
        targets=targets, preds=preds, probs=probs
    )


def print_metrics(m):
    print(f"\n  {'─'*50}")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'─'*50}")
    rows = [
        ('Accuracy',                    m['accuracy']),
        ('  (majority baseline)',       m['baseline_acc']),
        ('Sensitivity / Recall (Abn.)', m['recall']),
        ('Specificity (Normal)',        m['specificity']),
        ('Precision (Abn.)',            m['precision']),
        ('F1 (Abnormal)',               m['f1']),
        ('AUC-ROC',                     m['auc']),
        ('MCC',                         m['mcc']),
        ("Cohen's Kappa",               m['kappa']),
    ]
    for name, val in rows:
        bar = '█' * int(val * 20)
        print(f"  {name:<28} {val:>8.4f}  {bar}")
    print(f"\n  Confusion matrix (rows=true, cols=pred)  [{', '.join(CLASS_NAMES_S)}]")
    print(f"  {m['conf_matrix']}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(beats, labels, pids, train_idx, test_idx, tag, save_model_path=None):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    n_test_patients = len(np.unique(pids[test_idx]))
    print(f"  Train: {len(train_idx):,} beats | "
          f"Test: {len(test_idx):,} beats | "
          f"Test patients: {n_test_patients}")
    print(f"{'='*60}")

    X_tr = torch.tensor(beats[train_idx],  dtype=torch.float32)
    y_tr = torch.tensor(labels[train_idx], dtype=torch.long)
    X_te = torch.tensor(beats[test_idx],   dtype=torch.float32)
    y_te = torch.tensor(labels[test_idx],  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_te, y_te),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = ShallowCNN1D(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    counts    = np.bincount(labels[train_idx], minlength=NUM_CLASSES).astype(float)
    weights   = torch.tensor(1.0 / (counts + 1e-8), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    print(f"\n  Training {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        tgt, prd, _  = eval_model(model, test_loader)
        te_acc       = accuracy_score(tgt, prd)

        history['train_loss'].append(loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | "
                  f"Loss: {loss:.4f} | "
                  f"Train acc: {tr_acc:.3f} | "
                  f"Test acc:  {te_acc:.3f}")

    targets, preds, probs = eval_model(model, test_loader)
    metrics = compute_metrics(targets, preds, probs, tag)
    metrics['history'] = history
    print_metrics(metrics)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)
        print(f"\n  Model checkpoint saved -> {save_model_path}")
        np.save(os.path.join(RESULTS_DIR, 'stage1_binary_test_idx.npy'), test_idx)
        print(f"  Test-set indices saved -> results/stage1_binary_test_idx.npy")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SPLITS
# ══════════════════════════════════════════════════════════════════════════════
def beat_level_split(labels):
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)


def patient_stratified_split(labels, pids):
    gkf    = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(labels)), labels, groups=pids))
    return splits[0]


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_training_curves(m1, m2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Binary Fig 1 — Training Curves: Beat-level vs Patient-stratified',
                 fontsize=13, fontweight='bold', y=1.01)

    epochs = range(1, EPOCHS + 1)
    titles = ['Training Loss', 'Training Accuracy', 'Test Accuracy (per epoch)']
    keys   = ['train_loss',    'train_acc',          'test_acc']

    for ax, title, key in zip(axes, titles, keys):
        ax.plot(epochs, m1['history'][key], color=C_LEAKY,   lw=2,
                label='Beat-level (leaky)', marker='o', ms=3)
        ax.plot(epochs, m2['history'][key], color=C_CORRECT, lw=2,
                label='Patient-stratified', marker='s', ms=3)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=9)
        ax.set_facecolor('#FFFFFF')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    _save(fig, 'binary_fig1_training_curves.png')


def fig_confusion_matrices(m1, m2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Binary Fig 2 — Normalised Confusion Matrices',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, m, title, cmap in zip(
            axes,
            [m1, m2],
            ['Beat-level Split (LEAKY)', 'Patient-stratified Split (CORRECT)'],
            ['Reds', 'Blues']):

        cm_norm = m['conf_matrix'].astype(float)
        cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax,
                    xticklabels=CLASS_NAMES_S,
                    yticklabels=CLASS_NAMES_S,
                    cmap=cmap, vmin=0, vmax=1,
                    linewidths=0.5, linecolor='#DDDDDD',
                    annot_kws={'size': 11})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    _save(fig, 'binary_fig2_confusion_matrices.png')


def fig_roc_curves(m1, m2):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor('#F7F9FC')
    ax.set_facecolor('#FFFFFF')

    for m, label, col in zip([m1, m2],
                              ['Beat-level (Leaky)', 'Patient-stratified (Correct)'],
                              [C_LEAKY, C_CORRECT]):
        fpr, tpr, _ = roc_curve(m['targets'], m['probs'][:, 1])
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{label}  AUC={m["auc"]:.3f}')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_title('Binary Fig 3 — ROC Curve (Abnormal = positive class)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()
    _save(fig, 'binary_fig3_roc_curves.png')


def fig_summary_comparison(m1, m2):
    metrics = [
        ('Accuracy',    m1['accuracy'],    m2['accuracy']),
        ('Sensitivity\n(Abnormal)', m1['recall'], m2['recall']),
        ('Specificity\n(Normal)',   m1['specificity'], m2['specificity']),
        ('F1\n(Abnormal)', m1['f1'], m2['f1']),
        ('AUC-ROC',     m1['auc'],         m2['auc']),
        ('MCC',         m1['mcc'],         m2['mcc']),
        ("Cohen's\nKappa", m1['kappa'],    m2['kappa']),
    ]

    labels = [r[0] for r in metrics]
    vals1  = [r[1] for r in metrics]
    vals2  = [r[2] for r in metrics]

    x     = np.arange(len(labels))
    width = 0.30

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#F7F9FC')
    ax.set_facecolor('#FFFFFF')

    b1 = ax.bar(x - width/2, vals1, width, color=C_LEAKY,   alpha=0.88, label='Beat-level (LEAKY)')
    b2 = ax.bar(x + width/2, vals2, width, color=C_CORRECT, alpha=0.88, label='Patient-stratified (CORRECT)')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(m2['baseline_acc'], color='#999', linestyle=':', lw=1.5)
    ax.text(len(labels)-0.5, m2['baseline_acc']+0.015,
            f"majority-class floor ({m2['baseline_acc']*100:.1f}%)",
            fontsize=8, color='#666', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(
        'Binary Fig 4 — Stage 1 Summary (Normal vs Abnormal)\n'
        'Beat-level split inflates every metric vs. the honest patient-stratified evaluation',
        fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    _save(fig, 'binary_fig4_summary_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_comparison_text(m1, m2):
    rows = [
        ('Accuracy',                m1['accuracy'],    m2['accuracy']),
        ('Sensitivity (Abnormal)',  m1['recall'],       m2['recall']),
        ('Specificity (Normal)',    m1['specificity'],  m2['specificity']),
        ('F1 (Abnormal)',           m1['f1'],            m2['f1']),
        ('AUC-ROC',                 m1['auc'],           m2['auc']),
        ('MCC',                     m1['mcc'],           m2['mcc']),
        ("Cohen's Kappa",           m1['kappa'],         m2['kappa']),
    ]
    header = (
        f"\n{'='*65}\n"
        f"  STAGE 1 (BINARY) — FINAL COMPARISON TABLE\n"
        f"{'='*65}\n"
        f"  {'Metric':<25} {'Beat-level':>11} {'Pat-split':>11} {'Gap':>9}\n"
        f"  {'-'*58}"
    )
    body = ""
    for name, v1, v2 in rows:
        gap = v1 - v2
        direction = 'v' if gap > 0 else '^'
        body += (f"\n  {name:<25} {v1:>11.4f} {v2:>11.4f} "
                 f"{direction}{abs(gap):>7.4f}")

    acc_gap = m1['accuracy'] - m2['accuracy']
    conclusion = (
        f"\n\n{'='*65}\n"
        f"  CONCLUSION\n"
        f"{'='*65}\n"
        f"  Beat-level accuracy      : {m1['accuracy']*100:.2f}%\n"
        f"  Patient-split accuracy   : {m2['accuracy']*100:.2f}%\n"
        f"  Majority-class floor     : {m2['baseline_acc']*100:.2f}%  "
        f"(always predicting 'Normal')\n"
        f"  Leakage inflation        : {acc_gap*100:.2f} percentage points\n\n"
        f"  Patient-stratified accuracy must be read against the majority-class\n"
        f"  floor above, not against 0% — a model only beating that floor by a\n"
        f"  little is barely better than guessing 'Normal' every time.\n"
        f"{'='*65}\n"
    )
    return header + body + conclusion


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\nDevice : {DEVICE}")
    beats, labels, pids = load_data()
    print(f"Loaded  : {len(beats):,} beats | {len(np.unique(pids))} patients")
    print(f"Class balance: Normal {(labels==0).sum():,} "
          f"({(labels==0).mean()*100:.1f}%) | "
          f"Abnormal {(labels==1).sum():,} ({(labels==1).mean()*100:.1f}%)")

    # ── Experiment 1: Beat-level (leaky) ──────────────────────────────────────
    tr1, te1 = beat_level_split(labels)
    m1 = run_experiment(beats, labels, pids, tr1, te1,
                        tag="Experiment 1: Beat-level split (LEAKY)")

    # ── Experiment 2: Patient-stratified (correct) — this is the frozen ──────
    #    baseline model Stage 2's noise sweep will load.
    tr2, te2 = patient_stratified_split(labels, pids)
    model_path = os.path.join(MODEL_DIR, 'stage1_binary_patient_split.pt')
    m2 = run_experiment(beats, labels, pids, tr2, te2,
                        tag="Experiment 2: Patient-stratified split (CORRECT)",
                        save_model_path=model_path)

    comparison = build_comparison_text(m1, m2)
    print(comparison)

    print("\nGenerating figures...")
    fig_training_curves(m1, m2)
    fig_confusion_matrices(m1, m2)
    fig_roc_curves(m1, m2)
    fig_summary_comparison(m1, m2)
    print(f"\nAll figures saved to: {FIG_DIR}")

    out = os.path.join(RESULTS_DIR, 'stage1_binary_results.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("STAGE 1 (BINARY: Normal vs Abnormal) — ECG ARRHYTHMIA CLASSIFICATION\n")
        f.write("Noise Robustness and Generalization in ECG Signals\n")
        f.write("Galit Rothschild, Ariel University\n\n")
        f.write("EXPERIMENT 1: BEAT-LEVEL SPLIT (LEAKY)\n")
        f.write("="*60 + "\n")
        f.write(m1['report'] + "\n\n")
        f.write("EXPERIMENT 2: PATIENT-STRATIFIED SPLIT (CORRECT)\n")
        f.write("="*60 + "\n")
        f.write(m2['report'] + "\n\n")
        f.write(comparison)
    print(f"Text report saved -> {out}")
    print("\nStage 1 (binary) complete. Frozen model + test indices are ready for Stage 2.")


if __name__ == "__main__":
    main()
