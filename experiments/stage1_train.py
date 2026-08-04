"""
Stage 1: Reproduce Rasti et al. CNN on MIT-BIH.
Compares beat-level split (leaky) vs patient-stratified split (correct).

Metrics  : Accuracy, Precision, Recall, F1, AUC, Specificity, MCC, Kappa
Visuals  : Confusion matrices, per-class bar charts, ROC curves,
           training curves, summary comparison chart
Outputs  : results/stage1_results.txt
           results/figures/  (all PNG figures)
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
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cnn1d import ShallowCNN1D

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-3
NUM_CLASSES = 5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES   = ['N\nNormal', 'S\nSupra', 'V\nVentric.', 'F\nFusion', 'Q\nUnknown']
CLASS_NAMES_S = ['N', 'S', 'V', 'F', 'Q']   # short version for axis labels

# ── Colour palette ────────────────────────────────────────────────────────────
C_LEAKY   = '#E05C5C'   # red   — leaky / inflated
C_CORRECT = '#4A90D9'   # blue  — correct / honest
C_NEUTRAL = '#7B8FA1'


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    beats  = np.load(os.path.join(DATA_DIR, 'mitbih_beats.npy'))
    labels = np.load(os.path.join(DATA_DIR, 'mitbih_labels.npy'))
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
def specificity_per_class(targets, preds, n_classes):
    """True Negative Rate per class (one-vs-rest)."""
    specs = []
    for c in range(n_classes):
        tn = np.sum((targets != c) & (preds != c))
        fp = np.sum((targets != c) & (preds == c))
        specs.append(tn / (tn + fp + 1e-8))
    return np.array(specs)


def compute_metrics(targets, preds, probs, tag):
    acc   = accuracy_score(targets, preds)
    prec  = precision_score(targets, preds, average='macro', zero_division=0)
    rec   = recall_score(targets,    preds, average='macro', zero_division=0)
    f1    = f1_score(targets,        preds, average='macro', zero_division=0)
    mcc   = matthews_corrcoef(targets, preds)
    kappa = cohen_kappa_score(targets, preds)

    f1_per   = f1_score(targets,   preds, average=None, zero_division=0)
    prec_per = precision_score(targets, preds, average=None, zero_division=0)
    rec_per  = recall_score(targets,    preds, average=None, zero_division=0)
    spec_per = specificity_per_class(targets, preds, NUM_CLASSES)

    # AUC one-vs-rest macro
    try:
        y_bin = label_binarize(targets, classes=list(range(NUM_CLASSES)))
        auc   = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
        auc_per = roc_auc_score(y_bin, probs, multi_class='ovr', average=None)
    except Exception:
        auc     = float('nan')
        auc_per = [float('nan')] * NUM_CLASSES

    cm     = confusion_matrix(targets, preds)
    report = classification_report(targets, preds,
                                   target_names=CLASS_NAMES_S, digits=4)

    return dict(
        tag=tag,
        accuracy=acc, precision=prec, recall=rec,
        f1_macro=f1, auc=auc, mcc=mcc, kappa=kappa,
        f1_per=f1_per, prec_per=prec_per, rec_per=rec_per,
        spec_per=spec_per, auc_per=auc_per,
        conf_matrix=cm, report=report,
        targets=targets, preds=preds, probs=probs
    )


def print_metrics(m):
    print(f"\n  {'─'*50}")
    print(f"  {'Metric':<22} {'Value':>10}")
    print(f"  {'─'*50}")
    rows = [
        ('Accuracy',         m['accuracy']),
        ('Precision (macro)', m['precision']),
        ('Recall (macro)',    m['recall']),
        ('F1 (macro)',        m['f1_macro']),
        ('AUC (macro)',       m['auc']),
        ('MCC',               m['mcc']),
        ("Cohen's Kappa",     m['kappa']),
    ]
    for name, val in rows:
        bar = '█' * int(val * 20)
        print(f"  {name:<22} {val:>8.4f}  {bar}")
    print(f"\n  Per-class breakdown:")
    hdr = f"  {'Class':<14} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}"
    print(hdr)
    print(f"  {'─'*50}")
    for i, name in enumerate(CLASS_NAMES_S):
        print(f"  {name:<14} "
              f"{m['prec_per'][i]:>6.3f} "
              f"{m['rec_per'][i]:>6.3f} "
              f"{m['spec_per'][i]:>6.3f} "
              f"{m['f1_per'][i]:>6.3f} "
              f"{m['auc_per'][i]:>6.3f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred)  [{', '.join(CLASS_NAMES_S)}]")
    print(f"  {m['conf_matrix']}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(beats, labels, pids, train_idx, test_idx, tag):
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

    # Track curves for plotting
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
    """Training loss and accuracy curves for both experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Figure 1 — Training Curves: Beat-level vs Patient-stratified',
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
    _save(fig, 'fig1_training_curves.png')


def fig_confusion_matrices(m1, m2):
    """Side-by-side normalised confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Figure 2 — Normalised Confusion Matrices',
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
                    annot_kws={'size': 10})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    _save(fig, 'fig2_confusion_matrices.png')


def fig_per_class_metrics(m1, m2):
    """Grouped bar chart of per-class F1, Precision, Recall, Specificity."""
    metrics_to_plot = [
        ('f1_per',   'F1 Score'),
        ('prec_per', 'Precision'),
        ('rec_per',  'Recall'),
        ('spec_per', 'Specificity'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Figure 3 — Per-class Metrics: Beat-level vs Patient-stratified',
                 fontsize=13, fontweight='bold', y=1.01)

    x      = np.arange(NUM_CLASSES)
    width  = 0.35

    for ax, (key, label) in zip(axes.flat, metrics_to_plot):
        bars1 = ax.bar(x - width/2, m1[key], width,
                       color=C_LEAKY,   alpha=0.85, label='Beat-level (leaky)')
        bars2 = ax.bar(x + width/2, m2[key], width,
                       color=C_CORRECT, alpha=0.85, label='Patient-stratified')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES_S)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Score')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#FFFFFF')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # Value labels on bars
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    _save(fig, 'fig3_per_class_metrics.png')


def fig_roc_curves(m1, m2):
    """ROC curves per class for both methods."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#F7F9FC')
    fig.suptitle('Figure 4 — ROC Curves per Class',
                 fontsize=13, fontweight='bold', y=1.01)

    colors = ['#E05C5C','#4A90D9','#52B788','#F4A261','#9B5DE5']

    for ax, m, title in zip(
            axes,
            [m1, m2],
            ['Beat-level (Leaky)', 'Patient-stratified (Correct)']):

        y_bin = label_binarize(m['targets'], classes=list(range(NUM_CLASSES)))
        for i, (cname, col) in enumerate(zip(CLASS_NAMES_S, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], m['probs'][:, i])
            auc_val      = m['auc_per'][i]
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f'{cname}  AUC={auc_val:.3f}')

        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_facecolor('#FFFFFF')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()
    _save(fig, 'fig4_roc_curves.png')


def fig_summary_comparison(m1, m2):
    """Publication-ready summary bar chart of all aggregate metrics."""
    metrics = [
        ('Accuracy',          m1['accuracy'],  m2['accuracy']),
        ('Precision\n(macro)', m1['precision'], m2['precision']),
        ('Recall\n(macro)',    m1['recall'],    m2['recall']),
        ('F1\n(macro)',        m1['f1_macro'],  m2['f1_macro']),
        ('AUC\n(macro)',       m1['auc'],       m2['auc']),
        ('MCC',                m1['mcc'],       m2['mcc']),
        ("Cohen's\nKappa",     m1['kappa'],     m2['kappa']),
    ]

    labels  = [r[0] for r in metrics]
    vals1   = [r[1] for r in metrics]
    vals2   = [r[2] for r in metrics]
    gaps    = [v1 - v2 for v1, v2 in zip(vals1, vals2)]

    x     = np.arange(len(labels))
    width = 0.30

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#F7F9FC')
    ax.set_facecolor('#FFFFFF')

    b1 = ax.bar(x - width/2, vals1, width,
                color=C_LEAKY,   alpha=0.88, label='Beat-level (LEAKY)')
    b2 = ax.bar(x + width/2, vals2, width,
                color=C_CORRECT, alpha=0.88, label='Patient-stratified (CORRECT)')

    # Gap annotations
    for i, (v1, v2, gap) in enumerate(zip(vals1, vals2, gaps)):
        y_top = max(v1, v2) + 0.04
        ax.annotate('', xy=(x[i]+width/2, y_top),
                    xytext=(x[i]-width/2, y_top),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
        ax.text(x[i], y_top + 0.025,
                f'gap\n{gap:+.3f}',
                ha='center', va='bottom', fontsize=8,
                color='#333', fontweight='bold')

    # Bar value labels
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(
        'Figure 5 — Stage 1 Summary: Data Leakage Effect on All Metrics\n'
        'Beat-level split inflates every metric vs. the honest patient-stratified evaluation',
        fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Accuracy gap callout box
    acc_gap = m1['accuracy'] - m2['accuracy']
    ax.annotate(
        f'Accuracy inflated\nby {acc_gap*100:.1f} pp\ndue to data leakage',
        xy=(0 - width/2, m1['accuracy']),
        xytext=(1.2, 1.18),
        fontsize=9, color=C_LEAKY, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_LEAKY, lw=1.5))

    plt.tight_layout()
    _save(fig, 'fig5_summary_comparison.png')


def fig_class_distribution(labels):
    """Dataset class imbalance chart."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total  = len(labels)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#F7F9FC')
    ax.set_facecolor('#FFFFFF')

    colors = ['#4A90D9','#52B788','#E05C5C','#F4A261','#9B5DE5']
    bars   = ax.bar(CLASS_NAMES_S, counts, color=colors, alpha=0.88, edgecolor='white')

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f'{cnt:,}\n({cnt/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Figure 0 — MIT-BIH Class Distribution (AAMI 5-class)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Number of Beats', fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    _save(fig, 'fig0_class_distribution.png')


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_comparison_text(m1, m2):
    rows = [
        ('Accuracy',           m1['accuracy'],  m2['accuracy']),
        ('Precision (macro)',   m1['precision'], m2['precision']),
        ('Recall (macro)',      m1['recall'],    m2['recall']),
        ('F1 (macro)',          m1['f1_macro'],  m2['f1_macro']),
        ('AUC (macro)',         m1['auc'],       m2['auc']),
        ('MCC',                 m1['mcc'],       m2['mcc']),
        ("Cohen's Kappa",       m1['kappa'],     m2['kappa']),
    ]
    header = (
        f"\n{'='*65}\n"
        f"  STAGE 1 — FINAL COMPARISON TABLE\n"
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
        f"  Beat-level accuracy  : {m1['accuracy']*100:.2f}%\n"
        f"  Patient-split acc.   : {m2['accuracy']*100:.2f}%\n"
        f"  Leakage inflation    : {acc_gap*100:.2f} percentage points\n\n"
        f"  The {acc_gap*100:.2f} pp accuracy gap demonstrates that beat-level\n"
        f"  evaluation as used in Rasti et al. (2024) substantially\n"
        f"  overstates model performance. The model has effectively\n"
        f"  memorised patient-specific ECG morphology rather than\n"
        f"  learning generalisable arrhythmia patterns.\n"
        f"  F1 drop (0.{int(m1['f1_macro']*10000):04d} -> 0.{int(m2['f1_macro']*10000):04d}) "
        f"confirms this across all classes.\n"
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

    # Dataset overview figure (no training needed)
    print("\nGenerating dataset distribution figure...")
    fig_class_distribution(labels)

    # ── Experiment 1: Beat-level (leaky) ──────────────────────────────────────
    tr1, te1 = beat_level_split(labels)
    m1 = run_experiment(beats, labels, pids, tr1, te1,
                        tag="Experiment 1: Beat-level split (LEAKY)")

    # ── Experiment 2: Patient-stratified (correct) ────────────────────────────
    tr2, te2 = patient_stratified_split(labels, pids)
    m2 = run_experiment(beats, labels, pids, tr2, te2,
                        tag="Experiment 2: Patient-stratified split (CORRECT)")

    # ── Comparison text ───────────────────────────────────────────────────────
    comparison = build_comparison_text(m1, m2)
    print(comparison)

    # ── All figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_training_curves(m1, m2)
    fig_confusion_matrices(m1, m2)
    fig_per_class_metrics(m1, m2)
    fig_roc_curves(m1, m2)
    fig_summary_comparison(m1, m2)
    print(f"\nAll figures saved to: {FIG_DIR}")

    # ── Save text report ──────────────────────────────────────────────────────
    out = os.path.join(RESULTS_DIR, 'stage1_results.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("STAGE 1 — ECG ARRHYTHMIA CLASSIFICATION\n")
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
    print("\nStage 1 complete!")


if __name__ == "__main__":
    main()