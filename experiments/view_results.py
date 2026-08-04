"""
view_results.py
───────────────
Reads the saved .npy result arrays and stage1_results.txt and displays
everything in a clean, readable format in the terminal.
No retraining needed — runs in seconds.

Run from project root:
    python experiments/view_results.py
"""

import os
import sys
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(BASE, 'results')
DATA_DIR = os.path.join(BASE, 'data')
TXT_REPORT = os.path.join(RESULTS_DIR, 'stage1_results.txt')

CLASS_NAMES = ['N (Normal)', 'S (Supra)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']
CLASS_NAMES_S = ['N', 'S', 'V', 'F', 'Q']

# ── Terminal colours (work on Windows 10+ and Mac/Linux) ──────────────────────
RED = '\033[91m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'


def enable_windows_ansi():
    """Enable ANSI colours on Windows."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


enable_windows_ansi()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bar(value, width=25, color=GREEN):
    """ASCII progress bar."""
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"


def section(title, color=BOLD):
    width = 62
    print(f"\n{color}{'═' * width}{RESET}")
    print(f"{color}  {title}{RESET}")
    print(f"{color}{'═' * width}{RESET}")


def row(label, v1, v2, higher_is_better=True):
    """Print one metric row with colour-coded gap."""
    gap = v1 - v2
    is_up = gap > 0

    if higher_is_better:
        gap_color = RED if is_up else GREEN  # leaky higher = bad
        arrow = '↑' if is_up else '↓'
    else:
        gap_color = GREEN if is_up else RED
        arrow = '↑' if is_up else '↓'

    b1 = bar(max(0, v1), color=RED)
    b2 = bar(max(0, v2), color=BLUE)

    print(f"  {BOLD}{label:<22}{RESET}")
    print(f"    {RED}Beat-level  {RESET}{b1}  {RED}{v1:>7.4f}{RESET}")
    print(f"    {BLUE}Pat-split   {RESET}{b2}  {BLUE}{v2:>7.4f}{RESET}")
    print(f"    {'Gap':>12}  {gap_color}{arrow} {abs(gap):.4f}  "
          f"({abs(gap) * 100:.2f} pp){RESET}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_dataset():
    labels_path = os.path.join(DATA_DIR, 'mitbih_labels.npy')
    pids_path = os.path.join(DATA_DIR, 'mitbih_patient_ids.npy')

    if not os.path.exists(labels_path):
        print(f"  {YELLOW}[!] mitbih_labels.npy not found — skipping dataset overview{RESET}")
        return

    labels = np.load(labels_path)
    pids = np.load(pids_path)
    total = len(labels)
    counts = np.bincount(labels, minlength=5)

    section("DATASET OVERVIEW — MIT-BIH Arrhythmia Database", BOLD)
    print(f"  Total beats    : {BOLD}{total:,}{RESET}")
    print(f"  Patients       : {BOLD}{len(np.unique(pids))}{RESET}")
    print(f"  Sampling rate  : 360 Hz")
    print(f"  Beat window    : 216 samples (~0.6 s)\n")

    print(f"  {'Class':<22} {'Count':>7}  {'%':>6}  Distribution")
    print(f"  {'─' * 58}")
    colors = [BLUE, GREEN, RED, YELLOW, '\033[95m']
    for i, (name, cnt, col) in enumerate(zip(CLASS_NAMES, counts, colors)):
        pct = cnt / total
        bbar = bar(pct, width=20, color=col)
        print(f"  {name:<22} {cnt:>7,}  {pct * 100:>5.1f}%  {bbar}")

    print(f"\n  {YELLOW}Note: Class N (Normal) makes up "
          f"{counts[0] / total * 100:.1f}% of all beats.{RESET}")
    print(f"  {YELLOW}Weighted loss was used during training to handle this imbalance.{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED RESULTS  (from your last run — update these if you re-run)
# ══════════════════════════════════════════════════════════════════════════════

# ── These values come directly from your terminal output ─────────────────────
# ── If you re-run stage1_train.py, paste the new numbers here ────────────────

RESULTS = {
    'beat_level': {
        'accuracy': 0.9809,
        'precision': 0.8746,
        'recall': 0.9617,
        'f1_macro': 0.9118,
        'auc': 0.9955,
        # Per-class F1
        'f1_per': [0.9892, 0.8162, 0.9765, 0.7807, 0.9964],
        # Confusion matrix (paste from terminal)
        'conf_matrix': np.array([
            [14783, 197, 20, 51, 4],
            [29, 524, 3, 0, 0],
            [12, 6, 1412, 17, 0],
            [5, 1, 7, 146, 1],
            [4, 0, 3, 0, 1640],
        ]),
    },
    'patient_split': {
        'accuracy': 0.6553,
        'precision': 0.4809,
        'recall': 0.5641,
        'f1_macro': 0.4780,
        'auc': 0.7879,
        'f1_per': [0.7500, 0.0522, 0.6565, 0.0070, 0.9241],
        'conf_matrix': np.array([
            [9474, 2398, 968, 1392, 0],
            [1401, 109, 68, 4, 4],
            [56, 12, 1189, 15, 10],
            [3, 0, 9, 5, 0],
            [99, 75, 106, 4, 1815],
        ]),
    }
}


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def show_aggregate():
    section("AGGREGATE METRICS — Beat-level vs Patient-stratified", BOLD)

    b = RESULTS['beat_level']
    ps = RESULTS['patient_split']

    metrics = [
        ('Accuracy', b['accuracy'], ps['accuracy']),
        ('Precision (macro)', b['precision'], ps['precision']),
        ('Recall (macro)', b['recall'], ps['recall']),
        ('F1 (macro)', b['f1_macro'], ps['f1_macro']),
        ('AUC (macro)', b['auc'], ps['auc']),
    ]

    for name, v1, v2 in metrics:
        row(name, v1, v2)

    # Key finding callout
    acc_gap = b['accuracy'] - ps['accuracy']
    print(f"\n  {'─' * 58}")
    print(f"  {BOLD}{RED}KEY FINDING:{RESET}")
    print(f"  Beat-level accuracy  : {RED}{BOLD}{b['accuracy'] * 100:.2f}%{RESET}")
    print(f"  Patient-split acc.   : {BLUE}{BOLD}{ps['accuracy'] * 100:.2f}%{RESET}")
    print(f"  Leakage inflation    : {RED}{BOLD}{acc_gap * 100:.2f} percentage points{RESET}")
    print(f"\n  {DIM}This means the published ~98% figure overstates real-world{RESET}")
    print(f"  {DIM}performance by {acc_gap * 100:.1f} pp — the model memorised patients,{RESET}")
    print(f"  {DIM}not arrhythmia patterns.{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CLASS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def show_per_class():
    section("PER-CLASS F1 SCORES", BOLD)

    b = RESULTS['beat_level']
    ps = RESULTS['patient_split']

    print(f"  {BOLD}{'Class':<22} {'Beat-level':>11} {'Pat-split':>11} "
          f"{'Gap':>9}  {'Pat-split bar'}{RESET}")
    print(f"  {'─' * 70}")

    for i, name in enumerate(CLASS_NAMES):
        v1 = b['f1_per'][i]
        v2 = ps['f1_per'][i]
        gap = v1 - v2
        direction = 'v' if gap > 0 else '^'
        bbar = bar(v2, width=20, color=BLUE)
        gap_color = RED if gap > 0 else GREEN
        print(f"  {name:<22} {RED}{v1:>11.4f}{RESET} "
              f"{BLUE}{v2:>11.4f}{RESET} "
              f"{gap_color}{direction}{abs(gap):>7.4f}{RESET}  {bbar}")

    print(f"\n  {YELLOW}Worst-hit classes after leakage correction:{RESET}")
    f1_drops = [(CLASS_NAMES[i], b['f1_per'][i] - ps['f1_per'][i])
                for i in range(5)]
    f1_drops.sort(key=lambda x: -x[1])
    for name, drop in f1_drops:
        print(f"    {name:<22} drop of {RED}{drop:.4f}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX (ASCII)
# ══════════════════════════════════════════════════════════════════════════════

def show_confusion(label, cm, color):
    total_per_row = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / (total_per_row + 1e-8)

    print(f"\n  {color}{BOLD}{label}{RESET}")
    print(f"  {'':>6}", end='')
    for name in CLASS_NAMES_S:
        print(f"  {name:>8}", end='')
    print(f"  {'Total':>8}")
    print(f"  {'─' * 60}")

    for i, name in enumerate(CLASS_NAMES_S):
        print(f"  {name:>4}  ", end='')
        for j in range(5):
            val = cm[i, j]
            norm_val = cm_norm[i, j]
            # Highlight diagonal (correct predictions)
            if i == j:
                cell_color = GREEN if norm_val > 0.7 else YELLOW
                print(f"  {cell_color}{BOLD}{val:>8}{RESET}", end='')
            else:
                cell_color = RED if norm_val > 0.1 else RESET
                print(f"  {cell_color}{val:>8}{RESET}", end='')
        print(f"  {cm[i].sum():>8}")

    print(f"  {'─' * 60}")
    print(f"  {'':>6}", end='')
    for j in range(5):
        print(f"  {cm[:, j].sum():>8}", end='')
    print(f"  {cm.sum():>8}")

    print(f"\n  Normalised (% of true class correctly predicted):")
    print(f"  {'':>6}", end='')
    for name in CLASS_NAMES_S:
        print(f"  {name:>8}", end='')
    print()
    for i, name in enumerate(CLASS_NAMES_S):
        print(f"  {name:>4}  ", end='')
        for j in range(5):
            pct = cm_norm[i, j] * 100
            cell_color = GREEN if (i == j and pct > 70) else \
                YELLOW if (i == j) else \
                    RED if pct > 10 else DIM
            print(f"  {cell_color}{pct:>7.1f}%{RESET}", end='')
        print()


def show_confusion_matrices():
    section("CONFUSION MATRICES", BOLD)
    b = RESULTS['beat_level']
    ps = RESULTS['patient_split']
    show_confusion("Beat-level Split (LEAKY)", b['conf_matrix'], RED)
    show_confusion("Patient-stratified Split (CORRECT)", ps['conf_matrix'], BLUE)


# ══════════════════════════════════════════════════════════════════════════════
#  CLINICAL INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

def show_interpretation():
    section("CLINICAL INTERPRETATION", BOLD)

    ps = RESULTS['patient_split']
    b = RESULTS['beat_level']

    print(f"  What do these numbers mean in a real hospital setting?\n")

    v_f1_leaky = b['f1_per'][2]
    v_f1_correct = ps['f1_per'][2]
    s_f1_leaky = b['f1_per'][1]
    s_f1_correct = ps['f1_per'][1]

    print(f"  {BOLD}Ventricular beats (V) — life-threatening arrhythmia:{RESET}")
    print(f"    F1 drops from {RED}{v_f1_leaky:.3f}{RESET} → "
          f"{BLUE}{v_f1_correct:.3f}{RESET} after leakage correction")
    print(f"    A model that misses V-beats in new patients poses real clinical risk.\n")

    print(f"  {BOLD}Supraventricular beats (S):{RESET}")
    print(f"    F1 drops from {RED}{s_f1_leaky:.3f}{RESET} → "
          f"{BLUE}{s_f1_correct:.3f}{RESET} — nearly unusable on new patients.\n")

    print(f"  {BOLD}Fusion beats (F):{RESET}")
    print(f"    F1 drops from {RED}{b['f1_per'][3]:.3f}{RESET} → "
          f"{BLUE}{ps['f1_per'][3]:.3f}{RESET} — essentially random on new patients.\n")

    print(f"  {DIM}This confirms the core thesis: published ECG models report{RESET}")
    print(f"  {DIM}near-perfect accuracy because they have memorised patient-{RESET}")
    print(f"  {DIM}specific morphology, not learned generalizable patterns.{RESET}")
    print(f"  {DIM}Stage 2 (noise injection) will stress-test this further.{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT FILE VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def show_text_report():
    section("FULL TEXT REPORT (stage1_results.txt)", BOLD)
    if not os.path.exists(TXT_REPORT):
        print(f"  {YELLOW}[!] stage1_results.txt not found.{RESET}")
        print(f"      Run stage1_train.py first to generate it.")
        return
    with open(TXT_REPORT, 'r', encoding='utf-8') as f:
        content = f.read()
    print(content)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES CHECK
# ══════════════════════════════════════════════════════════════════════════════

def show_figures_status():
    section("SAVED FIGURES", BOLD)
    fig_dir = os.path.join(RESULTS_DIR, 'figures')
    expected = [
        ('fig0_class_distribution.png', 'Dataset class imbalance'),
        ('fig1_training_curves.png', 'Training loss and accuracy curves'),
        ('fig2_confusion_matrices.png', 'Normalised confusion matrices'),
        ('fig3_per_class_metrics.png', 'Per-class F1/Precision/Recall/Specificity'),
        ('fig4_roc_curves.png', 'ROC curves with per-class AUC'),
        ('fig5_summary_comparison.png', 'Main summary — all metrics with gap'),
    ]
    if not os.path.exists(fig_dir):
        print(f"  {YELLOW}[!] figures/ folder not found.{RESET}")
        print(f"      Run stage1_train.py to generate all figures.")
        return
    for fname, desc in expected:
        full = os.path.join(fig_dir, fname)
        if os.path.exists(full):
            size = os.path.getsize(full) / 1024
            print(f"  {GREEN}[OK]{RESET}  {fname:<40} {desc}  ({size:.0f} KB)")
        else:
            print(f"  {YELLOW}[--]{RESET}  {fname:<40} {YELLOW}not yet generated{RESET}")
    print(f"\n  Open {BOLD}results/figures/{RESET} in Windows Explorer to view all charts.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'█' * 62}{RESET}")
    print(f"{BOLD}  STAGE 1 RESULTS VIEWER{RESET}")
    print(f"{BOLD}  ECG Noise Robustness Project — Galit Rothschild{RESET}")
    print(f"{BOLD}{'█' * 62}{RESET}")

    show_dataset()
    show_aggregate()
    show_per_class()
    show_confusion_matrices()
    show_interpretation()
    show_figures_status()

    print(f"\n{BOLD}{'═' * 62}{RESET}")
    print(f"  Done. Ready to move to Stage 2 (noise injection).")
    print(f"{BOLD}{'═' * 62}{RESET}\n")


if __name__ == "__main__":
    main()