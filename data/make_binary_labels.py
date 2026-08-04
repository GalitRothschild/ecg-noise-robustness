"""
Derives binary (Normal / Abnormal) labels from the existing 5-class
AAMI labels, without re-downloading or re-extracting beats.

Mapping:
    N (0)                -> 0  Normal
    S, V, F, Q (1,2,3,4)  -> 1  Abnormal

This matches the original Rasti et al. (2024) task definition
(normal vs. abnormal), unlike the 5-class AAMI scheme used in the
Stage 1 exploratory run.

Reads : data/mitbih_labels.npy         (5-class, already extracted)
Writes: data/mitbih_labels_binary.npy  (binary)

Beats (mitbih_beats.npy) and patient IDs (mitbih_patient_ids.npy)
are reused as-is — only the label array changes.
"""

import os
import numpy as np

DATA_DIR = os.path.dirname(__file__)

CLASS_NAMES_5 = ['N', 'S', 'V', 'F', 'Q']


def main():
    labels_path = os.path.join(DATA_DIR, 'mitbih_labels.npy')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"{labels_path} not found. Run download_mitbih.py first."
        )

    labels_5class = np.load(labels_path)

    # 0 (N) stays Normal; everything else (S=1, V=2, F=3, Q=4) becomes Abnormal
    labels_binary = (labels_5class != 0).astype(np.int64)

    out_path = os.path.join(DATA_DIR, 'mitbih_labels_binary.npy')
    np.save(out_path, labels_binary)

    total = len(labels_binary)
    n_normal = int((labels_binary == 0).sum())
    n_abnormal = int((labels_binary == 1).sum())

    print("Binary label conversion complete.")
    print(f"  Total beats     : {total:,}")
    print(f"  Normal   (0)    : {n_normal:,}  ({n_normal/total*100:.1f}%)")
    print(f"  Abnormal (1)    : {n_abnormal:,}  ({n_abnormal/total*100:.1f}%)")
    print()
    print("  5-class breakdown folded into 'Abnormal':")
    counts_5 = np.bincount(labels_5class, minlength=5)
    for name, cnt in zip(CLASS_NAMES_5, counts_5):
        role = 'Normal' if name == 'N' else 'Abnormal'
        print(f"    {name} ({role:<8}): {cnt:>6,}  ({cnt/total*100:.1f}%)")
    print()
    print(f"  Majority-class ('always predict Normal') baseline accuracy: "
          f"{n_normal/total*100:.2f}%")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
