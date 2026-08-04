"""
Downloads MIT-BIH Arrhythmia Database via wfdb and extracts
labelled ECG beats with their patient (record) IDs.
Saves: data/mitbih_beats.npy, data/mitbih_labels.npy, data/mitbih_patient_ids.npy
"""

import os
import numpy as np
import wfdb
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
BEAT_LEN    = 216          # samples around each beat (at 360 Hz ≈ 0.6 s)
BEFORE_PEAK = 90           # samples before R-peak
AFTER_PEAK  = BEAT_LEN - BEFORE_PEAK

# AAMI standard 5-class mapping
# N=Normal, S=Supraventricular, V=Ventricular, F=Fusion, Q=Unknown/Paced
LABEL_MAP = {
    # N
    'N': 0, '.': 0, 'e': 0, 'j': 0,
    # S
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # V
    'V': 2, 'E': 2,
    # F
    'F': 3,
    # Q
    'Q': 4, '/': 4, 'f': 4, 'x': 4
}

MIT_BIH_RECORDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

DATA_DIR = os.path.join(os.path.dirname(__file__))   # saves inside data/


def extract_beats(record_name):
    """Download one record and return (beats, labels, patient_ids)."""
    try:
        record  = wfdb.rdrecord(record_name,  pn_dir='mitdb')
        annot   = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
    except Exception as e:
        print(f"  Could not load {record_name}: {e}")
        return [], [], []

    signal = record.p_signal[:, 0]   # use lead I (MLII)
    beats, labels, pids = [], [], []

    for peak, symbol in zip(annot.sample, annot.symbol):
        if symbol not in LABEL_MAP:
            continue                  # skip non-beat annotations
        start = peak - BEFORE_PEAK
        end   = peak + AFTER_PEAK
        if start < 0 or end > len(signal):
            continue                  # skip edge beats
        beat = signal[start:end]
        # Normalise each beat to zero-mean, unit-variance
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        beats.append(beat)
        labels.append(LABEL_MAP[symbol])
        pids.append(int(record_name))

    return beats, labels, pids


def main():
    all_beats, all_labels, all_pids = [], [], []

    print("Downloading and extracting MIT-BIH beats...")
    for rec in tqdm(MIT_BIH_RECORDS):
        b, l, p = extract_beats(rec)
        all_beats.extend(b)
        all_labels.extend(l)
        all_pids.extend(p)

    beats  = np.array(all_beats,  dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    pids   = np.array(all_pids,   dtype=np.int64)

    np.save(os.path.join(DATA_DIR, 'mitbih_beats.npy'),      beats)
    np.save(os.path.join(DATA_DIR, 'mitbih_labels.npy'),     labels)
    np.save(os.path.join(DATA_DIR, 'mitbih_patient_ids.npy'), pids)

    print(f"\nDone! Total beats: {len(beats)}")
    print(f"Label distribution: { {i: int((labels==i).sum()) for i in range(5)} }")
    print(f"Unique patients: {len(np.unique(pids))}")
    print(f"Saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()