"""
Downloads the PhysioNet Noise Stress Test Database (NSTDB) noise records:
    'bw' = baseline wander
    'ma' = muscle (EMG) artifact
    'em' = electrode motion artifact

All three are recorded at 360 Hz, two channels, same sampling rate as
MIT-BIH — so no resampling is needed to mix them with your extracted beats.

Saves one .npy per noise type (using channel 0), e.g.:
    data/nstdb_bw.npy
    data/nstdb_ma.npy
    data/nstdb_em.npy

Run this the same way you ran download_mitbih.py — from the project root,
inside your .venv, with internet access:

    python data/download_nstdb.py

NOTE: this requires reaching physionet.org, same as your MIT-BIH download.
If you're behind a restrictive network/proxy, use the same connection you
used for download_mitbih.py.
"""

import os
import numpy as np
import wfdb

DATA_DIR = os.path.dirname(__file__)

NSTDB_RECORDS = {
    'bw': 'bw',   # baseline wander
    'ma': 'ma',   # muscle artifact
    'em': 'em',   # electrode motion artifact
}


def download_noise_record(record_name):
    """Download one NSTDB record, return its first channel as a 1D array."""
    record = wfdb.rdrecord(record_name, pn_dir='nstdb')
    signal = record.p_signal[:, 0]
    return signal.astype(np.float32)


def main():
    print("Downloading PhysioNet Noise Stress Test Database (NSTDB)...")
    for tag, rec_name in NSTDB_RECORDS.items():
        print(f"  Fetching '{rec_name}'...")
        try:
            sig = download_noise_record(rec_name)
        except Exception as e:
            print(f"  Could not download {rec_name}: {e}")
            continue
        out_path = os.path.join(DATA_DIR, f'nstdb_{tag}.npy')
        np.save(out_path, sig)
        print(f"    {len(sig):,} samples ({len(sig)/360:.1f} s @ 360 Hz) -> {out_path}")

    print("\nDone. These three noise tracks + synthetic 50/60 Hz powerline")
    print("(generated on the fly, no download needed) are the 4 noise types")
    print("used by experiments/stage2_noise_sweep.py.")


if __name__ == "__main__":
    main()
