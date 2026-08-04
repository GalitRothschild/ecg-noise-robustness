# ECG Noise Robustness and Generalization

Final year B.Sc. Electrical and Electronic Engineering project, Ariel University. Author: Galit Rothschild.

This project reproduces a published shallow 1D CNN arrhythmia classifier on the MIT-BIH Arrhythmia Database, corrects its evaluation methodology using patient-stratified splitting instead of the leaky beat-level split common in the literature, and quantifies the resulting accuracy gap. The corrected model is then stress-tested under four realistic noise types from the PhysioNet Noise Stress Test Database, and two countermeasures are evaluated: signal preprocessing and noise-augmented training.

## Repository structure

data/ contains dataset download and label-preparation scripts (MIT-BIH, NSTDB). models/ contains the CNN model definition (cnn1d.py) and trained checkpoints. noise/ contains noise injection and SNR scaling utilities. preprocessing/ contains the bandpass, baseline-wander, notch filter, and normalization pipeline. experiments/ contains the four experiment stages: methodology correction, noise sweep, preprocessing ablation, and noise-augmented training, plus results plotting. results/ contains numerical results (txt/csv) and generated figures under results/figures/.

## Key results

Patient-stratified evaluation drops accuracy from 98.18% (leaky beat-level) to 76.02% (5-class), and from 99.00% to 75.14% (binary), confirming substantial data leakage in the standard evaluation protocol. Powerline interference collapses model sensitivity to 0.00 at 0 dB SNR; a targeted 60 Hz notch filter fully recovers it. Muscle artifact noise cannot be removed by filtering, since its spectrum overlaps the ECG signal itself; noise-augmented training recovers performance where preprocessing cannot.

See the full final report for methodology, literature review, and discussion.
