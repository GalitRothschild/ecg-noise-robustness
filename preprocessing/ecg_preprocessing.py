"""
Stage 3: preprocessing techniques, applied to noisy beats before they reach
the frozen model. Each one is testable independently and in combination,
matching the project proposal's ablation design (no preprocessing, each
technique alone, full pipeline).

Techniques:
    bandpass_filter        0.5-40 Hz Butterworth, zero-phase (proposal spec)
    remove_baseline_wander per-beat linear detrend (see note below)
    normalize_amplitude    re-apply z-score normalization AFTER noise is
                            added (the original clean beats were normalized
                            BEFORE noise injection in Stage 2, so the noisy
                            mixture's mean/std has drifted from 0/1 — this
                            re-centers and re-scales it)
    notch_60hz              notch at 60 Hz, added specifically because
                            Stage 2 found powerline interference to be by far
                            the most damaging noise type for this model —
                            this directly tests whether a targeted fix
                            resolves that specific failure mode

Note on baseline wander removal at the beat level: classic BW removal
(e.g. the two-pass median filter method) is designed for continuous,
multi-second recordings. Your beats are already isolated ~0.6 s windows
(216 samples), so the low-frequency drift that BW represents barely
completes a quarter-cycle within a single beat window. A per-beat linear
detrend (subtract the best-fit line) is the standard, defensible way to
handle residual drift at this scale — it removes a straight-line trend
without needing a multi-second context the beat window doesn't have.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

FS_DEFAULT = 360.0
NOTCH_QUALITY_DEFAULT = 2.0   # see notch_60hz docstring for why not Q=30


def bandpass_filter(beats, fs=FS_DEFAULT, low=0.5, high=40.0, order=2):
    """
    Zero-phase Butterworth bandpass, matching the project proposal's
    0.5-40 Hz spec. Operates on a batch of beats, shape (N, beat_len).

    order=2 (not higher) is deliberate: with only 216 samples per beat,
    a higher-order filter's required padding (padlen) can exceed the
    signal length and raise an error. Order 2 is standard for short
    ECG segments and still gives a clean roll-off.
    """
    beats = np.atleast_2d(np.asarray(beats, dtype=np.float64))
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')
    return sosfiltfilt(sos, beats, axis=-1)


def notch_60hz(beats, fs=FS_DEFAULT, freq=60.0, quality=None):
    """
    Notch filter at 60 Hz (US mains — see stage2_noise_sweep.py for why
    60 Hz and not 50 Hz for this dataset). Added specifically to test
    whether it fixes the powerline-driven sensitivity collapse found in
    Stage 2, beyond what the 0.5-40 Hz bandpass alone removes.

    quality defaults to 2.0 (NOT the textbook-narrow Q=30): a high-Q
    notch has a long impulse response and never fully settles within a
    216-sample (0.6 s) beat window, so it barely attenuates the tone in
    practice. Tested empirically on a 216-sample synthetic 60 Hz tone:
    Q=30 left ~36% of the 60 Hz band energy after filtfilt, Q=2 left
    ~9%. The wider notch removes more of the actual interference at the
    cost of also touching a slightly broader band around 60 Hz, an
    acceptable trade-off since that band is well above where QRS energy
    concentrates.
    """
    if quality is None:
        quality = NOTCH_QUALITY_DEFAULT
    beats = np.atleast_2d(np.asarray(beats, dtype=np.float64))
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, beats, axis=-1)


def remove_baseline_wander(beats):
    """
    Per-beat linear detrend: fit and subtract a degree-1 polynomial from
    each beat. See module docstring for why this (not the classic
    multi-second median-filter method) is the right scale for isolated
    216-sample beat windows.
    """
    beats = np.atleast_2d(np.asarray(beats, dtype=np.float64))
    n_beats, beat_len = beats.shape
    x = np.arange(beat_len)
    out = np.empty_like(beats)
    for i in range(n_beats):
        coeffs = np.polyfit(x, beats[i], deg=1)
        trend = np.polyval(coeffs, x)
        out[i] = beats[i] - trend
    return out


def normalize_amplitude(beats):
    """Re-apply per-beat z-score normalization (zero mean, unit variance)."""
    beats = np.atleast_2d(np.asarray(beats, dtype=np.float64))
    mean = beats.mean(axis=-1, keepdims=True)
    std  = beats.std(axis=-1, keepdims=True)
    return (beats - mean) / (std + 1e-8)


# Named pipelines for the ablation study: no-op, each technique alone,
# and the full stack. Order matters for the full pipeline: detrend first
# (removes slow drift), then bandpass (removes remaining out-of-band
# content including most of the 60 Hz energy), then notch (mops up
# whatever 60 Hz the bandpass didn't fully attenuate), then normalize last
# (rescales whatever is left to a consistent amplitude for the model).
PIPELINES = {
    'none':            lambda beats, fs: beats,
    'bandpass_only':   lambda beats, fs: bandpass_filter(beats, fs),
    'baseline_only':   lambda beats, fs: remove_baseline_wander(beats),
    'normalize_only':  lambda beats, fs: normalize_amplitude(beats),
    'notch_only':      lambda beats, fs: notch_60hz(beats, fs),
    'full_pipeline':   lambda beats, fs: normalize_amplitude(
                            notch_60hz(
                                bandpass_filter(
                                    remove_baseline_wander(beats), fs),
                                fs)),
}
