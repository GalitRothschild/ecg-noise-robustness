"""
Noise injection utilities for Stage 2.

Core idea: given a clean ECG beat and a noise segment (real, recorded
NSTDB noise, or synthetic powerline interference), scale the noise so the
resulting mixture has a specified signal-to-noise ratio (SNR, in dB), then
add it. This is the standard approach used across the ECG-denoising
literature (e.g. the original NSTDB papers, and most CNN noise-robustness
studies) — it lets you ask "how does accuracy change as SNR drops from
24 dB (near-clean) to 0 dB (noise power = signal power)?" in a controlled,
comparable way across noise types.

No dependency on wfdb/scipy here — this module is pure numpy so it can be
unit-tested and imported anywhere. The NSTDB recordings themselves are
fetched separately by data/download_nstdb.py.
"""

import numpy as np

SNR_LEVELS_DB = [24, 18, 12, 6, 0]   # matches the project proposal


def signal_power(x):
    """Mean squared amplitude (power) of a 1D signal."""
    x = np.asarray(x, dtype=np.float64)
    return np.mean(x ** 2)


def scale_noise_to_snr(clean, noise, snr_db):
    """
    Return `noise` rescaled so that mixing it with `clean` produces the
    requested SNR (in dB), where

        SNR_dB = 10 * log10( power(clean) / power(scaled_noise) )

    clean, noise : 1D arrays of the SAME LENGTH.
    snr_db       : target SNR in dB. Lower = noisier. 0 dB means the
                   noise has equal power to the clean signal.

    Returns the scaled noise array (same shape as input).
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if clean.shape != noise.shape:
        raise ValueError(
            f"clean and noise must be the same length, got "
            f"{clean.shape} vs {noise.shape}"
        )

    p_clean = signal_power(clean)
    p_noise = signal_power(noise)
    if p_noise < 1e-12:
        raise ValueError("noise segment has ~zero power, cannot scale to a target SNR")

    # Desired noise power such that 10*log10(p_clean / p_noise_target) = snr_db
    target_noise_power = p_clean / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / p_noise)
    return noise * scale


def add_noise_at_snr(clean, noise, snr_db):
    """Scale `noise` to hit `snr_db` against `clean`, then return clean+noise."""
    scaled = scale_noise_to_snr(clean, noise, snr_db)
    return clean + scaled


def combine_noise_segments(segments):
    """
    Sum multiple RAW (unscaled) noise segments into one combined noise
    signal, preserving their natural relative proportions.

    IMPORTANT: combine first, scale once. If you instead scaled each
    noise source to the target SNR individually and then summed the
    scaled results, the combined mixture's total power would be roughly
    N times higher than intended (noise powers add), so a "6 dB combined"
    condition built that way would actually be much noisier than 6 dB.
    Combining raw segments first and calling add_noise_at_snr() once on
    the sum keeps "6 dB" meaning the same thing whether it's 1 noise
    source or 4.
    """
    segments = [np.asarray(s, dtype=np.float64) for s in segments]
    lengths = {len(s) for s in segments}
    if len(lengths) != 1:
        raise ValueError(f"all segments must be the same length, got lengths {lengths}")
    return np.sum(segments, axis=0)


def random_noise_segment(noise_track, length, rng=None):
    """
    Pull a random contiguous `length`-sample segment out of a longer 1D
    noise recording (e.g. a full NSTDB channel), with wraparound so it
    always succeeds even if `length` is a decent fraction of the track.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_track = np.asarray(noise_track, dtype=np.float64)
    n = len(noise_track)
    if n == 0:
        raise ValueError("noise_track is empty")
    if length <= n:
        start = rng.integers(0, n - length + 1)
        return noise_track[start:start + length]
    # wraparound tile if the beat window is longer than the noise track
    reps = int(np.ceil(length / n))
    tiled = np.tile(noise_track, reps)
    start = rng.integers(0, len(tiled) - length + 1)
    return tiled[start:start + length]


def synthetic_powerline(length, fs=360.0, freq=50.0, rng=None):
    """
    Synthetic mains/powerline interference: a sinusoid at `freq` Hz
    (50 or 60 Hz) with a small random phase offset, unit amplitude
    (get scaled to target SNR by add_noise_at_snr like any other noise).
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(length) / fs
    phase = rng.uniform(0, 2 * np.pi)
    return np.sin(2 * np.pi * freq * t + phase)


def measured_snr_db(clean, mixture):
    """
    Sanity-check helper: given a clean signal and the noisy mixture that
    was produced from it, recover the achieved SNR. Should match the
    requested snr_db in add_noise_at_snr up to floating point precision.
    """
    clean = np.asarray(clean, dtype=np.float64)
    mixture = np.asarray(mixture, dtype=np.float64)
    added_noise = mixture - clean
    p_clean = signal_power(clean)
    p_noise = signal_power(added_noise)
    if p_noise < 1e-12:
        return float('inf')
    return 10 * np.log10(p_clean / p_noise)
