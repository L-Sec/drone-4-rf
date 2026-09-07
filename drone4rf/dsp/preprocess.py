"""IQ preprocessing: DC removal, Welch PSD, dBFS, artifact masking.

All power values in this project are *relative dBFS*: a full-scale
(amplitude 1.0) tone measures ~0 dBFS after coherent-gain normalization.
The HackRF is uncalibrated, so absolute dBm is deliberately not claimed.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as _fft
from scipy.signal import get_window

_EPS = 1e-20  # floor before log10 to avoid -inf


def remove_dc(samples: np.ndarray) -> np.ndarray:
    """Subtract the complex mean (bulk of the HackRF DC offset)."""
    return samples - samples.mean()


def clip_fraction(samples: np.ndarray, threshold: float = 0.95) -> float:
    """Fraction of samples with I or Q near full scale (ADC saturation)."""
    if len(samples) == 0:
        return 0.0
    clipped = (np.abs(samples.real) > threshold) | (np.abs(samples.imag) > threshold)
    return float(np.count_nonzero(clipped)) / len(samples)


def _segment_powers(
    samples: np.ndarray, fft_size: int, overlap: float, window: str
) -> np.ndarray:
    """Linear per-segment power spectra, fftshifted: shape (n_seg, n_bins)."""
    if len(samples) < fft_size:
        raise ValueError(f"need >= {fft_size} samples, got {len(samples)}")
    win = get_window(window, fft_size).astype(np.float32)
    coherent_gain = float(win.sum())
    step = max(1, int(fft_size * (1.0 - overlap)))
    segments = sliding_window_view(samples, fft_size)[::step]
    # scipy pocketfft: stays in complex64 (numpy upcasts to complex128),
    # releases the GIL, and parallelizes across segments.
    spectra = _fft.fft(segments * win, axis=1, workers=-1)
    power = np.abs(spectra) ** 2 / (coherent_gain**2)
    return np.fft.fftshift(power, axes=1)


def compute_psd(
    samples: np.ndarray,
    fft_size: int = 4096,
    overlap: float = 0.5,
    window: str = "hann",
) -> np.ndarray:
    """Welch power spectral density in relative dBFS, fftshifted.

    Segments of fft_size with the given overlap are windowed, FFT'd, and
    the squared magnitudes averaged. Normalizing by the window's coherent
    gain (sum of window samples) puts a full-scale tone at ~0 dBFS.
    """
    power = _segment_powers(samples, fft_size, overlap, window).mean(axis=0)
    return (10.0 * np.log10(power + _EPS)).astype(np.float64)


def spectrogram_and_psd(
    samples: np.ndarray,
    fft_size: int = 4096,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """One FFT pass yielding both the spectrogram and the Welch PSD.

    Returns (spec_db, psd_db): spec_db has shape (n_seg, n_bins) in dBFS
    and gives sub-chunk time resolution (fft_size*(1-overlap)/fs seconds
    per row) for burst/hop timing analysis; psd_db is the same average
    that compute_psd would return.
    """
    power = _segment_powers(samples, fft_size, overlap, window)
    psd_db = (10.0 * np.log10(power.mean(axis=0) + _EPS)).astype(np.float64)
    spec_db = (10.0 * np.log10(power + _EPS)).astype(np.float32)
    return spec_db, psd_db


def segment_step_seconds(fft_size: int, overlap: float, sample_rate: float) -> float:
    """Time between spectrogram rows."""
    return max(1, int(fft_size * (1.0 - overlap))) / sample_rate


def bin_frequencies(center_hz: float, sample_rate: float, fft_size: int) -> np.ndarray:
    """Absolute frequency of each fftshifted PSD bin."""
    return center_hz + np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))


def mask_dc_spike(psd_db: np.ndarray, mask_bins: int) -> np.ndarray:
    """Replace the center +/- mask_bins bins with the local neighbor median.

    The HackRF's residual DC/LO spike sits exactly at the center bin and
    must never reach the detectors as a 'signal'.
    """
    if mask_bins <= 0:
        return psd_db
    out = psd_db.copy()
    c = len(out) // 2
    lo, hi = c - mask_bins, c + mask_bins + 1
    # Neighborhood just outside the masked region provides the fill value.
    span = max(3 * mask_bins, 8)
    neighbors = np.concatenate([out[max(0, lo - span) : lo], out[hi : hi + span]])
    if len(neighbors):
        out[lo:hi] = np.median(neighbors)
    return out
