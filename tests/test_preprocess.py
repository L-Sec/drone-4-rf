import numpy as np

from drone4rf.dsp import preprocess

FS = 1e6
FFT = 1024


def _tone(freq: float, amp: float, n: int, fs: float = FS) -> np.ndarray:
    t = np.arange(n) / fs
    return (amp * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def test_tone_lands_in_correct_bin() -> None:
    # Put the tone exactly on a bin to avoid scalloping ambiguity.
    k = 100
    freq = k * FS / FFT
    rng = np.random.default_rng(0)
    samples = _tone(freq, 0.5, 16 * FFT) + (
        rng.normal(0, 1e-3, 16 * FFT) + 1j * rng.normal(0, 1e-3, 16 * FFT)
    ).astype(np.complex64)
    psd = preprocess.compute_psd(samples, FFT)
    freqs = preprocess.bin_frequencies(0.0, FS, FFT)
    peak_freq = freqs[int(np.argmax(psd))]
    assert abs(peak_freq - freq) <= FS / FFT  # within one bin


def test_full_scale_tone_is_about_zero_dbfs() -> None:
    freq = 64 * FS / FFT
    samples = _tone(freq, 1.0, 16 * FFT)
    psd = preprocess.compute_psd(samples, FFT)
    assert abs(psd.max()) < 1.0  # coherent-gain normalization => ~0 dBFS


def test_remove_dc_removes_offset() -> None:
    rng = np.random.default_rng(1)
    samples = (rng.normal(0, 0.01, 4096) + 1j * rng.normal(0, 0.01, 4096)).astype(
        np.complex64
    ) + (0.2 + 0.1j)
    out = preprocess.remove_dc(samples)
    assert abs(out.mean()) < 1e-3


def test_mask_dc_spike() -> None:
    psd = np.full(FFT, -80.0)
    c = FFT // 2
    psd[c - 1 : c + 2] = -20.0  # artificial DC/LO spike
    masked = preprocess.mask_dc_spike(psd, mask_bins=3)
    assert masked[c] < -70.0
    # Bins away from center untouched.
    assert masked[0] == -80.0


def test_clip_fraction() -> None:
    clean = (0.1 + 0.1j) * np.ones(1000, dtype=np.complex64)
    assert preprocess.clip_fraction(clean) == 0.0
    hot = np.ones(1000, dtype=np.complex64) * (1.0 + 1.0j)
    assert preprocess.clip_fraction(hot) == 1.0


def test_psd_rejects_short_input() -> None:
    import pytest

    with pytest.raises(ValueError):
        preprocess.compute_psd(np.zeros(10, dtype=np.complex64), FFT)
