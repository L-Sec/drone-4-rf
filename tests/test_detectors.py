import numpy as np

from drone4rf.config import CFARDetectorConfig, EnergyDetectorConfig
from drone4rf.detectors.base import group_bins
from drone4rf.detectors.cfar import OSCFARDetector
from drone4rf.detectors.energy import EnergyDetector

N = 1024
FS = 10e6
CENTER = 2_437e6


def _freqs() -> np.ndarray:
    return CENTER + np.fft.fftshift(np.fft.fftfreq(N, d=1 / FS))


def test_group_bins_merging_and_min_width() -> None:
    mask = np.zeros(N, dtype=bool)
    mask[10:15] = True
    mask[16:20] = True  # gap of 1 -> merged with previous run
    mask[500] = True  # isolated single bin -> dropped (min_bins=2)
    segs = group_bins(mask, min_bins=2, merge_gap_bins=2)
    assert segs == [(10, 20)]


def test_energy_detector_finds_injected_signal() -> None:
    rng = np.random.default_rng(0)
    psd = rng.normal(-80.0, 1.0, N)
    psd[300:308] = -55.0
    level = np.full(N, -80.0)
    thr = np.full(N, -70.0)
    det = EnergyDetector(EnergyDetectorConfig())
    out = det.detect(psd, thr, level, _freqs(), timestamp=0.0, duration_s=0.02)
    assert len(out) == 1
    d = out[0]
    expected_center = _freqs()[300:308].mean()
    assert abs(d.center_hz - expected_center) < 10 * FS / N
    assert d.snr_db > 20
    assert 0 < d.score <= 1
    assert "exceeded" in d.explanation


def test_energy_detector_quiet_on_noise() -> None:
    rng = np.random.default_rng(1)
    psd = rng.normal(-80.0, 1.0, N)
    level = np.full(N, -80.0)
    thr = np.full(N, -70.0)  # 10 dB margin over a sigma=1 floor
    det = EnergyDetector(EnergyDetectorConfig())
    assert det.detect(psd, thr, level, _freqs(), 0.0, 0.02) == []


def test_cfar_detects_tone_without_baseline() -> None:
    rng = np.random.default_rng(2)
    psd = rng.normal(-80.0, 1.0, N)
    psd[600:604] = -55.0
    det = OSCFARDetector(CFARDetectorConfig())
    out = det.detect(psd, _freqs(), timestamp=0.0, duration_s=0.02)
    assert len(out) == 1
    assert abs(out[0].center_hz - _freqs()[600:604].mean()) < 10 * FS / N


def test_cfar_quiet_on_noise() -> None:
    rng = np.random.default_rng(3)
    psd = rng.normal(-80.0, 1.0, N)
    det = OSCFARDetector(CFARDetectorConfig())
    # sigma=1 dB noise vs 9 dB offset: false alarms vanishingly unlikely.
    assert det.detect(psd, _freqs(), 0.0, 0.02) == []


def test_cfar_resolves_two_nearby_signals() -> None:
    rng = np.random.default_rng(4)
    psd = rng.normal(-80.0, 1.0, N)
    psd[400:404] = -55.0
    psd[440:444] = -55.0
    det = OSCFARDetector(CFARDetectorConfig())
    out = det.detect(psd, _freqs(), 0.0, 0.02)
    assert len(out) == 2  # OS statistic isn't dragged up by the neighbor
