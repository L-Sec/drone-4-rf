import math

import numpy as np

from drone4rf.analytics.temporal import compute_temporal_profile

SEG_STEP = 0.001  # 1 ms per spectrogram row


def _spec(n_seg: int = 100, n_bins: int = 64) -> np.ndarray:
    return np.full((n_seg, n_bins), -80.0, dtype=np.float32)


def test_periodic_bursts_profiled_correctly() -> None:
    spec = _spec()
    # Band bins 20:24 on for 5 segments every 20 segments -> 5 bursts.
    for start in range(0, 100, 20):
        spec[start : start + 5, 20:24] = -40.0
    prof = compute_temporal_profile(spec, 20, 24, SEG_STEP)
    assert abs(prof.duty - 0.25) < 0.05
    assert prof.n_bursts == 5
    assert abs(prof.burst_period_s - 0.020) < 0.002
    assert prof.period_cv < 0.05  # perfectly regular
    assert abs(prof.mean_burst_s - 0.005) < 0.002


def test_continuous_signal_is_duty_one() -> None:
    spec = _spec()
    spec[:, 30:34] = -40.0
    prof = compute_temporal_profile(spec, 30, 34, SEG_STEP)
    assert prof.duty == 1.0
    assert prof.n_bursts == 1
    assert math.isnan(prof.burst_period_s)


def test_single_burst_has_no_period() -> None:
    spec = _spec()
    spec[40:50, 10:14] = -40.0
    prof = compute_temporal_profile(spec, 10, 14, SEG_STEP)
    assert prof.n_bursts == 1
    assert abs(prof.duty - 0.10) < 0.03
    assert math.isnan(prof.burst_period_s)


def test_irregular_bursts_have_high_cv() -> None:
    spec = _spec(n_seg=200)
    for start in (0, 11, 47, 90, 170):  # erratic spacing
        spec[start : start + 4, 20:24] = -40.0
    prof = compute_temporal_profile(spec, 20, 24, SEG_STEP)
    assert prof.n_bursts == 5
    assert prof.period_cv > 0.5
