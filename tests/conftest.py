"""Shared test fixtures: small, fast configs and synthetic PSD helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drone4rf.config import (
    AppConfig,
    BaselineConfig,
    DetectionConfig,
    DeviceConfig,
    DSPConfig,
    TrackingConfig,
)

FS = 10e6
CENTER = 2_437e6
FFT = 1024


@pytest.fixture
def fast_config() -> AppConfig:
    """Small FFT/chunk config so pipeline tests run in milliseconds."""
    cfg = AppConfig(
        device=DeviceConfig(sample_rate=FS, center_freq_hz=CENTER),
        dsp=DSPConfig(fft_size=FFT, chunk_samples=65_536),
        detection=DetectionConfig(),
        baseline=BaselineConfig(),
        tracking=TrackingConfig(promote_hits=5, drop_after_misses=3),
    )
    cfg.validate()
    return cfg


def noise_psd(rng: np.random.Generator, n_bins: int = FFT, floor: float = -80.0,
              sigma: float = 1.0) -> np.ndarray:
    """A plausible dB-domain noise PSD frame."""
    return rng.normal(floor, sigma, n_bins)
