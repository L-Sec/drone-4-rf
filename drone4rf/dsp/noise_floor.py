"""Robust noise-floor estimation over a PSD frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Scale factor converting MAD to a Gaussian-equivalent standard deviation.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class NoiseFloorEstimate:
    floor_db: float  # median PSD level
    sigma_db: float  # MAD-derived spread


def estimate_noise_floor(psd_db: np.ndarray) -> NoiseFloorEstimate:
    """Median/MAD floor estimate - robust to narrowband signals occupying
    a minority of bins (the median ignores them, unlike the mean)."""
    med = float(np.median(psd_db))
    mad = float(np.median(np.abs(psd_db - med)))
    return NoiseFloorEstimate(floor_db=med, sigma_db=mad * MAD_TO_SIGMA)
