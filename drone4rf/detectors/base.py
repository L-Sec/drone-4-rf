"""Shared detection dataclass and bin-grouping utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detector firing on one PSD frame."""

    detector: str
    timestamp: float
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    avg_db: float
    snr_db: float
    duration_s: float  # observation window (chunk duration in Stage 2)
    score: float  # detector-specific confidence in [0, 1]
    flags: tuple[str, ...] = ()
    features: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


def group_bins(
    mask: np.ndarray, min_bins: int, merge_gap_bins: int
) -> list[tuple[int, int]]:
    """Group a boolean bin mask into [start, end) segments.

    Gaps up to merge_gap_bins are bridged (a modulated signal rarely
    exceeds threshold in every bin); runs shorter than min_bins are
    discarded as isolated noise excursions.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= merge_gap_bins + 1:
            prev = i
        else:
            segments.append((start, prev + 1))
            start = prev = i
    segments.append((start, prev + 1))
    return [(a, b) for a, b in segments if (b - a) >= min_bins]


def segment_stats(
    psd_db: np.ndarray,
    noise_ref_db: np.ndarray,
    freqs_hz: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    """Power/frequency statistics for a detected bin segment."""
    seg = psd_db[start:end]
    peak_i = start + int(np.argmax(seg))
    noise = float(np.median(noise_ref_db[start:end]))
    return {
        "center_hz": float(freqs_hz[peak_i]),
        "bandwidth_hz": float(abs(freqs_hz[end - 1] - freqs_hz[start])),
        "peak_db": float(seg.max()),
        "avg_db": float(seg.mean()),
        "snr_db": float(seg.max() - noise),
    }
