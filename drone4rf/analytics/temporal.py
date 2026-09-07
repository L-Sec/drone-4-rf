"""Sub-chunk temporal profiling of detected frequency segments.

A chunk's spectrogram (fft_size*(1-overlap)/fs seconds per row) gives
~0.1-0.4 ms time resolution: enough to resolve the burst structure of
RC control links (frame periods of 1-20 ms) that a chunk-averaged PSD
smears into apparent continuity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-20


@dataclass(frozen=True)
class TemporalProfile:
    duty: float  # fraction of segments the band was 'on'
    n_bursts: int  # distinct on-runs within the chunk
    mean_burst_s: float
    burst_period_s: float  # nan when fewer than 2 bursts
    period_cv: float  # coefficient of variation; nan when < 3 bursts


def compute_temporal_profile(
    spec_db: np.ndarray, bin_start: int, bin_end: int, seg_step_s: float
) -> TemporalProfile:
    """Profile the on/off timing of spectrogram bins [bin_start, bin_end).

    The band's per-segment power trace is thresholded halfway between its
    median (off level) and maximum (on level); if the swing is under 3 dB
    the signal is treated as continuous within the chunk.
    """
    band = spec_db[:, bin_start:bin_end]
    # Mean linear power per segment, back to dB.
    trace = 10.0 * np.log10(np.mean(10.0 ** (band / 10.0), axis=1) + _EPS)
    off_level = float(np.median(trace))
    on_level = float(trace.max())
    if on_level - off_level < 3.0:
        return TemporalProfile(
            duty=1.0,
            n_bursts=1,
            mean_burst_s=len(trace) * seg_step_s,
            burst_period_s=math.nan,
            period_cv=math.nan,
        )
    threshold = off_level + (on_level - off_level) / 2.0
    on = trace > threshold
    duty = float(np.mean(on))

    padded = np.concatenate([[False], on, [False]])
    rises = np.flatnonzero(~padded[:-1] & padded[1:])
    falls = np.flatnonzero(padded[:-1] & ~padded[1:])
    lengths = falls - rises
    n_bursts = len(rises)
    mean_burst_s = float(lengths.mean()) * seg_step_s if n_bursts else 0.0

    period_s = math.nan
    period_cv = math.nan
    if n_bursts >= 2:
        periods = np.diff(rises) * seg_step_s
        period_s = float(periods.mean())
        if n_bursts >= 3 and period_s > 0:
            period_cv = float(periods.std() / period_s)
    return TemporalProfile(
        duty=duty,
        n_bursts=n_bursts,
        mean_burst_s=mean_burst_s,
        burst_period_s=period_s,
        period_cv=period_cv,
    )
