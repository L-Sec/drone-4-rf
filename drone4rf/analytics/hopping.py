"""Frequency-hopping candidate detection.

A hopping link (FHSS RC control, some telemetry) shows up at chunk
granularity as many narrowband detections spread over distinct channels,
each with LOW within-chunk duty (the transmitter is elsewhere most of the
time), with roughly uniform channel bandwidths and often regular channel
spacing. Multiple independent continuous emitters look superficially
similar but have duty ~1.0 per channel - the duty test separates them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from drone4rf.config import HoppingConfig
from drone4rf.detectors.base import Detection


@dataclass(frozen=True)
class HopCandidate:
    timestamp: float
    freq_lo_hz: float
    freq_hi_hz: float
    n_channels: int
    channel_centers_hz: tuple[float, ...]
    mean_duty: float
    spacing_regularity: float  # 0 (irregular) .. 1 (perfectly regular)
    score: float
    explanation: str

    @property
    def center_hz(self) -> float:
        return (self.freq_lo_hz + self.freq_hi_hz) / 2


@dataclass
class _HopEvent:
    t: float
    center_hz: float
    bandwidth_hz: float
    duty: float


class HopAnalyzer:
    def __init__(self, cfg: HoppingConfig) -> None:
        self.cfg = cfg
        self._events: deque[_HopEvent] = deque(maxlen=4096)

    def ingest(self, detections: list[Detection]) -> None:
        for det in detections:
            if det.bandwidth_hz <= self.cfg.max_channel_bw_hz:
                duty = det.features.get("duty")
                self._events.append(
                    _HopEvent(
                        t=det.timestamp,
                        center_hz=det.center_hz,
                        bandwidth_hz=det.bandwidth_hz,
                        duty=float(duty) if duty is not None else 1.0,
                    )
                )

    def candidates(self, now: float) -> list[HopCandidate]:
        cutoff = now - self.cfg.window_s
        while self._events and self._events[0].t < cutoff:
            self._events.popleft()
        if not self._events:
            return []

        channels = self._cluster_channels(list(self._events))
        if len(channels) < self.cfg.min_channels:
            return []

        centers = np.array(sorted(c["center"] for c in channels))
        duties = np.array([c["duty"] for c in channels])
        # Duty complementarity: a hopper occupies each channel only a
        # fraction of the time. Continuous emitters fail this.
        low_duty_frac = float(np.mean(duties < 0.7))
        mean_duty = float(duties.mean())

        regularity = 0.0
        if len(centers) >= 3:
            spacings = np.diff(centers)
            mean_spacing = float(spacings.mean())
            if mean_spacing > 0:
                regularity = float(max(0.0, 1.0 - spacings.std() / mean_spacing))

        # Score: channel count (saturating at 8), low per-channel duty,
        # and spacing regularity, equally telling in practice.
        score = float(
            np.clip(
                0.4 * min(1.0, len(centers) / 8.0)
                + 0.4 * low_duty_frac
                + 0.2 * regularity,
                0.0,
                1.0,
            )
        )
        if low_duty_frac < 0.5:
            # Mostly-continuous channels: independent emitters or a
            # harmonic comb, not a link hopping between frequencies.
            score *= 0.3
        explanation = (
            f"{len(centers)} distinct narrowband channels between "
            f"{centers[0] / 1e6:.1f} and {centers[-1] / 1e6:.1f} MHz within "
            f"{self.cfg.window_s:.0f} s; mean per-channel duty {mean_duty:.2f}; "
            f"channel-spacing regularity {regularity:.2f}"
        )
        return [
            HopCandidate(
                timestamp=now,
                freq_lo_hz=float(centers[0]),
                freq_hi_hz=float(centers[-1]),
                n_channels=len(centers),
                channel_centers_hz=tuple(float(c) for c in centers),
                mean_duty=mean_duty,
                spacing_regularity=regularity,
                score=score,
                explanation=explanation,
            )
        ]

    def _cluster_channels(self, events: list[_HopEvent]) -> list[dict]:
        """Greedy 1-D clustering of event centers into channels."""
        tol = self.cfg.channel_tolerance_hz
        clusters: list[dict] = []
        for ev in sorted(events, key=lambda e: e.center_hz):
            if clusters and abs(ev.center_hz - clusters[-1]["center"]) <= tol:
                c = clusters[-1]
                c["n"] += 1
                c["center"] += (ev.center_hz - c["center"]) / c["n"]
                c["duty"] += (ev.duty - c["duty"]) / c["n"]
            else:
                clusters.append(
                    {"center": ev.center_hz, "duty": ev.duty, "n": 1}
                )
        return clusters
