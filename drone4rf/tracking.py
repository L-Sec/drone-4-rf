"""Frequency-clustered persistence tracking over per-frame detections.

Detections from any detector are clustered by center frequency into
tracks. A track that keeps re-appearing is promoted to *persistent*; when
it dies, its hit/observation ratio distinguishes continuous links from
repeated bursts. This is the Stage 2 seed of the Stage 4 fusion layer:
'one strong frame' never becomes an event on its own.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

from drone4rf.config import TrackingConfig
from drone4rf.detectors.base import Detection


@dataclass
class SignalTrack:
    track_id: int
    center_hz: float
    bandwidth_hz: float
    first_seen: float
    last_seen: float
    hits: int = 1
    consecutive_misses: int = 0
    # Frames elapsed while alive; incremented once per frame by the
    # tracker (including the creation frame), so it starts at 0.
    frames_observed: int = 0
    peak_db: float = -200.0
    max_snr_db: float = 0.0
    detectors: set[str] = field(default_factory=set)
    quality_flags: set[str] = field(default_factory=set)
    promoted: bool = False
    # Rolling per-hit feature history consumed by the Stage 4 analytics
    # (burst scoring, power-trend features). Bounded to cap memory.
    history: deque = field(default_factory=lambda: deque(maxlen=64))

    @property
    def occupancy(self) -> float:
        """Fraction of observed frames in which the signal was present.

        Near 1.0 => continuous link; low values => intermittent/bursty.
        """
        return self.hits / max(1, self.frames_observed)


@dataclass(frozen=True)
class TrackEvent:
    kind: str  # "track_persistent" | "track_closed"
    track: SignalTrack
    explanation: str


class SignalTracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self._tracks: list[SignalTrack] = []
        self._ids = itertools.count(1)

    @property
    def active_tracks(self) -> list[SignalTrack]:
        return list(self._tracks)

    def update(
        self,
        detections: list[Detection],
        now: float,
        window: tuple[float, float] | None = None,
    ) -> list[TrackEvent]:
        """Feed one frame's detections; returns promotion/closure events.

        In sweep mode, `window` is the (low, high) frequency span the
        receiver actually observed for this frame. Tracks outside the
        window are left untouched - a 2.4 GHz track must not accrue
        misses while the radio is tuned to 5.8 GHz.
        """
        events: list[TrackEvent] = []
        matched: set[int] = set()

        for det in detections:
            track = self._match(det)
            if track is None:
                track = SignalTrack(
                    track_id=next(self._ids),
                    center_hz=det.center_hz,
                    bandwidth_hz=det.bandwidth_hz,
                    first_seen=det.timestamp,
                    last_seen=det.timestamp,
                )
                self._tracks.append(track)
            elif id(track) not in matched:
                track.hits += 1
                track.consecutive_misses = 0
                track.last_seen = det.timestamp
                # Smooth the track center toward new observations.
                track.center_hz += 0.3 * (det.center_hz - track.center_hz)
                track.bandwidth_hz = max(track.bandwidth_hz, det.bandwidth_hz)
            track.peak_db = max(track.peak_db, det.peak_db)
            track.max_snr_db = max(track.max_snr_db, det.snr_db)
            track.detectors.add(det.detector)
            track.quality_flags.update(det.flags)
            track.history.append(
                {
                    "t": det.timestamp,
                    "snr_db": det.snr_db,
                    "bandwidth_hz": det.bandwidth_hz,
                    "duty": det.features.get("duty"),
                    "n_bursts": det.features.get("n_bursts"),
                    "period_cv": det.features.get("period_cv"),
                }
            )
            matched.add(id(track))

        survivors: list[SignalTrack] = []
        for track in self._tracks:
            in_window = window is None or (
                window[0] <= track.center_hz <= window[1]
            )
            if not in_window and id(track) not in matched:
                # Not observed this frame; neither hit nor miss.
                survivors.append(track)
                continue
            track.frames_observed += 1
            if id(track) not in matched:
                track.consecutive_misses += 1
            if not track.promoted and track.hits >= self.cfg.promote_hits:
                track.promoted = True
                events.append(
                    TrackEvent(
                        kind="track_persistent",
                        track=track,
                        explanation=self._describe(track, closing=False),
                    )
                )
            if track.consecutive_misses >= self.cfg.drop_after_misses:
                if track.promoted:
                    events.append(
                        TrackEvent(
                            kind="track_closed",
                            track=track,
                            explanation=self._describe(track, closing=True),
                        )
                    )
            else:
                survivors.append(track)
        self._tracks = survivors
        return events

    def _match(self, det: Detection) -> SignalTrack | None:
        tol = max(self.cfg.match_tolerance_hz, det.bandwidth_hz / 2)
        best: SignalTrack | None = None
        best_dist = tol
        for track in self._tracks:
            dist = abs(track.center_hz - det.center_hz)
            if dist <= best_dist:
                best, best_dist = track, dist
        return best

    def _describe(self, track: SignalTrack, closing: bool) -> str:
        behavior = (
            "continuous occupancy (persistent-link-like)"
            if track.occupancy > 0.8
            else "intermittent occupancy (repeated bursts)"
        )
        dets = "+".join(sorted(track.detectors))
        quality = (
            f"; quality warnings: {', '.join(sorted(track.quality_flags))}"
            if track.quality_flags
            else ""
        )
        phase = "signal ended after" if closing else "signal persisted across"
        return (
            f"{phase} {track.hits} detections in {track.frames_observed} frames "
            f"({behavior}); detectors agreeing: {dets}; peak SNR "
            f"{track.max_snr_db:.1f} dB{quality}. This is an unclassified RF "
            f"anomaly relative to the learned background, not a confirmed drone."
        )
