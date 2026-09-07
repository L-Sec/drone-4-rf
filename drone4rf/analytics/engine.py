"""Analytics orchestrator: feeds analyzers per chunk, runs fusion on an
interval, and decides which assessments are worth emitting.

Emission policy: an assessment is emitted when an entity first reaches
'possible' or above, whenever its category changes at that level, and
when a previously elevated entity de-escalates - quiet entities produce
no chatter (track events already cover plain anomalies).
"""

from __future__ import annotations

from collections import deque

from drone4rf.analytics.emi import EMIAnalyzer, EMIObservation
from drone4rf.analytics.fusion import (
    CATEGORY_POSSIBLE,
    DroneAssessment,
    FusionEngine,
    category_rank,
)
from drone4rf.analytics.hopping import HopAnalyzer
from drone4rf.config import AppConfig
from drone4rf.detectors.base import Detection
from drone4rf.tracking import SignalTrack


class AnalyticsEngine:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg.analytics
        self.enabled = self.cfg.enabled
        self.hop = HopAnalyzer(self.cfg.hopping)
        self.emi = EMIAnalyzer(self.cfg.emi)
        ml_scorer = None
        if cfg.ml.enabled:
            # Fail fast with an actionable message: enabling ML without a
            # trained model (or with a stale feature schema) is an
            # operator error, not something to paper over silently.
            from drone4rf.ml.models import MLScorer

            ml_scorer = MLScorer.from_config(cfg.ml)
        self.fusion = FusionEngine(self.cfg, ml_scorer=ml_scorer)
        self._emi_observations: deque[EMIObservation] = deque(maxlen=64)
        self._last_assess_time = 0.0
        self._last_emitted: dict[str, str] = {}
        self._last_emit_time: dict[str, float] = {}

    def observe(self, timestamp: float, detections: list[Detection]) -> None:
        """Per-chunk analyzer feeding; cheap (works on detections only)."""
        if not self.enabled or not detections:
            return
        self.hop.ingest(detections)
        obs = self.emi.analyze(timestamp, detections)
        if obs is not None:
            self._emi_observations.append(obs)

    def maybe_assess(
        self, now: float, tracks: list[SignalTrack]
    ) -> list[DroneAssessment]:
        """Run fusion on the configured interval; return emittable results."""
        if not self.enabled:
            return []
        if now - self._last_assess_time < self.cfg.assessment_interval_s:
            return []
        self._last_assess_time = now

        candidates = self.hop.candidates(now)
        assessments = self.fusion.assess(
            now, tracks, candidates, list(self._emi_observations)
        )
        return [a for a in assessments if self._should_emit(a, now)]

    def _should_emit(self, a: DroneAssessment, now: float) -> bool:
        """Emission with hysteresis: escalations are always announced;
        lateral changes and de-escalations respect the cooldown so a
        borderline entity flapping around a threshold does not spam."""
        key = f"{a.entity}:{round(a.center_hz / 1e6)}"
        prev = self._last_emitted.get(key)
        cooled = (
            now - self._last_emit_time.get(key, -1e18) >= self.cfg.emit_cooldown_s
        )
        possible_rank = category_rank(CATEGORY_POSSIBLE)
        rank = category_rank(a.category)
        prev_rank = category_rank(prev) if prev is not None else -1

        emit = False
        if rank >= possible_rank and a.category != prev:
            emit = rank > prev_rank or cooled
        elif prev_rank >= possible_rank and rank < prev_rank:
            emit = cooled  # de-escalation of a previously elevated entity
        if emit:
            self._last_emitted[key] = a.category
            self._last_emit_time[key] = now
        return emit
