"""Sweep scheduler: which center frequency to visit next, and for how long.

Each enabled band is divided into steps of usable bandwidth
(sample_rate * usable_fraction, so HackRF filter-edge roll-off overlaps
between adjacent steps). Steps carry a due time:

- selection picks the oldest-due step first, with priority breaking ties
  (use per-band revisit_s to control cadence, priority for contention);
- completing a visit reschedules the step revisit_s in the future;
- adaptive rescanning: when the pipeline reports detections on a step,
  its next visits use the (shorter) hot_revisit_s for hot_visits rounds,
  so suspicious frequencies are re-checked sooner.

Thread-safe: next_step/complete are called from the acquisition thread,
report_activity from the processing thread.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from drone4rf.config import SweepConfig


@dataclass
class SweepStep:
    step_id: str
    band: str
    center_hz: float
    dwell_chunks: int
    revisit_s: float
    priority: int
    next_due: float = 0.0
    visits: int = 0
    hot_visits_left: int = 0


class SweepScheduler:
    def __init__(
        self,
        cfg: SweepConfig,
        sample_rate: float,
        chunk_samples: int,
    ) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._steps: list[SweepStep] = []
        usable = sample_rate * cfg.usable_fraction
        for band in cfg.enabled_bands():
            span = band.stop_hz - band.start_hz
            n_steps = max(1, math.ceil(span / usable))
            dwell_chunks = max(1, round(band.dwell_s * sample_rate / chunk_samples))
            for k in range(n_steps):
                if span <= usable:
                    center = (band.start_hz + band.stop_hz) / 2
                else:
                    # Last step is pulled back so its window ends at the
                    # band edge instead of overshooting it.
                    center = min(
                        band.start_hz + usable * (k + 0.5),
                        band.stop_hz - usable / 2,
                    )
                self._steps.append(
                    SweepStep(
                        step_id=f"{band.name}/{k}",
                        band=band.name,
                        center_hz=center,
                        dwell_chunks=dwell_chunks,
                        revisit_s=band.revisit_s,
                        priority=band.priority,
                    )
                )
        if not self._steps:
            raise ValueError("sweep plan has no enabled bands")
        self._by_center: dict[int, SweepStep] = {
            int(round(s.center_hz)): s for s in self._steps
        }

    @property
    def steps(self) -> list[SweepStep]:
        with self._lock:
            return list(self._steps)

    def next_step(self, now: float) -> SweepStep:
        """Pick the next step: oldest due first, priority breaks ties.

        If nothing is due yet (all revisit timers pending), the soonest-due
        step is returned anyway - a passive scanner has nothing better to
        do than scan early.
        """
        with self._lock:
            due = [s for s in self._steps if s.next_due <= now]
            if due:
                return min(due, key=lambda s: (s.next_due, -s.priority, s.step_id))
            return min(self._steps, key=lambda s: (s.next_due, -s.priority))

    def complete(self, step_id: str, now: float) -> None:
        """Mark a visit finished and schedule the revisit."""
        with self._lock:
            step = self._find(step_id)
            step.visits += 1
            if step.hot_visits_left > 0:
                step.hot_visits_left -= 1
                revisit = min(step.revisit_s, self.cfg.adaptive.hot_revisit_s)
            else:
                revisit = step.revisit_s
            step.next_due = now + revisit

    def report_activity(self, center_hz: float, detections: int, now: float) -> None:
        """Processing thread feedback: activity marks the step 'hot'."""
        if detections <= 0 or not self.cfg.adaptive.enabled:
            return
        with self._lock:
            step = self._by_center.get(int(round(center_hz)))
            if step is None:
                return
            step.hot_visits_left = self.cfg.adaptive.hot_visits
            step.next_due = min(
                step.next_due, now + self.cfg.adaptive.hot_revisit_s
            )

    def _find(self, step_id: str) -> SweepStep:
        for s in self._steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)
