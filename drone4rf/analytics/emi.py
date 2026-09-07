"""Motor / ESC electromagnetic-interference analysis.

Electric propulsion (brushless motors + switching ESCs) tends to produce
comb-like spectra: harmonics of commutation/PWM rates at roughly even
spacing, drifting with motor speed. Many benign devices produce similar
combs, so this analyzer is SUPPORTING evidence only - its output carries
a mandatory list of competing explanations and the fusion engine caps
any assessment built on EMI alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone4rf.config import EMIConfig
from drone4rf.detectors.base import Detection

COMPETING_EXPLANATIONS = (
    "switching power supply",
    "LED driver",
    "VFD / industrial motor controller",
    "USB hub or computer peripheral",
    "vehicle electronics",
)


@dataclass(frozen=True)
class EMIObservation:
    timestamp: float
    center_hz: float
    spacing_hz: float
    n_harmonics: int
    score: float
    explanation: str
    competing: tuple[str, ...] = COMPETING_EXPLANATIONS


class EMIAnalyzer:
    def __init__(self, cfg: EMIConfig) -> None:
        self.cfg = cfg

    def analyze(
        self, timestamp: float, detections: list[Detection]
    ) -> EMIObservation | None:
        """Look for an evenly spaced comb among a chunk's narrow peaks."""
        peaks = sorted(
            d.center_hz
            for d in detections
            if d.bandwidth_hz <= self.cfg.max_peak_bw_hz
        )
        if len(peaks) < self.cfg.min_harmonics:
            return None
        freqs = np.array(peaks)
        spacings = np.diff(freqs)
        median_spacing = float(np.median(spacings))
        if median_spacing <= 0:
            return None
        # Count consecutive spacings that agree with the median within
        # tolerance: a run of k agreeing spacings = k+1 comb teeth.
        agree = np.abs(spacings - median_spacing) <= (
            self.cfg.spacing_tolerance * median_spacing
        )
        best_run = run = 0
        for a in agree:
            run = run + 1 if a else 0
            best_run = max(best_run, run)
        n_teeth = best_run + 1
        if n_teeth < self.cfg.min_harmonics:
            return None
        score = float(min(1.0, n_teeth / 8.0))
        center = float(freqs.mean())
        explanation = (
            f"harmonic comb: {n_teeth} peaks spaced ~{median_spacing / 1e3:.0f} kHz "
            f"around {center / 1e6:.2f} MHz - consistent with motor/ESC switching "
            f"noise, but equally with: {', '.join(COMPETING_EXPLANATIONS[:3])}"
        )
        return EMIObservation(
            timestamp=timestamp,
            center_hz=center,
            spacing_hz=median_spacing,
            n_harmonics=n_teeth,
            score=score,
            explanation=explanation,
        )
