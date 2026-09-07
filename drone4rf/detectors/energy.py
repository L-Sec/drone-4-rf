"""Energy detector: bins exceeding the adaptive baseline threshold."""

from __future__ import annotations

import numpy as np

from drone4rf.config import EnergyDetectorConfig
from drone4rf.detectors.base import Detection, group_bins, segment_stats


class EnergyDetector:
    name = "energy"

    def __init__(self, cfg: EnergyDetectorConfig) -> None:
        self.cfg = cfg

    def detect(
        self,
        psd_db: np.ndarray,
        threshold_db: np.ndarray,
        baseline_level_db: np.ndarray,
        freqs_hz: np.ndarray,
        timestamp: float,
        duration_s: float,
        quality_flags: tuple[str, ...] = (),
    ) -> list[Detection]:
        if not self.cfg.enabled:
            return []
        mask = psd_db > threshold_db
        detections: list[Detection] = []
        for start, end in group_bins(mask, self.cfg.min_bins, self.cfg.merge_gap_bins):
            stats = segment_stats(psd_db, baseline_level_db, freqs_hz, start, end)
            margin = float(np.max(psd_db[start:end] - threshold_db[start:end]))
            # Score saturates: 20 dB above threshold -> ~1.0.
            score = float(min(1.0, margin / 20.0))
            detections.append(
                Detection(
                    detector=self.name,
                    timestamp=timestamp,
                    duration_s=duration_s,
                    score=score,
                    flags=quality_flags,
                    features={
                        "margin_db": margin,
                        "bins": end - start,
                        "bin_start": start,
                        "bin_end": end,
                    },
                    explanation=(
                        f"power exceeded the learned background threshold by "
                        f"{margin:.1f} dB across {end - start} bins"
                    ),
                    **stats,
                )
            )
        return detections
