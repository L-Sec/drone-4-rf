"""Ordered-statistic CFAR detector across frequency bins.

OS-CFAR estimates the local noise level for each cell-under-test as an
order statistic (quantile) of surrounding training cells, excluding guard
cells adjacent to the CUT. Unlike cell-averaging CFAR, the order
statistic is not dragged upward by a strong signal inside the training
window, which preserves detection of closely spaced emitters.

This detector is baseline-independent: it works from frame-local
statistics, so it fires even before the background model has converged,
and serves as a cross-check on the energy detector.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from drone4rf.config import CFARDetectorConfig
from drone4rf.detectors.base import Detection, group_bins, segment_stats


class OSCFARDetector:
    name = "cfar"

    def __init__(self, cfg: CFARDetectorConfig) -> None:
        self.cfg = cfg

    def _noise_estimate(self, psd_db: np.ndarray) -> np.ndarray:
        """Per-bin OS noise estimate from training cells (dB domain).

        Working in dB is legitimate for an order statistic (monotone
        transform preserves quantiles).
        """
        g, t = self.cfg.guard_bins, self.cfg.train_bins
        half = g + t
        padded = np.pad(psd_db, half, mode="edge")
        windows = sliding_window_view(padded, 2 * half + 1)  # (n_bins, 2*half+1)
        training = np.concatenate([windows[:, :t], windows[:, -t:]], axis=1)
        return np.quantile(training, self.cfg.quantile, axis=1)

    def detect(
        self,
        psd_db: np.ndarray,
        freqs_hz: np.ndarray,
        timestamp: float,
        duration_s: float,
        quality_flags: tuple[str, ...] = (),
    ) -> list[Detection]:
        if not self.cfg.enabled:
            return []
        noise = self._noise_estimate(psd_db)
        mask = psd_db > (noise + self.cfg.offset_db)
        detections: list[Detection] = []
        for start, end in group_bins(mask, self.cfg.min_bins, self.cfg.merge_gap_bins):
            stats = segment_stats(psd_db, noise, freqs_hz, start, end)
            margin = float(
                np.max(psd_db[start:end] - noise[start:end]) - self.cfg.offset_db
            )
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
                        f"OS-CFAR: cell power {margin:.1f} dB beyond the "
                        f"{self.cfg.offset_db:.0f} dB margin over the local "
                        f"order-statistic noise estimate"
                    ),
                    **stats,
                )
            )
        return detections
