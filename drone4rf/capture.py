"""Short triggered IQ captures around notable events.

Privacy and storage stance: captures are DISABLED by default, only a few
chunks long (tens of milliseconds), capped by max_files, and contain raw
spectrum samples only - nothing is demodulated. Each .cf32 file gets a
.json sidecar with tuning metadata so it can be replayed via FileSource.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from drone4rf.config import CaptureConfig
from drone4rf.sdr.base import IQChunk

log = logging.getLogger(__name__)


class IQCaptureWriter:
    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._written = 0
        self._limit_warned = False

    @property
    def files_written(self) -> int:
        return self._written

    def write(self, chunk: IQChunk, reason: str = "") -> Path | None:
        """Write one chunk as interleaved float32 I/Q; returns the path,
        or None when capturing is disabled or the file cap is reached."""
        if not self.cfg.enabled:
            return None
        if self._written >= self.cfg.max_files:
            if not self._limit_warned:
                log.warning(
                    "capture limit reached (%d files); further captures skipped",
                    self.cfg.max_files,
                )
                self._limit_warned = True
            return None
        d = Path(self.cfg.dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        stem = f"capture_{stamp}_{chunk.center_hz / 1e6:.3f}MHz"
        path = d / f"{stem}.cf32"
        interleaved = np.empty(2 * len(chunk.samples), dtype=np.float32)
        interleaved[0::2] = chunk.samples.real
        interleaved[1::2] = chunk.samples.imag
        interleaved.tofile(path)
        sidecar = {
            "center_hz": chunk.center_hz,
            "sample_rate": chunk.sample_rate,
            "timestamp": chunk.timestamp,
            "seq": chunk.seq,
            "num_samples": len(chunk.samples),
            "format": "cf32",
            "reason": reason,
        }
        (d / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
        self._written += 1
        log.info("IQ capture written: %s (%s)", path, reason or "triggered")
        return path
