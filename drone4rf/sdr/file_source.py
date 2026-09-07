"""IQ file playback source.

Supported formats:
- 'cf32': interleaved float32 I/Q (GNU Radio / SoapySDR native).
- 'cs8' : interleaved signed 8-bit I/Q (hackrf_transfer output).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from drone4rf.sdr.base import IQChunk, SDRSource

_FORMATS = ("cf32", "cs8")


class FileSource(SDRSource):
    """Plays back a recorded IQ file as a stream of chunks."""

    name = "file"

    def __init__(
        self,
        path: str | Path,
        iq_format: str = "cf32",
        center_hz: float = 2_437e6,
        sample_rate: float = 10e6,
        loop: bool = False,
    ) -> None:
        super().__init__()
        if iq_format not in _FORMATS:
            raise ValueError(f"iq_format must be one of {_FORMATS}, got {iq_format!r}")
        self.path = Path(path)
        self.iq_format = iq_format
        self.center_hz = center_hz
        self.sample_rate = sample_rate
        self.loop = loop
        self._data: np.ndarray | None = None
        self._pos = 0
        self._seq = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"IQ file not found: {self.path}")
        if self.iq_format == "cf32":
            raw = np.fromfile(self.path, dtype=np.float32)
            if len(raw) % 2:
                raw = raw[:-1]
            self._data = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        else:  # cs8
            raw8 = np.fromfile(self.path, dtype=np.int8)
            if len(raw8) % 2:
                raw8 = raw8[:-1]
            # Normalize int8 full scale to unit full scale.
            self._data = (
                (raw8[0::2].astype(np.float32) + 1j * raw8[1::2].astype(np.float32))
                / 127.0
            ).astype(np.complex64)
        if len(self._data) == 0:
            raise ValueError(f"IQ file is empty: {self.path}")
        self._pos = 0
        self._seq = 0

    def read_chunk(self, num_samples: int) -> IQChunk | None:
        if self._data is None:
            raise RuntimeError("FileSource.read_chunk called before open()")
        if self._pos >= len(self._data):
            if not self.loop:
                return None
            self._pos = 0
        end = min(self._pos + num_samples, len(self._data))
        samples = self._data[self._pos : end]
        self._pos = end
        if len(samples) < num_samples:
            # Zero-pad the tail chunk so downstream FFT sizing is uniform.
            samples = np.concatenate(
                [samples, np.zeros(num_samples - len(samples), dtype=np.complex64)]
            )
        self._seq += 1
        self.health.chunks_produced += 1
        return IQChunk(
            samples=samples,
            center_hz=self.center_hz,
            sample_rate=self.sample_rate,
            timestamp=time.time(),
            seq=self._seq,
        )

    def close(self) -> None:
        self._data = None
