"""Abstract IQ source interface shared by hardware, simulator, and files.

All sources are RECEIVE-ONLY: the interface has no transmit methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np


@dataclass(frozen=True)
class IQChunk:
    """One contiguous block of complex baseband samples."""

    samples: np.ndarray  # complex64, unit full scale (|I|,|Q| <= 1.0)
    center_hz: float
    sample_rate: float
    timestamp: float  # host wall-clock (time.time()) at chunk start
    seq: int  # monotone chunk counter from the source
    overflow: bool = False  # source reported dropped samples before this chunk


@dataclass
class SourceHealth:
    """Cumulative source statistics, exposed to the UI layer."""

    chunks_produced: int = 0
    overflows: int = 0
    reconnects: int = 0
    retunes: int = 0
    notes: list[str] = field(default_factory=list)


class SDRSource(ABC):
    """Receive-only IQ sample source."""

    name: str = "abstract"

    def __init__(self) -> None:
        self.health = SourceHealth()

    @abstractmethod
    def open(self) -> None:
        """Acquire the device/file and start streaming."""

    @abstractmethod
    def read_chunk(self, num_samples: int) -> IQChunk | None:
        """Return the next chunk, or None when the source is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Stop streaming and release resources. Must be idempotent."""

    @property
    def supports_retune(self) -> bool:
        """True when retune() can change the RX center frequency."""
        return False

    def retune(self, center_hz: float) -> None:
        """Change the RX center frequency (sweep mode)."""
        raise NotImplementedError(f"{self.name} source cannot retune")

    def __enter__(self) -> "SDRSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
