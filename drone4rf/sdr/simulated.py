"""Deterministic synthetic IQ source for tests, demos, and CI.

Generates complex Gaussian noise plus configurable emitters:

- ToneEmitter: continuous carrier at a frequency offset.
- BurstEmitter: carrier gated by an on/off duty pattern (repeated bursts).
- HopperEmitter: carrier that hops over a channel list on a fixed period,
  optionally gated (models FHSS control links).
- WidebandEmitter: band-limited noise block (models video links / OFDM).
- EMICombEmitter: harmonic comb with per-harmonic decay (models motor /
  ESC switching interference).

Sample-index bookkeeping is continuous across chunks so phases, burst
timing, and hop sequences are seamless - essential for burst/hop
detectors to see realistic timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from drone4rf.sdr.base import IQChunk, SDRSource


@dataclass(frozen=True)
class ToneEmitter:
    offset_hz: float
    amplitude: float  # linear, relative to full scale 1.0


@dataclass(frozen=True)
class BurstEmitter:
    offset_hz: float
    amplitude: float
    burst_s: float  # on-time per period
    period_s: float  # burst repetition period


@dataclass(frozen=True)
class HopperEmitter:
    channel_offsets_hz: tuple[float, ...]
    amplitude: float
    hop_period_s: float
    duty: float = 1.0  # fraction of each hop that is transmitted
    seed: int = 1234  # hop-sequence PRNG (fixed => repeatable sequence)


@dataclass(frozen=True)
class WidebandEmitter:
    offset_hz: float
    bandwidth_hz: float
    amplitude: float


@dataclass(frozen=True)
class EMICombEmitter:
    fundamental_hz: float  # harmonic spacing (e.g. ESC switching rate image)
    amplitude: float
    n_harmonics: int = 8
    decay: float = 0.7  # amplitude ratio between successive harmonics


# typing.Union (not the | operator): this alias is evaluated at runtime,
# and live-HackRF use runs under Python 3.9 (PothosSDR bindings).
Emitter = Union[
    ToneEmitter, BurstEmitter, HopperEmitter, WidebandEmitter, EMICombEmitter
]


@dataclass
class SimScenario:
    noise_std: float = 0.01  # per-component Gaussian noise sigma
    emitters: list[Emitter] = field(default_factory=list)
    seed: int = 42
    # When True, emitter frequencies (offset_hz, channel_offsets_hz, and
    # EMI harmonic positions) are ABSOLUTE RF frequencies; the source
    # translates them relative to its current center and silences
    # emitters that fall outside the tuned window. Enables realistic
    # multi-band sweep testing.
    frequencies_absolute: bool = False


def demo_scenario() -> SimScenario:
    """CW tone + bursty emitter + 8-channel hopper + weak EMI comb."""
    return SimScenario(
        noise_std=0.01,
        emitters=[
            ToneEmitter(offset_hz=1.2e6, amplitude=0.05),
            BurstEmitter(offset_hz=-2.0e6, amplitude=0.08, burst_s=0.004, period_s=0.02),
            HopperEmitter(
                channel_offsets_hz=tuple(-3.5e6 + i * 1.0e6 for i in range(8)),
                amplitude=0.06,
                hop_period_s=0.008,
                duty=0.8,
            ),
            EMICombEmitter(fundamental_hz=0.45e6, amplitude=0.02, n_harmonics=6),
        ],
    )


def sweep_demo_scenario() -> SimScenario:
    """Absolute-frequency scenario spanning the default sweep bands:
    a continuous tone + hopper in 2.4 GHz ISM and a wideband 'video
    link' in the 5.8 GHz region."""
    return SimScenario(
        noise_std=0.01,
        frequencies_absolute=True,
        emitters=[
            ToneEmitter(offset_hz=2.442e9, amplitude=0.06),
            HopperEmitter(
                channel_offsets_hz=tuple(2.405e9 + i * 10e6 for i in range(8)),
                amplitude=0.06,
                hop_period_s=0.008,
                duty=0.8,
            ),
            WidebandEmitter(offset_hz=5.80e9, bandwidth_hz=6e6, amplitude=0.05),
        ],
    )


class SimulatedSource(SDRSource):
    """Generates IQ chunks from a SimScenario; infinite unless duration set."""

    name = "simulated"

    def __init__(
        self,
        scenario: SimScenario | None = None,
        center_hz: float = 2_437e6,
        sample_rate: float = 10e6,
        duration_s: float | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else demo_scenario()
        self.center_hz = center_hz
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self._rng: np.random.Generator | None = None
        self._sample_index = 0
        self._seq = 0
        self._hop_tables: dict[int, np.ndarray] = {}

    def open(self) -> None:
        self._rng = np.random.default_rng(self.scenario.seed)
        self._sample_index = 0
        self._seq = 0
        # Precompute a long pseudorandom hop sequence per hopper so the
        # channel choice is a pure function of hop index (chunk-seamless).
        self._hop_tables = {}
        for i, em in enumerate(self.scenario.emitters):
            if isinstance(em, HopperEmitter):
                rng = np.random.default_rng(em.seed)
                self._hop_tables[i] = rng.integers(
                    0, len(em.channel_offsets_hz), size=65536
                )

    def read_chunk(self, num_samples: int) -> IQChunk | None:
        if self._rng is None:
            raise RuntimeError("SimulatedSource.read_chunk called before open()")
        fs = self.sample_rate
        if self.duration_s is not None and self._sample_index / fs >= self.duration_s:
            return None

        n0 = self._sample_index
        t = (n0 + np.arange(num_samples)) / fs  # absolute time per sample

        out = self._rng.normal(0.0, self.scenario.noise_std, num_samples).astype(
            np.float32
        ) + 1j * self._rng.normal(0.0, self.scenario.noise_std, num_samples).astype(
            np.float32
        )
        out = out.astype(np.complex64)

        absolute = self.scenario.frequencies_absolute
        half_bw = fs / 2

        def to_offset(freq_hz: float) -> float | None:
            """Translate a configured frequency to a baseband offset, or
            None when it falls outside the currently tuned window."""
            off = freq_hz - self.center_hz if absolute else freq_hz
            return off if abs(off) <= half_bw else None

        for i, em in enumerate(self.scenario.emitters):
            if isinstance(em, ToneEmitter):
                off = to_offset(em.offset_hz)
                if off is None:
                    continue
                out += (em.amplitude * np.exp(2j * np.pi * off * t)).astype(
                    np.complex64
                )
            elif isinstance(em, BurstEmitter):
                off = to_offset(em.offset_hz)
                if off is None:
                    continue
                gate = (np.mod(t, em.period_s) < em.burst_s).astype(np.float32)
                out += (
                    em.amplitude * gate * np.exp(2j * np.pi * off * t)
                ).astype(np.complex64)
            elif isinstance(em, HopperEmitter):
                hop_idx = (t / em.hop_period_s).astype(np.int64)
                table = self._hop_tables[i]
                chan = table[hop_idx % len(table)]
                channels = np.asarray(em.channel_offsets_hz, dtype=np.float64)
                if absolute:
                    channels = channels - self.center_hz
                offs = channels[chan]
                gate = (
                    np.mod(t, em.hop_period_s) < em.duty * em.hop_period_s
                ).astype(np.float32)
                # Hops that land outside the tuned window are inaudible.
                gate *= (np.abs(offs) <= half_bw).astype(np.float32)
                out += (em.amplitude * gate * np.exp(2j * np.pi * offs * t)).astype(
                    np.complex64
                )
            elif isinstance(em, WidebandEmitter):
                off = to_offset(em.offset_hz)
                if off is None:
                    continue
                # Band-limited noise: white noise brickwall-filtered in the
                # frequency domain, then shifted to the target offset.
                wn = self._rng.normal(0, 1, num_samples) + 1j * self._rng.normal(
                    0, 1, num_samples
                )
                spec = np.fft.fft(wn)
                f = np.fft.fftfreq(num_samples, d=1 / fs)
                spec[np.abs(f) > em.bandwidth_hz / 2] = 0
                bl = np.fft.ifft(spec)
                rms = np.sqrt(np.mean(np.abs(bl) ** 2)) or 1.0
                out += (
                    em.amplitude * (bl / rms) * np.exp(2j * np.pi * off * t)
                ).astype(np.complex64)
            elif isinstance(em, EMICombEmitter):
                comb = np.zeros(num_samples, dtype=np.complex64)
                for h in range(1, em.n_harmonics + 1):
                    off = to_offset(em.fundamental_hz * h)
                    if off is None:
                        continue
                    a = em.amplitude * (em.decay ** (h - 1))
                    comb += (a * np.exp(2j * np.pi * off * t)).astype(np.complex64)
                out += comb

        self._sample_index += num_samples
        self._seq += 1
        self.health.chunks_produced += 1
        return IQChunk(
            samples=out,
            center_hz=self.center_hz,
            sample_rate=fs,
            timestamp=time.time(),
            seq=self._seq,
        )

    @property
    def supports_retune(self) -> bool:
        return True

    def retune(self, center_hz: float) -> None:
        self.center_hz = center_hz
        self.health.retunes += 1

    def close(self) -> None:
        self._rng = None
