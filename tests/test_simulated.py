import numpy as np

from drone4rf.dsp import preprocess
from drone4rf.sdr.simulated import (
    BurstEmitter,
    HopperEmitter,
    SimScenario,
    SimulatedSource,
    ToneEmitter,
    demo_scenario,
)

FS = 10e6


def test_tone_appears_at_configured_offset() -> None:
    src = SimulatedSource(
        SimScenario(noise_std=0.001, emitters=[ToneEmitter(1.0e6, 0.1)]),
        center_hz=0.0,
        sample_rate=FS,
    )
    with src:
        chunk = src.read_chunk(65_536)
    assert chunk is not None
    psd = preprocess.compute_psd(chunk.samples, 1024)
    freqs = preprocess.bin_frequencies(0.0, FS, 1024)
    assert abs(freqs[int(np.argmax(psd))] - 1.0e6) < 2 * FS / 1024


def test_burst_duty_cycle() -> None:
    duty = 0.2
    src = SimulatedSource(
        SimScenario(
            noise_std=0.001,
            emitters=[BurstEmitter(0.5e6, 0.5, burst_s=duty * 0.01, period_s=0.01)],
        ),
        sample_rate=FS,
    )
    with src:
        chunk = src.read_chunk(int(FS * 0.1))  # 10 burst periods
    assert chunk is not None
    on_fraction = np.mean(np.abs(chunk.samples) > 0.25)
    assert abs(on_fraction - duty) < 0.05


def test_hopper_visits_multiple_channels() -> None:
    channels = tuple(-3e6 + i * 1e6 for i in range(6))
    hop_period = 0.002
    src = SimulatedSource(
        SimScenario(
            noise_std=0.001,
            emitters=[HopperEmitter(channels, 0.2, hop_period_s=hop_period)],
        ),
        sample_rate=FS,
    )
    hop_samples = int(hop_period * FS)
    seen: set[float] = set()
    with src:
        for _ in range(24):  # 24 consecutive hop intervals
            chunk = src.read_chunk(hop_samples)
            assert chunk is not None
            spec = np.abs(np.fft.fftshift(np.fft.fft(chunk.samples)))
            freqs = np.fft.fftshift(np.fft.fftfreq(hop_samples, d=1 / FS))
            peak = freqs[int(np.argmax(spec))]
            # Snap to the nearest configured channel.
            nearest = min(channels, key=lambda c: abs(c - peak))
            assert abs(nearest - peak) < 100e3
            seen.add(nearest)
    assert len(seen) >= 3  # the PRNG sequence must actually hop around


def test_absolute_mode_emitter_follows_retune() -> None:
    """An absolute-frequency tone is visible only when the source is
    tuned to a window containing it."""
    tone_abs = 2.442e9
    src = SimulatedSource(
        SimScenario(
            noise_std=0.001,
            frequencies_absolute=True,
            emitters=[ToneEmitter(offset_hz=tone_abs, amplitude=0.2)],
        ),
        center_hz=2.440e9,
        sample_rate=FS,
    )
    with src:
        chunk = src.read_chunk(65_536)
        assert chunk is not None
        psd = preprocess.compute_psd(chunk.samples, 1024)
        freqs = preprocess.bin_frequencies(2.440e9, FS, 1024)
        assert abs(freqs[int(np.argmax(psd))] - tone_abs) < 2 * FS / 1024

        src.retune(5.8e9)  # tone far outside the window -> silence
        chunk2 = src.read_chunk(65_536)
        assert chunk2 is not None
        assert chunk2.center_hz == 5.8e9
        psd2 = preprocess.compute_psd(chunk2.samples, 1024)
        assert psd2.max() < -40.0  # nothing but the noise floor


def test_determinism_with_fixed_seed() -> None:
    def first_chunk() -> np.ndarray:
        src = SimulatedSource(demo_scenario(), sample_rate=FS)
        with src:
            chunk = src.read_chunk(8192)
        assert chunk is not None
        return chunk.samples

    assert np.array_equal(first_chunk(), first_chunk())


def test_duration_limit_ends_stream() -> None:
    src = SimulatedSource(demo_scenario(), sample_rate=FS, duration_s=0.01)
    with src:
        first = src.read_chunk(int(FS * 0.01))
        second = src.read_chunk(1024)
    assert first is not None
    assert second is None
