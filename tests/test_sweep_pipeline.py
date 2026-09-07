"""End-to-end sweep tests: two bands, retunes, exclusions, captures."""

from pathlib import Path

from drone4rf.config import (
    AppConfig,
    BandConfig,
    BaselineConfig,
    CaptureConfig,
    DeviceConfig,
    DSPConfig,
    ExclusionRange,
    SweepConfig,
    TrackingConfig,
)
from drone4rf.events import EventStore
from drone4rf.pipeline import ScannerPipeline
from drone4rf.scheduler import SweepScheduler
from drone4rf.sdr.simulated import SimScenario, SimulatedSource, ToneEmitter

FS = 10e6
# Tones sit ~1 MHz off their band's step center: a tone exactly at the
# step center would coincide with DC and be removed by the DC-spike mask.
TONE_A = 2.4035e9  # band_a step center is 2.4025 GHz
TONE_B = 5.7265e9  # band_b step center is 5.7275 GHz


def _sweep_config(tmp_path: Path, exclusions=(), capture=None) -> AppConfig:
    cfg = AppConfig(
        device=DeviceConfig(sample_rate=FS, center_freq_hz=2.4e9),
        dsp=DSPConfig(fft_size=1024, chunk_samples=65_536),
        baseline=BaselineConfig(),
        tracking=TrackingConfig(promote_hits=4, drop_after_misses=6),
        sweep=SweepConfig(
            bands=(
                BandConfig(name="band_a", start_hz=2.400e9, stop_hz=2.405e9,
                           dwell_s=0.04),
                BandConfig(name="band_b", start_hz=5.725e9, stop_hz=5.730e9,
                           dwell_s=0.04),
            ),
            exclusions=tuple(exclusions),
            capture=capture or CaptureConfig(),
            environments_dir=str(tmp_path / "envs"),
        ),
    )
    cfg.validate()
    return cfg


def _scenario() -> SimScenario:
    return SimScenario(
        noise_std=0.005,
        frequencies_absolute=True,
        emitters=[
            ToneEmitter(offset_hz=TONE_A, amplitude=0.1),
            ToneEmitter(offset_hz=TONE_B, amplitude=0.1),
        ],
    )


def _run(cfg: AppConfig, tmp_path: Path, seconds: float = 2.0):
    source = SimulatedSource(
        _scenario(),
        center_hz=cfg.device.center_freq_hz,
        sample_rate=FS,
        duration_s=seconds,
    )
    scheduler = SweepScheduler(cfg.sweep, FS, cfg.dsp.chunk_samples)
    store = EventStore(tmp_path / "events.db")
    pipeline = ScannerPipeline(cfg, source, store=store, scheduler=scheduler)
    with source:
        stats = pipeline.run()
    return pipeline, store, stats


def test_sweep_visits_both_bands_and_detects_both_tones(tmp_path) -> None:
    cfg = _sweep_config(tmp_path)
    pipeline, store, stats = _run(cfg, tmp_path)

    assert stats.steps_visited >= 4  # multiple visits to both steps
    events = store.recent(limit=100)
    persistent = [e for e in events if e["kind"] == "track_persistent"]
    freqs = sorted(e["center_hz"] for e in persistent)
    assert any(abs(f - TONE_A) < 1e6 for f in freqs), f"tone A missing: {freqs}"
    assert any(abs(f - TONE_B) < 1e6 for f in freqs), f"tone B missing: {freqs}"
    store.close()


def test_track_survives_absence_while_other_band_observed(tmp_path) -> None:
    """The band-A track must not be closed for misses accrued while the
    receiver was tuned to band B (window-aware tracking)."""
    cfg = _sweep_config(tmp_path)
    pipeline, store, stats = _run(cfg, tmp_path)
    closures = [
        e for e in store.recent(limit=100) if e["kind"] == "track_closed"
    ]
    # Tones are continuous: no track should ever close mid-run.
    assert closures == []
    store.close()


def test_exclusion_range_suppresses_detections(tmp_path) -> None:
    excl = ExclusionRange(start_hz=TONE_B - 2e6, stop_hz=TONE_B + 2e6)
    cfg = _sweep_config(tmp_path, exclusions=[excl])
    pipeline, store, stats = _run(cfg, tmp_path)
    persistent = [
        e for e in store.recent(limit=100) if e["kind"] == "track_persistent"
    ]
    assert any(abs(e["center_hz"] - TONE_A) < 1e6 for e in persistent)
    assert not any(abs(e["center_hz"] - TONE_B) < 1e6 for e in persistent)
    store.close()


def test_triggered_capture_writes_files_with_sidecars(tmp_path) -> None:
    capture = CaptureConfig(
        enabled=True, dir=str(tmp_path / "caps"), max_files=6, chunks_per_event=2
    )
    cfg = _sweep_config(tmp_path, capture=capture)
    pipeline, store, stats = _run(cfg, tmp_path)

    caps = sorted((tmp_path / "caps").glob("*.cf32"))
    sidecars = sorted((tmp_path / "caps").glob("*.json"))
    assert caps, "expected at least one triggered capture"
    assert len(caps) <= capture.max_files
    assert len(sidecars) == len(caps)
    # Promotion events reference their capture file.
    persistent = [
        e for e in store.recent(limit=100) if e["kind"] == "track_persistent"
    ]
    assert any(e["capture_path"] for e in persistent)
    store.close()


def test_capture_disabled_by_default(tmp_path) -> None:
    cfg = _sweep_config(tmp_path)
    pipeline, store, stats = _run(cfg, tmp_path, seconds=1.0)
    assert stats.captures_written == 0
    store.close()


def test_sweep_requires_retunable_source(tmp_path) -> None:
    import pytest

    from drone4rf.sdr.file_source import FileSource

    cfg = _sweep_config(tmp_path)
    scheduler = SweepScheduler(cfg.sweep, FS, cfg.dsp.chunk_samples)
    src = FileSource(tmp_path / "x.cf32")
    with pytest.raises(ValueError, match="retune"):
        ScannerPipeline(cfg, src, scheduler=scheduler)
