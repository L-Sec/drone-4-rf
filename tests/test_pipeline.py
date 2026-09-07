"""End-to-end tests: simulated IQ -> DSP -> detectors -> tracks -> SQLite."""

from pathlib import Path

import numpy as np

from drone4rf.config import AppConfig
from drone4rf.events import EventStore
from drone4rf.pipeline import ScannerPipeline
from drone4rf.sdr.simulated import (
    SimScenario,
    SimulatedSource,
    ToneEmitter,
)


def _run(cfg: AppConfig, scenario: SimScenario, tmp_path: Path, seconds: float = 0.5):
    src = SimulatedSource(
        scenario,
        center_hz=cfg.device.center_freq_hz,
        sample_rate=cfg.device.sample_rate,
        duration_s=seconds,
    )
    store = EventStore(tmp_path / "events.db")
    pipeline = ScannerPipeline(cfg, src, store=store)
    with src:
        stats = pipeline.run()
    return pipeline, store, stats


def test_tone_produces_persistent_track_event(fast_config: AppConfig, tmp_path) -> None:
    scenario = SimScenario(
        noise_std=0.005, emitters=[ToneEmitter(offset_hz=1.5e6, amplitude=0.1)]
    )
    pipeline, store, stats = _run(fast_config, scenario, tmp_path)

    assert stats.chunks_processed >= 10
    assert stats.detections > 0
    events = store.recent(limit=50)
    persistent = [e for e in events if e["kind"] == "track_persistent"]
    assert persistent, "tone should have been promoted to a persistent track"
    ev = persistent[0]
    assert abs(ev["center_hz"] - (fast_config.device.center_freq_hz + 1.5e6)) < 200e3
    assert ev["explanation"]
    assert ev["confidence_label"] == "unclassified_rf_activity"
    assert ev["config_version"]
    store.close()


def test_noise_only_produces_no_track_events(fast_config: AppConfig, tmp_path) -> None:
    scenario = SimScenario(noise_std=0.005, emitters=[])
    _, store, stats = _run(fast_config, scenario, tmp_path)
    assert stats.chunks_processed >= 10
    assert store.count() == 0
    store.close()


def test_clipping_is_flagged(fast_config: AppConfig, tmp_path) -> None:
    # Amplitude 1.0 tone + noise pushes many samples past the clip threshold.
    scenario = SimScenario(
        noise_std=0.02, emitters=[ToneEmitter(offset_hz=0.8e6, amplitude=1.2)]
    )
    _, store, stats = _run(fast_config, scenario, tmp_path, seconds=0.2)
    assert stats.clipped_chunks > 0
    assert any("overload" in w for w in stats.warnings)
    store.close()


def test_noise_floor_estimate_is_sane(fast_config: AppConfig, tmp_path) -> None:
    scenario = SimScenario(noise_std=0.005, emitters=[])
    _, store, stats = _run(fast_config, scenario, tmp_path, seconds=0.2)
    assert np.isfinite(stats.noise_floor_db)
    assert -140 < stats.noise_floor_db < -40
    store.close()
