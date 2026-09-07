"""Stage 5 tests: spectrum callback, event feedback, GUI smoke test."""

import os
import time

import numpy as np
import pytest

from drone4rf.config import AppConfig
from drone4rf.events import EventStore
from drone4rf.pipeline import ScannerPipeline, SpectrumFrame
from drone4rf.sdr.simulated import SimScenario, SimulatedSource, ToneEmitter


def test_on_spectrum_callback_delivers_frames(fast_config: AppConfig, tmp_path) -> None:
    frames: list[SpectrumFrame] = []
    scenario = SimScenario(
        noise_std=0.005, emitters=[ToneEmitter(offset_hz=1.5e6, amplitude=0.1)]
    )
    source = SimulatedSource(
        scenario,
        center_hz=fast_config.device.center_freq_hz,
        sample_rate=fast_config.device.sample_rate,
        duration_s=0.2,
    )
    pipeline = ScannerPipeline(fast_config, source, on_spectrum=frames.append)
    with source:
        pipeline.run()
    assert len(frames) >= 10
    f = frames[-1]
    assert len(f.psd_db) == fast_config.dsp.fft_size
    assert len(f.freqs_hz) == fast_config.dsp.fft_size
    assert f.center_hz == fast_config.device.center_freq_hz
    assert np.isfinite(f.noise_floor_db)
    assert f.threshold_db is not None  # baseline ready by frame 10+


def test_event_feedback_roundtrip(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    from drone4rf.detectors.base import Detection

    det = Detection(
        detector="energy", timestamp=time.time(), center_hz=2.4e9,
        bandwidth_hz=1e5, peak_db=-50.0, avg_db=-55.0, snr_db=20.0,
        duration_s=0.02, score=0.5,
    )
    store.log_detection(det, "simulated")
    row = store.recent(limit=1)[0]
    assert row["user_feedback"] is None
    store.set_feedback(row["id"], "false_positive")
    assert store.recent(limit=1)[0]["user_feedback"] == "false_positive"
    store.close()


@pytest.mark.skipif(
    os.environ.get("DW_SKIP_GUI_TESTS") == "1", reason="GUI tests disabled"
)
def test_gui_smoke_offscreen(fast_config: AppConfig, tmp_path, monkeypatch) -> None:
    """Construct the main window offscreen, feed it a spectrum frame, and
    exercise the poll paths without a running pipeline."""
    pytest.importorskip("PyQt6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from drone4rf.config import (
        AppConfig as AC,
        DeviceConfig,
        DSPConfig,
        StorageConfig,
    )
    from drone4rf.gui.app import MainWindow

    cfg = AC(
        device=DeviceConfig(),
        dsp=DSPConfig(fft_size=1024, chunk_samples=65_536),
        storage=StorageConfig(database_path=str(tmp_path / "events.db")),
    )
    cfg.validate()

    app = QApplication.instance() or QApplication([])
    window = MainWindow(cfg)

    n = cfg.dsp.fft_size
    frame = SpectrumFrame(
        timestamp=time.time(),
        center_hz=2.437e9,
        sample_rate=10e6,
        freqs_hz=2.437e9 + np.linspace(-5e6, 5e6, n),
        psd_db=np.random.default_rng(0).normal(-80, 1, n),
        threshold_db=np.full(n, -70.0),
        noise_floor_db=-80.0,
    )
    # Inject through the same callback the pipeline uses.
    window.controller._on_spectrum(frame)
    window._poll_spectrum()
    assert window._wf_buffer is not None
    assert window._wf_buffer.shape[1] == n

    window._poll_slow()  # no pipeline: must not raise
    window._refresh_events()  # empty DB: must not raise
    window.close()
    app.processEvents()
