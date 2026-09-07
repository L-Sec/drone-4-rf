"""UI-agnostic scanner session controller.

Owns the pipeline lifecycle for interactive front-ends (the Qt desktop
GUI and the browser dashboard) and bridges worker-thread callbacks into
thread-safe, MULTI-CONSUMER containers:

- the latest spectrum frame is published with a sequence number, so any
  number of viewers can render it without consuming it from each other;
- assessments and track events accumulate in bounded logs keyed by a
  monotone sequence, so each viewer reads "everything after my cursor".

No UI toolkit is imported here; pipeline threads never touch UI objects.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

from drone4rf.analytics import DroneAssessment
from drone4rf.baseline import BaselineBank, BaselineModel
from drone4rf.config import AppConfig
from drone4rf.detectors.base import Detection
from drone4rf.events import EventStore
from drone4rf.pipeline import PipelineStats, ScannerPipeline, SpectrumFrame
from drone4rf.scheduler import SweepScheduler
from drone4rf.session import SessionRecorder, waveform_payload
from drone4rf.sdr.base import IQChunk, SDRSource
from drone4rf.sdr.simulated import (
    SimulatedSource,
    demo_scenario,
    sweep_demo_scenario,
)
from drone4rf.tracking import SignalTrack, TrackEvent

log = logging.getLogger(__name__)

LOG_DEPTH = 300
# Environment names become directory names; keep them filesystem-safe so
# a name arriving from a network request cannot escape the data dir.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
FOCUS_FREQUENCY_RANGE_HZ = (1e6, 6e9)


class ControllerError(RuntimeError):
    """Invalid control request (bad parameters, already running, ...)."""


def validate_environment_name(name: str) -> str:
    if not _SAFE_NAME.match(name or ""):
        raise ControllerError(
            "environment name must be 1-64 characters of letters, digits, "
            "dot, dash or underscore"
        )
    return name


class PipelineController:
    """Runs at most one scanning session; safe to drive from any thread."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.pipeline: ScannerPipeline | None = None
        self._source: SDRSource | None = None
        self._thread: threading.Thread | None = None
        self._store: EventStore | None = None
        self._calibrating = False
        self._environment = "default"
        self._mode = "sweep"
        self._recorder: SessionRecorder | None = None

        self._frame: SpectrumFrame | None = None
        self._frame_seq = 0
        self._waveform: dict | None = None
        self._waveform_seq = 0
        self._last_waveform_at = 0.0
        self._assessments: deque[tuple[int, int, DroneAssessment]] = deque(
            maxlen=LOG_DEPTH
        )
        self._assess_seq = 0
        self._track_events: deque[tuple[int, TrackEvent]] = deque(maxlen=LOG_DEPTH)
        self._track_seq = 0

        self.last_error: str | None = None
        self.status_message = "idle"

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def available_environments(self) -> list[str]:
        root = Path(self.cfg.sweep.environments_dir)
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir() if (p / "manifest.json").exists()
        )

    def start(
        self,
        source_name: str = "hackrf",
        mode: str = "sweep",
        environment: str = "default",
        frozen: bool = False,
        calibrate: bool = False,
        center_hz: float | None = None,
    ) -> None:
        if self.running:
            raise ControllerError("a scan is already running")
        if source_name not in ("hackrf", "sim"):
            raise ControllerError(f"unknown source {source_name!r}")
        if mode not in ("scan", "sweep"):
            raise ControllerError(f"unknown mode {mode!r}")
        environment = validate_environment_name(environment)
        if calibrate and frozen:
            raise ControllerError("cannot calibrate against a frozen baseline")
        if center_hz is not None:
            center_hz = float(center_hz)
            lo, hi = FOCUS_FREQUENCY_RANGE_HZ
            if not math.isfinite(center_hz) or not lo <= center_hz <= hi:
                raise ControllerError("center frequency must be between 1 and 6000 MHz")
            if mode != "scan":
                raise ControllerError("center frequency focus requires scan mode")
            if frozen and center_hz != self.cfg.device.center_freq_hz:
                raise ControllerError(
                    "focused scans at a new center need an adaptive baseline"
                )

        cfg = (
            replace(
                self.cfg,
                device=replace(self.cfg.device, center_freq_hz=center_hz),
            )
            if center_hz is not None else self.cfg
        )
        sweep = mode == "sweep"
        if sweep and not cfg.sweep.enabled_bands():
            raise ControllerError(
                "sweep mode needs at least one enabled band in the config"
            )
        self.last_error = None
        self._calibrating = calibrate
        self._environment = environment
        self._mode = mode

        if source_name == "sim":
            source: SDRSource = SimulatedSource(
                sweep_demo_scenario() if sweep else demo_scenario(),
                center_hz=cfg.device.center_freq_hz,
                sample_rate=cfg.device.sample_rate,
            )
        else:
            from drone4rf.sdr.hackrf import HackRFSource

            source = HackRFSource(cfg.device)
        self._source = source

        scheduler = None
        bank = None
        baseline = None
        if sweep:
            scheduler = SweepScheduler(
                cfg.sweep, cfg.device.sample_rate, cfg.dsp.chunk_samples
            )
            if frozen:
                env_dir = Path(cfg.sweep.environments_dir) / environment
                bank = BaselineBank.load(env_dir, cfg.baseline, cfg.dsp.fft_size)
                bank.freeze_all()
        elif frozen:
            baseline = BaselineModel.load(
                Path(cfg.baseline.path),
                cfg.baseline,
                cfg.device.center_freq_hz,
                cfg.device.sample_rate,
                cfg.dsp.fft_size,
            )
            baseline.frozen = True

        self._store = EventStore(cfg.storage.database_path) if not calibrate else None
        self.pipeline = ScannerPipeline(
            cfg,
            source,
            store=self._store,
            baseline=baseline,
            bank=bank,
            scheduler=scheduler,
            on_detection=self._on_detection,
            on_iq_chunk=self._on_iq_chunk,
            on_spectrum=self._on_spectrum,
            on_track_event=self._on_track_event,
            on_stored_assessment=self._on_stored_assessment,
        )
        self.status_message = (
            f"{'calibrating' if calibrate else 'scanning'} "
            f"({mode}, {source_name}"
            + (
                f", {cfg.device.center_freq_hz / 1e6:.3f} MHz"
                if center_hz is not None else ""
            )
            + (f", env '{environment}' frozen" if frozen else "")
            + ")"
        )
        self._thread = threading.Thread(
            target=self._run, name="dw-session", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=15.0)
        if self._store is not None:
            self._store.close()
            self._store = None
        self.stop_recording()
        if not self.status_message.startswith(("baseline", "environment", "error")):
            self.status_message = "idle"

    def _run(self) -> None:
        assert self.pipeline is not None and self._source is not None
        try:
            with self._source:
                self.pipeline.run()
            if self._calibrating:
                self._save_calibration()
            elif not self.status_message.startswith("error"):
                self.status_message = "idle"
        except Exception as exc:  # surfaced in the UI, never crashes it
            log.exception("scan session failed")
            self.last_error = str(exc)
            self.status_message = f"error: {exc}"

    def _save_calibration(self) -> None:
        assert self.pipeline is not None
        cfg = self.cfg
        if self.pipeline.bank is not None:
            if self.pipeline.bank.any_ready():
                env_dir = Path(cfg.sweep.environments_dir) / self._environment
                self.pipeline.bank.save(env_dir, environment=self._environment)
                ready = sum(1 for m in self.pipeline.bank.models.values() if m.ready)
                self.status_message = (
                    f"environment '{self._environment}' saved "
                    f"({ready} step baselines)"
                )
            else:
                self.status_message = "calibration too short - nothing saved"
        elif self.pipeline.baseline is not None:
            if self.pipeline.baseline.ready:
                self.pipeline.baseline.save(cfg.baseline.path)
                self.status_message = f"baseline saved to {cfg.baseline.path}"
            else:
                self.status_message = "calibration too short - nothing saved"

    # -- pipeline callbacks (worker threads) --------------------------------

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recorder is not None

    def start_recording(
        self,
        path: Path,
        metadata: dict | None = None,
        *,
        capture_cf32: bool = False,
        capture_wav: bool = False,
        capture_csv: bool = False,
    ) -> None:
        if not self.running:
            raise ControllerError("start a scan before recording a workshop session")
        if (capture_cf32 or capture_wav) and self._mode != "scan":
            raise ControllerError(
                "continuous IQ/WAV recording requires single-band scan mode"
            )
        with self._lock:
            if self._recorder is not None:
                raise ControllerError("a workshop session is already recording")
            recorder = SessionRecorder(
                path,
                {**(metadata or {}), "mode": self._mode},
                capture_cf32=capture_cf32,
                capture_wav=capture_wav,
                capture_csv=capture_csv,
            )
            self._recorder = recorder

    def stop_recording(self) -> dict | None:
        with self._lock:
            recorder = self._recorder
            self._recorder = None
        if recorder is None:
            return None
        stats = self.snapshot()
        runtime = (
            {
                "pipeline_queue_drops": stats.queue_drops,
                "source_overflows": stats.source_overflows,
                "clipped_chunks": stats.clipped_chunks,
                "chunks_processed": stats.chunks_processed,
            }
            if stats is not None else {}
        )
        return recorder.close(runtime=runtime)

    def record_label(self, label: dict) -> None:
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            recorder.label(label)

    def _on_detection(self, det: Detection) -> None:
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            recorder.detection(det)

    def _on_iq_chunk(self, chunk: IQChunk) -> None:
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            recorder.iq(chunk)

        now = time.monotonic()
        with self._lock:
            if now - self._last_waveform_at < 0.08:
                return
            self._last_waveform_at = now
        payload = waveform_payload(chunk)
        with self._lock:
            self._waveform = payload
            self._waveform_seq += 1
            recorder = self._recorder
        if recorder is not None:
            recorder.waveform(payload)

    def _on_spectrum(self, frame: SpectrumFrame) -> None:
        with self._lock:
            self._frame = frame
            self._frame_seq += 1
            recorder = self._recorder
        if recorder is not None:
            recorder.spectrum(frame)

    def _on_track_event(self, ev: TrackEvent) -> None:
        with self._lock:
            self._track_seq += 1
            self._track_events.append((self._track_seq, ev))
            recorder = self._recorder
        if recorder is not None:
            recorder.track(ev)

    def _on_stored_assessment(self, event_id: int, a: DroneAssessment) -> None:
        with self._lock:
            self._assess_seq += 1
            self._assessments.append((self._assess_seq, event_id, a))
            recorder = self._recorder
        if recorder is not None:
            recorder.assessment(event_id, a)

    # -- viewer reads (any thread, non-destructive) --------------------------

    def peek_spectrum(self) -> tuple[int, SpectrumFrame] | None:
        """Latest frame with its sequence; None before the first frame."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame_seq, self._frame

    def peek_waveform(self) -> tuple[int, dict] | None:
        with self._lock:
            if self._waveform is None:
                return None
            return self._waveform_seq, self._waveform

    def assessments_since(
        self, cursor: int
    ) -> list[tuple[int, int, DroneAssessment]]:
        with self._lock:
            return [
                (s, event_id, a)
                for s, event_id, a in self._assessments
                if s > cursor
            ]

    def track_events_since(self, cursor: int) -> list[tuple[int, TrackEvent]]:
        with self._lock:
            return [(s, e) for s, e in self._track_events if s > cursor]

    def snapshot(self) -> PipelineStats | None:
        return self.pipeline.snapshot() if self.pipeline is not None else None

    def active_tracks(self) -> list[SignalTrack]:
        return self.pipeline.tracker.active_tracks if self.pipeline else []
