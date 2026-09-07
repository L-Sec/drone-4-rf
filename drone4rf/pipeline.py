"""Scanner pipeline: acquisition thread -> bounded queue -> processing.

Two operating modes share one processing chain:

- **single-band** (Stage 2): the source stays on one center frequency
  and one BaselineModel adapts to it;
- **sweep** (Stage 3): a SweepScheduler drives the acquisition thread
  through retune/settle/dwell cycles across configured bands, each
  visited center gets its own baseline from a BaselineBank, the tracker
  only judges tracks inside the currently observed window, and step
  activity feeds back into adaptive revisit scheduling.

Threading contract (see docs/DESIGN.md §3):
- the acquisition thread reads chunks (and, in sweep mode, retunes);
  on a full queue it drops the OLDEST chunk and counts the drop;
- the processing thread runs DSP/detect/track/store;
- shutdown is a single Event; both threads are joined on stop().
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from drone4rf.analytics import AnalyticsEngine, DroneAssessment
from drone4rf.analytics.temporal import compute_temporal_profile
from drone4rf.baseline import BaselineBank, BaselineModel
from drone4rf.capture import IQCaptureWriter
from drone4rf.config import AppConfig
from drone4rf.detectors import Detection, EnergyDetector, OSCFARDetector
from drone4rf.dsp import preprocess
from drone4rf.dsp.noise_floor import estimate_noise_floor
from drone4rf.events import EventStore
from drone4rf.scheduler import SweepScheduler
from drone4rf.sdr.base import IQChunk, SDRSource
from drone4rf.tracking import SignalTracker, TrackEvent

log = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    chunks_processed: int = 0
    queue_drops: int = 0
    source_overflows: int = 0
    clipped_chunks: int = 0
    detections: int = 0
    track_events: int = 0
    steps_visited: int = 0
    captures_written: int = 0
    assessments: int = 0
    noise_floor_db: float = float("nan")
    current_center_hz: float = float("nan")
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpectrumFrame:
    """One processed PSD frame, published to UI consumers."""

    timestamp: float
    center_hz: float
    sample_rate: float
    freqs_hz: np.ndarray
    psd_db: np.ndarray
    threshold_db: np.ndarray | None  # None until the baseline is ready
    noise_floor_db: float


@dataclass
class _BandContext:
    """Per-(center, sample_rate) processing state, created lazily."""

    freqs: np.ndarray
    baseline: BaselineModel


class ScannerPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        source: SDRSource,
        store: EventStore | None = None,
        baseline: BaselineModel | None = None,
        bank: BaselineBank | None = None,
        scheduler: SweepScheduler | None = None,
        on_detection: Callable[[Detection], None] | None = None,
        on_track_event: Callable[[TrackEvent], None] | None = None,
        on_assessment: Callable[[DroneAssessment], None] | None = None,
        on_stored_assessment: Callable[[int, DroneAssessment], None] | None = None,
        on_spectrum: Callable[[SpectrumFrame], None] | None = None,
        on_iq_chunk: Callable[[IQChunk], None] | None = None,
        update_baseline: bool = True,
        log_raw_detections: bool = False,
    ) -> None:
        self.cfg = cfg
        self.source = source
        self.store = store
        self.scheduler = scheduler
        self.on_detection = on_detection
        self.on_track_event = on_track_event
        self.on_assessment = on_assessment
        self.on_stored_assessment = on_stored_assessment
        self.on_spectrum = on_spectrum
        self.on_iq_chunk = on_iq_chunk
        self.update_baseline = update_baseline
        # Raw per-frame detections are numerous; by default only track
        # events (persistence/closure) are written to the database.
        self.log_raw_detections = log_raw_detections

        self.stats = PipelineStats()
        if scheduler is not None:
            if not source.supports_retune:
                raise ValueError(
                    f"source '{source.name}' cannot retune; sweep mode "
                    "requires a retunable source (hackrf or sim)"
                )
            self.bank = bank or BaselineBank(cfg.baseline, cfg.dsp.fft_size)
            self.baseline: BaselineModel | None = None
        else:
            self.bank = None
            self.baseline = baseline or BaselineModel(
                n_bins=cfg.dsp.fft_size,
                cfg=cfg.baseline,
                center_hz=cfg.device.center_freq_hz,
                sample_rate=cfg.device.sample_rate,
            )

        self.tracker = SignalTracker(cfg.tracking)
        self.analytics = AnalyticsEngine(cfg)
        self.capture_writer = IQCaptureWriter(cfg.sweep.capture)
        self._seg_step_s = preprocess.segment_step_seconds(
            cfg.dsp.fft_size, cfg.dsp.overlap, cfg.device.sample_rate
        )
        self._energy = EnergyDetector(cfg.detection.energy)
        self._cfar = OSCFARDetector(cfg.detection.cfar)
        self._contexts: dict[tuple[int, int], _BandContext] = {}
        self._pending_captures: dict[int, int] = {}  # center key -> chunks left
        self._chunk_duration = cfg.dsp.chunk_samples / cfg.device.sample_rate
        self._tuned_center: float | None = None

        self._queue: queue.Queue[IQChunk] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._acq_thread: threading.Thread | None = None
        self._proc_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def run(self, duration_s: float | None = None) -> PipelineStats:
        """Run until the source ends, duration elapses, or stop() is called."""
        self._stop.clear()
        acquire = self._acquire_sweep if self.scheduler else self._acquire
        self._acq_thread = threading.Thread(
            target=acquire, name="dw-acquire", daemon=True
        )
        self._proc_thread = threading.Thread(
            target=self._process, name="dw-process", daemon=True
        )
        deadline = time.time() + duration_s if duration_s is not None else None
        try:
            self._acq_thread.start()
            self._proc_thread.start()
            while not self._stop.is_set():
                if deadline is not None and time.time() >= deadline:
                    break
                if not self._acq_thread.is_alive() and self._queue.empty():
                    break
                time.sleep(0.1)
        finally:
            self.stop()
        return self.stats

    def stop(self) -> None:
        self._stop.set()
        for t in (self._acq_thread, self._proc_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)

    # -- acquisition ------------------------------------------------------

    def _enqueue(self, chunk: IQChunk) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest chunk to keep latency bounded.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)
            with self._lock:
                self.stats.queue_drops += 1

    def _acquire(self) -> None:
        """Single-band acquisition: read and enqueue until exhausted."""
        try:
            while not self._stop.is_set():
                chunk = self.source.read_chunk(self.cfg.dsp.chunk_samples)
                if chunk is None:
                    log.info("source exhausted")
                    break
                self._enqueue(chunk)
        except Exception:
            log.exception("acquisition thread failed")
        finally:
            self._stop_when_drained()

    def _acquire_sweep(self) -> None:
        """Sweep acquisition: retune / settle / dwell / reschedule."""
        assert self.scheduler is not None
        n = self.cfg.dsp.chunk_samples
        try:
            while not self._stop.is_set():
                step = self.scheduler.next_step(time.time())
                if self._tuned_center != step.center_hz:
                    self.source.retune(step.center_hz)
                    self._tuned_center = step.center_hz
                    # Discard settle chunks: LO transient after retune
                    # must not reach the detectors.
                    for _ in range(self.cfg.sweep.settle_chunks):
                        if self.source.read_chunk(n) is None:
                            log.info("source exhausted during settle")
                            return
                for _ in range(step.dwell_chunks):
                    if self._stop.is_set():
                        return
                    chunk = self.source.read_chunk(n)
                    if chunk is None:
                        log.info("source exhausted")
                        return
                    self._enqueue(chunk)
                self.scheduler.complete(step.step_id, time.time())
                with self._lock:
                    self.stats.steps_visited += 1
        except Exception:
            log.exception("sweep acquisition thread failed")
        finally:
            self._stop_when_drained()

    def _stop_when_drained(self) -> None:
        """Let the processor finish queued chunks, then signal stop."""
        while not self._queue.empty() and not self._stop.is_set():
            time.sleep(0.05)
        self._stop.set()

    # -- processing -------------------------------------------------------

    def _process(self) -> None:
        try:
            while True:
                try:
                    chunk = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        return
                    continue
                self._process_chunk(chunk)
        except Exception:
            log.exception("processing thread failed")
            self._stop.set()

    def _context_for(self, chunk: IQChunk) -> _BandContext:
        key = (int(round(chunk.center_hz)), int(round(chunk.sample_rate)))
        ctx = self._contexts.get(key)
        if ctx is None:
            freqs = preprocess.bin_frequencies(
                chunk.center_hz, chunk.sample_rate, self.cfg.dsp.fft_size
            )
            if self.bank is not None:
                model = self.bank.get(chunk.center_hz, chunk.sample_rate)
            else:
                assert self.baseline is not None
                model = self.baseline
            ctx = _BandContext(freqs=freqs, baseline=model)
            self._contexts[key] = ctx
        return ctx

    def _process_chunk(self, chunk: IQChunk) -> None:
        cfg = self.cfg
        if self.on_iq_chunk is not None:
            self.on_iq_chunk(chunk)
        ctx = self._context_for(chunk)
        quality: list[str] = []
        if chunk.overflow:
            quality.append("usb_overflow")
            with self._lock:
                self.stats.source_overflows += 1

        clip = preprocess.clip_fraction(chunk.samples)
        if clip > cfg.detection.clip_fraction_warn:
            quality.append("clipping")
            with self._lock:
                self.stats.clipped_chunks += 1
                msg = (
                    f"receiver overload: {clip * 100:.2f}% of samples near full "
                    f"scale - reduce VGA/LNA gain"
                )
                if msg not in self.stats.warnings:
                    self.stats.warnings.append(msg)

        samples = preprocess.remove_dc(chunk.samples)
        # One FFT pass yields both the averaged PSD (detection) and the
        # spectrogram (sub-chunk burst/hop timing for Stage 4 analytics).
        spec_db, psd = preprocess.spectrogram_and_psd(
            samples, cfg.dsp.fft_size, cfg.dsp.overlap, cfg.dsp.window
        )
        psd = preprocess.mask_dc_spike(psd, cfg.dsp.dc_mask_bins)

        nf = estimate_noise_floor(psd)
        if self.update_baseline:
            ctx.baseline.update(psd)

        if self.on_spectrum is not None:
            self.on_spectrum(
                SpectrumFrame(
                    timestamp=chunk.timestamp,
                    center_hz=chunk.center_hz,
                    sample_rate=chunk.sample_rate,
                    freqs_hz=ctx.freqs,
                    psd_db=psd,
                    threshold_db=(
                        ctx.baseline.threshold_db() if ctx.baseline.ready else None
                    ),
                    noise_floor_db=nf.floor_db,
                )
            )

        detections: list[Detection] = []
        flags = tuple(quality)
        if ctx.baseline.ready:
            detections += self._energy.detect(
                psd,
                ctx.baseline.threshold_db(),
                ctx.baseline.level_db(),
                ctx.freqs,
                chunk.timestamp,
                self._chunk_duration,
                flags,
            )
        detections += self._cfar.detect(
            psd, ctx.freqs, chunk.timestamp, self._chunk_duration, flags
        )
        # Operator-configured exclusion ranges (known local emitters).
        if cfg.sweep.exclusions:
            detections = [
                d for d in detections if not cfg.sweep.is_excluded(d.center_hz)
            ]

        # Sub-chunk temporal profiles (duty, burst count, period) feed
        # the burst/hop analytics via detection features and track history.
        for det in detections:
            a = det.features.get("bin_start")
            b = det.features.get("bin_end")
            if a is not None and b is not None:
                prof = compute_temporal_profile(spec_db, a, b, self._seg_step_s)
                det.features["duty"] = prof.duty
                det.features["n_bursts"] = prof.n_bursts
                det.features["burst_period_s"] = prof.burst_period_s
                det.features["period_cv"] = prof.period_cv
        self.analytics.observe(chunk.timestamp, detections)

        window = (
            chunk.center_hz - chunk.sample_rate / 2,
            chunk.center_hz + chunk.sample_rate / 2,
        )
        track_events = self.tracker.update(
            detections, chunk.timestamp, window=window
        )
        if self.scheduler is not None:
            self.scheduler.report_activity(
                chunk.center_hz, len(detections), time.time()
            )

        capture_path = self._maybe_capture(chunk, track_events)
        assessments = self.analytics.maybe_assess(
            time.time(), self.tracker.active_tracks
        )

        with self._lock:
            self.stats.chunks_processed += 1
            self.stats.detections += len(detections)
            self.stats.track_events += len(track_events)
            self.stats.assessments += len(assessments)
            self.stats.noise_floor_db = nf.floor_db
            self.stats.current_center_hz = chunk.center_hz
            self.stats.captures_written = self.capture_writer.files_written

        for det in detections:
            if self.store is not None and self.log_raw_detections:
                self.store.log_detection(det, self.source.name)
            if self.on_detection is not None:
                self.on_detection(det)
        for ev in track_events:
            if self.store is not None:
                self.store.log_track_event(
                    ev,
                    self.source.name,
                    capture_path=capture_path if ev.kind == "track_persistent" else None,
                )
            if self.on_track_event is not None:
                self.on_track_event(ev)
        for assessment in assessments:
            event_id = None
            if self.store is not None:
                event_id = self.store.log_assessment(assessment, self.source.name)
            if event_id is not None and self.on_stored_assessment is not None:
                self.on_stored_assessment(event_id, assessment)
            if self.on_assessment is not None:
                self.on_assessment(assessment)

    def _maybe_capture(
        self, chunk: IQChunk, track_events: list[TrackEvent]
    ) -> str | None:
        """Write triggered IQ captures: the promoting chunk plus the next
        chunks_per_event-1 chunks observed at the same center."""
        center_key = int(round(chunk.center_hz))
        promoted = [e for e in track_events if e.kind == "track_persistent"]
        if promoted:
            reason = (
                f"track_persistent @ {promoted[0].track.center_hz / 1e6:.3f} MHz"
            )
            path = self.capture_writer.write(chunk, reason=reason)
            if path is not None:
                self._pending_captures[center_key] = (
                    self.cfg.sweep.capture.chunks_per_event - 1
                )
            return str(path) if path is not None else None
        pending = self._pending_captures.get(center_key, 0)
        if pending > 0:
            self.capture_writer.write(chunk, reason="follow-up")
            self._pending_captures[center_key] = pending - 1
        return None

    # -- introspection ----------------------------------------------------

    def snapshot(self) -> PipelineStats:
        with self._lock:
            return PipelineStats(
                chunks_processed=self.stats.chunks_processed,
                queue_drops=self.stats.queue_drops,
                source_overflows=self.stats.source_overflows,
                clipped_chunks=self.stats.clipped_chunks,
                detections=self.stats.detections,
                track_events=self.stats.track_events,
                steps_visited=self.stats.steps_visited,
                captures_written=self.stats.captures_written,
                assessments=self.stats.assessments,
                noise_floor_db=self.stats.noise_floor_db,
                current_center_hz=self.stats.current_center_hz,
                warnings=list(self.stats.warnings),
            )
