"""Synchronized recording and deterministic replay of RF survey sessions.

The compressed session stream always contains processed spectra, waveform
previews, and detector output. Optional sidecars retain continuous raw IQ as
Inspectrum-compatible CF32, stereo PCM16 IQ/WAV, and detector-ready CSV.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import queue
import re
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from drone4rf import __version__
from drone4rf.analytics import DroneAssessment
from drone4rf.detectors.base import Detection
from drone4rf.pipeline import SpectrumFrame
from drone4rf.sdr.base import IQChunk
from drone4rf.tracking import TrackEvent

SESSION_SCHEMA_VERSION = "1.0"
SESSION_EXTENSION = ".dwsession"
SPECTRUM_POINTS = 1024
SPECTRUM_INTERVAL_S = 0.08
WAVEFORM_POINTS = 512
IQ_QUEUE_CHUNKS = 16
WAV_MAX_DATA_BYTES = 1_000_000_000
CSV_FIELDS = (
    "record_type", "relative_s", "timestamp", "center_hz", "sample_rate",
    "detector", "bandwidth_hz", "peak_db", "avg_db", "median_db",
    "noise_floor_db", "snr_db", "duration_s", "score", "flags",
    "features_json", "explanation",
)
_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class SessionError(ValueError):
    """Invalid workshop-session name, file, or operation."""


def validate_session_name(value: str) -> str:
    """Return a safe filename (without an extension)."""
    name = str(value or "").strip()
    if name.endswith(SESSION_EXTENSION):
        name = name[: -len(SESSION_EXTENSION)]
    if not _SAFE_SESSION_NAME.fullmatch(name) or name in (".", ".."):
        raise SessionError(
            "session name must be 1-80 letters, digits, dots, dashes or underscores"
        )
    return name


def default_session_name() -> str:
    return datetime.now(timezone.utc).strftime("workshop-%Y%m%dT%H%M%SZ")


def session_path(root: Path, name: str) -> Path:
    return root / (validate_session_name(name) + SESSION_EXTENSION)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _decimate_peak(values: np.ndarray, target: int = SPECTRUM_POINTS) -> list[float]:
    n = len(values)
    if n <= target:
        return [round(float(v), 1) for v in values]
    bucket = n // target
    usable = bucket * target
    reduced = values[:usable].reshape(target, bucket).max(axis=1)
    if usable < n:
        reduced[-1] = max(reduced[-1], float(values[usable:].max()))
    return [round(float(v), 1) for v in reduced]


def spectrum_payload(frame: SpectrumFrame) -> dict[str, Any]:
    return {
        "center_mhz": frame.center_hz / 1e6,
        "f0_mhz": float(frame.freqs_hz[0] / 1e6),
        "f1_mhz": float(frame.freqs_hz[-1] / 1e6),
        "psd": _decimate_peak(frame.psd_db),
        "thr": (
            _decimate_peak(frame.threshold_db)
            if frame.threshold_db is not None
            else None
        ),
        "noise_db": _num(frame.noise_floor_db),
    }


def waveform_payload(chunk: IQChunk, target: int = WAVEFORM_POINTS) -> dict[str, Any]:
    """Small raw-IQ preview for the live UI and compact replay stream."""
    samples = np.asarray(chunk.samples, dtype=np.complex64)
    if len(samples) > target:
        start = (len(samples) - target) // 2
        samples = samples[start : start + target]
    return {
        "timestamp": chunk.timestamp,
        "center_mhz": chunk.center_hz / 1e6,
        "sample_rate_msps": chunk.sample_rate / 1e6,
        "window_us": len(samples) / chunk.sample_rate * 1e6,
        "i": [round(float(value), 4) for value in samples.real],
        "q": [round(float(value), 4) for value in samples.imag],
    }


def detection_payload(det: Detection) -> dict[str, Any]:
    return _json_safe({
        "detector": det.detector, "timestamp": det.timestamp,
        "center_hz": det.center_hz, "bandwidth_hz": det.bandwidth_hz,
        "peak_db": det.peak_db, "avg_db": det.avg_db, "snr_db": det.snr_db,
        "duration_s": det.duration_s, "score": det.score,
        "flags": list(det.flags), "features": det.features,
        "explanation": det.explanation,
    })


def track_payload(event: TrackEvent) -> dict[str, Any]:
    track = event.track
    return _json_safe({
        "kind": event.kind, "explanation": event.explanation,
        "track": {
            "track_id": track.track_id, "center_hz": track.center_hz,
            "bandwidth_hz": track.bandwidth_hz, "first_seen": track.first_seen,
            "last_seen": track.last_seen, "hits": track.hits,
            "frames_observed": track.frames_observed, "peak_db": track.peak_db,
            "max_snr_db": track.max_snr_db, "occupancy": track.occupancy,
            "detectors": sorted(track.detectors),
            "quality_flags": sorted(track.quality_flags),
        },
    })


def assessment_payload(event_id: int, assessment: DroneAssessment) -> dict[str, Any]:
    return _json_safe({
        "event_id": event_id, "timestamp": assessment.timestamp,
        "category": assessment.category, "confidence": assessment.confidence,
        "center_hz": assessment.center_hz, "freq_span_hz": assessment.freq_span_hz,
        "entity": assessment.entity, "observations": assessment.observations,
        "explanation": assessment.explanation, "evidence": assessment.evidence,
        "penalties": assessment.penalties,
    })


class SessionRecorder:
    """Thread-safe session writer with optional synchronized sidecars."""

    def __init__(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
        *,
        capture_cf32: bool = False,
        capture_wav: bool = False,
        capture_csv: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise SessionError(f"session already exists: {self.path.name}")
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._started_mono = time.monotonic()
        self._last_spectrum = -SPECTRUM_INTERVAL_S
        self._counts: dict[str, int] = {}
        self._csv_sample_rate: float | None = None
        self._csv_noise_floor: float | None = None
        self._capture_cf32 = bool(capture_cf32)
        self._capture_wav = bool(capture_wav)
        self._capture_csv = bool(capture_csv)
        self._iq_queue: queue.Queue[IQChunk | None] = queue.Queue(
            maxsize=IQ_QUEUE_CHUNKS
        )
        self._iq_thread: threading.Thread | None = None
        self._iq_dropped_chunks = 0
        self._iq_samples = 0
        self._iq_error: str | None = None
        self._cf32_path = self.path.with_suffix(".cf32")
        self._wav_path = self.path.with_suffix(".iq.wav")
        self._wav_paths: list[Path] = []
        self._csv_path = self.path.with_suffix(".detector.csv")
        self._iq_meta_path = self.path.with_suffix(".iq.json")
        artifact_paths = []
        if self._capture_cf32:
            artifact_paths.append(self._cf32_path)
        if self._capture_wav:
            artifact_paths.append(self._wav_path)
        if self._capture_csv:
            artifact_paths.append(self._csv_path)
        if self._capture_cf32 or self._capture_wav:
            artifact_paths.append(self._iq_meta_path)
        existing = [item.name for item in artifact_paths if item.exists()]
        if existing:
            raise SessionError(f"session artifact already exists: {existing[0]}")

        self._csv_file = None
        self._csv_writer = None
        if self._capture_csv:
            self._csv_file = self._csv_path.open(
                "x", encoding="utf-8", newline=""
            )
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=CSV_FIELDS
            )
            self._csv_writer.writeheader()

        self._file = gzip.open(self.path, "wt", encoding="utf-8", newline="\n")
        self._write({
            "type": "header", "schema_version": SESSION_SCHEMA_VERSION,
            "drone4rf_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": _json_safe({
                **(metadata or {}),
                "capture_cf32": self._capture_cf32,
                "capture_wav": self._capture_wav,
                "capture_csv": self._capture_csv,
            }),
        }, timed=False)
        if self._capture_cf32 or self._capture_wav:
            self._iq_thread = threading.Thread(
                target=self._write_iq, name="dw-iq-writer", daemon=True
            )
            self._iq_thread.start()

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._started_mono, 4)

    def _write(self, record: dict[str, Any], *, timed: bool = True) -> None:
        # A pipeline callback may already hold a recorder reference while the
        # control thread is closing it. The recorder lock serializes both; a
        # late callback becomes a harmless no-op instead of writing to gzip.
        if self._file.closed:
            return
        if timed:
            record = {"t": self._elapsed(), **record}
        self._file.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
        self._file.write("\n")
        kind = record["type"]
        self._counts[kind] = self._counts.get(kind, 0) + 1

    def _csv_row(self, **values: Any) -> None:
        if self._csv_writer is None:
            return
        row = {field: "" for field in CSV_FIELDS}
        row.update({key: value for key, value in values.items() if key in row})
        self._csv_writer.writerow(row)

    def iq(self, chunk: IQChunk) -> None:
        """Queue every raw chunk for sidecars without blocking DSP."""
        if self._iq_thread is None or self._closing.is_set():
            return
        try:
            self._iq_queue.put_nowait(chunk)
        except queue.Full:
            self._iq_dropped_chunks += 1

    def waveform(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._write({"type": "waveform", "data": payload})

    def _write_iq(self) -> None:
        cf32_file = None
        wav_file = None
        center_hz = None
        sample_rate = None
        wav_data_bytes = 0
        wav_part = 0

        def open_wav(rate: float):
            nonlocal wav_part, wav_data_bytes
            wav_part += 1
            path = (
                self._wav_path
                if wav_part == 1
                else self.path.with_suffix(f".iq.part{wav_part:03d}.wav")
            )
            if path.exists():
                raise SessionError(f"session artifact already exists: {path.name}")
            handle = wave.open(str(path), "wb")
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(int(round(rate)))
            self._wav_paths.append(path)
            wav_data_bytes = 0
            return handle

        try:
            if self._capture_cf32:
                cf32_file = self._cf32_path.open("xb")
            while True:
                chunk = self._iq_queue.get()
                if chunk is None:
                    break
                if sample_rate is None:
                    sample_rate = float(chunk.sample_rate)
                    center_hz = float(chunk.center_hz)
                    if self._capture_wav:
                        wav_file = open_wav(sample_rate)
                elif (
                    float(chunk.sample_rate) != sample_rate
                    or float(chunk.center_hz) != center_hz
                ):
                    raise SessionError(
                        "raw IQ recording cannot span sample-rate or tuning changes"
                    )

                samples = np.asarray(chunk.samples, dtype=np.complex64)
                if cf32_file is not None:
                    interleaved = np.empty(len(samples) * 2, dtype="<f4")
                    interleaved[0::2] = samples.real
                    interleaved[1::2] = samples.imag
                    cf32_file.write(interleaved.tobytes())
                if wav_file is not None:
                    pcm = np.empty(len(samples) * 2, dtype="<i2")
                    pcm[0::2] = np.rint(
                        np.clip(samples.real, -1.0, 1.0) * 32767.0
                    ).astype(np.int16)
                    pcm[1::2] = np.rint(
                        np.clip(samples.imag, -1.0, 1.0) * 32767.0
                    ).astype(np.int16)
                    if wav_data_bytes + pcm.nbytes > WAV_MAX_DATA_BYTES:
                        wav_file.close()
                        wav_file = open_wav(sample_rate)
                    wav_file.writeframesraw(pcm.tobytes())
                    wav_data_bytes += pcm.nbytes
                self._iq_samples += len(samples)
        except Exception as exc:  # surfaced in the sidecar/session manifest
            self._iq_error = str(exc)
        finally:
            if cf32_file is not None:
                cf32_file.close()
            if wav_file is not None:
                wav_file.close()
            if self._capture_cf32 or self._capture_wav:
                self._iq_meta_path.write_text(
                    json.dumps({
                        "format": "complex_iq",
                        "layout": "interleaved I,Q",
                        "cf32_dtype": "little-endian float32",
                        "wav_dtype": "little-endian PCM16 stereo (I left, Q right)",
                        "center_hz": center_hz,
                        "sample_rate": sample_rate,
                        "num_complex_samples": self._iq_samples,
                        "dropped_chunks": self._iq_dropped_chunks,
                        "writer_error": self._iq_error,
                        "wav_parts": [path.name for path in self._wav_paths],
                    }, indent=2),
                    encoding="utf-8",
                )

    def spectrum(self, frame: SpectrumFrame) -> None:
        now = time.monotonic() - self._started_mono
        with self._lock:
            if now - self._last_spectrum < SPECTRUM_INTERVAL_S:
                return
            self._last_spectrum = now
            self._write({"type": "spectrum", "data": spectrum_payload(frame)})
            self._csv_sample_rate = frame.sample_rate
            self._csv_noise_floor = _num(frame.noise_floor_db)
            self._csv_row(
                record_type="spectrum", relative_s=self._elapsed(),
                timestamp=frame.timestamp, center_hz=frame.center_hz,
                sample_rate=frame.sample_rate,
                peak_db=_num(np.max(frame.psd_db)),
                avg_db=_num(np.mean(frame.psd_db)),
                median_db=_num(np.median(frame.psd_db)),
                noise_floor_db=_num(frame.noise_floor_db),
                features_json=json.dumps({
                    "threshold_mean_db": (
                        _num(np.mean(frame.threshold_db))
                        if frame.threshold_db is not None else None
                    )
                }, separators=(",", ":")),
            )

    def detection(self, det: Detection) -> None:
        with self._lock:
            self._write({"type": "detection", "data": detection_payload(det)})
            self._csv_row(
                record_type="detection", relative_s=self._elapsed(),
                timestamp=det.timestamp, center_hz=det.center_hz,
                sample_rate=self._csv_sample_rate,
                detector=det.detector, bandwidth_hz=det.bandwidth_hz,
                peak_db=det.peak_db, avg_db=det.avg_db, snr_db=det.snr_db,
                noise_floor_db=self._csv_noise_floor,
                duration_s=det.duration_s, score=det.score,
                flags="|".join(det.flags),
                features_json=json.dumps(
                    _json_safe(det.features), separators=(",", ":")
                ), explanation=det.explanation,
            )

    def track(self, event: TrackEvent) -> None:
        with self._lock:
            self._write({"type": "track", "data": track_payload(event)})
            track = event.track
            self._csv_row(
                record_type=event.kind, relative_s=self._elapsed(),
                timestamp=track.last_seen, center_hz=track.center_hz,
                sample_rate=self._csv_sample_rate,
                detector="+".join(sorted(track.detectors)),
                bandwidth_hz=track.bandwidth_hz, peak_db=track.peak_db,
                snr_db=track.max_snr_db,
                noise_floor_db=self._csv_noise_floor,
                duration_s=max(0.0, track.last_seen - track.first_seen),
                flags="|".join(sorted(track.quality_flags)),
                features_json=json.dumps({
                    "hits": track.hits,
                    "frames_observed": track.frames_observed,
                    "occupancy": track.occupancy,
                }, separators=(",", ":")),
                explanation=event.explanation,
            )

    def assessment(self, event_id: int, assessment: DroneAssessment) -> None:
        with self._lock:
            self._write({
                "type": "assessment",
                "data": assessment_payload(event_id, assessment),
            })
            self._csv_row(
                record_type="assessment", relative_s=self._elapsed(),
                timestamp=assessment.timestamp,
                center_hz=assessment.center_hz,
                detector="fusion", bandwidth_hz=assessment.freq_span_hz,
                score=assessment.confidence,
                features_json=json.dumps({
                    "event_id": event_id,
                    "entity": assessment.entity,
                    "observations": assessment.observations,
                    "category": assessment.category,
                    "evidence": assessment.evidence,
                    "penalties": assessment.penalties,
                }, separators=(",", ":")),
                explanation=assessment.explanation,
            )

    def label(self, label: dict[str, Any]) -> None:
        with self._lock:
            self._write({"type": "label", "data": _json_safe(label)})

    def close(self, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        self._closing.set()
        if self._iq_thread is not None and self._iq_thread.is_alive():
            self._iq_queue.put(None)
            self._iq_thread.join(timeout=30.0)
            if self._iq_thread.is_alive():
                self._iq_error = "IQ writer did not stop within 30 seconds"
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        with self._lock:
            if self._file.closed:
                return dict(self._counts)
            artifacts = []
            kinds = (
                (self._cf32_path, "iq_cf32", "complex float32 I/Q"),
                (self._csv_path, "detector_csv", "detector summary CSV"),
                (self._iq_meta_path, "iq_metadata", "IQ metadata JSON"),
            )
            for artifact_path, kind, description in kinds:
                if artifact_path.is_file():
                    artifacts.append({
                        "name": artifact_path.name,
                        "kind": kind,
                        "description": description,
                        "size_bytes": artifact_path.stat().st_size,
                    })
            for artifact_path in self._wav_paths:
                if artifact_path.is_file():
                    artifacts.append({
                        "name": artifact_path.name,
                        "kind": "iq_wav",
                        "description": "PCM16 stereo I/Q",
                        "size_bytes": artifact_path.stat().st_size,
                    })
            self._write({
                "type": "end",
                "duration_s": round(time.monotonic() - self._started_mono, 3),
                "counts": dict(self._counts),
                "artifacts": artifacts,
                "iq_samples": self._iq_samples,
                "iq_dropped_chunks": self._iq_dropped_chunks,
                "iq_error": self._iq_error,
                "runtime": _json_safe(runtime or {}),
            }, timed=False)
            self._file.close()
            return dict(self._counts)


def read_session(path: Path) -> Iterator[dict[str, Any]]:
    """Yield validated records; corrupt or truncated files fail clearly."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            first = json.loads(stream.readline())
            if first.get("type") != "header":
                raise SessionError("session header is missing")
            if first.get("schema_version") != SESSION_SCHEMA_VERSION:
                raise SessionError(
                    f"unsupported session schema {first.get('schema_version')!r}"
                )
            yield first
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict) or "type" not in record:
                        raise SessionError("invalid session record")
                    yield record
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionError(f"cannot read session: {exc}") from exc


def session_info(path: Path) -> dict[str, Any]:
    header = None
    end = None
    for record in read_session(path):
        header = header or record
        if record["type"] == "end":
            end = record
    if header is None:
        raise SessionError("empty session")
    return {
        "name": path.name[: -len(SESSION_EXTENSION)],
        "created_at": header.get("created_at"),
        "duration_s": (end or {}).get("duration_s"),
        "counts": (end or {}).get("counts", {}),
        "artifacts": (end or {}).get("artifacts", []),
        "iq_samples": (end or {}).get("iq_samples", 0),
        "iq_dropped_chunks": (end or {}).get("iq_dropped_chunks", 0),
        "iq_error": (end or {}).get("iq_error"),
        "runtime": (end or {}).get("runtime", {}),
        "size_bytes": path.stat().st_size,
        "metadata": header.get("metadata", {}),
    }
