"""HTTP + Server-Sent-Events server for the browser dashboard.

Transport choices and why:
- stdlib ``ThreadingHTTPServer``: zero dependencies, and the workload
  (a handful of viewers on a LAN) never justifies an async framework.
- Server-Sent Events rather than WebSockets: the data flow is one-way
  (scanner -> browser), SSE reconnects automatically, and it needs no
  extra library on either side. Control actions are ordinary POSTs.
- Spectra are peak-decimated to ~1024 points before transport, which
  keeps a frame near 6 KB while preserving narrow peaks (a mean would
  hide exactly the signals this tool exists to find).

Local-service hardening (this process controls radio hardware):
- binds 127.0.0.1 by default; a non-loopback bind prints a warning;
- validates the Host header, which blocks DNS-rebinding attacks;
- state-changing endpoints require JSON content-type plus a per-run
  token embedded in the served page, so another site open in the same
  browser cannot start scans or edit the event database;
- no CORS headers are ever sent, so cross-origin reads are refused.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from drone4rf import LEGAL_NOTICE, __version__
from drone4rf.config import AppConfig
from drone4rf.control import ControllerError, PipelineController
from drone4rf.events import EventStore
from drone4rf.labels import (
    LabelValidationError,
    export_csv,
    vocabulary_payload,
)
from drone4rf.session import (
    SESSION_EXTENSION,
    SessionError,
    default_session_name,
    read_session,
    session_info,
    session_path,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SPECTRUM_POINTS = 1024
SPECTRUM_INTERVAL_S = 0.08  # ~12 fps
STATE_INTERVAL_S = 0.5
MAX_BODY_BYTES = 64 * 1024


def _num(value: float | None) -> float | None:
    """JSON-safe number: non-finite values become null, not NaN."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def decimate_peak(values: np.ndarray, target: int = SPECTRUM_POINTS) -> list[float]:
    """Reduce a spectrum to <= target points, keeping the peak of each
    bucket so narrowband signals survive the downsampling."""
    n = len(values)
    if n <= target:
        return [round(float(v), 1) for v in values]
    bucket = n // target
    usable = bucket * target
    reduced = values[:usable].reshape(target, bucket).max(axis=1)
    if usable < n:  # fold the remainder into the final bucket
        reduced[-1] = max(reduced[-1], float(values[usable:].max()))
    return [round(float(v), 1) for v in reduced]


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, cfg: AppConfig, host: str) -> None:
        super().__init__(addr, handler)
        self.cfg = cfg
        self.controller = PipelineController(cfg)
        self.store = EventStore(cfg.storage.database_path)
        self.token = secrets.token_urlsafe(24)
        self.allowed_hosts = {"localhost", "127.0.0.1", "::1", host.lower()}
        self._index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.sessions_dir = Path(cfg.storage.database_path).parent / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._replay_lock = threading.Lock()
        self._replay: tuple[Path, float, int] | None = None
        self._replay_generation = 0

    def index_html(self) -> bytes:
        html = self._index.replace("__DW_TOKEN__", self.token)
        html = html.replace(
            "__DW_LABEL_VOCAB__",
            json.dumps(vocabulary_payload(), ensure_ascii=False),
        )
        return html.encode("utf-8")

    def handle_error(self, request, client_address) -> None:
        """Closing a browser tab resets its keep-alive/SSE socket; that is
        routine, not an error. Logging a traceback per closed tab would
        bury genuine failures in a server that runs for days."""
        import sys

        exc = sys.exc_info()[1]
        if isinstance(
            exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)
        ):
            log.debug("client %s disconnected: %s", client_address[0], exc)
            return
        super().handle_error(request, client_address)

    def shutdown_all(self) -> None:
        self.stop_replay()
        self.controller.stop()
        self.store.close()

    def list_sessions(self) -> list[dict]:
        result = []
        for path in sorted(
            self.sessions_dir.glob(f"*{SESSION_EXTENSION}"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                result.append(session_info(path))
            except SessionError as exc:
                result.append({"name": path.stem, "error": str(exc)})
        return result

    def start_replay(self, path: Path, speed: float) -> int:
        with self._replay_lock:
            self._replay_generation += 1
            self._replay = (path, speed, self._replay_generation)
            return self._replay_generation

    def stop_replay(self) -> None:
        with self._replay_lock:
            self._replay_generation += 1
            self._replay = None

    def replay_snapshot(self) -> tuple[Path, float, int] | None:
        with self._replay_lock:
            return self._replay


class Handler(BaseHTTPRequestHandler):
    server_version = f"drone4rf/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host.strip("[]").lower() in self.server.allowed_hosts

    def _send_json(self, payload, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_download(
        self, body: bytes, content_type: str, filename: str
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Disposition", f'attachment; filename="{filename}"'
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        """Stream a potentially multi-gigabyte local capture without buffering."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header(
            "Content-Disposition", f'attachment; filename="{path.name}"'
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                self.wfile.write(block)

    def _read_json(self) -> dict:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if ctype != "application/json":
            raise ControllerError("Content-Type must be application/json")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ControllerError("request body too large")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ControllerError("request body must be a JSON object")
        return data

    def _token_ok(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-DW-Token") or "", self.server.token
        )

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "host not allowed")
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            body = self.server.index_html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._send_json(self._status_payload())
        elif path == "/api/events":
            self._send_json({"events": self.server.store.recent(limit=200)})
        elif path == "/api/sessions":
            if not self._token_ok():
                self._error(HTTPStatus.FORBIDDEN, "missing or invalid session token")
                return
            self._send_json({"sessions": self.server.list_sessions()})
        elif path == "/api/sessions/artifact":
            if not self._token_ok():
                self._error(HTTPStatus.FORBIDDEN, "missing or invalid session token")
                return
            query = parse_qs(parsed.query)
            session_name = query.get("session", [""])[0]
            artifact_name = query.get("file", [""])[0]
            try:
                session_file = session_path(
                    self.server.sessions_dir, session_name
                )
            except SessionError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if not session_file.is_file():
                self._error(HTTPStatus.NOT_FOUND, "session not found")
                return
            try:
                info = session_info(session_file)
            except SessionError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            allowed = {item["name"] for item in info.get("artifacts", [])}
            if (
                artifact_name not in allowed
                or Path(artifact_name).name != artifact_name
            ):
                self._error(HTTPStatus.BAD_REQUEST, "artifact not in session")
                return
            artifact = self.server.sessions_dir / artifact_name
            if not artifact.is_file():
                self._error(HTTPStatus.NOT_FOUND, "artifact not found")
                return
            content_type = {
                ".wav": "audio/wav",
                ".csv": "text/csv; charset=utf-8",
                ".json": "application/json",
            }.get(artifact.suffix.lower(), "application/octet-stream")
            self._send_file(artifact, content_type)
        elif path == "/api/labels/export":
            if not self._token_ok():
                self._error(HTTPStatus.FORBIDDEN, "missing or invalid session token")
                return
            records = self.server.store.export_labels()
            export_format = parse_qs(parsed.query).get("format", ["json"])[0]
            if export_format == "json":
                self._send_json({"labels": records})
            elif export_format == "csv":
                self._send_download(
                    export_csv(records).encode("utf-8"),
                    "text/csv; charset=utf-8",
                    "drone4rf-labels.csv",
                )
            else:
                self._error(HTTPStatus.BAD_REQUEST, "format must be json or csv")
        elif path == "/api/signatures/export":
            if not self._token_ok():
                self._error(HTTPStatus.FORBIDDEN, "missing or invalid session token")
                return
            query = parse_qs(parsed.query)
            labeled_only = query.get("labeled_only", ["0"])[0].lower() in (
                "1", "true", "yes",
            )
            records = self.server.store.export_signatures(
                labeled_only=labeled_only
            )
            export_format = query.get("format", ["json"])[0]
            if export_format == "json":
                self._send_json({"signatures": records})
            elif export_format == "csv":
                self._send_download(
                    export_csv(records).encode("utf-8"),
                    "text/csv; charset=utf-8",
                    "drone4rf-rf-signatures.csv",
                )
            else:
                self._error(HTTPStatus.BAD_REQUEST, "format must be json or csv")
        elif path == "/api/stream":
            self._stream()
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "host not allowed")
            return
        if not self._token_ok():
            self._error(HTTPStatus.FORBIDDEN, "missing or invalid session token")
            return
        path = urlparse(self.path).path
        controller = self.server.controller
        try:
            body = self._read_json()
            if path == "/api/start":
                self.server.stop_replay()
                center_mhz = body.get("center_mhz")
                controller.start(
                    source_name=str(body.get("source", "hackrf")),
                    mode=str(body.get("mode", "sweep")),
                    environment=str(body.get("environment", "default")),
                    frozen=bool(body.get("frozen", False)),
                    calibrate=bool(body.get("calibrate", False)),
                    center_hz=(
                        float(center_mhz) * 1e6
                        if center_mhz not in (None, "") else None
                    ),
                )
                self._send_json({"ok": True, "status": controller.status_message})
            elif path == "/api/stop":
                controller.stop()
                self._send_json({"ok": True, "status": controller.status_message})
            elif path == "/api/session/record/start":
                name = str(body.get("name") or default_session_name())
                path = session_path(self.server.sessions_dir, name)
                controller.start_recording(
                    path,
                    {
                        "source": str(body.get("source") or "dashboard"),
                        "environment": str(body.get("environment") or "default"),
                    },
                    capture_cf32=bool(body.get("capture_cf32", False)),
                    capture_wav=bool(body.get("capture_wav", False)),
                    capture_csv=bool(body.get("capture_csv", False)),
                )
                self._send_json({"ok": True, "name": path.stem})
            elif path == "/api/session/record/stop":
                counts = controller.stop_recording()
                self._send_json({"ok": True, "counts": counts or {}})
            elif path == "/api/session/replay/start":
                if controller.running:
                    raise ControllerError(
                        "stop the live scan before replaying a session"
                    )
                path = session_path(self.server.sessions_dir, str(body.get("name", "")))
                if not path.is_file():
                    raise FileNotFoundError(path.name)
                # Validate before switching SSE viewers to the file.
                next(read_session(path))
                speed = float(body.get("speed", 1.0))
                if not 0.1 <= speed <= 20.0:
                    raise ControllerError("replay speed must be between 0.1 and 20")
                self.server.start_replay(path, speed)
                self._send_json({"ok": True, "name": path.stem, "speed": speed})
            elif path == "/api/session/replay/stop":
                self.server.stop_replay()
                self._send_json({"ok": True})
            elif path == "/api/feedback":
                event_id = int(body.get("event_id", 0))
                if "verdict" in body:
                    stored = self.server.store.set_structured_feedback(
                        event_id,
                        verdict=body.get("verdict"),
                        drone_model=body.get("drone_model"),
                        fp_class=body.get("fp_class"),
                        fp_class_detail=body.get("fp_class_detail"),
                        label_notes=body.get("label_notes"),
                        labeled_by=body.get("labeled_by"),
                    )
                else:
                    stored = self.server.store.set_feedback(
                        event_id, str(body.get("label", ""))
                    )
                if stored is not None:
                    controller.record_label(stored)
                self._send_json({"ok": True, "label": stored})
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except ControllerError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except LabelValidationError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except SessionError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc).strip("'"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"bad request: {exc}")
        except FileNotFoundError as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"not found: {exc}")
        except Exception as exc:  # never leak a traceback to the browser
            log.exception("request failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    # -- payloads --------------------------------------------------------------

    def _status_payload(self) -> dict:
        cfg = self.server.cfg
        controller = self.server.controller
        return {
            "version": __version__,
            "legal_notice": LEGAL_NOTICE,
            "running": controller.running,
            "status": controller.status_message,
            "error": controller.last_error,
            "environments": controller.available_environments(),
            "device": {
                "driver": cfg.device.driver,
                "center_mhz": cfg.device.center_freq_hz / 1e6,
                "sample_rate_msps": cfg.device.sample_rate / 1e6,
                "lna_db": cfg.device.lna_gain_db,
                "vga_db": cfg.device.vga_gain_db,
                "amp": cfg.device.amp_enabled,
            },
            "bands": [
                {
                    "name": b.name,
                    "start_mhz": b.start_hz / 1e6,
                    "stop_mhz": b.stop_hz / 1e6,
                }
                for b in cfg.sweep.enabled_bands()
            ],
            "ml_enabled": cfg.ml.enabled,
            "analytics_enabled": cfg.analytics.enabled,
            "recording": controller.recording,
            "replaying": self.server.replay_snapshot() is not None,
        }

    def _state_payload(self, assess_cursor: int) -> tuple[dict, int]:
        controller = self.server.controller
        stats = controller.snapshot()
        new = controller.assessments_since(assess_cursor)
        if new:
            assess_cursor = new[-1][0]
        payload = {
            "running": controller.running,
            "recording": controller.recording,
            "replaying": False,
            "status": controller.status_message,
            "error": controller.last_error,
            "stats": (
                {
                    "chunks": stats.chunks_processed,
                    "detections": stats.detections,
                    "track_events": stats.track_events,
                    "assessments": stats.assessments,
                    "drops": stats.queue_drops,
                    "overflows": stats.source_overflows,
                    "clipped": stats.clipped_chunks,
                    "steps": stats.steps_visited,
                    "captures": stats.captures_written,
                    "tuned_mhz": (
                        _num(stats.current_center_hz / 1e6)
                        if _num(stats.current_center_hz) is not None
                        else None
                    ),
                    "noise_db": _num(stats.noise_floor_db),
                    "warnings": stats.warnings,
                }
                if stats is not None
                else None
            ),
            "tracks": [
                {
                    "mhz": round(t.center_hz / 1e6, 4),
                    "bw_khz": round(t.bandwidth_hz / 1e3, 1),
                    "snr": round(t.max_snr_db, 1),
                    "hits": t.hits,
                    "occupancy": round(t.occupancy, 2),
                    "detectors": "+".join(sorted(t.detectors)),
                }
                for t in sorted(
                    controller.active_tracks(), key=lambda t: -t.max_snr_db
                )[:40]
            ],
            "assessments": [
                {
                    "seq": seq,
                    "event_id": event_id,
                    "category": a.category,
                    "confidence": round(a.confidence, 3),
                    "center_mhz": round(a.center_hz / 1e6, 4),
                    "span_mhz": round(a.freq_span_hz / 1e6, 3),
                    "entity": a.entity,
                    "observations": a.observations,
                    "explanation": a.explanation,
                    "evidence": {k: round(v, 3) for k, v in a.evidence.items()},
                    "penalties": {k: round(v, 3) for k, v in a.penalties.items()},
                    "time": time.strftime(
                        "%H:%M:%S", time.localtime(a.timestamp)
                    ),
                }
                for seq, event_id, a in new
            ],
        }
        return payload, assess_cursor

    def _spectrum_payload(self, frame) -> dict:
        return {
            "center_mhz": frame.center_hz / 1e6,
            "f0_mhz": float(frame.freqs_hz[0] / 1e6),
            "f1_mhz": float(frame.freqs_hz[-1] / 1e6),
            "psd": decimate_peak(frame.psd_db),
            "thr": (
                decimate_peak(frame.threshold_db)
                if frame.threshold_db is not None
                else None
            ),
            "noise_db": _num(frame.noise_floor_db),
        }

    # -- SSE ---------------------------------------------------------------------

    def _stream(self) -> None:
        # The body has no length and never ends; keep-alive cannot apply.
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        replay = self.server.replay_snapshot()
        if replay is not None:
            self._stream_replay(*replay)
            return

        controller = self.server.controller
        frame_seq_sent = -1
        waveform_seq_sent = -1
        assess_cursor = 0
        last_state = 0.0
        try:
            while not getattr(self.server, "_stopping", False):
                if self.server.replay_snapshot() is not None:
                    return
                now = time.time()
                item = controller.peek_spectrum()
                if item is not None and item[0] != frame_seq_sent:
                    frame_seq_sent = item[0]
                    self._sse("spectrum", self._spectrum_payload(item[1]))
                waveform = controller.peek_waveform()
                if waveform is not None and waveform[0] != waveform_seq_sent:
                    waveform_seq_sent = waveform[0]
                    self._sse("waveform", waveform[1])
                if now - last_state >= STATE_INTERVAL_S:
                    last_state = now
                    payload, assess_cursor = self._state_payload(assess_cursor)
                    self._sse("state", payload)
                time.sleep(SPECTRUM_INTERVAL_S)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log.debug("SSE client disconnected")
        except Exception:
            log.exception("SSE stream failed")

    def _stream_replay(self, path: Path, speed: float, generation: int) -> None:
        """Replay a processed session through the same SSE events as live RF."""
        started = time.monotonic()
        tracks: dict[int, dict] = {}
        pending_assessments: list[dict] = []
        stats = {"chunks": 0, "detections": 0, "track_events": 0,
                 "assessments": 0, "drops": 0, "overflows": 0, "clipped": 0,
                 "steps": 0, "captures": 0, "tuned_mhz": None,
                 "noise_db": None, "warnings": []}
        last_state = 0.0

        def active() -> bool:
            replay = self.server.replay_snapshot()
            return replay is not None and replay[2] == generation

        def send_state(status: str) -> None:
            nonlocal pending_assessments, last_state
            self._sse("state", {
                "running": False, "replaying": True, "recording": False,
                "status": status, "error": None, "stats": stats,
                "tracks": list(tracks.values())[:40],
                "assessments": pending_assessments,
            })
            pending_assessments = []
            last_state = time.monotonic()

        try:
            for record in read_session(path):
                if not active():
                    return
                target = float(record.get("t", 0.0)) / speed
                while active() and time.monotonic() - started < target:
                    time.sleep(min(0.05, target - (time.monotonic() - started)))
                kind = record["type"]
                data = record.get("data", {})
                if kind == "spectrum":
                    stats["chunks"] += 1
                    stats["tuned_mhz"] = data.get("center_mhz")
                    stats["noise_db"] = data.get("noise_db")
                    self._sse("spectrum", data)
                elif kind == "waveform":
                    self._sse("waveform", data)
                elif kind == "detection":
                    stats["detections"] += 1
                elif kind == "track":
                    stats["track_events"] += 1
                    track = data.get("track", {})
                    track_id = int(track.get("track_id", 0))
                    if data.get("kind") == "track_closed":
                        tracks.pop(track_id, None)
                    else:
                        tracks[track_id] = {
                            "mhz": round(float(track.get("center_hz", 0)) / 1e6, 4),
                            "bw_khz": round(
                                float(track.get("bandwidth_hz", 0)) / 1e3, 1
                            ),
                            "snr": round(float(track.get("max_snr_db", 0)), 1),
                            "hits": int(track.get("hits", 0)),
                            "occupancy": round(float(track.get("occupancy", 0)), 2),
                            "detectors": "+".join(track.get("detectors", [])),
                        }
                elif kind == "assessment":
                    stats["assessments"] += 1
                    pending_assessments.append({
                        "seq": stats["assessments"], "event_id": data.get("event_id"),
                        "category": data.get("category"),
                        "confidence": round(float(data.get("confidence", 0)), 3),
                        "center_mhz": round(float(data.get("center_hz", 0)) / 1e6, 4),
                        "span_mhz": round(float(data.get("freq_span_hz", 0)) / 1e6, 3),
                        "entity": data.get("entity"),
                        "observations": data.get("observations", 0),
                        "explanation": data.get("explanation", ""),
                        "evidence": data.get("evidence", {}),
                        "penalties": data.get("penalties", {}),
                        "time": time.strftime(
                            "%H:%M:%S",
                            time.localtime(float(data.get("timestamp", 0))),
                        ),
                    })
                state_due = time.monotonic() - last_state >= STATE_INTERVAL_S
                if pending_assessments or state_due:
                    send_state(f"replaying '{path.stem}' at {speed:g}x")
            send_state(f"replay complete: '{path.stem}'")
            while active() and not getattr(self.server, "_stopping", False):
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log.debug("replay SSE client disconnected")
        except Exception:
            log.exception("session replay failed")

    def _sse(self, event: str, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False)
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def run_server(
    cfg: AppConfig,
    host: str = "127.0.0.1",
    port: int = 8731,
    open_browser: bool = True,
) -> int:
    """Serve the dashboard until interrupted. Returns a process exit code."""
    httpd = DashboardServer((host, port), Handler, cfg, host)
    httpd._stopping = False
    actual_port = httpd.server_address[1]
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{actual_port}/"

    print(f"Drone 4-RF dashboard: {url}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "⚠ bound to a non-loopback address: anyone who can reach this "
            "host can start scans and read your event database. Use a "
            "trusted network only.",
        )
    print("press Ctrl+C to stop")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        httpd._stopping = True
        httpd.shutdown_all()
        httpd.server_close()
    return 0
