"""Browser dashboard: routing, hardening, and a live SSE round-trip."""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from drone4rf.config import (
    AppConfig,
    DeviceConfig,
    DSPConfig,
    StorageConfig,
    TrackingConfig,
)
from drone4rf.control import ControllerError, validate_environment_name
from drone4rf.web.server import DashboardServer, Handler, decimate_peak


@pytest.fixture
def server(tmp_path):
    cfg = AppConfig(
        device=DeviceConfig(sample_rate=10e6, center_freq_hz=2.437e9),
        dsp=DSPConfig(fft_size=1024, chunk_samples=65_536),
        tracking=TrackingConfig(promote_hits=4, drop_after_misses=8),
        storage=StorageConfig(database_path=str(tmp_path / "events.db")),
    )
    cfg.validate()
    httpd = DashboardServer(("127.0.0.1", 0), Handler, cfg, "127.0.0.1")
    httpd._stopping = False
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    httpd.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield httpd
    httpd._stopping = True
    httpd.shutdown()
    httpd.shutdown_all()
    httpd.server_close()


def _get(server, path, host=None, token=None, timeout=5):
    req = urllib.request.Request(server.base + path)
    if host:
        req.add_header("Host", host)
    if token is not None:
        req.add_header("X-DW-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def _post(server, path, body, token=None, ctype="application/json", timeout=15):
    req = urllib.request.Request(
        server.base + path, data=json.dumps(body).encode(), method="POST"
    )
    req.add_header("Content-Type", ctype)
    if token is not None:
        req.add_header("X-DW-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


# -- decimation ---------------------------------------------------------------


def test_decimate_preserves_narrow_peaks() -> None:
    psd = np.full(4096, -95.0)
    psd[1234] = -30.0  # a single-bin signal must survive downsampling
    out = decimate_peak(psd, 1024)
    assert len(out) == 1024
    assert max(out) == -30.0


def test_decimate_handles_short_and_ragged_inputs() -> None:
    assert len(decimate_peak(np.zeros(100), 1024)) == 100
    ragged = np.full(1030, -90.0)
    ragged[1029] = -10.0  # peak inside the remainder must be folded in
    out = decimate_peak(ragged, 512)
    assert len(out) == 512 and max(out) == -10.0


# -- routing ------------------------------------------------------------------


def test_index_serves_token_and_no_placeholder(server) -> None:
    status, body = _get(server, "/")
    assert status == 200
    assert server.token in body
    assert "__DW_TOKEN__" not in body
    assert "__DW_LABEL_VOCAB__" not in body
    assert "confirmed_drone" in body
    assert "Export RF signatures" in body
    assert "Workshop session" in body
    assert "Stop replay" in body
    assert "CF32 (Inspectrum)" in body
    assert "IQ waveform" in body
    assert "detector CSV" in body
    assert "Band focus" in body
    assert "Visible span MHz" in body
    assert "receive-only" in body.lower()


def test_status_endpoint(server) -> None:
    _, body = _get(server, "/api/status")
    data = json.loads(body)
    assert data["running"] is False
    assert data["device"]["driver"] == "hackrf"
    assert "legal_notice" in data
    assert data["ml_enabled"] is False


def test_unknown_route_404(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/api/nope")
    assert exc.value.code == 404


# -- hardening ----------------------------------------------------------------


def test_foreign_host_header_rejected(server) -> None:
    """Blocks DNS-rebinding: a hostile name resolving to 127.0.0.1."""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/api/status", host="evil.example.com")
    assert exc.value.code == 403


def test_post_without_token_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/api/start", {"source": "sim"})
    assert exc.value.code == 403
    assert server.controller.running is False


def test_post_with_wrong_token_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/api/start", {"source": "sim"}, token="nope")
    assert exc.value.code == 403


def test_non_json_content_type_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server, "/api/stop", {}, token=server.token,
            ctype="application/x-www-form-urlencoded",
        )
    assert exc.value.code == 400


def test_environment_name_path_traversal_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server,
            "/api/start",
            {"source": "sim", "mode": "sweep", "environment": "../../etc"},
            token=server.token,
        )
    assert exc.value.code == 400
    assert server.controller.running is False


def test_validate_environment_name_unit() -> None:
    assert validate_environment_name("home") == "home"
    for bad in ("../etc", "a/b", "", "x" * 65, "we;rm -rf"):
        with pytest.raises(ControllerError):
            validate_environment_name(bad)


def test_bad_source_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/api/start", {"source": "gnuradio"}, token=server.token)
    assert exc.value.code == 400


# -- control + streaming -------------------------------------------------------


def test_start_stream_stop_roundtrip(server) -> None:
    status, data = _post(
        server,
        "/api/start",
        {"source": "sim", "mode": "scan", "frozen": False},
        token=server.token,
    )
    assert status == 200 and data["ok"] is True
    assert server.controller.running is True

    # Read the SSE stream until a spectrum frame and a state frame arrive.
    saw = set()
    req = urllib.request.Request(server.base + "/api/stream")
    deadline = time.time() + 20
    with urllib.request.urlopen(req, timeout=20) as stream:
        event = None
        while time.time() < deadline and {"spectrum", "waveform", "state"} - saw:
            line = stream.readline().decode().strip()
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                payload = json.loads(line[6:])
                saw.add(event)
                if event == "spectrum":
                    assert len(payload["psd"]) <= 1024
                    assert payload["center_mhz"] == pytest.approx(2437.0)
                elif event == "waveform":
                    assert len(payload["i"]) == len(payload["q"])
                    assert len(payload["i"]) <= 512
                else:
                    assert payload["running"] is True
    assert saw == {"spectrum", "waveform", "state"}

    _, data = _post(server, "/api/stop", {}, token=server.token)
    assert data["ok"] is True
    assert server.controller.running is False


def test_focused_scan_uses_requested_center_without_mutating_config(server) -> None:
    _, data = _post(
        server, "/api/start",
        {"source": "sim", "mode": "scan", "center_mhz": 915, "frozen": False},
        token=server.token,
    )
    assert data["ok"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        frame = server.controller.peek_spectrum()
        if frame is not None:
            assert frame[1].center_hz == pytest.approx(915e6)
            break
        time.sleep(0.05)
    else:
        pytest.fail("focused scan did not publish a spectrum frame")
    _post(server, "/api/stop", {}, token=server.token)
    assert server.cfg.device.center_freq_hz == pytest.approx(2.437e9)


@pytest.mark.parametrize(
    "center", [0, 6001, float("nan"), "not-a-number", {"mhz": 915}]
)
def test_focused_scan_rejects_invalid_center(server, center) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server, "/api/start",
            {"source": "sim", "mode": "scan", "center_mhz": center},
            token=server.token,
        )
    assert exc.value.code == 400
    assert server.controller.running is False


def test_focused_scan_requires_scan_mode_and_adaptive_baseline(server) -> None:
    for payload in (
        {"source": "sim", "mode": "sweep", "center_mhz": 915},
        {"source": "sim", "mode": "scan", "center_mhz": 915, "frozen": True},
    ):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(server, "/api/start", payload, token=server.token)
        assert exc.value.code == 400
        assert server.controller.running is False


def test_workshop_recording_requires_scan_and_lists_session(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server, "/api/session/record/start", {"name": "class-a"},
            token=server.token,
        )
    assert exc.value.code == 400

    _post(
        server, "/api/start", {"source": "sim", "mode": "scan"},
        token=server.token,
    )
    _post(
        server, "/api/session/record/start",
        {"name": "class-a", "capture_csv": True},
        token=server.token,
    )
    deadline = time.time() + 8
    while time.time() < deadline and server.controller.peek_spectrum() is None:
        time.sleep(0.05)
    _, result = _post(
        server, "/api/session/record/stop", {}, token=server.token
    )
    _post(server, "/api/stop", {}, token=server.token)
    assert result["counts"]["spectrum"] >= 1

    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/api/sessions")
    assert exc.value.code == 403
    _, body = _get(server, "/api/sessions", token=server.token)
    sessions = json.loads(body)["sessions"]
    assert sessions[0]["name"] == "class-a"
    assert sessions[0]["counts"]["spectrum"] >= 1
    csv_artifact = next(
        item for item in sessions[0]["artifacts"]
        if item["kind"] == "detector_csv"
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(
            server,
            "/api/sessions/artifact?session=class-a&file="
            + csv_artifact["name"],
        )
    assert exc.value.code == 403
    _, csv_body = _get(
        server,
        "/api/sessions/artifact?session=class-a&file=" + csv_artifact["name"],
        token=server.token,
    )
    assert "record_type" in csv_body and "spectrum" in csv_body
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(
            server,
            "/api/sessions/artifact?session=class-a&file=../events.db",
            token=server.token,
        )
    assert exc.value.code == 400


def test_workshop_replay_streams_recorded_waterfall(server) -> None:
    from drone4rf.pipeline import SpectrumFrame
    from drone4rf.session import SessionRecorder

    recorder = SessionRecorder(server.sessions_dir / "lesson.dwsession")
    recorder.spectrum(SpectrumFrame(
        timestamp=time.time(), center_hz=915e6, sample_rate=2e6,
        freqs_hz=np.linspace(914e6, 916e6, 256),
        psd_db=np.full(256, -80.0), threshold_db=None, noise_floor_db=-95.0,
    ))
    recorder.waveform({
        "timestamp": time.time(), "center_mhz": 915.0,
        "sample_rate_msps": 2.0, "i": [0.1, -0.1], "q": [0.2, -0.2],
    })
    recorder.close()

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server, "/api/session/replay/start", {"name": "../lesson"},
            token=server.token,
        )
    assert exc.value.code == 400

    _post(
        server, "/api/session/replay/start", {"name": "lesson", "speed": 10},
        token=server.token,
    )
    req = urllib.request.Request(server.base + "/api/stream")
    saw = set()
    with urllib.request.urlopen(req, timeout=10) as stream:
        event = None
        deadline = time.time() + 8
        while time.time() < deadline and {"spectrum", "waveform", "state"} - saw:
            line = stream.readline().decode().strip()
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                payload = json.loads(line[6:])
                saw.add(event)
                if event == "spectrum":
                    assert payload["center_mhz"] == 915.0
                elif event == "waveform":
                    assert payload["i"] == [0.1, -0.1]
                elif event == "state":
                    assert payload["replaying"] is True
    assert saw == {"spectrum", "waveform", "state"}
    _post(server, "/api/session/replay/stop", {}, token=server.token)


def test_events_and_feedback(server) -> None:
    import time as _t

    from drone4rf.detectors.base import Detection

    det = Detection(
        detector="energy", timestamp=_t.time(), center_hz=2.44e9,
        bandwidth_hz=1e5, peak_db=-50.0, avg_db=-55.0, snr_db=20.0,
        duration_s=0.02, score=0.5,
    )
    server.store.log_detection(det, "simulated")
    _, body = _get(server, "/api/events")
    events = json.loads(body)["events"]
    assert events and events[0]["user_feedback"] is None

    event_id = events[0]["id"]
    _post(
        server, "/api/feedback",
        {"event_id": event_id, "label": "false_positive"},
        token=server.token,
    )
    _, body = _get(server, "/api/events")
    assert json.loads(body)["events"][0]["user_feedback"] == "false_positive"


def test_unknown_feedback_label_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server, "/api/feedback", {"event_id": 1, "label": "definitely_a_drone"},
            token=server.token,
        )
    assert exc.value.code == 400


def _add_review_event(server) -> int:
    from drone4rf.detectors.base import Detection

    det = Detection(
        detector="energy", timestamp=time.time(), center_hz=2.44e9,
        bandwidth_hz=1e5, peak_db=-50.0, avg_db=-55.0, snr_db=20.0,
        duration_s=0.02, score=0.5, features={"duty": 0.4},
    )
    server.store.log_detection(det, "simulated")
    return server.store.recent(limit=1)[0]["id"]


def test_structured_feedback_roundtrip_and_relabel_is_idempotent(server) -> None:
    event_id = _add_review_event(server)
    _, first = _post(
        server,
        "/api/feedback",
        {
            "event_id": event_id,
            "verdict": "confirmed_drone",
            "drone_model": "DJI Mini 4 Pro",
            "label_notes": "authorized flight",
            "labeled_by": "field-team-1",
        },
        token=server.token,
    )
    assert first["label"]["verdict"] == "confirmed_drone"
    assert first["label"]["user_feedback"] == "drone"

    _, second = _post(
        server,
        "/api/feedback",
        {
            "event_id": event_id,
            "verdict": "false_positive",
            "fp_class": "other",
            "fp_class_detail": "wireless camera",
            "labeled_by": "field-team-2",
        },
        token=server.token,
    )
    assert second["label"]["verdict"] == "false_positive"
    assert server.store.count() == 1
    row = server.store.recent(limit=1)[0]
    assert row["drone_model"] is None
    assert row["fp_class"] == "other"
    assert row["fp_class_detail"] == "wireless camera"
    assert row["user_feedback"] == "false_positive"


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "definitely_a_drone", "drone_model": "x"},
        {"verdict": "confirmed_drone"},
        {"verdict": "false_positive"},
        {"verdict": "false_positive", "fp_class": "wifi-ish"},
    ],
)
def test_structured_feedback_validation(server, payload) -> None:
    event_id = _add_review_event(server)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(
            server,
            "/api/feedback",
            {"event_id": event_id, **payload},
            token=server.token,
        )
    assert exc.value.code == 400


def test_unsure_clears_conditionally_irrelevant_fields(server) -> None:
    event_id = _add_review_event(server)
    _, data = _post(
        server,
        "/api/feedback",
        {
            "event_id": event_id,
            "verdict": "unsure",
            "drone_model": "should be cleared",
            "fp_class": "wifi_ap",
            "label_notes": "needs a second observer",
        },
        token=server.token,
    )
    assert data["label"]["drone_model"] is None
    assert data["label"]["fp_class"] is None
    assert data["label"]["user_feedback"] == ""


def test_label_export_requires_token_and_returns_features(server) -> None:
    event_id = _add_review_event(server)
    _post(
        server,
        "/api/feedback",
        {
            "event_id": event_id,
            "verdict": "false_positive",
            "fp_class": "wifi_ap",
            "labeled_by": "site-alpha",
        },
        token=server.token,
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/api/labels/export")
    assert exc.value.code == 403

    status, body = _get(
        server, "/api/labels/export", token=server.token
    )
    labels = json.loads(body)["labels"]
    assert status == 200
    assert len(labels) == 1
    assert labels[0]["event_id"] == event_id
    assert labels[0]["features"] == {"duty": 0.4}
    assert labels[0]["fp_class"] == "wifi_ap"


def test_label_export_csv(server) -> None:
    event_id = _add_review_event(server)
    _post(
        server,
        "/api/feedback",
        {"event_id": event_id, "verdict": "confirmed_drone", "drone_model": "custom"},
        token=server.token,
    )
    status, body = _get(
        server, "/api/labels/export?format=csv", token=server.token
    )
    assert status == 200
    assert "event_id" in body and "confirmed_drone" in body


def test_rf_signature_export_includes_unlabeled_and_requires_token(server) -> None:
    event_id = _add_review_event(server)
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/api/signatures/export")
    assert exc.value.code == 403

    status, body = _get(
        server, "/api/signatures/export", token=server.token
    )
    signatures = json.loads(body)["signatures"]
    assert status == 200
    assert len(signatures) == 1
    assert signatures[0]["event_id"] == event_id
    assert signatures[0]["signature_schema_version"] == "1.0"
    assert signatures[0]["features"] == {"duty": 0.4}
    assert signatures[0]["verdict"] is None


def test_rf_signature_export_csv_and_labeled_filter(server) -> None:
    event_id = _add_review_event(server)
    _post(
        server,
        "/api/feedback",
        {"event_id": event_id, "verdict": "false_positive", "fp_class": "zigbee"},
        token=server.token,
    )
    _add_review_event(server)  # remains unlabeled and must be filtered out
    status, body = _get(
        server,
        "/api/signatures/export?format=csv&labeled_only=1",
        token=server.token,
    )
    assert status == 200
    assert "signature_schema_version" in body
    assert "zigbee" in body
    assert body.count("\n") == 2  # header plus one labeled signature
