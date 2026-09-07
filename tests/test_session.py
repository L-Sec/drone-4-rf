"""Processed RF workshop-session recording format."""

import time
import csv
import json
import wave

import numpy as np
import pytest

from drone4rf.analytics import DroneAssessment
from drone4rf.detectors.base import Detection
from drone4rf.pipeline import SpectrumFrame
from drone4rf.sdr.base import IQChunk
from drone4rf.session import (
    SESSION_SCHEMA_VERSION,
    SessionError,
    SessionRecorder,
    read_session,
    session_info,
    session_path,
    validate_session_name,
    waveform_payload,
)
from drone4rf.tracking import SignalTrack, TrackEvent


def _frame() -> SpectrumFrame:
    return SpectrumFrame(
        timestamp=time.time(), center_hz=2.437e9, sample_rate=10e6,
        freqs_hz=np.linspace(2.432e9, 2.442e9, 4096),
        psd_db=np.r_[np.full(2048, -95.0), -30.0, np.full(2047, -95.0)],
        threshold_db=np.full(4096, -70.0), noise_floor_db=-96.5,
    )


def test_session_roundtrip_records_waterfall_and_rf_events(tmp_path) -> None:
    path = session_path(tmp_path, "class-a")
    recorder = SessionRecorder(path, {"source": "sim"})
    recorder.spectrum(_frame())
    recorder.detection(Detection(
        detector="energy", timestamp=time.time(), center_hz=2.437e9,
        bandwidth_hz=200e3, peak_db=-30.0, avg_db=-50.0, snr_db=40.0,
        duration_s=0.02, score=0.8, flags=("clipping",),
        features={"duty": np.float64(0.4)}, explanation="test emitter",
    ))
    track = SignalTrack(
        track_id=7, center_hz=2.437e9, bandwidth_hz=200e3,
        first_seen=time.time() - 1, last_seen=time.time(), hits=5,
        frames_observed=8, peak_db=-30.0, max_snr_db=40.0,
        detectors={"energy"}, promoted=True,
    )
    recorder.track(TrackEvent("track_persistent", track, "persistent test emitter"))
    recorder.assessment(42, DroneAssessment(
        timestamp=time.time(), entity="track", center_hz=2.437e9,
        freq_span_hz=200e3, confidence=0.75,
        category="probable_drone_activity", evidence={"burst": 0.8},
        observations=5, explanation="workshop example",
    ))
    recorder.label({"event_id": 42, "verdict": "unsure", "labeled_by": "student"})
    recorder.close()

    records = list(read_session(path))
    assert records[0]["schema_version"] == SESSION_SCHEMA_VERSION
    by_type = {record["type"]: record for record in records}
    assert len(by_type["spectrum"]["data"]["psd"]) == 1024
    assert max(by_type["spectrum"]["data"]["psd"]) == -30.0
    assert by_type["detection"]["data"]["features"] == {"duty": 0.4}
    assert by_type["assessment"]["data"]["event_id"] == 42
    assert by_type["label"]["data"]["labeled_by"] == "student"
    info = session_info(path)
    assert info["name"] == "class-a"
    assert info["counts"]["spectrum"] == 1
    assert info["size_bytes"] < 20_000


@pytest.mark.parametrize("name", ("../escape", "a/b", "", ".", "x" * 81))
def test_session_names_reject_path_traversal(name) -> None:
    with pytest.raises(SessionError):
        validate_session_name(name)


def test_synchronized_cf32_wav_and_detector_csv_artifacts(tmp_path) -> None:
    path = session_path(tmp_path, "rf-lab")
    recorder = SessionRecorder(
        path,
        {"mode": "scan"},
        capture_cf32=True,
        capture_wav=True,
        capture_csv=True,
    )
    samples = np.array(
        [0.25 + 0.5j, -0.5 - 0.25j, 1.0 - 1.0j], dtype=np.complex64
    )
    chunk = IQChunk(
        samples=samples, center_hz=915e6, sample_rate=2e6,
        timestamp=1234.5, seq=7,
    )
    recorder.iq(chunk)
    recorder.waveform({"timestamp": 1234.5, "i": [0.25], "q": [0.5]})
    recorder.spectrum(_frame())
    recorder.detection(Detection(
        detector="energy", timestamp=1234.5, center_hz=915e6,
        bandwidth_hz=100e3, peak_db=-35.0, avg_db=-50.0, snr_db=25.0,
        duration_s=0.03, score=0.9, features={"duty_cycle": 0.6},
    ))
    recorder.close()

    cf32 = np.fromfile(tmp_path / "rf-lab.cf32", dtype="<f4")
    assert cf32.tolist() == pytest.approx(
        [0.25, 0.5, -0.5, -0.25, 1.0, -1.0]
    )
    with wave.open(str(tmp_path / "rf-lab.iq.wav"), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 2_000_000
        assert wav.getnframes() == 3
        pcm = np.frombuffer(wav.readframes(3), dtype="<i2")
    assert pcm.tolist() == pytest.approx(
        [8192, 16384, -16384, -8192, 32767, -32767], abs=1
    )

    with (tmp_path / "rf-lab.detector.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["record_type"] for row in rows] == ["spectrum", "detection"]
    assert json.loads(rows[1]["features_json"]) == {"duty_cycle": 0.6}

    metadata = json.loads((tmp_path / "rf-lab.iq.json").read_text())
    assert metadata["layout"] == "interleaved I,Q"
    assert metadata["num_complex_samples"] == 3
    assert metadata["dropped_chunks"] == 0
    info = session_info(path)
    assert {item["kind"] for item in info["artifacts"]} == {
        "iq_cf32", "iq_wav", "detector_csv", "iq_metadata",
    }
    assert any(record["type"] == "waveform" for record in read_session(path))


def test_waveform_preview_is_a_contiguous_time_window() -> None:
    samples = np.arange(1024, dtype=np.float32).astype(np.complex64)
    payload = waveform_payload(IQChunk(
        samples=samples, center_hz=100e6, sample_rate=2e6,
        timestamp=1.0, seq=1,
    ))
    assert payload["i"] == pytest.approx(samples.real[256:768], abs=0.001)
    assert payload["window_us"] == pytest.approx(256.0)
