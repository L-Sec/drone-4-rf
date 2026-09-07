"""Structured crowd-survey labels, migration, and export."""

import json
import sqlite3
import time

from drone4rf.detectors.base import Detection
from drone4rf.events import EventStore


def _event(store: EventStore) -> int:
    store.log_detection(
        Detection(
            detector="energy",
            timestamp=time.time(),
            center_hz=2.44e9,
            bandwidth_hz=1e5,
            peak_db=-50.0,
            avg_db=-55.0,
            snr_db=20.0,
            duration_s=0.02,
            score=0.5,
            features={"duty": 0.25},
        ),
        "simulated",
    )
    return store.recent(limit=1)[0]["id"]


def test_v07_database_migrates_in_place_and_preserves_rows(tmp_path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts_utc TEXT, "
        "center_hz REAL, user_feedback TEXT)"
    )
    conn.execute(
        "INSERT INTO events (id, ts_utc, center_hz, user_feedback) "
        "VALUES (7, '2026-01-01T00:00:00+00:00', 2437000000, 'drone')"
    )
    conn.commit()
    conn.close()

    store = EventStore(db)
    store.close()

    conn = sqlite3.connect(str(db))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    row = conn.execute(
        "SELECT id, user_feedback, verdict, drone_model FROM events WHERE id = 7"
    ).fetchone()
    conn.close()
    assert {
        "verdict", "drone_model", "fp_class", "fp_class_detail",
        "label_notes", "labeled_by", "labeled_at",
    } <= columns
    assert row == (7, "drone", "confirmed_drone", "other/unknown")


def test_structured_feedback_overwrites_without_adding_rows(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    event_id = _event(store)
    first = store.set_structured_feedback(
        event_id,
        verdict="confirmed_drone",
        drone_model="DJI Mavic 3",
        labeled_by="survey-a",
    )
    second = store.set_structured_feedback(
        event_id,
        verdict="false_positive",
        fp_class="other",
        fp_class_detail="wireless camera",
        label_notes="stationary emitter",
        labeled_by="survey-b",
    )
    row = store.recent(limit=1)[0]
    exported = store.export_labels()
    store.close()

    assert first["verdict"] == "confirmed_drone"
    assert second["verdict"] == "false_positive"
    assert row["user_feedback"] == "false_positive"
    assert row["drone_model"] is None
    assert row["fp_class"] == "other"
    assert row["fp_class_detail"] == "wireless camera"
    assert len(exported) == 1
    assert exported[0]["event_id"] == event_id
    assert exported[0]["features"] == {"duty": 0.25}


def test_export_labels_cli_writes_json(tmp_path) -> None:
    from drone4rf.cli import build_parser

    db = tmp_path / "events.db"
    store = EventStore(db)
    event_id = _event(store)
    store.set_structured_feedback(
        event_id,
        verdict="false_positive",
        fp_class="ble",
        labeled_by="cli-surveyor",
    )
    store.close()

    destination = tmp_path / "labels.json"
    args = build_parser().parse_args(
        [
            "ml", "export-labels", "--db", str(db), "--out",
            str(destination), "--format", "json",
        ]
    )
    assert args.func(args) == 0
    records = json.loads(destination.read_text(encoding="utf-8"))
    assert records[0]["event_id"] == event_id
    assert records[0]["fp_class"] == "ble"


def test_signature_export_includes_unlabeled_events_and_can_filter(tmp_path) -> None:
    store = EventStore(tmp_path / "events.db")
    unlabeled_id = _event(store)
    labeled_id = _event(store)
    store.set_structured_feedback(
        labeled_id,
        verdict="confirmed_drone",
        drone_model="generic 2.4 GHz control link",
    )
    all_signatures = store.export_signatures()
    labeled_signatures = store.export_signatures(labeled_only=True)
    store.close()

    assert [row["event_id"] for row in all_signatures] == [
        unlabeled_id, labeled_id,
    ]
    assert all_signatures[0]["signature_schema_version"] == "1.0"
    assert all_signatures[0]["features"] == {"duty": 0.25}
    assert all_signatures[0]["verdict"] is None
    assert [row["event_id"] for row in labeled_signatures] == [labeled_id]


def test_export_signatures_cli_writes_csv(tmp_path) -> None:
    from drone4rf.cli import build_parser

    db = tmp_path / "events.db"
    store = EventStore(db)
    event_id = _event(store)
    store.close()

    destination = tmp_path / "signatures.csv"
    args = build_parser().parse_args(
        [
            "ml", "export-signatures", "--db", str(db), "--out",
            str(destination), "--format", "csv",
        ]
    )
    assert args.func(args) == 0
    content = destination.read_text(encoding="utf-8")
    assert "signature_schema_version" in content
    assert str(event_id) in content
