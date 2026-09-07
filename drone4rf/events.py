"""SQLite event store (WAL mode, single writer)."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drone4rf import __version__
from drone4rf.analytics.fusion import DroneAssessment
from drone4rf.detectors.base import Detection
from drone4rf.labels import (
    SIGNATURE_SCHEMA_VERSION,
    VERDICT_TO_LEGACY,
    legacy_label_payload,
    validate_structured_label,
)
from drone4rf.tracking import TrackEvent

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id               INTEGER PRIMARY KEY,
  ts_utc           TEXT NOT NULL,
  ts_local         TEXT NOT NULL,
  source           TEXT NOT NULL,
  detector         TEXT NOT NULL,
  kind             TEXT NOT NULL,
  center_hz        REAL NOT NULL,
  bandwidth_hz     REAL,
  peak_db          REAL,
  avg_db           REAL,
  snr_db           REAL,
  duration_s       REAL,
  score            REAL,
  confidence_label TEXT,
  explanation      TEXT,
  features_json    TEXT,
  quality_json     TEXT,
  config_version   TEXT,
  capture_path     TEXT,
  user_feedback    TEXT,
  verdict          TEXT,
  drone_model      TEXT,
  fp_class         TEXT,
  fp_class_detail  TEXT,
  label_notes      TEXT,
  labeled_by       TEXT,
  labeled_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_freq ON events (center_hz);
"""


def _timestamps(epoch: float) -> tuple[str, str]:
    utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return utc.isoformat(), utc.astimezone().isoformat()


class EventStore:
    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # The pipeline writes from its processing thread while the UI may
        # query from the main thread: share one connection under a lock.
        self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Migrations for databases created before newer columns existed.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
        migrations = (
            ("capture_path", "TEXT"),
            ("user_feedback", "TEXT"),
            ("verdict", "TEXT"),
            ("drone_model", "TEXT"),
            ("fp_class", "TEXT"),
            ("fp_class_detail", "TEXT"),
            ("label_notes", "TEXT"),
            ("labeled_by", "TEXT"),
            ("labeled_at", "TEXT"),
        )
        for name, sql_type in migrations:
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE events ADD COLUMN {name} {sql_type}"
                )
        # Preserve old feedback and make it immediately useful to the new
        # survey/export paths. Unknown legacy values remain untouched.
        self._conn.execute(
            "UPDATE events SET verdict = 'confirmed_drone', "
            "drone_model = COALESCE(drone_model, 'other/unknown') "
            "WHERE verdict IS NULL AND user_feedback IN ('drone', 'true_positive')"
        )
        self._conn.execute(
            "UPDATE events SET verdict = 'false_positive', "
            "fp_class = COALESCE(fp_class, 'other') "
            "WHERE verdict IS NULL AND user_feedback IN "
            "('false_positive', 'not_drone')"
        )
        self._conn.commit()

    def log_detection(self, det: Detection, source: str) -> None:
        ts_utc, ts_local = _timestamps(det.timestamp)
        self._insert(
            ts_utc=ts_utc,
            ts_local=ts_local,
            source=source,
            detector=det.detector,
            kind="detection",
            center_hz=det.center_hz,
            bandwidth_hz=det.bandwidth_hz,
            peak_db=det.peak_db,
            avg_db=det.avg_db,
            snr_db=det.snr_db,
            duration_s=det.duration_s,
            score=det.score,
            confidence_label="unclassified_rf_activity",
            explanation=det.explanation,
            features_json=json.dumps(det.features),
            quality_json=json.dumps(list(det.flags)),
        )

    def log_track_event(
        self, ev: TrackEvent, source: str, capture_path: str | None = None
    ) -> None:
        t = ev.track
        ts_utc, ts_local = _timestamps(t.last_seen)
        self._insert(
            ts_utc=ts_utc,
            ts_local=ts_local,
            source=source,
            detector="+".join(sorted(t.detectors)) or "tracker",
            kind=ev.kind,
            center_hz=t.center_hz,
            bandwidth_hz=t.bandwidth_hz,
            peak_db=t.peak_db,
            avg_db=None,
            snr_db=t.max_snr_db,
            duration_s=max(0.0, t.last_seen - t.first_seen),
            score=min(1.0, t.hits / 20.0),
            confidence_label="unclassified_rf_activity",
            explanation=ev.explanation,
            features_json=json.dumps(
                {"hits": t.hits, "frames": t.frames_observed, "occupancy": t.occupancy}
            ),
            quality_json=json.dumps(sorted(t.quality_flags)),
            capture_path=capture_path,
        )

    def log_assessment(self, a: DroneAssessment, source: str) -> int:
        ts_utc, ts_local = _timestamps(a.timestamp)
        return self._insert(
            ts_utc=ts_utc,
            ts_local=ts_local,
            source=source,
            detector="fusion",
            kind="assessment",
            center_hz=a.center_hz,
            bandwidth_hz=a.freq_span_hz,
            peak_db=None,
            avg_db=None,
            snr_db=None,
            duration_s=None,
            score=a.confidence,
            confidence_label=a.category,
            explanation=a.explanation,
            features_json=json.dumps(
                {
                    "entity": a.entity,
                    "observations": a.observations,
                    "evidence": a.evidence,
                    "penalties": a.penalties,
                }
            ),
            quality_json="[]",
            capture_path=None,
        )

    def _insert(self, **row: Any) -> int:
        row["config_version"] = __version__
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO events ({cols}) VALUES ({marks})", tuple(row.values())
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_structured_feedback(
        self,
        event_id: int,
        *,
        verdict: Any,
        drone_model: Any = None,
        fp_class: Any = None,
        fp_class_detail: Any = None,
        label_notes: Any = None,
        labeled_by: Any = None,
    ) -> dict[str, Any]:
        """Validate and idempotently store a survey label on an event."""
        label = validate_structured_label(
            verdict=verdict,
            drone_model=drone_model,
            fp_class=fp_class,
            fp_class_detail=fp_class_detail,
            label_notes=label_notes,
            labeled_by=labeled_by,
        )
        labeled_at = datetime.now(timezone.utc).isoformat()
        legacy = VERDICT_TO_LEGACY[label["verdict"]]
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET user_feedback = ?, verdict = ?, "
                "drone_model = ?, fp_class = ?, fp_class_detail = ?, "
                "label_notes = ?, labeled_by = ?, labeled_at = ? WHERE id = ?",
                (
                    legacy,
                    label["verdict"],
                    label["drone_model"],
                    label["fp_class"],
                    label["fp_class_detail"],
                    label["label_notes"],
                    label["labeled_by"],
                    labeled_at,
                    int(event_id),
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                raise KeyError(f"event {event_id} not found")
            self._conn.commit()
        return {
            "event_id": int(event_id),
            **label,
            "labeled_at": labeled_at,
            "user_feedback": legacy,
        }

    def set_feedback(self, event_id: int, feedback: str) -> dict[str, Any] | None:
        """Keep the v0.7 flat-label API working via structured defaults."""
        label = legacy_label_payload(feedback)
        if label is None:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE events SET user_feedback = '', verdict = NULL, "
                    "drone_model = NULL, fp_class = NULL, fp_class_detail = NULL, "
                    "label_notes = NULL, labeled_by = NULL, labeled_at = NULL "
                    "WHERE id = ?",
                    (int(event_id),),
                )
                if cur.rowcount != 1:
                    self._conn.rollback()
                    raise KeyError(f"event {event_id} not found")
                self._conn.commit()
            return None
        return self.set_structured_feedback(event_id, **label)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            )

    def export_labels(self) -> list[dict[str, Any]]:
        """Return survey-grade labeled events with parsed feature payloads."""
        columns = (
            "id", "ts_utc", "source", "detector", "kind", "center_hz",
            "bandwidth_hz", "snr_db", "duration_s", "score",
            "confidence_label", "features_json", "quality_json",
            "config_version", "verdict", "drone_model", "fp_class",
            "fp_class_detail", "label_notes", "labeled_by", "labeled_at",
        )
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(columns)} FROM events "
                "WHERE verdict IS NOT NULL AND verdict != '' ORDER BY id"
            ).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            item = dict(zip(columns, row))
            item["event_id"] = item.pop("id")
            item["features"] = _parse_json(item.pop("features_json"), {})
            item["quality_flags"] = _parse_json(item.pop("quality_json"), [])
            records.append(item)
        return records

    def export_signatures(
        self, *, labeled_only: bool = False
    ) -> list[dict[str, Any]]:
        """Return versioned RF signatures for all stored event kinds.

        Unlike ``export_labels``, this includes unlabeled observations so a
        survey team can retain the complete captured signature corpus.
        """
        columns = (
            "id", "ts_utc", "source", "detector", "kind", "center_hz",
            "bandwidth_hz", "peak_db", "avg_db", "snr_db", "duration_s",
            "score", "confidence_label", "explanation", "features_json",
            "quality_json", "config_version", "verdict", "drone_model",
            "fp_class", "fp_class_detail", "label_notes", "labeled_by",
            "labeled_at",
        )
        where = (
            " WHERE verdict IS NOT NULL AND verdict != ''"
            if labeled_only
            else ""
        )
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(columns)} FROM events{where} ORDER BY id"
            ).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            item = dict(zip(columns, row))
            item["signature_schema_version"] = SIGNATURE_SCHEMA_VERSION
            item["event_id"] = item.pop("id")
            item["features"] = _parse_json(item.pop("features_json"), {})
            item["quality_flags"] = _parse_json(item.pop("quality_json"), [])
            records.append(item)
        return records

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _parse_json(value: str | None, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return fallback
