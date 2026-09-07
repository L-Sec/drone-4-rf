"""Dataset construction from the events database, with provenance.

Labels come exclusively from operator feedback on assessment rows. Structured
`verdict` values are preferred, with the v0.7 `user_feedback` column retained
as a compatibility fallback:

- 'false_positive'          -> class 0 (not drone)
- 'drone' / 'true_positive' -> class 1 (drone-related)
- anything else / NULL      -> -1 (unlabeled; kept for anomaly training)

No label is ever fabricated: rows without operator feedback stay
unlabeled, and the classifier trainer refuses datasets without enough
genuine examples of both classes.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from drone4rf import __version__
from drone4rf.events import EventStore
from drone4rf.ml.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    vector_from_assessment_features,
)

LABEL_NOT_DRONE = 0
LABEL_DRONE = 1
LABEL_UNLABELED = -1

_FEEDBACK_LABELS = {
    "false_positive": LABEL_NOT_DRONE,
    "not_drone": LABEL_NOT_DRONE,
    "drone": LABEL_DRONE,
    "true_positive": LABEL_DRONE,
}
_VERDICT_LABELS = {
    "confirmed_drone": LABEL_DRONE,
    "false_positive": LABEL_NOT_DRONE,
    "unsure": LABEL_UNLABELED,
}


@dataclass
class DatasetMeta:
    created_utc: str = ""
    source_db: str = ""
    site: str = ""
    feature_version: str = FEATURE_VERSION
    config_version: str = __version__
    n_total: int = 0
    n_drone: int = 0
    n_not_drone: int = 0
    n_unlabeled: int = 0
    event_ids: list = field(default_factory=list)


def build_dataset(
    db_path: str | Path, site: str = "unspecified-site"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetMeta]:
    """Extract (X, y, sites, meta) from a drone4rf events database."""
    # Opening through EventStore applies additive migrations before the
    # direct read below, including for databases created by v0.7.
    store = EventStore(db_path)
    store.close()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, bandwidth_hz, features_json, verdict, user_feedback "
            "FROM events WHERE kind = 'assessment'"
        ).fetchall()
    finally:
        conn.close()

    vectors, labels, ids = [], [], []
    for event_id, bandwidth_hz, features_json, verdict, feedback in rows:
        try:
            features = json.loads(features_json or "{}")
        except json.JSONDecodeError:
            continue
        vec = vector_from_assessment_features(features, bandwidth_hz)
        if vec is None:
            continue
        vectors.append(vec)
        structured = (verdict or "").strip().lower()
        if structured:
            labels.append(_VERDICT_LABELS.get(structured, LABEL_UNLABELED))
        else:
            labels.append(
                _FEEDBACK_LABELS.get(
                    (feedback or "").strip().lower(), LABEL_UNLABELED
                )
            )
        ids.append(int(event_id))

    X = np.vstack(vectors) if vectors else np.empty((0, len(FEATURE_NAMES)))
    y = np.asarray(labels, dtype=np.int64)
    sites = np.asarray([site] * len(y))
    meta = DatasetMeta(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_db=str(db_path),
        site=site,
        n_total=len(y),
        n_drone=int(np.sum(y == LABEL_DRONE)),
        n_not_drone=int(np.sum(y == LABEL_NOT_DRONE)),
        n_unlabeled=int(np.sum(y == LABEL_UNLABELED)),
        event_ids=ids,
    )
    return X, y, sites, meta


def merge_datasets(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, DatasetMeta]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetMeta]:
    """Combine per-site datasets (enables site-independent evaluation)."""
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    sites = np.concatenate([p[2] for p in parts])
    meta = DatasetMeta(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_db=";".join(p[3].source_db for p in parts),
        site=";".join(p[3].site for p in parts),
        n_total=len(y),
        n_drone=int(np.sum(y == LABEL_DRONE)),
        n_not_drone=int(np.sum(y == LABEL_NOT_DRONE)),
        n_unlabeled=int(np.sum(y == LABEL_UNLABELED)),
        event_ids=[i for p in parts for i in p[3].event_ids],
    )
    return X, y, sites, meta


def save_dataset(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    meta: DatasetMeta,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        X=X,
        y=y,
        sites=sites,
        feature_names=np.asarray(FEATURE_NAMES),
        feature_version=FEATURE_VERSION,
        meta_json=json.dumps(asdict(meta)),
    )


def load_dataset(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetMeta]:
    with np.load(Path(path), allow_pickle=False) as data:
        if str(data["feature_version"]) != FEATURE_VERSION:
            raise ValueError(
                f"dataset feature version {data['feature_version']} does not "
                f"match current {FEATURE_VERSION}; rebuild the dataset"
            )
        meta = DatasetMeta(**json.loads(str(data["meta_json"])))
        return data["X"], data["y"], data["sites"], meta
