"""Stage 6 tests: features, dataset, training, inference, fusion hookup."""

import json
import math
import time

import numpy as np
import pytest

pytest.importorskip("sklearn")

from drone4rf.analytics.fusion import (
    CATEGORY_POSSIBLE,
    FusionEngine,
    category_rank,
)
from drone4rf.config import AnalyticsConfig, AppConfig, MLConfig
from drone4rf.events import EventStore
from drone4rf.ml import dataset as ds
from drone4rf.ml import models as ml_models
from drone4rf.ml.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    vector_from_assessment_features,
    vector_from_scores,
)
from drone4rf.ml.synth import SYNTHETIC_MARKER, synthetic_dataset


# -- features ---------------------------------------------------------------


def test_feature_vector_shape_and_order() -> None:
    fv = vector_from_scores(
        {"hopping": 0.9, "anomaly": 0.5}, {"bt_similarity": 0.3}, 10, 1e6
    )
    assert fv.shape == (len(FEATURE_NAMES),)
    assert fv[FEATURE_NAMES.index("hopping")] == 0.9
    assert fv[FEATURE_NAMES.index("bt_similarity")] == 0.3
    assert fv[FEATURE_NAMES.index("observations_log")] == 1.0  # log10(10)


def test_ml_terms_never_leak_into_features() -> None:
    fv = vector_from_scores(
        {"hopping": 0.9, "ml": 0.99, "ml_anomaly": 0.8},
        {"ml_background": 0.7},
        5,
        1e6,
    )
    # 'ml*' keys are not in FEATURE_NAMES and must be silently ignored.
    assert not math.isclose(fv.max(), 0.99)
    assert 0.99 not in fv and 0.8 not in fv and 0.7 not in fv


def test_vector_from_assessment_features_rejects_foreign_payloads() -> None:
    assert vector_from_assessment_features({"hits": 5}, 1e6) is None


# -- dataset ------------------------------------------------------------------


def _make_labeled_db(tmp_path, n_drone=6, n_fp=6):
    """Events DB with labeled assessment rows via the real EventStore."""
    from drone4rf.analytics.fusion import DroneAssessment

    store = EventStore(tmp_path / "events.db")
    rng = np.random.default_rng(0)
    ids_by_label = {"drone": [], "false_positive": [], None: []}
    for i in range(n_drone + n_fp + 3):
        drone_like = i < n_drone
        a = DroneAssessment(
            timestamp=time.time(),
            entity="hopper" if drone_like else "track",
            center_hz=2.44e9,
            freq_span_hz=5e6,
            confidence=0.7,
            category="possible_drone_activity",
            evidence={
                "hopping": float(rng.uniform(0.6, 1.0) if drone_like else rng.uniform(0, 0.2)),
                "anomaly": float(rng.uniform(0.5, 1.0)),
                "burst": float(rng.uniform(0.5, 1.0) if drone_like else 0.0),
            },
            penalties={
                "bt_similarity": float(0.0 if drone_like else rng.uniform(0.5, 1.0)),
                "wifi_similarity": float(0.0 if drone_like else rng.uniform(0.3, 1.0)),
            },
            observations=10,
            explanation="test",
        )
        store.log_assessment(a, "simulated")
        row_id = store.recent(limit=1)[0]["id"]
        if i < n_drone:
            store.set_feedback(row_id, "drone")
            ids_by_label["drone"].append(row_id)
        elif i < n_drone + n_fp:
            store.set_feedback(row_id, "false_positive")
            ids_by_label["false_positive"].append(row_id)
    store.close()
    return tmp_path / "events.db", ids_by_label


def test_build_dataset_from_events_db(tmp_path) -> None:
    db, _ = _make_labeled_db(tmp_path)
    X, y, sites, meta = ds.build_dataset(db, site="test-site")
    assert meta.n_drone == 6
    assert meta.n_not_drone == 6
    assert meta.n_unlabeled == 3
    assert X.shape == (15, len(FEATURE_NAMES))
    assert meta.source_db == str(db)
    assert len(meta.event_ids) == 15  # provenance: exact rows recorded


def test_dataset_save_load_roundtrip(tmp_path) -> None:
    db, _ = _make_labeled_db(tmp_path)
    X, y, sites, meta = ds.build_dataset(db, site="s1")
    path = tmp_path / "dataset.npz"
    ds.save_dataset(path, X, y, sites, meta)
    X2, y2, sites2, meta2 = ds.load_dataset(path)
    assert np.allclose(X, X2)
    assert np.array_equal(y, y2)
    assert meta2.site == "s1"


# -- training ------------------------------------------------------------------


def test_training_refuses_insufficient_labels(tmp_path) -> None:
    db, _ = _make_labeled_db(tmp_path, n_drone=2, n_fp=6)
    X, y, sites, meta = ds.build_dataset(db)
    with pytest.raises(ml_models.MLError, match="at least 5 labeled"):
        ml_models.train_classifier(X, y, sites=sites)


def test_synthetic_training_and_metrics() -> None:
    X, y, sites, meta = synthetic_dataset(n_per_class=100, seed=1)
    bundle = ml_models.train_classifier(X, y, sites=sites, meta=meta)
    m = bundle["metrics"]
    assert m["f1_drone"] > 0.9  # separable by construction
    assert m["site_independent_f1_drone"] is not None  # two synth sites
    assert bundle["calibrated"] is True
    assert bundle["feature_version"] == FEATURE_VERSION
    assert SYNTHETIC_MARKER in bundle["provenance"]["dataset"]["source_db"]


def test_model_save_load_and_version_guard(tmp_path) -> None:
    X, y, sites, meta = synthetic_dataset(n_per_class=60, seed=2)
    bundle = ml_models.train_classifier(X, y, sites=sites, meta=meta)
    path = tmp_path / "model.joblib"
    ml_models.save_model(bundle, path)

    loaded = ml_models.load_model(path, expected_kind="classifier")
    assert loaded["metrics"]["f1_drone"] == bundle["metrics"]["f1_drone"]
    with pytest.raises(ml_models.MLError, match="'classifier'"):
        ml_models.load_model(path, expected_kind="anomaly")

    # Feature-version mismatch must refuse to load.
    import joblib

    stale = dict(bundle, feature_version="0.0")
    joblib.dump(stale, tmp_path / "stale.joblib")
    with pytest.raises(ml_models.MLError, match="feature schema"):
        ml_models.load_model(tmp_path / "stale.joblib")


def test_missing_model_is_clear_error(tmp_path) -> None:
    with pytest.raises(ml_models.MLError, match="no model at"):
        ml_models.load_model(tmp_path / "nope.joblib")


def test_unknown_class_rejection() -> None:
    X, y, sites, meta = synthetic_dataset(n_per_class=100, seed=3)
    bundle = ml_models.train_classifier(X, y, sites=sites, meta=meta)
    scorer = ml_models.MLScorer(bundle, None, min_confidence=0.6)

    clear = scorer.score(X[0])  # a training-like drone example
    assert not clear.rejected and clear.p_drone is not None

    # A vector engineered to sit between the classes should be rejected
    # by a high enough threshold.
    ambiguous = (X[:100].mean(axis=0) + X[100:].mean(axis=0)) / 2
    strict = ml_models.MLScorer(bundle, None, min_confidence=0.95)
    r = strict.score(ambiguous)
    assert r.rejected or 0.05 < r.p_drone < 0.95


def test_anomaly_model_flags_novelty() -> None:
    X, y, sites, meta = synthetic_dataset(n_per_class=100, seed=4)
    bundle = ml_models.train_anomaly(X, meta=meta)
    scorer = ml_models.MLScorer(
        ml_models.train_classifier(X, y, sites=sites), bundle, 0.6
    )
    weird = np.full(len(FEATURE_NAMES), 9.0)  # far outside training space
    novel = scorer.score(weird)
    normal = scorer.score(X[0])
    assert novel.anomaly01 is not None and normal.anomaly01 is not None
    assert novel.anomaly01 > normal.anomaly01


# -- fusion integration ----------------------------------------------------------


class _StubScorer:
    calibrated = True
    version = "stub"

    def __init__(self, p_drone, rejected=False, anomaly=None):
        self._r = ml_models.MLScore(
            p_drone=None if rejected else p_drone,
            rejected=rejected,
            anomaly01=anomaly,
            model_version="stub-model",
        )

    def score(self, fv):
        return self._r


def _bursty_track():
    from tests.test_fusion import _track

    return _track(1, 2.42e9, duty=0.3, n_bursts=3, cv=0.1)


def test_fusion_ml_evidence_raises_category() -> None:
    cfg = AnalyticsConfig()
    base = FusionEngine(cfg).assess(10.0, [_bursty_track()], [], [])[0]
    boosted = FusionEngine(cfg, ml_scorer=_StubScorer(0.95)).assess(
        10.0, [_bursty_track()], [], []
    )[0]
    assert boosted.confidence > base.confidence
    assert boosted.evidence["ml"] == 0.95
    assert "ML classifier" in boosted.explanation


def test_fusion_ml_background_lowers_category() -> None:
    cfg = AnalyticsConfig()
    base = FusionEngine(cfg).assess(10.0, [_bursty_track()], [], [])[0]
    damped = FusionEngine(cfg, ml_scorer=_StubScorer(0.05)).assess(
        10.0, [_bursty_track()], [], []
    )[0]
    assert damped.confidence < base.confidence
    assert damped.penalties["ml_background"] == 0.95


def test_fusion_ml_rejection_changes_nothing() -> None:
    cfg = AnalyticsConfig()
    base = FusionEngine(cfg).assess(10.0, [_bursty_track()], [], [])[0]
    same = FusionEngine(cfg, ml_scorer=_StubScorer(None, rejected=True)).assess(
        10.0, [_bursty_track()], [], []
    )[0]
    assert same.confidence == base.confidence
    assert "abstained" in same.explanation


def test_analytics_engine_fails_fast_without_model(tmp_path) -> None:
    from drone4rf.analytics import AnalyticsEngine

    cfg = AppConfig(
        ml=MLConfig(enabled=True, model_path=str(tmp_path / "missing.joblib"))
    )
    cfg.validate()
    with pytest.raises(ml_models.MLError, match="no model at"):
        AnalyticsEngine(cfg)


def test_ml_disabled_by_default() -> None:
    cfg = AppConfig()
    assert cfg.ml.enabled is False
    from drone4rf.analytics import AnalyticsEngine

    engine = AnalyticsEngine(cfg)
    assert engine.fusion.ml_scorer is None
