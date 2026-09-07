"""Training, persistence, and CPU-only inference of the ML models.

Model bundles are joblib files carrying the fitted estimator plus
feature-schema version, provenance, and measured cross-validation
metrics. Loading refuses feature-version mismatches. Everything here is
scikit-learn (CPU-only); the import is deferred so the core scanner
never requires it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from drone4rf import __version__
from drone4rf.config import MLConfig
from drone4rf.ml.dataset import (
    LABEL_DRONE,
    LABEL_NOT_DRONE,
    LABEL_UNLABELED,
    DatasetMeta,
)
from drone4rf.ml.features import FEATURE_NAMES, FEATURE_VERSION

log = logging.getLogger(__name__)

_SKLEARN_HELP = (
    "scikit-learn is required for the ML subsystem: pip install .[ml]"
)

# Below this many examples per class, probability calibration would be
# fit on too little data to be meaningful.
MIN_PER_CLASS = 5
MIN_FOR_CALIBRATION = 15


class MLError(RuntimeError):
    """Raised for ML training/loading problems, with actionable messages."""


def _import_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise MLError(_SKLEARN_HELP) from exc
    import joblib

    return joblib


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray | None = None,
    meta: DatasetMeta | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Train a calibrated Random Forest on the LABELED subset of (X, y).

    Handles class imbalance via class_weight='balanced', measures
    stratified cross-validation metrics, additionally measures
    site-independent (GroupKFold) metrics when >= 2 sites are present,
    and calibrates probabilities (sigmoid) when both classes have enough
    examples. Refuses to train without genuine labels for both classes.
    """
    _import_sklearn()
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import (
        GroupKFold,
        StratifiedKFold,
        cross_val_predict,
    )

    labeled = y != LABEL_UNLABELED
    X_l, y_l = X[labeled], y[labeled]
    sites_l = sites[labeled] if sites is not None else None
    n_drone = int(np.sum(y_l == LABEL_DRONE))
    n_not = int(np.sum(y_l == LABEL_NOT_DRONE))
    if min(n_drone, n_not) < MIN_PER_CLASS:
        raise MLError(
            f"need at least {MIN_PER_CLASS} labeled examples of each class; "
            f"have drone={n_drone}, not_drone={n_not}. Label events with "
            f"'drone4rf ml label' or the GUI's false-positive button."
        )

    base = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=random_state
    )

    n_splits = min(5, n_drone, n_not)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(base, X_l, y_l, cv=skf)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_l, y_pred, labels=[LABEL_NOT_DRONE, LABEL_DRONE], zero_division=0
    )
    metrics: dict[str, Any] = {
        "cv_splits": n_splits,
        "precision_not_drone": float(precision[0]),
        "recall_not_drone": float(recall[0]),
        "f1_not_drone": float(f1[0]),
        "precision_drone": float(precision[1]),
        "recall_drone": float(recall[1]),
        "f1_drone": float(f1[1]),
    }

    # Site-independent testing: train on some sites, test on others.
    if sites_l is not None and len(np.unique(sites_l)) >= 2:
        gkf = GroupKFold(n_splits=min(len(np.unique(sites_l)), 5))
        y_site_pred = cross_val_predict(base, X_l, y_l, cv=gkf, groups=sites_l)
        _, _, f1_site, _ = precision_recall_fscore_support(
            y_l, y_site_pred, labels=[LABEL_NOT_DRONE, LABEL_DRONE],
            zero_division=0,
        )
        metrics["site_independent_f1_drone"] = float(f1_site[1])
        metrics["site_independent_f1_not_drone"] = float(f1_site[0])
    else:
        metrics["site_independent_f1_drone"] = None  # single-site data

    calibrated = min(n_drone, n_not) >= MIN_FOR_CALIBRATION
    if calibrated:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=n_splits)
        model.fit(X_l, y_l)
    else:
        log.warning(
            "too few examples for probability calibration (%d/%d); "
            "using raw forest probabilities - treat confidences with care",
            n_drone, n_not,
        )
        model = base.fit(X_l, y_l)

    import sklearn

    return {
        "kind": "classifier",
        "model": model,
        "calibrated": calibrated,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "classes": {"not_drone": LABEL_NOT_DRONE, "drone": LABEL_DRONE},
        "metrics": metrics,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "drone4rf_version": __version__,
        "provenance": {
            "n_drone": n_drone,
            "n_not_drone": n_not,
            "sites": sorted(set(map(str, sites_l))) if sites_l is not None else [],
            "dataset": (meta.__dict__ if meta is not None else {}),
        },
    }


def train_anomaly(
    X: np.ndarray, meta: DatasetMeta | None = None, random_state: int = 0
) -> dict[str, Any]:
    """Isolation Forest over ALL feature vectors (labels not required):
    scores how unusual an entity's behavioral profile is for this site."""
    _import_sklearn()
    from sklearn.ensemble import IsolationForest

    if len(X) < 20:
        raise MLError(
            f"need at least 20 feature vectors to train the anomaly model; "
            f"have {len(X)}"
        )
    import sklearn

    model = IsolationForest(
        n_estimators=200, random_state=random_state, contamination="auto"
    ).fit(X)
    return {
        "kind": "anomaly",
        "model": model,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "drone4rf_version": __version__,
        "provenance": {
            "n_samples": int(len(X)),
            "dataset": (meta.__dict__ if meta is not None else {}),
        },
    }


def save_model(bundle: dict[str, Any], path: str | Path) -> None:
    joblib = _import_sklearn()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)
    log.info("model bundle (%s) saved to %s", bundle["kind"], p)


def load_model(path: str | Path, expected_kind: str | None = None) -> dict[str, Any]:
    joblib = _import_sklearn()
    p = Path(path)
    if not p.exists():
        raise MLError(
            f"no model at {p}; train one with 'drone4rf ml train' or set "
            f"ml.enabled: false"
        )
    bundle = joblib.load(p)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise MLError(
            f"model at {p} was trained with feature schema "
            f"{bundle.get('feature_version')} but the code uses "
            f"{FEATURE_VERSION}; retrain the model"
        )
    if expected_kind and bundle.get("kind") != expected_kind:
        raise MLError(
            f"model at {p} is a '{bundle.get('kind')}' bundle, "
            f"expected '{expected_kind}'"
        )
    return bundle


@dataclass(frozen=True)
class MLScore:
    p_drone: float | None  # None when rejected as unknown
    rejected: bool
    anomaly01: float | None  # None when no anomaly model is loaded
    model_version: str


class MLScorer:
    """CPU-only runtime inference over fusion feature vectors."""

    def __init__(
        self,
        classifier_bundle: dict[str, Any],
        anomaly_bundle: dict[str, Any] | None,
        min_confidence: float,
    ) -> None:
        self._clf = classifier_bundle
        self._anom = anomaly_bundle
        self.min_confidence = min_confidence
        self.version = classifier_bundle["trained_utc"]
        self.calibrated = bool(classifier_bundle.get("calibrated"))

    @classmethod
    def from_config(cls, cfg: MLConfig) -> "MLScorer":
        clf = load_model(cfg.model_path, expected_kind="classifier")
        anom = (
            load_model(cfg.anomaly_model_path, expected_kind="anomaly")
            if cfg.anomaly_model_path
            else None
        )
        return cls(clf, anom, cfg.min_calibrated_confidence)

    def score(self, features: np.ndarray) -> MLScore:
        fv = features.reshape(1, -1)
        model = self._clf["model"]
        proba = model.predict_proba(fv)[0]
        classes = list(model.classes_)
        p_drone = float(proba[classes.index(LABEL_DRONE)])
        p_max = float(proba.max())
        # Unknown-class rejection: an uncertain classifier stays silent.
        rejected = p_max < self.min_confidence

        anomaly01 = None
        if self._anom is not None:
            df = float(self._anom["model"].decision_function(fv)[0])
            # decision_function: positive = inlier. Map to [0, 1] novelty.
            anomaly01 = float(np.clip(0.5 - df, 0.0, 1.0))
        return MLScore(
            p_drone=None if rejected else p_drone,
            rejected=rejected,
            anomaly01=anomaly01,
            model_version=self.version,
        )
