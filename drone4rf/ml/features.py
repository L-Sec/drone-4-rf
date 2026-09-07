"""Versioned feature schema shared by training and inference.

The feature vector is derived from the same evidence/penalty scores the
fusion engine computes deterministically - assessment rows in the events
database therefore double as training examples, and runtime featurization
is guaranteed to match training featurization as long as FEATURE_VERSION
agrees. Bump FEATURE_VERSION whenever FEATURE_NAMES or any derivation
changes; models refuse to load against a different version.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

FEATURE_VERSION = "1.0"

FEATURE_NAMES: tuple[str, ...] = (
    "anomaly",
    "burst",
    "hopping",
    "emi",
    "multiband",
    "wifi_similarity",
    "bt_similarity",
    "fixed_emitter",
    "observations_log",
    "span_log",
)


def vector_from_scores(
    evidence: Mapping[str, float],
    penalties: Mapping[str, float],
    observations: int,
    span_hz: float,
) -> np.ndarray:
    """Build the feature vector from fusion scores.

    Only the fixed names above are consumed; extra keys (including any
    'ml' terms a live model added to an assessment) are ignored, so
    training on model-annotated history cannot leak the model's own
    output into its features.
    """
    values = [
        float(evidence.get("anomaly", 0.0)),
        float(evidence.get("burst", 0.0)),
        float(evidence.get("hopping", 0.0)),
        float(evidence.get("emi", 0.0)),
        float(evidence.get("multiband", 0.0)),
        float(penalties.get("wifi_similarity", 0.0)),
        float(penalties.get("bt_similarity", 0.0)),
        float(penalties.get("fixed_emitter", 0.0)),
        math.log10(max(1, observations)),
        math.log10(max(1.0, span_hz)),
    ]
    return np.asarray(values, dtype=np.float64)


def vector_from_assessment_features(
    features: Mapping[str, Any], bandwidth_hz: float | None
) -> np.ndarray | None:
    """Featurize a stored assessment row's features_json payload.

    Returns None when the payload does not carry fusion scores (e.g.
    rows written by other event kinds).
    """
    evidence = features.get("evidence")
    penalties = features.get("penalties")
    if not isinstance(evidence, dict) or not isinstance(penalties, dict):
        return None
    return vector_from_scores(
        evidence,
        penalties,
        int(features.get("observations", 1)),
        float(bandwidth_hz or 0.0),
    )
