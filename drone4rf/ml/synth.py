"""Synthetic feature data - STRICTLY for pipeline testing.

This generates fabricated feature vectors so the dataset/training/
inference plumbing can be exercised end-to-end without real captures.
Models trained on it carry provenance marking them SYNTHETIC and must
never be deployed as if they reflected real drone behavior.
"""

from __future__ import annotations

import numpy as np

from drone4rf.ml.dataset import (
    LABEL_DRONE,
    LABEL_NOT_DRONE,
    DatasetMeta,
)
from drone4rf.ml.features import FEATURE_NAMES

SYNTHETIC_MARKER = "SYNTHETIC-PIPELINE-TEST-ONLY"


def synthetic_dataset(
    n_per_class: int = 200, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetMeta]:
    """Two-class fabricated data with plausible (but invented) structure.

    'drone-like': strong hopping/burst/multiband evidence, low background
    similarity. 'background-like': the reverse. Split across two fake
    sites so the site-independent evaluation path is exercised too.
    """
    rng = np.random.default_rng(seed)
    d = len(FEATURE_NAMES)

    def block(n: int, drone: bool) -> np.ndarray:
        X = np.zeros((n, d))
        hi = lambda: rng.uniform(0.5, 1.0, n)  # noqa: E731
        lo = lambda: rng.uniform(0.0, 0.3, n)  # noqa: E731
        X[:, 0] = hi()  # anomaly
        X[:, 1] = hi() if drone else lo()  # burst
        X[:, 2] = hi() if drone else lo()  # hopping
        X[:, 3] = rng.uniform(0, 0.6, n)  # emi (ambiguous by design)
        X[:, 4] = (rng.uniform(0, 1, n) < (0.5 if drone else 0.05)).astype(float)
        X[:, 5] = lo() if drone else hi()  # wifi similarity
        X[:, 6] = lo() if drone else hi()  # bt similarity
        X[:, 7] = lo() if drone else rng.uniform(0.0, 0.8, n)  # fixed
        X[:, 8] = rng.uniform(0.5, 2.5, n)  # observations_log
        X[:, 9] = rng.uniform(4.0, 7.5, n)  # span_log
        return X

    X = np.vstack([block(n_per_class, True), block(n_per_class, False)])
    y = np.concatenate(
        [
            np.full(n_per_class, LABEL_DRONE, dtype=np.int64),
            np.full(n_per_class, LABEL_NOT_DRONE, dtype=np.int64),
        ]
    )
    sites = np.asarray(
        ["synth-site-a", "synth-site-b"] * (len(y) // 2), dtype=object
    ).astype(str)
    meta = DatasetMeta(
        source_db=SYNTHETIC_MARKER,
        site=SYNTHETIC_MARKER,
        n_total=len(y),
        n_drone=n_per_class,
        n_not_drone=n_per_class,
        n_unlabeled=0,
    )
    return X, y, sites, meta
