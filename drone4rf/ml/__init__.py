"""Optional machine-learning subsystem (Stage 6).

Strict separation of concerns per the design: signal DETECTION and
FEATURE EXTRACTION live in the deterministic pipeline; this package only
CLASSIFIES fixed feature vectors and CALIBRATES its confidence. The
subsystem is disabled until the operator trains a model on their own
labeled observations; synthetic data exists strictly for pipeline
testing and is marked as such in model provenance.

scikit-learn is imported lazily so the core scanner has no hard
dependency on it; inference is CPU-only.
"""
