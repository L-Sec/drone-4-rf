"""Stage 4 drone-oriented analytics: behavioral feature analyzers and the
explainable confidence fusion engine.

Nothing in this package identifies a drone with certainty - analyzers
produce scores in [0, 1] with explanations, and the fusion engine's
vocabulary is capped at 'high_confidence_drone_activity'. 'Confirmed'
does not exist here by design.
"""

from drone4rf.analytics.engine import AnalyticsEngine
from drone4rf.analytics.fusion import DroneAssessment

__all__ = ["AnalyticsEngine", "DroneAssessment"]
