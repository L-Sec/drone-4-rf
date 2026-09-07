"""Detection layer: each detector maps a PSD frame to Detection objects."""

from drone4rf.detectors.base import Detection
from drone4rf.detectors.cfar import OSCFARDetector
from drone4rf.detectors.energy import EnergyDetector

__all__ = ["Detection", "EnergyDetector", "OSCFARDetector"]
