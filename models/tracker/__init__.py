"""
TriNetra-AMRF — Object Tracking Models
=========================================

Multi-object tracking algorithms for persistent identity association.

  - DeepSORT (deep appearance + Kalman filter)  -> deepsort_tracker.FusionTracker  [done, Phase 6.1]
  - ByteTrack (byte-level association for low-confidence detections)  [planned]
  - Custom re-identification (ReID) feature extractor  [planned]

Self-test: python -m utils.check_tracker
"""

from .deepsort_tracker import FusionTracker  # noqa: F401
