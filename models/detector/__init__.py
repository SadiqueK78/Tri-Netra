"""
TriNetra-AMRF — Object Detection Models (Layer 4)
==================================================

Multi-scale object detection architectures for dual-modality surveillance:
  - FusionDetector — assembled dual-modal detector (Phase 4): TRC-gated
    mid-fusion feeding a pretrained YOLOv8 neck + head.

Phase 5 adds the training loop, loss wiring, and mAP evaluation.
"""

from .fusion_detector import FusionDetector

__all__ = ["FusionDetector"]
