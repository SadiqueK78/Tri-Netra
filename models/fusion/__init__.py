"""
TriNetra-AMRF — Cross-Modal Fusion Models (Phase 4 — Layer 3)
=============================================================

Adaptive fusion modules combining visible and thermal feature maps:
  - ThermalStem     — 1-channel input stem (mean-of-RGB warm start)
  - DualBackbone    — parallel visible/thermal YOLOv8 backbones (P3/P4/P5 taps)
  - TRCGatedFusion  — TRC-gated fusion block (gated | concatenation | attention)
  - FusionNeck      — three fusion blocks across scales
  - freeze helpers  — staged transfer-learning schedule (freeze/unfreeze)
"""

from .backbone import DualBackbone, load_yolo_model
from .freeze import (
    apply_stage,
    describe_freeze_state,
    freeze_modules,
    param_counts,
    refreeze_bn,
    unfreeze_modules,
)
from .fusion_block import TRCGatedFusion
from .fusion_neck import FusionNeck
from .thermal_stem import ThermalStem

__all__ = [
    "DualBackbone",
    "load_yolo_model",
    "ThermalStem",
    "TRCGatedFusion",
    "FusionNeck",
    "freeze_modules",
    "unfreeze_modules",
    "apply_stage",
    "refreeze_bn",
    "param_counts",
    "describe_freeze_state",
]
