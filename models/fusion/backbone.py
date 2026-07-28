#!/usr/bin/env python3
"""
TriNetra-AMRF — Dual YOLOv8 Backbone (Phase 4.1)
================================================

Purpose (Phase 4 — Layer 3: Adaptive Cross-Modal Fusion):
    Split a COCO-pretrained YOLOv8 into **two parallel backbone streams**:

      * visible stream — layers 0–9 with the pretrained weights (frozen by
        default in Stage 1);
      * thermal stream — same architecture, layer-0 stem swapped for a
        1-channel ``ThermalStem``; remaining layers optionally warm-started
        from the visible weights (config: ``fusion.backbone.thermal_warm_start``).

    Each stream exposes taps at the three backbone output scales used by the
    YOLO neck (verified on ultralytics 8.4.89 / yolov8m, 23 modules):

        P3 = layer 4 (stride 8) · P4 = layer 6 (stride 16) · P5 = layer 9/SPPF (stride 32)

Public API:
    load_yolo_model(weights) -> ultralytics DetectionModel
    DualBackbone(det_model, cfg) — forward(vis, thr) -> ((v3,v4,v5), (t3,t4,t5))
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .thermal_stem import ThermalStem

# Backbone slice and P-scale tap indices for the YOLOv8 architecture family.
BACKBONE_END = 10          # layers 0–9 are the backbone (10–21 neck, 22 head)
TAP_LAYERS = (4, 6, 9)     # P3 / P4 / P5(SPPF) outputs consumed by the neck

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yolo_model(weights: str):
    """Load a pretrained ultralytics YOLO and return its ``DetectionModel``.

    Accepts a path relative to the project root (e.g. ``weights/yolov8m.pt``).
    """
    from ultralytics import YOLO

    p = Path(weights)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return YOLO(str(p)).model  # DetectionModel (nn.Module, .model = layer list)


class _BackboneStream(nn.Module):
    """One YOLOv8 backbone (layers 0–9) returning the P3/P4/P5 tap outputs."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        taps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in TAP_LAYERS:
                taps.append(x)
        return tuple(taps)  # (P3, P4, P5)


class DualBackbone(nn.Module):
    """Two YOLOv8 backbones (visible + thermal) exposing P3/P4/P5 taps.

    - visible stream: initialized from pretrained COCO weights (frozen by default)
    - thermal stream: same architecture; stem swapped for 1-channel input,
      remaining layers optionally warm-started from the visible weights.
    Returns ``((v3, v4, v5), (t3, t4, t5))``.
    """

    def __init__(self, det_model: nn.Module, fusion_cfg: dict):
        """
        Args:
            det_model:  A loaded ultralytics ``DetectionModel`` (its layers are
                        *shared*, not copied, for the visible stream so the
                        pretrained neck/head in ``FusionDetector`` stay in sync).
            fusion_cfg: The ``fusion`` config block (``backbone.*`` keys).
        """
        super().__init__()
        bcfg = fusion_cfg.get("backbone", {})
        warm_start = bool(bcfg.get("thermal_warm_start", True))
        stem_init = str(bcfg.get("thermal_stem_init", "mean3ch"))

        full = det_model.model  # nn.Sequential of 23 modules
        vis_layers = nn.ModuleList(full[i] for i in range(BACKBONE_END))

        # Thermal stream: deep-copy carries the visible weights (warm start).
        # If warm_start is off, re-randomize everything except the stem slot.
        thr_layers = nn.ModuleList(copy.deepcopy(vis_layers))
        if not warm_start:
            for m in thr_layers.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    m.reset_parameters()
        # Swap layer 0 for the 1-channel stem (mean-init reads the *pretrained*
        # stem, so pass the visible layer 0 regardless of warm_start).
        thr_layers[0] = ThermalStem(vis_layers[0], init=stem_init)

        self.visible = _BackboneStream(vis_layers)
        self.thermal = _BackboneStream(thr_layers)

        # Resolve per-scale channel counts from the model (width-multiple
        # aware — do NOT hard-code; e.g. yolov8m gives (192, 384, 576)).
        self.channels = self._probe_channels()

    @torch.no_grad()
    def _probe_channels(self) -> Tuple[int, ...]:
        """Dummy forward to read the P3/P4/P5 channel counts at build time."""
        was_training = self.visible.training
        self.visible.eval()
        dev = next(self.visible.parameters()).device
        feats = self.visible(torch.zeros(1, 3, 64, 64, device=dev))
        if was_training:
            self.visible.train()
        return tuple(f.shape[1] for f in feats)

    def forward(
        self, visible: torch.Tensor, thermal: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Args:
            visible: [B, 3, H, W] normalized RGB.
            thermal: [B, 1, H, W] normalized thermal.

        Returns:
            ((v3, v4, v5), (t3, t4, t5)) feature tuples at strides 8/16/32.
        """
        return self.visible(visible), self.thermal(thermal)
