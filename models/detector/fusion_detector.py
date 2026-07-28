#!/usr/bin/env python3
"""
TriNetra-AMRF — Assembled Dual-Modal Fusion Detector (Phase 4.5)
================================================================

Purpose (Phase 4 — Layer 3 → Layer 4 hand-off):
    End-to-end dual-modal detector:

        visible, thermal, trc
          -> DualBackbone            (vis P3/P4/P5, thr P3/P4/P5)
          -> FusionNeck (× TRC gate) -> fused P3/P4/P5
          -> YOLO neck  (layers 10–21) [pretrained]
          -> Detect head (layer 22)    [pretrained]
          -> detections

    The pretrained neck + head are the *original* ultralytics modules, run via
    the same from-index (`layer.f`) wiring as ``DetectionModel.forward`` — but
    with the backbone tap slots (4/6/9) seeded with the **fused** features.

    Consumes the Phase 2/3 dataloader batch:
        visible [B,3,H,W] · thermal [B,1,H,W] · targets[i]['trc'] (scalar).

Usage:
    cfg = yaml.safe_load(open('configs/default.yaml'))
    model = FusionDetector(cfg)
    model.freeze_pretrained()                # Stage 1
    preds = model(visible, thermal, trc)     # trc: [B] tensor (or None)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..fusion.backbone import BACKBONE_END, TAP_LAYERS, DualBackbone, load_yolo_model
from ..fusion.freeze import (
    apply_stage,
    describe_freeze_state,
    freeze_modules,
    param_counts,
    refreeze_bn,
    unfreeze_modules,
)
from ..fusion.fusion_neck import FusionNeck


class FusionDetector(nn.Module):
    """End-to-end dual-modal detector (TRC-gated mid-fusion + pretrained YOLO)."""

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Full parsed config (reads the ``fusion`` block; the base model
                 comes from ``fusion.backbone.base_model``).
        """
        super().__init__()
        self.cfg = cfg
        fcfg = cfg.get("fusion", {})
        weights = fcfg.get("backbone", {}).get("base_model", "weights/yolov8m.pt")

        det_model = load_yolo_model(weights)  # ultralytics DetectionModel
        full = det_model.model                # 23 modules (0–9 bb, 10–21 neck, 22 head)

        # Layer 3 — dual backbone + TRC-gated fusion at P3/P4/P5.
        self.backbone = DualBackbone(det_model, fcfg)
        self.fusion_neck = FusionNeck(self.backbone.channels, fcfg)

        # Pretrained neck (10–21) + Detect head (22), kept as original modules
        # so their `f`/`i` from-index attributes drive the forward wiring.
        self.neck = nn.ModuleList(full[i] for i in range(BACKBONE_END, len(full) - 1))
        self.head = full[-1]

        # Ultralytics metadata (names, strides) useful for Phase 5 / inference.
        self.names = getattr(det_model, "names", None)
        self.stride = getattr(det_model, "stride", None)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        visible: torch.Tensor,
        thermal: torch.Tensor,
        trc: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            visible: [B, 3, H, W] normalized RGB.
            thermal: [B, 1, H, W] normalized thermal.
            trc:     [B] Thermal Reliability Confidence per image (or a
                     [B,1,H,W] spatial map). ``None`` → gate bypassed (trc≡1).

        Returns:
            The Detect head output — training mode: list of 3 per-scale
            tensors; eval mode: (decoded preds, per-scale tensors).
        """
        vis_feats, thr_feats = self.backbone(visible, thermal)
        fused = self.fusion_neck(vis_feats, thr_feats, trc)

        # Seed the saved-outputs table with fused features standing in for the
        # original backbone taps (layers 4/6/9), then run neck + head exactly
        # as DetectionModel.forward does (from-index wiring).
        y = {tap: f for tap, f in zip(TAP_LAYERS, fused)}
        x = fused[-1]  # layer-9 (SPPF/P5) output feeds layer 10
        for layer in list(self.neck) + [self.head]:
            f = layer.f
            if isinstance(f, int):
                x = x if f == -1 else y[f]
            else:
                x = [x if j == -1 else y[j] for j in f]
            x = layer(x)
            y[layer.i] = x
        return x

    @staticmethod
    def trc_from_targets(targets: List[dict], device=None) -> torch.Tensor:
        """Stack the per-sample ``targets[i]['trc']`` scalars into a [B] tensor.

        Prefers the spatial ``trc_map`` when every sample carries one
        (``preprocessing.trc.spatial_map: true``) — returned as [B,1,H,W].
        """
        if targets and all("trc_map" in t for t in targets):
            maps = torch.stack([torch.as_tensor(t["trc_map"], dtype=torch.float32)
                                for t in targets])
            if maps.dim() == 3:
                maps = maps.unsqueeze(1)
            return maps.to(device) if device is not None else maps
        trc = torch.stack([torch.as_tensor(t.get("trc", 1.0), dtype=torch.float32)
                           for t in targets])
        return trc.to(device) if device is not None else trc

    # ── staged transfer-learning schedule (Phase 4.7) ────────────────────────
    def freeze_pretrained(self) -> None:
        """Stage 1: freeze visible backbone + neck + head (COCO weights);
        the thermal stream and fusion blocks remain trainable."""
        apply_stage(self, self.cfg, stage=1)

    def unfreeze(self, names=("neck", "head")) -> None:
        """Stage 2: unfreeze the given modules (default: neck + head)."""
        unfreeze_modules(self, names)

    def train(self, mode: bool = True):
        """Standard ``train()`` but keeps frozen-module BatchNorms in eval."""
        super().train(mode)
        if mode:
            refreeze_bn(self)
        return self

    # ── introspection helpers ────────────────────────────────────────────────
    def param_summary(self) -> str:
        trainable, frozen = param_counts(self)
        total = trainable + frozen
        lines = [f"FusionDetector parameters: {total:,} total — "
                 f"{trainable:,} trainable ({100 * trainable / total:.1f}%), "
                 f"{frozen:,} frozen"]
        lines += ["  " + l for l in describe_freeze_state(self)]
        return "\n".join(lines)


__all__ = ["FusionDetector", "freeze_modules", "unfreeze_modules"]
