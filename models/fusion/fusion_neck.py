#!/usr/bin/env python3
"""
TriNetra-AMRF — Multi-Scale Fusion Wiring (Phase 4.4)
=====================================================

Purpose (Phase 4 — Layer 3: Adaptive Cross-Modal Fusion):
    Hold one ``TRCGatedFusion`` block per backbone scale (P3/P4/P5, strides
    8/16/32) and produce the fused feature tuple the pretrained YOLO neck
    expects. Per-scale channel counts are resolved from the loaded model at
    build time (width-multiple aware), never hard-coded.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .fusion_block import TRCGatedFusion


class FusionNeck(nn.Module):
    """Three TRC-gated fusion blocks producing the (P3, P4, P5) tuple."""

    def __init__(self, channels: Sequence[int], fusion_cfg: dict):
        """
        Args:
            channels:   Per-scale channel counts, e.g. ``(192, 384, 576)`` for
                        yolov8m — from ``DualBackbone.channels``.
            fusion_cfg: The ``fusion`` config block (``strategy``, ``trc_gate.*``).
        """
        super().__init__()
        gcfg = fusion_cfg.get("trc_gate", {})
        kwargs = dict(
            strategy=str(fusion_cfg.get("strategy", "gated")),
            trc_enabled=bool(gcfg.get("enabled", True)),
            learnable_alpha=bool(gcfg.get("learnable_alpha", True)),
            residual_visible=bool(gcfg.get("residual_visible", True)),
        )
        self.fuse3 = TRCGatedFusion(channels[0], **kwargs)
        self.fuse4 = TRCGatedFusion(channels[1], **kwargs)
        self.fuse5 = TRCGatedFusion(channels[2], **kwargs)

    def forward(
        self,
        vis_feats: Tuple[torch.Tensor, ...],
        thr_feats: Tuple[torch.Tensor, ...],
        trc,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            vis_feats: (v3, v4, v5) from the visible backbone.
            thr_feats: (t3, t4, t5) from the thermal backbone.
            trc:       [B] scalars (or spatial map) — shared across scales.

        Returns:
            (f3, f4, f5) fused features, channel-identical to ``vis_feats``.
        """
        f3 = self.fuse3(vis_feats[0], thr_feats[0], trc)
        f4 = self.fuse4(vis_feats[1], thr_feats[1], trc)
        f5 = self.fuse5(vis_feats[2], thr_feats[2], trc)
        return f3, f4, f5
