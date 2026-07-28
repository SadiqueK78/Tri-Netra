#!/usr/bin/env python3
"""
TriNetra-AMRF — Thermal Input Stem (Phase 4.2)
==============================================

Purpose (Phase 4 — Layer 3: Adaptive Cross-Modal Fusion):
    YOLOv8's pretrained stem (layer 0) is a ``Conv`` expecting 3-channel RGB.
    The thermal branch feeds 1-channel imagery, so its stem must be replaced.

    Two init modes (config: ``fusion.backbone.thermal_stem_init``):
      * ``mean3ch`` (recommended) — average the pretrained 3-ch kernels over the
        channel dim ``[C,3,k,k] -> [C,1,k,k]``. Low-level edge/blob filters
        transfer across modalities, so this is a sensible warm start.
      * ``random`` — fresh Kaiming-init Conv (train fully from scratch).

    The BatchNorm + SiLU that follow the conv are copied from the pretrained
    stem in both modes (BN running stats will adapt during Stage-1 training).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ThermalStem(nn.Module):
    """1-channel replacement for YOLOv8's 3-channel ``Conv`` stem.

    Mirrors the ultralytics ``Conv`` structure (conv → bn → act) so it drops
    into the thermal backbone at layer 0 unchanged.
    """

    def __init__(self, pretrained_stem: nn.Module, init: str = "mean3ch"):
        """
        Args:
            pretrained_stem: The ultralytics ``Conv`` module at layer 0 of a
                             loaded YOLOv8 (attributes ``.conv``/``.bn``/``.act``).
            init:            ``mean3ch`` | ``random`` (see module docstring).
        """
        super().__init__()
        src: nn.Conv2d = pretrained_stem.conv
        if src.in_channels != 3:
            raise ValueError(f"Expected a 3-ch pretrained stem, got {src.in_channels}-ch.")

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=src.out_channels,
            kernel_size=src.kernel_size,
            stride=src.stride,
            padding=src.padding,
            dilation=src.dilation,
            bias=src.bias is not None,
        )
        if init == "mean3ch":
            with torch.no_grad():
                # [C,3,k,k] -> [C,1,k,k]: average the RGB kernels.
                self.conv.weight.copy_(src.weight.mean(dim=1, keepdim=True))
                if src.bias is not None:
                    self.conv.bias.copy_(src.bias)
        elif init != "random":
            raise ValueError(f"Unknown thermal_stem_init: {init!r} (mean3ch|random)")

        # Reuse the pretrained BN + activation (copied, so the visible stem
        # is untouched when the thermal branch trains).
        self.bn = copy.deepcopy(pretrained_stem.bn)
        self.act = copy.deepcopy(pretrained_stem.act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] -> [B,C,H/2,W/2] (same downsampling as the RGB stem)."""
        return self.act(self.bn(self.conv(x)))
