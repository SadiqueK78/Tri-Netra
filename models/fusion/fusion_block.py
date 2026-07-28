#!/usr/bin/env python3
"""
TriNetra-AMRF — TRC-Gated Fusion Block (Phase 4.3 — the novelty)
================================================================

Purpose (Phase 4 — Layer 3: Adaptive Cross-Modal Fusion):
    Combine one scale's visible + thermal feature maps, with the Phase-3
    **Thermal Reliability Confidence (TRC)** score gating the thermal branch:

        fused = Fuse( P_vis , gate(trc) · P_thr )

    High trc → full thermal contribution; low trc → thermal is suppressed
    toward zero and the fused feature falls back to visible. The gate accepts
    a per-frame scalar ``[B]`` (Phase 3 default) or a spatial map ``[B,1,h,w]``
    (region-adaptive; auto-resized to each scale).

Strategies (config: ``fusion.strategy``):
    * ``gated``         — TRC gate + 3×3 conv on the thermal branch, added
                          residually to the visible features (default).
    * ``concatenation`` — concat([P_vis, trc·P_thr]) → 1×1 conv back to the
                          visible channel count.
    * ``attention``     — cross-attention (query = vis, key/value = thr),
                          trc-scaled, added residually.

Design invariants (so the FROZEN pretrained neck keeps working):
    * Output channels == visible input channels at every scale.
    * Residual safety (``fused = P_vis + block(trc·P_thr)``): an untrained
      block degrades gracefully to (near) visible-only features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _broadcast_trc(trc: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Normalize a TRC tensor to something broadcastable over ``like`` [B,C,h,w].

    Accepts [B] / [B,1] / [B,1,1,1] scalars or a [B,1,H,W] spatial map (the map
    is bilinearly resized to the feature scale).
    """
    if trc.dim() == 1:                       # [B]
        return trc.view(-1, 1, 1, 1)
    if trc.dim() == 2:                       # [B,1]
        return trc.view(-1, 1, 1, 1)
    if trc.dim() == 3:                       # [B,H,W] spatial map, no channel dim
        trc = trc.unsqueeze(1)
    if trc.dim() == 4:                       # [B,1,H,W]
        if trc.shape[-2:] != like.shape[-2:]:
            trc = F.interpolate(trc, size=like.shape[-2:], mode="bilinear",
                                align_corners=False)
        return trc
    raise ValueError(f"Unsupported trc shape: {tuple(trc.shape)}")


class TRCGatedFusion(nn.Module):
    """Adaptive fusion of one scale's visible & thermal feature maps.

    fused = Fuse( P_vis , gate(trc) * P_thr )
    """

    def __init__(
        self,
        channels: int,
        strategy: str = "gated",
        trc_enabled: bool = True,
        learnable_alpha: bool = True,
        residual_visible: bool = True,
        attn_heads: int = 4,
    ):
        """
        Args:
            channels:         Visible P-scale channel count (= output channels;
                              the frozen neck expects exactly this).
            strategy:         'gated' | 'concatenation' | 'attention'.
            trc_enabled:      If False the gate is bypassed (trc ≡ 1) — the
                              "fusion without TRC" ablation baseline.
            learnable_alpha:  Wrap the gate in a learnable scalar α (init 1.0):
                              gate = α·trc — the network learns how strongly
                              to obey TRC.
            residual_visible: fused = P_vis + block(trc·P_thr) (safe default).
                              If False, the block output replaces P_vis.
            attn_heads:       Head count for the 'attention' strategy.
        """
        super().__init__()
        if strategy not in ("gated", "concatenation", "attention"):
            raise ValueError(f"Unknown fusion strategy: {strategy!r}")
        self.channels = channels
        self.strategy = strategy
        self.trc_enabled = trc_enabled
        self.residual_visible = residual_visible

        # gate = α · trc  (α frozen at 1.0 when not learnable)
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=learnable_alpha)

        if strategy == "gated":
            # Project the gated thermal features. The final conv starts at a
            # small magnitude so fused ≈ P_vis at step 0 (near-identity, safe
            # for the frozen neck) while the TRC gate is still observably wired
            # (trc=1 vs trc=0 must change the output — see check_fusion.py).
            self.proj = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
            )
            nn.init.normal_(self.proj[-1].weight, std=1e-2)
            nn.init.zeros_(self.proj[-1].bias)
        elif strategy == "concatenation":
            self.proj = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:  # attention
            self.norm_q = nn.GroupNorm(1, channels)
            self.norm_kv = nn.GroupNorm(1, channels)
            self.attn = nn.MultiheadAttention(channels, attn_heads, batch_first=True)
            self.proj = nn.Conv2d(channels, channels, 1)
            nn.init.normal_(self.proj.weight, std=1e-2)
            nn.init.zeros_(self.proj.bias)

    # ── gate ─────────────────────────────────────────────────────────────────
    def _gate(self, p_thr: torch.Tensor, trc) -> torch.Tensor:
        """Multiply the thermal features by α·trc (broadcast or spatial)."""
        if not self.trc_enabled or trc is None:
            return p_thr * self.alpha
        if not torch.is_tensor(trc):
            trc = torch.tensor(float(trc), device=p_thr.device)
        if trc.dim() == 0:
            trc = trc.expand(p_thr.shape[0])
        trc = _broadcast_trc(trc.to(p_thr.dtype).to(p_thr.device), p_thr)
        return p_thr * (self.alpha * trc)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, p_vis: torch.Tensor, p_thr: torch.Tensor, trc) -> torch.Tensor:
        """
        Args:
            p_vis: [B, C, h, w] visible features at this scale.
            p_thr: [B, C, h, w] thermal features at this scale.
            trc:   [B] scalar per image, [B,1,1,1], or [B,1,H,W] spatial map
                   (also accepts a python float / 0-dim tensor). None → no gate.

        Returns:
            fused [B, C, h, w] — same shape as ``p_vis``.
        """
        t = self._gate(p_thr, trc)

        if self.strategy == "gated":
            out = self.proj(t)
        elif self.strategy == "concatenation":
            out = self.proj(torch.cat([p_vis, t], dim=1))
        else:  # attention: query = visible, key/value = gated thermal
            b, c, h, w = p_vis.shape
            q = self.norm_q(p_vis).flatten(2).transpose(1, 2)   # [B, hw, C]
            kv = self.norm_kv(t).flatten(2).transpose(1, 2)     # [B, hw, C]
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
            out = self.proj(attn_out.transpose(1, 2).reshape(b, c, h, w))

        # Residual safety: for 'gated'/'attention' the projection is zero-init,
        # so fused == P_vis exactly at step 0. 'concatenation' already consumes
        # P_vis inside the concat; its residual keeps the visible identity path.
        if self.residual_visible:
            return p_vis + out
        return out
