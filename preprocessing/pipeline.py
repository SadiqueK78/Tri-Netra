"""
TriNetra-AMRF — Unified Preprocess Transform  [NEW]
===================================================

Purpose (Phase 3 — Preprocessing & Reliability):
    A single callable that runs the Layer-2 chain

        3.1 align → 3.2 denoise → 3.3 normalize → 3.4 TRC

    on **uint8 arrays, BEFORE augmentation**, and plugs into the Phase 2
    ``DualModalDataset`` (which already accepts a preprocessing hook).

Contract:
    __call__(visible_u8, thermal_u8) -> (visible_u8, thermal_u8, extras)
        visible_u8 : HxWx3 uint8  (pre-conditioned RGB)
        thermal_u8 : HxWx1 uint8  (aligned / denoised / normalized thermal)
        extras     : {"trc": float,
                      "trc_map": np.ndarray[H,W]|None,
                      "trc_components": {..},
                      "align_error": float|None}

    Keeping the output uint8 preserves the ordering contract with Phase 2:
    Albumentations' spatial + ImageNet ``Normalize`` still runs *after* this,
    unchanged.

Integration (see PHASE3_IMPLEMENTATION.md, Open Question #1):
    (a) Online  — call inside ``DualModalDataset.__getitem__`` before aug
                  (the default; alignment is a no-op for LLVIP, filters cheap).
    (b) Offline — a batch cache writer (``run_preprocess.py``) could persist
                  processed images + ``trc.json`` once; not needed yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .alignment import align_thermal_to_visible, load_homography
from .denoise import denoise
from .normalize import normalize
from .trc import compute_trc


class PreprocessPipeline:
    """align → denoise → normalize → TRC, on uint8 arrays, before augmentation."""

    def __init__(self, cfg: dict, homography: Optional[np.ndarray] = None):
        """
        Args:
            cfg:        Full config dict (reads the ``preprocessing`` block).
            homography: Optional 3x3 thermal→visible matrix. If ``None``, it is
                        loaded from ``datasets.calibration_dir/homography.npy``
                        when present; a missing/identity matrix means the
                        alignment step is a pass-through (LLVIP).
        """
        self.cfg = cfg
        pcfg = cfg.get("preprocessing", {})
        self.denoise_enabled = "denoise" in pcfg
        self.normalize_enabled = "normalize" in pcfg
        tcfg = pcfg.get("trc", {})
        self.trc_enabled = bool(tcfg.get("enabled", True))
        self.trc_spatial = bool(tcfg.get("spatial_map", False))

        # Resolve a homography once (cheap; identity for LLVIP).
        if homography is None:
            calib_dir = cfg.get("datasets", {}).get("calibration_dir")
            if calib_dir is not None:
                try:
                    homography = load_homography(calib_dir)
                except Exception:
                    homography = None
        self.homography = homography

    # ── the chain ────────────────────────────────────────────────────────────
    def __call__(
        self,
        visible: np.ndarray,
        thermal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Run the full Layer-2 chain on one uint8 pair.

        Args:
            visible: HxWx3 uint8 RGB.
            thermal: HxWx1 (or HxW) uint8 thermal.

        Returns:
            (visible_u8, thermal_u8, extras) — see module docstring.
        """
        if thermal.ndim == 2:
            thermal = thermal[:, :, None]

        # 3.1 — spatial alignment (pass-through for identity / LLVIP).
        thermal, align_error = align_thermal_to_visible(
            visible, thermal, homography=self.homography
        )

        # 3.2 — modality-aware denoising.
        if self.denoise_enabled:
            visible = denoise(visible, "visible", self.cfg)
            thermal = denoise(thermal, "thermal", self.cfg, guide=visible)

        # 3.3 — intensity normalization (pre-conditioning, stays uint8).
        if self.normalize_enabled:
            visible = normalize(visible, "visible", self.cfg)
            thermal = normalize(thermal, "thermal", self.cfg)

        # Keep thermal as HxWx1 uint8 for the downstream transform.
        if thermal.ndim == 2:
            thermal = thermal[:, :, None]
        if thermal.dtype != np.uint8:
            thermal = np.clip(thermal, 0, 255).astype(np.uint8)
        if visible.dtype != np.uint8:
            visible = np.clip(visible, 0, 255).astype(np.uint8)

        # 3.4 — Thermal Reliability Confidence (computed on the conditioned
        # thermal so the score reflects what fusion will actually see).
        extras: Dict[str, object] = {
            "trc": 1.0,
            "trc_map": None,
            "trc_components": {},
            "align_error": align_error,
        }
        if self.trc_enabled:
            trc_out = compute_trc(
                thermal, visible=visible, align_error=align_error, cfg=self.cfg
            )
            extras["trc"] = float(trc_out["trc"])
            extras["trc_map"] = trc_out["trc_map"]
            extras["trc_components"] = trc_out["components"]

        return visible, thermal, extras


def build_preprocess_pipeline(cfg: dict) -> Optional[PreprocessPipeline]:
    """Factory: return a pipeline if any preprocessing stage is configured/enabled.

    Returns ``None`` when the ``preprocessing`` block is absent, so callers can
    treat "no preprocessing" as a clean opt-out.
    """
    pcfg = cfg.get("preprocessing")
    if not pcfg:
        return None
    return PreprocessPipeline(cfg)
