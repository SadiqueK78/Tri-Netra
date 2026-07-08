"""
TriNetra-AMRF — Preprocessing Package (Phase 3 — Layer 2)
=========================================================

Preprocessing & Reliability pipeline for dual-modality (RGB + Thermal) imagery.
Sits between the Phase 2 DataLoader and the Phase 4 fusion module.

Chain (uint8 arrays, before augmentation):
    1. alignment.py — cross-modal spatial registration (pass-through for LLVIP)
    2. denoise.py   — modality-aware noise reduction
    3. normalize.py — intensity/contrast normalization (CLAHE / clip / z-score)
    4. trc.py       — Thermal Reliability Confidence (novel; feeds Phase 4)
    5. pipeline.py  — PreprocessPipeline wiring 1-4 into the DataLoader

Typical usage:
    from preprocessing import PreprocessPipeline
    pre = PreprocessPipeline(cfg)
    vis_u8, thr_u8, extras = pre(visible_u8, thermal_u8)
    trc = extras["trc"]   # in [0, 1]
"""

from .alignment import (
    align_thermal_to_visible,
    estimate_homography,
    load_homography,
)
from .denoise import denoise
from .normalize import normalize
from .trc import compute_trc
from .pipeline import PreprocessPipeline, build_preprocess_pipeline

__all__ = [
    "align_thermal_to_visible",
    "estimate_homography",
    "load_homography",
    "denoise",
    "normalize",
    "compute_trc",
    "PreprocessPipeline",
    "build_preprocess_pipeline",
]
