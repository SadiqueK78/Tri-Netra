"""
TriNetra-AMRF — Intensity & Contrast Normalization
====================================================

Purpose (Phase 3 — Preprocessing & Reliability):
    Harmonizes the very different dynamic ranges of visible (0-255 RGB) and
    thermal imagery so both modalities present consistent, well-conditioned
    inputs to the downstream network.

Pipeline Position:
    Layer 2 (Preprocessing & Reliability) → after denoising, before augmentation.

Supported Methods:
    - clahe          : Contrast-Limited Adaptive Histogram Equalization
                       (visible: applied on the L channel to lift shadow detail;
                        thermal: applied directly).
    - minmax         : rescale to full 0-255 range.
    - zscore         : per-image standardization (returned as uint8 for the
                       downstream uint8 → Albumentations.Normalize path).
    - percentile_clip: clip to [p_lo, p_hi] percentiles then rescale — the
                       thermal default, suppressing hot/cold outliers.

Ordering note (vs. Phase 2):
    Phase 2's ``Albumentations.Normalize`` (ImageNet mean/std) is the *final*
    tensor-normalization for the network. This module is a *pre-conditioning*
    pass on the **uint8 image** that happens **before** augmentation. To keep
    that contract, every method here returns a **uint8** array.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ── channel helpers ──────────────────────────────────────────────────────────
def _has_trailing_channel(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 1


def _restore_shape(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    if _has_trailing_channel(like) and out.ndim == 2:
        return out[:, :, None]
    return out


# ── methods ──────────────────────────────────────────────────────────────────
def _clahe_gray(gray: np.ndarray, clip_limit: float, tile: Tuple[int, int]
                ) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile))
    return clahe.apply(gray)


def _clahe(image: np.ndarray, clip_limit: float, tile: Tuple[int, int]
           ) -> np.ndarray:
    """CLAHE — on the L channel for RGB, directly for single-channel."""
    if image.ndim == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = _clahe_gray(lab[:, :, 0], clip_limit, tile)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    work = image[:, :, 0] if _has_trailing_channel(image) else image
    out = _clahe_gray(work, clip_limit, tile)
    return _restore_shape(out, image)


def _minmax(image: np.ndarray) -> np.ndarray:
    out = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def _zscore(image: np.ndarray) -> np.ndarray:
    """Per-image z-score, remapped to uint8 (mean→128, ±2.5σ→[0,255])."""
    arr = image.astype(np.float32)
    mean, std = float(arr.mean()), float(arr.std())
    if std < 1e-6:
        return image.astype(np.uint8)
    z = (arr - mean) / std
    out = np.clip(z * (255.0 / 5.0) + 128.0, 0, 255)  # ±2.5σ spans full range
    return out.astype(np.uint8)


def _percentile_clip(image: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    """Clip to [p_lo, p_hi] percentiles, then rescale to 0-255."""
    arr = image.astype(np.float32)
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi - lo < 1e-6:
        return image.astype(np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
    return out.astype(np.uint8)


# ── public API ───────────────────────────────────────────────────────────────
def normalize(image: np.ndarray, modality: str, cfg: dict) -> np.ndarray:
    """Intensity-normalize a uint8 image, returning uint8.

    Method selection:
        * ``thermal`` always gets a percentile clip first (outlier suppression)
          when ``thermal_percentile_clip`` is configured, regardless of the
          global method — hot pixels otherwise dominate any rescale.
        * The configured ``method`` (default ``clahe``) is then applied.

    Args:
        image:    HxW / HxWx1 / HxWx3 uint8 array.
        modality: "visible" or "thermal".
        cfg:      Full config; reads ``cfg['preprocessing']['normalize']``.

    Returns:
        Normalized uint8 array, same shape as the input.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV (cv2) is required for normalization.")

    ncfg = cfg.get("preprocessing", {}).get("normalize", {})
    method = str(ncfg.get("method", "clahe")).lower()

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Thermal outlier suppression (pre-step, before the main method).
    if modality == "thermal":
        clip = ncfg.get("thermal_percentile_clip")
        if clip:
            p_lo, p_hi = float(clip[0]), float(clip[1])
            image = _percentile_clip(image, p_lo, p_hi)
            # If the configured method IS percentile_clip, we're done.
            if method == "percentile_clip":
                return image

    if method == "clahe":
        clip_limit = float(ncfg.get("clip_limit", 2.0))
        tile = tuple(ncfg.get("tile_grid_size", (8, 8)))
        return _clahe(image, clip_limit=clip_limit, tile=tile)
    if method == "minmax":
        return _minmax(image)
    if method == "zscore":
        return _zscore(image)
    if method == "percentile_clip":
        clip = ncfg.get("percentile_clip", ncfg.get("thermal_percentile_clip", [1, 99]))
        return _percentile_clip(image, float(clip[0]), float(clip[1]))

    return image  # unknown method → no-op
