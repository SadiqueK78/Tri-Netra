"""
TriNetra-AMRF — Noise Reduction Pipeline
==========================================

Purpose (Phase 3 — Preprocessing & Reliability):
    Applies modality-aware denoising to both visible and thermal streams.
    Thermal imagery is particularly susceptible to fixed-pattern noise (FPN),
    non-uniformity noise (NUC residuals), and temporal noise — this module
    addresses those artifacts while preserving edge detail critical for
    downstream detection.

Pipeline Position:
    Layer 2 (Preprocessing & Reliability) → after alignment, before normalization

Supported Methods:
    - bilateral : edge-preserving spatial smoothing (default; good for both)
    - nlm       : non-local means (visible low-light grain)
    - gaussian  : lightweight, fast blur
    - guided    : visible-guided thermal denoise (edge-aware, cross-modal)

Design note:
    Keep denoising **mild** — over-smoothing erases the small-pedestrian edges
    LLVIP depends on. Strengths come from ``cfg['preprocessing']['denoise']``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ── channel helpers ──────────────────────────────────────────────────────────
def _has_trailing_channel(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 1


def _restore_shape(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Give ``out`` the same (H,W,1) trailing singleton as ``like`` if needed."""
    if _has_trailing_channel(like) and out.ndim == 2:
        return out[:, :, None]
    return out


# ── individual filters ───────────────────────────────────────────────────────
def _bilateral(image: np.ndarray, d: int, sigma_color: float, sigma_space: float
               ) -> np.ndarray:
    work = image[:, :, 0] if _has_trailing_channel(image) else image
    out = cv2.bilateralFilter(work, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return _restore_shape(out, image)


def _gaussian(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1  # kernel size must be odd
    out = cv2.GaussianBlur(image, (k, k), sigmaX=sigma)
    return _restore_shape(out, image)


def _nlm(image: np.ndarray, strength: float) -> np.ndarray:
    """Non-local means. Colored variant for 3-ch, grayscale for 1-ch."""
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=strength, hColor=strength,
            templateWindowSize=7, searchWindowSize=21,
        )
    work = image[:, :, 0] if _has_trailing_channel(image) else image
    out = cv2.fastNlMeansDenoising(
        work, None, h=strength, templateWindowSize=7, searchWindowSize=21,
    )
    return _restore_shape(out, image)


def _guided(image: np.ndarray, guide: Optional[np.ndarray], radius: int, eps: float
            ) -> np.ndarray:
    """Edge-aware guided filter; ``guide`` (visible) steers thermal smoothing.

    Falls back to bilateral when no guide is supplied or the ximgproc contrib
    module is unavailable.
    """
    if guide is None:
        return _bilateral(image, d=5, sigma_color=50.0, sigma_space=50.0)

    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None:  # contrib not installed → graceful fallback
        return _bilateral(image, d=5, sigma_color=50.0, sigma_space=50.0)

    src = image[:, :, 0] if _has_trailing_channel(image) else image
    g = guide
    if g.ndim == 3 and g.shape[2] == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    elif _has_trailing_channel(g):
        g = g[:, :, 0]
    # Guided filter wants matching spatial dims.
    if g.shape[:2] != src.shape[:2]:
        g = cv2.resize(g, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)

    out = ximgproc.guidedFilter(guide=g, src=src, radius=radius, eps=eps)
    return _restore_shape(out, image)


# ── public API ───────────────────────────────────────────────────────────────
def denoise(
    image: np.ndarray,
    modality: str,
    cfg: dict,
    guide: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Denoise a uint8 image, dispatching on the configured method.

    Args:
        image:    HxW, HxWx1, or HxWx3 uint8 array.
        modality: "visible" or "thermal" (selects the ``apply_to`` gate; the
                  method itself is shared unless the config differs).
        cfg:      Full config dict; reads ``cfg['preprocessing']['denoise']``.
        guide:    Optional guide image (visible) for the "guided" method.

    Returns:
        Denoised uint8 array with the same shape as ``image``. If the modality
        is not in ``apply_to``, the input is returned unchanged.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV (cv2) is required for denoising.")

    dcfg = cfg.get("preprocessing", {}).get("denoise", {})
    apply_to = dcfg.get("apply_to", ["visible", "thermal"])
    if modality not in apply_to:
        return image

    method = str(dcfg.get("method", "bilateral")).lower()
    strength = float(dcfg.get("strength", 10))

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if method == "bilateral":
        d = int(dcfg.get("bilateral_d", 5))
        return _bilateral(image, d=d, sigma_color=strength, sigma_space=strength)
    if method == "gaussian":
        ksize = int(dcfg.get("gaussian_ksize", 5))
        sigma = float(dcfg.get("gaussian_sigma", strength / 10.0))
        return _gaussian(image, ksize=ksize, sigma=sigma)
    if method == "nlm":
        return _nlm(image, strength=strength)
    if method == "guided":
        radius = int(dcfg.get("guided_radius", 4))
        eps = float(dcfg.get("guided_eps", 1e-2 * (255 ** 2)))
        return _guided(image, guide=guide, radius=radius, eps=eps)

    # Unknown method → no-op rather than raising in the data pipeline.
    return image
