"""
TriNetra-AMRF — Cross-Modal Spatial Alignment
==============================================

Purpose (Phase 3 — Preprocessing & Reliability):
    Spatially aligns thermal imagery to the visible (RGB) reference frame
    using calibration data. Supports homography (and, via a supplied 2x3
    matrix, affine/rigid) transformation models. Ensures pixel-level
    correspondence between the two modalities before fusion.

Pipeline Position:
    Layer 2 (Preprocessing & Reliability) → feeds into Layer 3 (Fusion)

Expected Inputs:
    - Visible (RGB) image: np.ndarray (H, W, 3)
    - Thermal (IR) image:  np.ndarray (H, W) or (H, W, 1/3)
    - Homography:          np.ndarray (3x3) — from calibration or estimated

Expected Outputs:
    - Aligned thermal image registered to the visible frame
    - Alignment quality metric (reprojection error / inlier ratio)

Note on LLVIP:
    LLVIP frames are already pixel-registered; Phase 2 wrote an identity
    ``homography.npy``. For an identity (or ``None``) homography this module is
    a **pass-through** — it returns the thermal image unchanged with a zero
    reprojection error. The real warping path exists so un-calibrated sensor
    pairs (FLIR ADAS, custom rigs) work later without re-plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:  # cv2 is an env dependency; import lazily-tolerant so the module loads.
    import cv2
except Exception:  # pragma: no cover - environment without OpenCV
    cv2 = None


# ── helpers ──────────────────────────────────────────────────────────────────
def _is_identity(H: Optional[np.ndarray], tol: float = 1e-6) -> bool:
    """True if H is None or numerically the 3x3 identity."""
    if H is None:
        return True
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        return False
    return bool(np.allclose(H, np.eye(3), atol=tol))


def load_homography(calibration_dir: str | Path, name: str = "homography.npy"
                    ) -> Optional[np.ndarray]:
    """Load a saved 3x3 homography, or return ``None`` if absent.

    ``None`` is treated downstream as "no warp needed" (identity).
    """
    path = Path(calibration_dir) / name
    if not path.is_file():
        return None
    H = np.load(str(path))
    return np.asarray(H, dtype=np.float64)


# ── main API ─────────────────────────────────────────────────────────────────
def align_thermal_to_visible(
    visible: np.ndarray,
    thermal: np.ndarray,
    homography: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Warp ``thermal`` into the ``visible`` reference frame.

    Args:
        visible:    HxWx3 (or HxW) reference image; only its H, W are used.
        thermal:    HxW / HxWx1 / HxWx3 image to warp.
        homography: 3x3 matrix mapping thermal→visible coordinates. If ``None``
                    or the identity, the thermal image is returned unchanged
                    (LLVIP pass-through).

    Returns:
        (aligned_thermal, reproj_error):
            aligned_thermal — same channel layout as the input ``thermal``.
            reproj_error    — ``0.0`` for the identity pass-through, ``None``
                              when it cannot be computed, else a float carried
                              from ``estimate_homography`` if available.
    """
    # Identity / no calibration → pass-through (the LLVIP case).
    if _is_identity(homography):
        return thermal, 0.0

    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV (cv2) is required for non-identity alignment.")

    H_img, W_img = visible.shape[:2]
    H = np.asarray(homography, dtype=np.float64)

    aligned = cv2.warpPerspective(
        thermal, H, (W_img, H_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # warpPerspective drops a trailing singleton channel; restore it so callers
    # that expect HxWx1 thermal keep a stable shape.
    if thermal.ndim == 3 and aligned.ndim == 2:
        aligned = aligned[:, :, None]
    return aligned, None


def estimate_homography(
    visible: np.ndarray,
    thermal: np.ndarray,
    method: str = "orb",
    max_features: int = 2000,
    ransac_thresh: float = 5.0,
) -> Tuple[Optional[np.ndarray], float]:
    """Feature-based homography estimation for un-calibrated sensor pairs.

    Matches keypoints between the (grayscale) visible and thermal images and
    fits a homography with RANSAC. Cross-modal matching is inherently hard —
    this is a best-effort utility for rigs without factory calibration, not the
    LLVIP path.

    Args:
        visible:      HxWx3 or HxW reference image.
        thermal:      HxW / HxWx1 / HxWx3 image to register.
        method:       "orb" (default, license-free) or "sift".
        max_features: Feature budget for the detector.
        ransac_thresh: RANSAC reprojection threshold in pixels.

    Returns:
        (H, inlier_ratio): 3x3 homography (thermal→visible) and the fraction of
        matches RANSAC kept as inliers. ``(None, 0.0)`` if estimation fails.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV (cv2) is required for homography estimation.")

    vis_g = _to_gray_u8(visible)
    thr_g = _to_gray_u8(thermal)

    if method.lower() == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(thr_g, None)  # src = thermal
    kp2, des2 = detector.detectAndCompute(vis_g, None)  # dst = visible
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0.0

    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 4:
        return None, 0.0

    matches = sorted(matches, key=lambda m: m.distance)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None, 0.0

    inlier_ratio = float(mask.sum()) / float(len(mask)) if len(mask) else 0.0
    return H.astype(np.float64), inlier_ratio


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Coerce an image to single-channel uint8 for feature detection."""
    arr = image
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return arr
