"""
TriNetra-AMRF — Thermal Reliability Confidence (TRC)  [NEW — novel]
==================================================================

Purpose (Phase 3 — Preprocessing & Reliability):
    The centerpiece of Layer 2. Produces a scalar (and optional spatial map)
    ``trc ∈ [0, 1]`` estimating how much the *thermal* modality should be
    trusted for a given frame. This is the signal that drives **adaptive**
    fusion in Phase 4: when thermal is degraded (sensor saturation, thermal
    crossover at dawn/dusk, blur/defocus, misalignment), ``trc`` drops and the
    fusion module down-weights the thermal feature branch.

Design:
    * **Unsupervised & fast** — no training, no labels; runs inside the data
      pipeline. Each cue is a cheap, deterministic image statistic.
    * Every component is normalized to ``[0, 1]`` where **1 = reliable**.
    * Final ``trc`` is a config-weighted mean of the enabled components.
    * An optional ``trc_map`` (per-region grid, upsampled to HxW) enables
      spatially-adaptive fusion in a later Phase 4 iteration.

Components (all higher = more reliable):
    contrast    : normalized histogram entropy — flat/saturated thermal → low.
    gradient    : Laplacian-variance sharpness — blur/defocus → low.
    saturation  : 1 − fraction of clipped (0/255) pixels — sensor clipping → low.
    mutual_info : RGB↔IR mutual information — thermal-crossover risk → low.
    align_error : maps reprojection error → reliability — misalignment → low.

Hand-off:
    The DataLoader target dict gains a ``trc`` field (and optionally
    ``trc_map``); Phase 4 fusion multiplies the thermal branch by (a broadcast
    of) ``trc``. See the architecture note in PHASE3_IMPLEMENTATION.md.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# Default component weights (used when cfg is absent). Mirrors default.yaml.
_DEFAULT_WEIGHTS = {
    "contrast": 0.2,
    "gradient": 0.2,
    "saturation": 0.2,
    "mutual_info": 0.2,
    "align_error": 0.2,
}

# Sharpness normalization constant: lap_var / (lap_var + K) → [0,1].
# Calibrated to LLVIP thermal, whose Laplacian variance is naturally low
# (median ≈ 33 — IR is smooth/low-frequency). K = 15 maps a median-sharpness
# clean frame to ≈ 0.7 while a blurred frame (variance → single digits) still
# collapses toward 0. Raising K would wrongly penalize valid, smooth thermal.
_GRADIENT_K = 15.0
# Reprojection-error scale (pixels): reliability = exp(-err / SCALE).
_ALIGN_ERR_SCALE = 5.0


# ── small utilities ──────────────────────────────────────────────────────────
def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Coerce any image to single-channel uint8."""
    arr = image
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if cv2 is not None \
                else arr.mean(axis=2)
        else:
            arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        rng = a.max() - a.min()
        arr = (((a - a.min()) / rng) * 255.0).astype(np.uint8) if rng > 1e-6 \
            else np.zeros_like(a, dtype=np.uint8)
    return arr


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ── individual reliability cues (each → [0,1], 1 = reliable) ──────────────────
def contrast_score(gray: np.ndarray) -> float:
    """Normalized Shannon entropy of the intensity histogram.

    A flat (dead) or saturated thermal frame concentrates its histogram into a
    few bins → low entropy → low reliability. Full-range detail → entropy ≈ 8
    bits → score ≈ 1.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))       # 0 .. 8 bits
    return _clip01(entropy / 8.0)


def gradient_score(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness, saturating to [0,1].

    Blur / defocus collapses high-frequency energy → low Laplacian variance →
    low score. Uses ``v / (v + K)`` so the score is bounded and monotonic.
    """
    if cv2 is not None:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
    else:  # pragma: no cover
        lap = np.gradient(np.gradient(gray.astype(np.float64), axis=0), axis=0)
    v = float(lap.var())
    return _clip01(v / (v + _GRADIENT_K))


def saturation_score(gray: np.ndarray, low: int = 0, high: int = 255,
                     tol: int = 2) -> float:
    """1 − fraction of pixels clipped at the sensor floor/ceiling.

    Large regions pinned at 0 or 255 indicate sensor saturation / dead area →
    low reliability.
    """
    g = gray
    clipped = (g <= low + tol) | (g >= high - tol)
    ratio = float(clipped.mean())
    return _clip01(1.0 - ratio)


def mutual_info_score(thermal_gray: np.ndarray, visible_gray: np.ndarray,
                      bins: int = 32) -> float:
    """Normalized mutual information between thermal and visible intensities.

    Thermal crossover (dawn/dusk, when scene and body temperatures equalize)
    destroys the structural correspondence between RGB and IR → low MI → low
    reliability. Normalized by min(H_t, H_v) to land in ``[0, 1]``.
    """
    if thermal_gray.shape != visible_gray.shape:
        if cv2 is not None:
            visible_gray = cv2.resize(visible_gray,
                                      (thermal_gray.shape[1], thermal_gray.shape[0]),
                                      interpolation=cv2.INTER_AREA)
        else:  # pragma: no cover
            return 0.5

    hist_2d, _, _ = np.histogram2d(
        thermal_gray.ravel(), visible_gray.ravel(),
        bins=bins, range=[[0, 256], [0, 256]],
    )
    pxy = hist_2d / hist_2d.sum() if hist_2d.sum() > 0 else hist_2d
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    nz = pxy > 0
    px_i = px[:, None]
    py_j = py[None, :]
    denom = px_i * py_j
    mi = np.sum(pxy[nz] * np.log2(pxy[nz] / denom[nz]))

    def _entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    h_t, h_v = _entropy(px), _entropy(py)
    norm = min(h_t, h_v)
    if norm < 1e-6:
        return 0.0
    return _clip01(mi / norm)


def align_error_score(align_error: Optional[float]) -> float:
    """Map a reprojection error (pixels) to reliability via ``exp(-err/scale)``.

    ``None`` (unknown / not estimated) is treated as neutral-good (1.0), since
    the LLVIP pass-through reports 0.0 and pre-calibrated rigs have no error to
    penalize.
    """
    if align_error is None:
        return 1.0
    return _clip01(float(np.exp(-abs(align_error) / _ALIGN_ERR_SCALE)))


# ── spatial map ──────────────────────────────────────────────────────────────
def _compute_trc_map(thermal_gray: np.ndarray, visible_gray: Optional[np.ndarray],
                     weights: Dict[str, float], grid: int = 8) -> np.ndarray:
    """Per-region TRC on a ``grid × grid`` tiling, upsampled to HxW (float32).

    Only the local cues (contrast, gradient, saturation, mutual_info) vary
    spatially; the frame-level align_error is folded in as a constant.
    """
    H, W = thermal_gray.shape[:2]
    gh, gw = max(1, H // grid), max(1, W // grid)
    cells = np.zeros((grid, grid), dtype=np.float32)

    w_con = weights.get("contrast", 0.0)
    w_grad = weights.get("gradient", 0.0)
    w_sat = weights.get("saturation", 0.0)
    w_mi = weights.get("mutual_info", 0.0)
    local_wsum = w_con + w_grad + w_sat + w_mi

    for r in range(grid):
        for c in range(grid):
            y0, x0 = r * gh, c * gw
            y1 = H if r == grid - 1 else (r + 1) * gh
            x1 = W if c == grid - 1 else (c + 1) * gw
            t_patch = thermal_gray[y0:y1, x0:x1]
            acc = (w_con * contrast_score(t_patch)
                   + w_grad * gradient_score(t_patch)
                   + w_sat * saturation_score(t_patch))
            if w_mi > 0 and visible_gray is not None:
                v_patch = visible_gray[y0:y1, x0:x1]
                acc += w_mi * mutual_info_score(t_patch, v_patch)
            cells[r, c] = acc / local_wsum if local_wsum > 0 else 0.0

    if cv2 is not None:
        return cv2.resize(cells, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.kron(cells, np.ones((gh, gw), dtype=np.float32))[:H, :W]


# ── public API ───────────────────────────────────────────────────────────────
def compute_trc(
    thermal: np.ndarray,
    visible: Optional[np.ndarray] = None,
    align_error: Optional[float] = None,
    cfg: Optional[dict] = None,
) -> Dict[str, object]:
    """Compute the Thermal Reliability Confidence for one frame.

    Args:
        thermal:     HxW / HxWx1 / HxWx3 thermal image (uint8 or float).
        visible:     Optional HxWx3 / HxW visible image (enables the mutual_info
                     cue; if absent that component is dropped and weights
                     renormalize over the remaining cues).
        align_error: Optional reprojection error in pixels (from alignment 3.1).
        cfg:         Full config; reads ``cfg['preprocessing']['trc']`` for
                     component weights and ``spatial_map``.

    Returns:
        dict with:
            "trc"        : float in [0, 1] — the frame-level reliability.
            "trc_map"    : np.ndarray[H, W] float32 in [0,1], or None.
            "components" : {name: score} for every enabled, computable cue.
    """
    tcfg = (cfg or {}).get("preprocessing", {}).get("trc", {}) if cfg else {}
    weights = dict(_DEFAULT_WEIGHTS)
    weights.update(tcfg.get("components", {}) or {})
    want_map = bool(tcfg.get("spatial_map", False))

    thr_gray = _to_gray_u8(thermal)
    vis_gray = _to_gray_u8(visible) if visible is not None else None

    # Compute each enabled cue; skip mutual_info when visible is unavailable.
    components: Dict[str, float] = {}
    if weights.get("contrast", 0) > 0:
        components["contrast"] = contrast_score(thr_gray)
    if weights.get("gradient", 0) > 0:
        components["gradient"] = gradient_score(thr_gray)
    if weights.get("saturation", 0) > 0:
        components["saturation"] = saturation_score(thr_gray)
    if weights.get("mutual_info", 0) > 0 and vis_gray is not None:
        components["mutual_info"] = mutual_info_score(thr_gray, vis_gray)
    if weights.get("align_error", 0) > 0:
        components["align_error"] = align_error_score(align_error)

    # Weighted mean over the components we actually computed (renormalized).
    wsum = sum(weights[k] for k in components)
    if wsum > 0:
        trc = sum(weights[k] * v for k, v in components.items()) / wsum
    else:
        trc = 1.0  # no cues enabled → default to fully-trusted
    trc = _clip01(trc)

    trc_map = None
    if want_map:
        grid = int(tcfg.get("grid", 8))
        trc_map = _compute_trc_map(thr_gray, vis_gray, weights, grid=grid)
        # Fold the frame-level align_error into the local map multiplicatively-ish
        # via a weighted blend so a misaligned frame drags the whole map down.
        if "align_error" in components and weights.get("align_error", 0) > 0:
            ae = components["align_error"]
            w_ae = weights["align_error"]
            local_w = wsum - w_ae
            if local_w > 0:
                trc_map = (local_w * trc_map + w_ae * ae) / wsum
        trc_map = np.clip(trc_map, 0.0, 1.0).astype(np.float32)

    return {"trc": trc, "trc_map": trc_map, "components": components}
