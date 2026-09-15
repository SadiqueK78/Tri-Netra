#!/usr/bin/env python3
"""
TriNetra-AMRF — DeepSORT Tracking Wrapper (Phase 6.1)
======================================================

Purpose (Phase 6 — Layer 5: Tracking & Behaviour Analysis):
    Wrap ``deep_sort_realtime``'s ``DeepSort`` so it consumes exactly the
    per-frame detection format ``training.evaluate.decode_predictions``
    already produces — ``[n, 6]`` tensors of ``(x1, y1, x2, y2, conf, cls)``
    in resized-frame pixel coordinates, NMS'd, one tensor per image — and
    turns a *sequence* of those into persistent track identities.

    This module is detector-agnostic: it never touches ``FusionDetector``
    directly, only its decoded output, so it works the same whether the
    detections came from the fusion model, a baseline, or ground truth (useful
    for testing the tracker in isolation, since Phase 5 has not produced a
    trained checkpoint yet).

Config (``tracking`` block in ``configs/default.yaml``):
    algorithm:     only "deepsort" is implemented so far.
    max_age:       frames a track survives with no matching detection.
    min_hits:      consecutive detections required before a track is
                   "confirmed" (DeepSort's ``n_init``).
    iou_threshold: gating distance for motion-only association
                   (DeepSort's ``max_iou_distance``).
    embedder:      appearance embedder for re-identification ("mobilenet",
                   the deep_sort_realtime default) or ``null``/``none`` to
                   disable appearance matching and fall back to motion+IoU
                   only (faster, no image crops needed — useful when no real
                   frame is available, e.g. testing against synthetic boxes).

Usage:
    tracker = FusionTracker(cfg)
    for visible, thermal, targets in loader:            # temporal sequence
        decoded = model(visible, thermal, trc)[0]
        dets = decode_predictions(decoded, cfg, keep_classes)[0]  # one image
        tracks = tracker.update(dets, frame=frame_bgr_uint8)      # frame optional
        # tracks: List[{"track_id", "bbox_xyxy", "cls", "conf"}]

Self-test (no dataset, no detector needed):
    python -m utils.check_tracker
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


class FusionTracker:
    """Persistent multi-object tracker over a sequence of per-frame detections."""

    def __init__(self, cfg: dict):
        tcfg = cfg.get("tracking", {})
        algorithm = str(tcfg.get("algorithm", "deepsort")).lower()
        if algorithm != "deepsort":
            raise NotImplementedError(
                f"tracking.algorithm={algorithm!r} is not implemented yet "
                "(only 'deepsort' — bytetrack is planned, see models/tracker/__init__.py)."
            )

        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._cfg = dict(tcfg)
        embedder = tcfg.get("embedder", "mobilenet")
        embedder = None if str(embedder).lower() in ("none", "null", "") else embedder
        self._no_embedder = embedder is None

        self.tracker = DeepSort(
            max_age=int(tcfg.get("max_age", 30)),
            n_init=int(tcfg.get("min_hits", 3)),
            max_iou_distance=float(tcfg.get("iou_threshold", 0.3)),
            embedder=embedder,
            embedder_gpu=torch.cuda.is_available(),
            bgr=True,
        )

    def reset(self) -> None:
        """Start a fresh sequence (e.g. a new camera/dataset scene) — DeepSort
        keeps no public reset, so rebuild it from the same config instead."""
        self.__init__({"tracking": self._cfg})

    def update(
        self,
        detections: torch.Tensor,
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Advance the tracker by one frame.

        Args:
            detections: ``[n, 6]`` tensor ``(x1, y1, x2, y2, conf, cls)`` for
                        this single frame — e.g. one element of
                        ``training.evaluate.decode_predictions``'s return list.
                        An empty/``None`` tensor is a valid "nothing detected
                        this frame" input.
            frame:      ``HxWx3`` uint8 BGR image, required only when the
                        configured embedder is not disabled (appearance
                        re-identification crops patches from it).

        Returns:
            One dict per *confirmed* track: ``track_id`` (str), ``bbox_xyxy``
            (list[float]), ``cls`` (int or None), ``conf`` (float or None).
        """
        raw = []
        if detections is not None and len(detections):
            for x1, y1, x2, y2, conf, cls in detections.detach().cpu().tolist():
                raw.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                           float(conf), int(cls)))

        # With appearance matching disabled, deep_sort_realtime still demands
        # *something* per detection (real embedder or a frame to crop from) —
        # hand it identical neutral embeddings so association falls back to
        # pure motion/IoU, matching the "no frame needed" promise of
        # embedder=null. A zero vector looks neutral but isn't: the library
        # L2-normalizes embeddings for cosine distance, and 0/||0|| is NaN,
        # which silently corrupts the assignment cost matrix. A non-zero
        # constant vector normalizes to a fixed direction instead, so every
        # pair is tied at cosine distance 0 (uninformative, as intended)
        # without ever producing NaN.
        embeds = [[1.0]] * len(raw) if (self._no_embedder and frame is None) else None
        tracks = self.tracker.update_tracks(raw, embeds=embeds, frame=frame)

        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, top, r, b = t.to_ltrb()
            out.append({
                "track_id": t.track_id,
                "bbox_xyxy": [float(l), float(top), float(r), float(b)],
                "cls": int(t.det_class) if t.det_class is not None else None,
                "conf": float(t.det_conf) if t.det_conf is not None else None,
            })
        return out
