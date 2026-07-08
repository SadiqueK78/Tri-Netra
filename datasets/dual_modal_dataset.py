#!/usr/bin/env python3
"""
TriNetra-AMRF — Dual-Modality PyTorch Dataset
=============================================

Purpose (Phase 2.5):
    The core data class. Loads paired visible (RGB) + thermal (single-channel)
    images together with their COCO annotations, applies a synchronized
    transform, and returns model-ready tensors.

Per ``__getitem__`` it returns ``(visible, thermal, targets)`` where:

    visible : FloatTensor [3, H, W]  — normalized RGB
    thermal : FloatTensor [1, H, W]  — normalized single-channel thermal
    targets : dict with
        boxes    : FloatTensor [N, 4]  — COCO [x, y, w, h] in the *resized* frame
        labels   : LongTensor  [N]     — unified category IDs
        image_id : LongTensor  []      — scalar COCO image id
        orig_size: LongTensor  [2]     — (height, width) before transform

Pairing convention: visible and thermal share an identical filename, so the
same ``file_name`` from the COCO JSON is looked up in both directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DualModalDataset(Dataset):
    """Paired visible + thermal detection dataset backed by a COCO JSON."""

    def __init__(
        self,
        visible_dir: str,
        thermal_dir: str,
        annotations_json: str,
        transform: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
        require_pairs: bool = True,
    ):
        """
        Args:
            visible_dir:      Directory of RGB images for this split.
            thermal_dir:      Directory of thermal images for this split.
            annotations_json: COCO JSON for this split (from split_dataset.py).
            transform:        A ``DualModalTransform`` (or compatible callable):
                              (vis_np, thr_np, boxes, labels) ->
                              (vis_t, thr_t, boxes_t, labels_t).
                              If None, images are returned as raw CHW float
                              tensors in [0, 1] with no resizing.
            preprocess:       Optional Phase 3 ``PreprocessPipeline`` (or
                              compatible callable) run on the uint8 pair BEFORE
                              ``transform``: (vis_u8, thr_u8) ->
                              (vis_u8, thr_u8, extras). Its ``extras['trc']``
                              (and optional ``trc_map``) are attached to the
                              target dict for Phase 4 fusion.
            require_pairs:    If True, drop any image whose visible or thermal
                              file is missing (with a warning) instead of
                              erroring at read time.
        """
        # pycocotools is the canonical COCO reader; import lazily so the module
        # imports even in environments where it's absent.
        from pycocotools.coco import COCO

        self.visible_dir = Path(visible_dir)
        self.thermal_dir = Path(thermal_dir)
        self.transform = transform
        self.preprocess = preprocess

        self.coco = COCO(str(annotations_json))
        self.cat_ids = sorted(self.coco.getCatIds())

        # Build the working list of image ids, optionally filtering to those
        # whose files actually exist on disk in BOTH modalities.
        image_ids = sorted(self.coco.getImgIds())
        self.image_ids = []
        n_missing = 0
        for img_id in image_ids:
            info = self.coco.loadImgs(img_id)[0]
            fname = info["file_name"]
            if require_pairs:
                if not (self.visible_dir / fname).exists() or not (self.thermal_dir / fname).exists():
                    n_missing += 1
                    continue
            self.image_ids.append(img_id)

        if n_missing:
            print(f"  [WARN] DualModalDataset: skipped {n_missing} image(s) "
                  f"missing a visible/thermal file under {self.visible_dir.parent}.")
        if not self.image_ids:
            raise RuntimeError(
                f"No usable image pairs found.\n"
                f"  visible_dir = {self.visible_dir}\n"
                f"  thermal_dir = {self.thermal_dir}\n"
                f"  annotations = {annotations_json}"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    # ── loading helpers ──────────────────────────────────────────────────────
    def _load_visible(self, fname: str) -> np.ndarray:
        """Load an RGB image as HxWx3 uint8."""
        with Image.open(self.visible_dir / fname) as im:
            return np.array(im.convert("RGB"))

    def _load_thermal(self, fname: str) -> np.ndarray:
        """Load a thermal image as HxWx1 uint8 (LLVIP stores it as 3-ch JPG)."""
        with Image.open(self.thermal_dir / fname) as im:
            arr = np.array(im.convert("L"))  # collapse to single channel
        return arr[:, :, None]

    def _load_targets(self, img_id: int):
        """Return (boxes_xywh, labels) lists for one image from COCO anns."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([float(x), float(y), float(w), float(h)])
            labels.append(int(a["category_id"]))
        return boxes, labels

    # ── main accessor ────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        fname = info["file_name"]

        visible = self._load_visible(fname)   # HxWx3
        thermal = self._load_thermal(fname)   # HxWx1
        orig_h, orig_w = visible.shape[:2]
        boxes, labels = self._load_targets(img_id)

        # ── Phase 3 preprocessing (align → denoise → normalize → TRC) ────────
        # Runs on the uint8 pair BEFORE augmentation; produces the reliability
        # score attached to targets below.
        extras: Dict[str, object] = {}
        if self.preprocess is not None:
            visible, thermal, extras = self.preprocess(visible, thermal)

        if self.transform is not None:
            vis_t, thr_t, boxes_t, labels_t = self.transform(visible, thermal, boxes, labels)
        else:
            # No transform: raw CHW tensors in [0, 1]; boxes untouched.
            vis_t = torch.from_numpy(visible).permute(2, 0, 1).float() / 255.0
            thr_t = torch.from_numpy(thermal).permute(2, 0, 1).float() / 255.0
            boxes_t = (torch.as_tensor(boxes, dtype=torch.float32)
                       if boxes else torch.zeros((0, 4), dtype=torch.float32))
            labels_t = (torch.as_tensor(labels, dtype=torch.int64)
                        if labels else torch.zeros((0,), dtype=torch.int64))

        targets: Dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(img_id, dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }

        # Phase 3 → Phase 4 hand-off: expose the Thermal Reliability Confidence
        # (scalar, and optional per-region map) on the target dict. Defaults to
        # a fully-trusted 1.0 when no preprocessing pipeline is attached.
        trc_val = float(extras.get("trc", 1.0)) if extras else 1.0
        targets["trc"] = torch.tensor(trc_val, dtype=torch.float32)
        # NOTE: trc_map (when spatial_map is enabled) is computed at the
        # ORIGINAL resolution, before the augmentation transform's crop/flip/
        # resize — it does not yet track those spatial ops. spatial_map is off
        # by default and the per-region map is a deferred Phase 4 iteration
        # (see PHASE3_IMPLEMENTATION.md Open Question #2). The scalar `trc`, a
        # frame-level trust score, is augmentation-invariant and is the primary
        # Phase 4 hand-off.
        trc_map = extras.get("trc_map") if extras else None
        if trc_map is not None:
            targets["trc_map"] = torch.as_tensor(np.asarray(trc_map), dtype=torch.float32)

        return vis_t, thr_t, targets

    # ── convenience ──────────────────────────────────────────────────────────
    @property
    def num_classes(self) -> int:
        return len(self.cat_ids)

    def category_names(self) -> Dict[int, str]:
        return {c["id"]: c["name"] for c in self.coco.loadCats(self.cat_ids)}
