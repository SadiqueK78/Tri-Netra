#!/usr/bin/env python3
"""
TriNetra-AMRF — Dual-Modality Augmentations
===========================================

Purpose (Phase 2.6):
    Provide synchronized augmentations for paired visible (RGB) + thermal (1-ch)
    imagery using Albumentations >= 2.0.

The central constraint: any *spatial* transform (crop, flip, rotate, resize)
MUST be applied identically to both modalities and to the bounding boxes, or
the pixel-level correspondence between RGB and thermal breaks. *Photometric*
transforms (color jitter, sensor noise) are modality-specific and applied
independently.

Design — three composed stages inside ``DualModalTransform``:

    Stage A  (shared spatial)   : one A.Compose with bbox_params and
                                  additional_targets={'thermal': 'image'} so the
                                  SAME geometric transform hits visible, thermal,
                                  and boxes together.
    Stage B1 (visible pixels)   : ColorJitter / gamma → Normalize(ImageNet) → tensor
    Stage B2 (thermal pixels)   : GaussNoise / brightness → Normalize(0.5) → tensor

Both val and train share Stage B (deterministic); only Stage A differs.

The transform is callable:

    vis_t, thr_t, boxes, labels = transform(visible_np, thermal_np, boxes, labels)

where inputs are HxWxC uint8 numpy arrays and ``boxes`` are COCO [x, y, w, h].
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class DualModalTransform:
    """Callable that applies synchronized spatial + per-modality pixel transforms."""

    def __init__(self, spatial: A.Compose, visible_pixel: A.Compose, thermal_pixel: A.Compose):
        self.spatial = spatial
        self.visible_pixel = visible_pixel
        self.thermal_pixel = thermal_pixel

    def __call__(
        self,
        visible: np.ndarray,
        thermal: np.ndarray,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visible: HxWx3 uint8 RGB array.
            thermal: HxWx1 uint8 array.
            boxes:   list of COCO [x, y, w, h] boxes (absolute pixels).
            labels:  list of integer category IDs, parallel to boxes.

        Returns:
            (visible_tensor[3,H,W], thermal_tensor[1,H,W],
             boxes_tensor[N,4] COCO xywh, labels_tensor[N]) — all torch tensors.
        """
        boxes = [list(map(float, b)) for b in boxes]
        labels = list(labels)

        # ── Stage A: shared geometry (boxes + both modalities together) ──────
        spatial_out = self.spatial(
            image=visible,
            thermal=thermal,
            bboxes=boxes,
            labels=labels,
        )
        vis = spatial_out["image"]
        thr = spatial_out["thermal"]
        out_boxes = spatial_out["bboxes"]
        out_labels = spatial_out["labels"]

        # ── Stage B: independent photometric + normalize + to-tensor ─────────
        vis_t = self.visible_pixel(image=vis)["image"]
        thr_t = self.thermal_pixel(image=thr)["image"]

        # Boxes may be empty after an aggressive crop — keep shapes well-formed.
        if len(out_boxes) > 0:
            boxes_t = torch.as_tensor(np.asarray(out_boxes, dtype=np.float32))
            labels_t = torch.as_tensor(np.asarray(out_labels, dtype=np.int64))
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        return vis_t, thr_t, boxes_t, labels_t


# ── Builders ─────────────────────────────────────────────────────────────────
def _bbox_params() -> A.BboxParams:
    """COCO bbox params shared by train/val spatial stages.

    ``clip=True`` keeps boxes inside the image after geometric transforms;
    ``min_visibility`` drops boxes that are almost entirely cropped away.
    """
    return A.BboxParams(
        format="coco",
        label_fields=["labels"],
        min_area=1.0,
        min_visibility=0.1,
        clip=True,
    )


def _visible_pixel_stage(cfg: dict, train: bool) -> A.Compose:
    norm = cfg["datasets"]["normalize"]
    steps: List[A.BasicTransform] = []
    if train and cfg["datasets"]["augmentation"].get("color_jitter", True):
        steps.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5))
        steps.append(A.RandomGamma(gamma_limit=(80, 120), p=0.3))
    steps.append(A.Normalize(mean=tuple(norm["visible_mean"]), std=tuple(norm["visible_std"]),
                             max_pixel_value=255.0))
    steps.append(ToTensorV2())
    return A.Compose(steps)


def _thermal_pixel_stage(cfg: dict, train: bool) -> A.Compose:
    norm = cfg["datasets"]["normalize"]
    aug = cfg["datasets"]["augmentation"]
    steps: List[A.BasicTransform] = []
    if train:
        if aug.get("thermal_gauss_noise", True):
            # std_range is a fraction of the 0..1 range in Albumentations 2.x.
            steps.append(A.GaussNoise(std_range=(0.02, 0.08), p=0.3))
        steps.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3))
    steps.append(A.Normalize(mean=tuple(norm["thermal_mean"]), std=tuple(norm["thermal_std"]),
                             max_pixel_value=255.0))
    steps.append(ToTensorV2())
    return A.Compose(steps)


def get_train_transforms(cfg: dict) -> DualModalTransform:
    """Build the training transform (random spatial + per-modality photometric)."""
    ds = cfg["datasets"]
    w, h = ds["image_size"]  # config stores [width, height]
    aug = ds["augmentation"]

    spatial_steps: List[A.BasicTransform] = []
    if aug.get("random_resized_crop", True):
        scale = tuple(aug.get("crop_scale", (0.6, 1.0)))
        spatial_steps.append(
            A.RandomResizedCrop(size=(h, w), scale=scale, ratio=(0.75, 1.3333), p=1.0)
        )
    else:
        spatial_steps.append(A.Resize(height=h, width=w, p=1.0))
    spatial_steps.append(A.HorizontalFlip(p=aug.get("horizontal_flip_prob", 0.5)))
    rot = aug.get("rotation_limit", 0)
    if rot:
        # Rotate visible & thermal together; border reflect avoids black wedges.
        spatial_steps.append(A.Affine(rotate=(-rot, rot), fit_output=False, p=0.3))

    spatial = A.Compose(
        spatial_steps,
        bbox_params=_bbox_params(),
        additional_targets={"thermal": "image"},
    )
    return DualModalTransform(
        spatial=spatial,
        visible_pixel=_visible_pixel_stage(cfg, train=True),
        thermal_pixel=_thermal_pixel_stage(cfg, train=True),
    )


def get_val_transforms(cfg: dict) -> DualModalTransform:
    """Build the deterministic val/test transform (resize + normalize only)."""
    ds = cfg["datasets"]
    w, h = ds["image_size"]
    spatial = A.Compose(
        [A.Resize(height=h, width=w, p=1.0)],
        bbox_params=_bbox_params(),
        additional_targets={"thermal": "image"},
    )
    return DualModalTransform(
        spatial=spatial,
        visible_pixel=_visible_pixel_stage(cfg, train=False),
        thermal_pixel=_thermal_pixel_stage(cfg, train=False),
    )


def build_transforms(cfg: dict, split: str) -> DualModalTransform:
    """Return the appropriate transform for a split name."""
    return get_train_transforms(cfg) if split == "train" else get_val_transforms(cfg)
