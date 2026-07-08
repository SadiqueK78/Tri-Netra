"""
TriNetra-AMRF — Datasets Package
================================

Phase 2 data pipeline for paired visible (RGB) + thermal detection data.

Modules:
    download            — verify raw datasets & write calibration artifacts
    convert_annotations — source annotations (VOC/KAIST/…) → unified COCO JSON
    split_dataset       — train/val/test splitting into the canonical layout
    dual_modal_dataset  — DualModalDataset (paired loading + COCO targets)
    augmentations       — DualModalTransform (synchronized spatial + pixel aug)
    dataloader          — create_dataloaders() factory + collate

Typical usage:
    import yaml
    from datasets import create_dataloaders
    cfg = yaml.safe_load(open("configs/default.yaml"))
    train_dl, val_dl, test_dl = create_dataloaders(cfg)
"""

from .dual_modal_dataset import DualModalDataset
from .dataloader import create_dataloaders, dual_modal_collate_fn
from .augmentations import (
    DualModalTransform,
    build_transforms,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "DualModalDataset",
    "create_dataloaders",
    "dual_modal_collate_fn",
    "DualModalTransform",
    "build_transforms",
    "get_train_transforms",
    "get_val_transforms",
]
