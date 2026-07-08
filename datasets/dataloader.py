#!/usr/bin/env python3
"""
TriNetra-AMRF — DataLoader Factory
==================================

Purpose (Phase 2.6):
    Assemble ``DualModalDataset`` instances for each split and wrap them in
    PyTorch ``DataLoader``s with a collate function that handles the
    variable number of objects per image.

The collate keeps ``targets`` as a *list of dicts* (one per sample) — the
standard convention for detection — while stacking the fixed-shape visible and
thermal tensors into batches.

Public API:
    create_dataloaders(config) -> (train_loader, val_loader, test_loader)
    dual_modal_collate_fn(batch)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .augmentations import build_transforms
from .dual_modal_dataset import DualModalDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Phase 3 preprocessing factory. Imported defensively so the data pipeline
# still works if the preprocessing package is unavailable (returns no pipeline).
try:
    from preprocessing.pipeline import build_preprocess_pipeline
except Exception:  # pragma: no cover
    def build_preprocess_pipeline(cfg):  # type: ignore
        return None


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def dual_modal_collate_fn(batch):
    """
    Collate a list of ``(visible, thermal, targets)`` samples into a batch.

    Returns:
        visible : FloatTensor [B, 3, H, W]
        thermal : FloatTensor [B, 1, H, W]
        targets : list[dict] of length B (variable-length boxes/labels kept as-is)
    """
    visibles, thermals, targets = zip(*batch)
    visible_batch = torch.stack(visibles, dim=0)
    thermal_batch = torch.stack(thermals, dim=0)
    return visible_batch, thermal_batch, list(targets)


def _make_split_dataset(cfg: dict, split: str) -> Optional[DualModalDataset]:
    """Build a dataset for one split, or None if its annotation JSON is absent."""
    ann_dir = resolve(cfg["datasets"]["annotations_dir"])
    ann_json = ann_dir / f"{split}.json"
    if not ann_json.is_file():
        print(f"  [INFO] No annotations for split '{split}' ({ann_json}); skipping.")
        return None

    visible_dir = resolve(cfg["datasets"]["visible_dir"]) / split
    thermal_dir = resolve(cfg["datasets"]["thermal_dir"]) / split
    transform = build_transforms(cfg, split)
    # Phase 3: attach the Layer-2 preprocessing pipeline (align→denoise→
    # normalize→TRC) when a `preprocessing` block is present in the config.
    preprocess = build_preprocess_pipeline(cfg)
    return DualModalDataset(
        visible_dir=str(visible_dir),
        thermal_dir=str(thermal_dir),
        annotations_json=str(ann_json),
        transform=transform,
        preprocess=preprocess,
    )


def create_dataloaders(
    cfg: dict,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Build (train_loader, val_loader, test_loader) from a config dict.

    Args:
        cfg:         Parsed config (from configs/default.yaml).
        batch_size:  Override for cfg['training']['batch_size'].
        num_workers: Override for cfg['training']['num_workers'].

    Any split whose annotation JSON is missing yields ``None`` in its slot,
    so this works before every split has been generated.
    """
    train_cfg = cfg.get("training", {})
    bs = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 16))
    nw = num_workers if num_workers is not None else int(train_cfg.get("num_workers", 4))
    pin = torch.cuda.is_available()

    loaders = []
    for split in ("train", "val", "test"):
        ds = _make_split_dataset(cfg, split)
        if ds is None:
            loaders.append(None)
            continue
        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=nw,
            pin_memory=pin,
            drop_last=(split == "train"),
            collate_fn=dual_modal_collate_fn,
            persistent_workers=(nw > 0),
            prefetch_factor=(2 if nw > 0 else None),
        )
        print(f"  [OK] {split} loader: {len(ds)} samples, "
              f"{len(loader)} batches (bs={bs}, workers={nw})")
        loaders.append(loader)

    return tuple(loaders)  # type: ignore[return-value]
