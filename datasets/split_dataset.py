#!/usr/bin/env python3
"""
TriNetra-AMRF — Train / Val / Test Splitting
============================================

Purpose (Phase 2.4):
    Turn the master COCO JSON (``instances_all.json``) plus the raw LLVIP tree
    into the canonical, self-contained dataset layout:

        datasets/
        ├── visible/{train,val,test}/*.jpg
        ├── thermal/{train,val,test}/*.jpg   (single-scene pair = same filename)
        └── annotations/{train,val,test}.json

Splitting rules (config-driven):
    * respect_official_test = True  →  keep the source's official test set
      exactly, and split the official *train* images into train/val using the
      train:val proportion from ``split_ratios``.
    * Otherwise  →  pool everything and do a fresh train/val/test split by the
      three ``split_ratios``.
    * Deterministic (fixed ``seed``) and "stratified" on a coarse key
      (has-objects vs. negative) so both splits keep a similar positive ratio.
    * Matched: because visible & thermal share a filename, copying by filename
      guarantees the pair lands in the same split.

Usage:
    python datasets/split_dataset.py --config configs/default.yaml
    python datasets/split_dataset.py --symlink        # try symlinks (Win: dev mode)
    python datasets/split_dataset.py --dry-run        # report only, copy nothing
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Windows consoles/pipes default to cp1252 and choke on box-drawing/emoji
# output; force UTF-8 so scripts never crash on a print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ── Config / path helpers ────────────────────────────────────────────────────
def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ── Raw-image locator ────────────────────────────────────────────────────────
def build_raw_image_index(cfg: dict) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Map each image stem -> its raw visible path and raw thermal path.

    Scans the source's official train/ and test/ subfolders when it has them
    (LLVIP); otherwise scans the visible/thermal roots directly, since not
    every source ships a pre-split layout (M3FD is flat: all 4,200 pairs
    sit straight under Vis/ and Ir/, no train/test subfolders).
    Returns (visible_index, thermal_index).
    """
    src = cfg["datasets"]["sources"][cfg["datasets"]["active_source"]]
    root = resolve(src["root"])
    vis_root = root / src["visible_subdir"]
    thr_root = root / src["thermal_subdir"]

    if "official_train_dir" in src and "official_test_dir" in src:
        subdirs = (src["official_train_dir"], src["official_test_dir"])
        bases = [(vis_root / sub, thr_root / sub) for sub in subdirs]
    else:
        bases = [(vis_root, thr_root)]

    vis_index: Dict[str, Path] = {}
    thr_index: Dict[str, Path] = {}
    for vis_base, thr_base in bases:
        for base, index in ((vis_base, vis_index), (thr_base, thr_index)):
            if not base.is_dir():
                continue
            for p in base.iterdir():
                if p.suffix.lower() in _IMG_EXTS:
                    index[p.stem] = p
    return vis_index, thr_index


# ── Split assignment ─────────────────────────────────────────────────────────
def _stratify_key(image_id: int, anns_by_image: Dict[int, list]) -> str:
    """Coarse stratification bucket: 'pos' if the image has ≥1 box else 'neg'."""
    return "pos" if anns_by_image.get(image_id) else "neg"


def assign_splits(
    coco: dict,
    ratios: Dict[str, float],
    respect_official_test: bool,
    seed: int,
) -> Dict[str, List[dict]]:
    """
    Assign each image record to 'train' / 'val' / 'test'.

    Returns {split_name -> [image_record, ...]}.
    """
    rng = random.Random(seed)
    anns_by_image: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    images = coco["images"]
    result: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    if respect_official_test and any("split" in img for img in images):
        # Official test images go straight to test; official train images are
        # divided into train/val by the train:val proportion.
        official_train = [img for img in images if img.get("split") != "test"]
        result["test"] = [img for img in images if img.get("split") == "test"]

        val_frac = ratios["val"] / (ratios["train"] + ratios["val"])
        _stratified_two_way(official_train, anns_by_image, val_frac, rng,
                            small="val", large="train", result=result)
    else:
        # Fresh three-way split over everything.
        _stratified_three_way(images, anns_by_image, ratios, rng, result)

    return result


def _stratified_two_way(images, anns_by_image, small_frac, rng, small, large, result):
    """Split `images` into two buckets, stratified by pos/neg, deterministically."""
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for img in images:
        buckets[_stratify_key(img["id"], anns_by_image)].append(img)
    for _key, group in buckets.items():
        group_sorted = sorted(group, key=lambda x: x["id"])
        rng.shuffle(group_sorted)
        n_small = round(len(group_sorted) * small_frac)
        result[small].extend(group_sorted[:n_small])
        result[large].extend(group_sorted[n_small:])


def _stratified_three_way(images, anns_by_image, ratios, rng, result):
    """Split `images` into train/val/test, stratified by pos/neg."""
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for img in images:
        buckets[_stratify_key(img["id"], anns_by_image)].append(img)
    for _key, group in buckets.items():
        group_sorted = sorted(group, key=lambda x: x["id"])
        rng.shuffle(group_sorted)
        n = len(group_sorted)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        result["train"].extend(group_sorted[:n_train])
        result["val"].extend(group_sorted[n_train:n_train + n_val])
        result["test"].extend(group_sorted[n_train + n_val:])


# ── Per-split COCO builder ───────────────────────────────────────────────────
def build_split_coco(coco: dict, split_images: List[dict]) -> dict:
    """Assemble a standalone COCO dict for one split (images + their anns)."""
    keep_ids = {img["id"] for img in split_images}
    anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    # Strip the internal 'split' tag from the emitted image records.
    clean_images = [{k: v for k, v in img.items() if k != "split"} for img in split_images]
    return {
        "info": coco.get("info", {}),
        "categories": coco["categories"],
        "images": clean_images,
        "annotations": anns,
    }


# ── File placement ───────────────────────────────────────────────────────────
def place_file(src: Path, dst: Path, use_symlink: bool) -> None:
    """Copy or symlink `src` -> `dst`, creating parents. Falls back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            # Windows without Developer Mode/admin — silently fall back to copy.
            pass
    shutil.copy2(src, dst)


# ── Stats ────────────────────────────────────────────────────────────────────
def print_split_stats(coco: dict, splits: Dict[str, List[dict]]) -> None:
    """Print per-split image counts and per-class object counts."""
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    print("\n" + "=" * 64)
    print("  Split Statistics")
    print("=" * 64)
    header = f"  {'split':<8}{'images':>9}{'objects':>9}   per-class"
    print(header)
    print("  " + "-" * 60)
    for split in ("train", "val", "test"):
        imgs = splits[split]
        per_class: Dict[int, int] = defaultdict(int)
        n_obj = 0
        for img in imgs:
            for a in anns_by_image.get(img["id"], []):
                per_class[a["category_id"]] += 1
                n_obj += 1
        pc = ", ".join(f"{cat_name.get(cid, cid)}={cnt}"
                       for cid, cnt in sorted(per_class.items())) or "—"
        print(f"  {split:<8}{len(imgs):>9}{n_obj:>9}   {pc}")
    total = sum(len(splits[s]) for s in ("train", "val", "test"))
    print("  " + "-" * 60)
    print(f"  {'total':<8}{total:>9}")


# ── Driver ───────────────────────────────────────────────────────────────────
def split_dataset(cfg: dict, use_symlink: bool = False, dry_run: bool = False) -> None:
    master_path = resolve(cfg["datasets"]["master_annotations"])
    if not master_path.is_file():
        raise FileNotFoundError(
            f"Master annotations not found: {master_path}\n"
            f"Run: python datasets/convert_annotations.py "
            f"--source {cfg['datasets']['active_source']} --validate"
        )
    with open(master_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ratios = cfg["datasets"]["split_ratios"]
    seed = int(cfg["datasets"].get("seed", 42))
    respect = bool(cfg["datasets"].get("respect_official_test", True))

    print("=" * 64)
    print("  Splitting dataset")
    print("=" * 64)
    print(f"  Master JSON      : {master_path.name} ({len(coco['images'])} images)")
    print(f"  respect_official : {respect}")
    print(f"  ratios           : {ratios}")
    print(f"  seed             : {seed}")

    splits = assign_splits(coco, ratios, respect, seed)
    print_split_stats(coco, splits)

    if dry_run:
        print("\n  [dry-run] No files copied, no JSON written.")
        return

    vis_index, thr_index = build_raw_image_index(cfg)
    vis_out_root = resolve(cfg["datasets"]["visible_dir"])
    thr_out_root = resolve(cfg["datasets"]["thermal_dir"])
    ann_out_root = resolve(cfg["datasets"]["annotations_dir"])
    ann_out_root.mkdir(parents=True, exist_ok=True)

    verb = "Linking" if use_symlink else "Copying"
    missing = 0
    for split, imgs in splits.items():
        print(f"\n  {verb} {len(imgs)} pair(s) -> {split}/ ...")
        for i, img in enumerate(imgs, 1):
            stem = Path(img["file_name"]).stem
            vis_src = vis_index.get(stem)
            thr_src = thr_index.get(stem)
            if vis_src is None or thr_src is None:
                missing += 1
                continue
            place_file(vis_src, vis_out_root / split / img["file_name"], use_symlink)
            place_file(thr_src, thr_out_root / split / img["file_name"], use_symlink)
            if i % 1000 == 0 or i == len(imgs):
                print(f"     {i}/{len(imgs)}")

        split_coco = build_split_coco(coco, imgs)
        out_json = ann_out_root / f"{split}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(split_coco, f)
        print(f"     wrote {out_json}")

    if missing:
        print(f"\n  [WARN] {missing} image(s) had no raw file and were skipped.")
    print("\n  [OK] Split complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split paired dataset into train/val/test.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink instead of copy (Windows needs Developer Mode/admin).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the split without copying files or writing JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_dataset(cfg, use_symlink=args.symlink, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
