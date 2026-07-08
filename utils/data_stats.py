#!/usr/bin/env python3
"""
TriNetra-AMRF — Dataset Statistics
==================================

Purpose (Phase 2.7):
    Summarize a COCO-format split: image/annotation counts, class
    distribution, COCO small/medium/large size buckets, negatives (images with
    no objects), and bbox aspect-ratio spread.

Usage:
    python utils/data_stats.py --split train
    python utils/data_stats.py --json datasets/annotations/val.json
    python utils/data_stats.py --all          # train + val + test summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

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

# COCO area thresholds (in pixels^2).
_SMALL = 32 ** 2
_MEDIUM = 96 ** 2


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def compute_dataset_statistics(annotations_json: str | Path, quiet: bool = False) -> dict:
    """
    Compute and (optionally) print statistics for one COCO JSON.

    Returns a dict of the computed metrics.
    """
    with open(annotations_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    n_images = len(coco["images"])
    anns = coco["annotations"]
    n_ann = len(anns)

    per_class = defaultdict(int)
    size_buckets = {"small": 0, "medium": 0, "large": 0}
    aspect_ratios = []
    images_with_ann = set()

    for a in anns:
        per_class[a["category_id"]] += 1
        images_with_ann.add(a["image_id"])
        _, _, w, h = a["bbox"]
        area = a.get("area", w * h)
        if area < _SMALL:
            size_buckets["small"] += 1
        elif area < _MEDIUM:
            size_buckets["medium"] += 1
        else:
            size_buckets["large"] += 1
        if h > 0:
            aspect_ratios.append(w / h)

    n_neg = n_images - len(images_with_ann)
    avg_ar = sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0.0
    avg_obj = n_ann / n_images if n_images else 0.0

    stats = {
        "json": str(annotations_json),
        "images": n_images,
        "annotations": n_ann,
        "avg_objects_per_image": round(avg_obj, 2),
        "negatives": n_neg,
        "per_class": {cat_name.get(k, k): v for k, v in sorted(per_class.items())},
        "size_buckets": size_buckets,
        "avg_aspect_ratio": round(avg_ar, 3),
    }

    if not quiet:
        _print_stats(stats)
    return stats


def _bar(count: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * count / total))
    return "█" * filled + "·" * (width - filled)


def _print_stats(s: dict) -> None:
    name = Path(s["json"]).name
    print("=" * 64)
    print(f"  Dataset Statistics — {name}")
    print("=" * 64)
    print(f"  Images                : {s['images']}")
    print(f"  Annotations           : {s['annotations']}")
    print(f"  Avg objects / image   : {s['avg_objects_per_image']}")
    print(f"  Negative images       : {s['negatives']}")
    print(f"  Avg bbox aspect ratio : {s['avg_aspect_ratio']}")

    print("\n  Class distribution:")
    total_obj = s["annotations"] or 1
    for cls, cnt in s["per_class"].items():
        print(f"    {str(cls):<14}{cnt:>8}  {_bar(cnt, total_obj)}")

    print("\n  Object size (COCO area buckets):")
    sb = s["size_buckets"]
    tb = sum(sb.values()) or 1
    for k in ("small", "medium", "large"):
        print(f"    {k:<14}{sb[k]:>8}  {_bar(sb[k], tb)}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute COCO dataset statistics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--json", default=None, help="Explicit COCO JSON path.")
    parser.add_argument("--split", default=None, help="Split name (train/val/test).")
    parser.add_argument("--all", action="store_true", help="Summarize all three splits.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ann_dir = resolve(cfg["datasets"]["annotations_dir"])

    if args.json:
        compute_dataset_statistics(resolve(args.json))
        return 0

    if args.all:
        for split in ("train", "val", "test"):
            jp = ann_dir / f"{split}.json"
            if jp.is_file():
                compute_dataset_statistics(jp)
            else:
                print(f"  [INFO] {jp} not found; skipping.")
        return 0

    split = args.split or "train"
    jp = ann_dir / f"{split}.json"
    if not jp.is_file():
        print(f"[ERROR] {jp} not found. Run datasets/split_dataset.py first.")
        return 1
    compute_dataset_statistics(jp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
