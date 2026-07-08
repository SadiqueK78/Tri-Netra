#!/usr/bin/env python3
"""
TriNetra-AMRF — Paired Data Visualization
=========================================

Purpose (Phase 2.7):
    Sanity-check the data pipeline visually:
      * ``visualize_pair``  — RGB | thermal | overlay, with color-coded boxes,
        drawn straight from a split's COCO JSON (no transforms).
      * ``visualize_batch`` — a grid of samples pulled from a DataLoader, which
        confirms that synchronized augmentation and normalization behave.

Figures are written to files (Agg backend) so this runs on headless machines.

Usage:
    python utils/visualize.py --split train --num-samples 6
    python utils/visualize.py --batch --split train --num-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Windows consoles/pipes default to cp1252 and choke on box-drawing/emoji
# output; force UTF-8 so scripts never crash on a print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Allow `python utils/visualize.py` to import the top-level `datasets` package
# (when run as a script, sys.path[0] is utils/, not the project root).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Distinct colors per unified category id (1..6).
_CLASS_COLORS = {
    1: "#00E5FF",  # person
    2: "#FFEA00",  # car
    3: "#FF6D00",  # bus
    4: "#FF1744",  # truck
    5: "#D500F9",  # motorcycle
    6: "#76FF03",  # cyclist
}
_DEFAULT_COLOR = "#FFFFFF"


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _draw_boxes(ax, boxes, labels, cat_name: Dict[int, str]) -> None:
    """Draw COCO [x, y, w, h] boxes with per-class colors and labels."""
    for (x, y, w, h), lbl in zip(boxes, labels):
        color = _CLASS_COLORS.get(int(lbl), _DEFAULT_COLOR)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.5))
        ax.text(x, max(0, y - 3), cat_name.get(int(lbl), str(lbl)),
                color="black", fontsize=7,
                bbox=dict(facecolor=color, edgecolor="none", pad=0.5, alpha=0.85))


def visualize_pair(
    visible_path: str | Path,
    thermal_path: str | Path,
    boxes: List[List[float]],
    labels: List[int],
    cat_name: Dict[int, str],
    output_path: Optional[str | Path] = None,
    title: str = "",
):
    """
    Render [visible | thermal | overlay] side-by-side with boxes.

    ``boxes`` are COCO [x, y, w, h] in the ORIGINAL image coordinate frame.
    """
    vis = np.array(Image.open(visible_path).convert("RGB"))
    thr = np.array(Image.open(thermal_path).convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(vis)
    axes[0].set_title(f"Visible (RGB){'  —  ' + title if title else ''}")
    _draw_boxes(axes[0], boxes, labels, cat_name)

    axes[1].imshow(thr, cmap="inferno")
    axes[1].set_title("Thermal (IR)")
    _draw_boxes(axes[1], boxes, labels, cat_name)

    # Overlay: blend grayscale-thermal (as heat) over the RGB frame.
    thr_rgb = plt.get_cmap("inferno")(thr / 255.0)[:, :, :3]
    overlay = (0.55 * (vis / 255.0) + 0.45 * thr_rgb)
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay (0.55·RGB + 0.45·IR)")
    _draw_boxes(axes[2], boxes, labels, cat_name)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return output_path
    plt.close(fig)
    return None


def visualize_from_split(cfg: dict, split: str, num_samples: int, out_dir: Path) -> None:
    """Pull the first ``num_samples`` images from a split's COCO JSON and render."""
    ann_json = resolve(cfg["datasets"]["annotations_dir"]) / f"{split}.json"
    if not ann_json.is_file():
        print(f"[ERROR] {ann_json} not found. Run split_dataset.py first.")
        return
    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image: Dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    vis_dir = resolve(cfg["datasets"]["visible_dir"]) / split
    thr_dir = resolve(cfg["datasets"]["thermal_dir"]) / split

    # Prefer images that actually have objects, for a more informative preview.
    imgs = [im for im in coco["images"] if anns_by_image.get(im["id"])][:num_samples]
    if not imgs:
        imgs = coco["images"][:num_samples]

    out_dir.mkdir(parents=True, exist_ok=True)
    for im in imgs:
        anns = anns_by_image.get(im["id"], [])
        boxes = [a["bbox"] for a in anns]
        labels = [a["category_id"] for a in anns]
        out = out_dir / f"{split}_{Path(im['file_name']).stem}_pair.png"
        visualize_pair(vis_dir / im["file_name"], thr_dir / im["file_name"],
                       boxes, labels, cat_name, output_path=out,
                       title=f"{im['file_name']} ({len(boxes)} obj)")
        print(f"  wrote {out}")


def _denormalize(t: np.ndarray, mean, std) -> np.ndarray:
    """Undo Normalize for display; t is CxHxW. Returns HxWxC in [0, 1]."""
    arr = t.copy()
    for c in range(arr.shape[0]):
        arr[c] = arr[c] * std[c] + mean[c]
    arr = np.clip(arr, 0, 1)
    return np.transpose(arr, (1, 2, 0))


def visualize_batch(cfg: dict, split: str, num_samples: int, out_path: Path) -> None:
    """Grid-visualize one batch from the DataLoader (post-augmentation)."""
    import torch  # local import keeps module import light
    from datasets.dataloader import create_dataloaders

    loaders = create_dataloaders(cfg, batch_size=num_samples, num_workers=0)
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]
    if loader is None:
        print(f"[ERROR] No loader for split '{split}'.")
        return

    visible, thermal, targets = next(iter(loader))
    norm = cfg["datasets"]["normalize"]
    n = min(num_samples, visible.shape[0])

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        vis_img = _denormalize(visible[i].numpy(), norm["visible_mean"], norm["visible_std"])
        thr_img = _denormalize(thermal[i].numpy(), norm["thermal_mean"], norm["thermal_std"])[:, :, 0]
        n_obj = int(targets[i]["boxes"].shape[0])

        axes[0, i].imshow(vis_img)
        axes[0, i].set_title(f"vis #{i} ({n_obj} obj)", fontsize=8)
        # Boxes are COCO xywh in the resized frame — draw directly.
        for (x, y, w, h), lbl in zip(targets[i]["boxes"].tolist(), targets[i]["labels"].tolist()):
            axes[0, i].add_patch(Rectangle((x, y), w, h, fill=False,
                                           edgecolor=_CLASS_COLORS.get(int(lbl), _DEFAULT_COLOR),
                                           linewidth=1.2))
        axes[1, i].imshow(thr_img, cmap="inferno")
        axes[1, i].set_title(f"thermal #{i}", fontsize=8)
        for ax in (axes[0, i], axes[1, i]):
            ax.axis("off")

    fig.suptitle(f"Augmented batch — split '{split}'", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize paired dual-modal data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--batch", action="store_true",
                        help="Visualize a post-augmentation DataLoader batch instead of raw pairs.")
    parser.add_argument("--out-dir", default="runs/visualizations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve(args.out_dir)

    if args.batch:
        visualize_batch(cfg, args.split, args.num_samples, out_dir / f"batch_{args.split}.png")
    else:
        visualize_from_split(cfg, args.split, args.num_samples, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
