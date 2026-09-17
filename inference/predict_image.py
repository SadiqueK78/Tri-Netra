#!/usr/bin/env python3
"""
TriNetra-AMRF — Single-Pair Inference & Detection Visualization
=================================================================

Purpose:
    Run a trained FusionDetector (or baseline) checkpoint on one
    visible+thermal image pair and save an annotated image with drawn
    detection boxes — for demos, reviewer walkthroughs, and quick manual
    sanity checks. This is a lightweight sibling to Phase 9's planned
    realtime/video pipelines (inference/realtime.py, inference/video.py),
    which handle live camera streams; this script handles exactly one pair.

Note on inputs:
    FusionDetector needs BOTH a visible and a thermal frame of the same
    scene — feeding it a single ordinary photo isn't meaningful. Use a
    matched pair from datasets/visible/<split>/ + datasets/thermal/<split>/
    (any file with the same name in both), or your own aligned pair. To
    test on a single arbitrary RGB photo with no thermal counterpart, use
    the RGB-only baseline checkpoint instead (weights/ablation/poc/rgb/best.pt)
    — it ignores the thermal argument entirely.

Usage:
    python inference/predict_image.py \\
        --checkpoint weights/ablation/poc/fusion_trc/best.pt \\
        --visible datasets/visible/test/00000.png \\
        --thermal datasets/thermal/test/00000.png \\
        --output runs/predictions/demo.jpg

    # Lower the confidence threshold to see more (lower-confidence) boxes:
    python inference/predict_image.py ... --conf 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_BOX_COLOR = (255, 64, 64)


def predict(
    checkpoint: str,
    visible_path: str,
    thermal_path: str,
    config_path: str = str(DEFAULT_CONFIG),
    output_path: str = "runs/predictions/output.jpg",
    conf_override: float | None = None,
    device: str | None = None,
):
    """Run one image pair through a checkpoint; save + return the annotated image."""
    from datasets.augmentations import build_transforms
    from preprocessing.pipeline import build_preprocess_pipeline
    from training.evaluate import decode_predictions, load_checkpoint_model
    from training.loss_adapter import get_class_id_map

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if conf_override is not None:
        cfg.setdefault("detection", {})["confidence_threshold"] = float(conf_override)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_checkpoint_model(checkpoint, cfg, device)
    model.eval()

    vis_img = Image.open(visible_path).convert("RGB")
    thr_img = Image.open(thermal_path).convert("L")
    orig_w, orig_h = vis_img.size
    vis_arr = np.array(vis_img)
    thr_arr = np.array(thr_img)[:, :, None]

    trc = 1.0
    preprocess = build_preprocess_pipeline(cfg)
    if preprocess is not None:
        vis_arr, thr_arr, extras = preprocess(vis_arr, thr_arr)
        trc = float(extras.get("trc", 1.0))

    transform = build_transforms(cfg, "val")
    vis_t, thr_t, _, _ = transform(vis_arr, thr_arr, [], [])
    vis_t = vis_t.unsqueeze(0).to(device)
    thr_t = thr_t.unsqueeze(0).to(device)
    trc_t = torch.tensor([trc], dtype=torch.float32, device=device)

    class_map = get_class_id_map(cfg)                       # unified -> head index
    head_classes = sorted(set(class_map.values()))
    coco_from_head = {v: k for k, v in class_map.items()}
    id_to_name = {c["id"]: c["name"] for c in cfg["datasets"]["categories"]}

    with torch.no_grad():
        out = model(vis_t, thr_t, trc_t)
    decoded = out[0] if isinstance(out, (list, tuple)) else out
    dets = decode_predictions(decoded, cfg, keep_classes=head_classes)[0]

    resized_h, resized_w = vis_t.shape[-2:]
    sx, sy = orig_w / resized_w, orig_h / resized_h

    draw = ImageDraw.Draw(vis_img)
    n_det = 0
    for x1, y1, x2, y2, conf, cls in ([] if dets is None else dets.cpu().tolist()):
        cat_id = coco_from_head.get(int(cls))
        if cat_id is None:
            continue
        name = id_to_name.get(cat_id, str(cat_id))
        box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        draw.rectangle(box, outline=_BOX_COLOR, width=3)
        label = f"{name} {conf:.2f}"
        draw.rectangle([box[0], max(0, box[1] - 14), box[0] + 7 * len(label), max(0, box[1])],
                       fill=_BOX_COLOR)
        draw.text((box[0] + 2, max(0, box[1] - 13)), label, fill="white")
        n_det += 1

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_img.save(out_path)
    print(f"[OK] {n_det} detection(s)  TRC={trc:.3f}  device={device}  -> {out_path}")
    return out_path, n_det


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a TriNetra checkpoint on one visible+thermal pair.")
    ap.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint.")
    ap.add_argument("--visible", required=True, help="Path to the visible (RGB) image.")
    ap.add_argument("--thermal", required=True, help="Path to the paired thermal image.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--output", default="runs/predictions/output.jpg")
    ap.add_argument("--conf", type=float, default=None,
                     help="Override detection.confidence_threshold (default: config value, 0.25).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    predict(args.checkpoint, args.visible, args.thermal, args.config, args.output,
            conf_override=args.conf, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
