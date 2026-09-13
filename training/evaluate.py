#!/usr/bin/env python3
"""
TriNetra-AMRF — Detection Evaluation (Phase 5.4 / 5.7)
======================================================

Purpose (Phase 5 — Layer 4: Detection Integration & Training):
    COCO-metric evaluation of the trained ``FusionDetector`` (and the Phase-5
    baselines) on any split:

      * decode eval-mode head output → NMS → COCO detection JSON → COCOeval
        (mAP@0.5, mAP@0.5:0.95, small/medium/large breakdown);
      * TRC-binned analysis (5.7): per-bin mAP for degraded (<0.5), marginal
        (0.5–0.7), and clean (≥0.7) frames — where the TRC gate must earn its
        keep;
      * ``--stress``: inject the Phase-3 blur/saturate corruptions into the
        thermal stream at load time and re-score — the controlled experiment
        for fusion-with-TRC vs fusion-without-TRC.

    Predictions are produced in the resized frame (640×512) and scaled back to
    each image's ``orig_size`` so they match the ground-truth JSON coordinates.

Usage:
    python training/evaluate.py --checkpoint weights/fusion/best.pt --split test
    python training/evaluate.py --checkpoint weights/fusion/best.pt --split test --stress
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# Windows consoles default to cp1252 and choke on arrows/emoji; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.loss_adapter import get_class_id_map                # noqa: E402

# TRC bins for the degraded-frame analysis (5.7): [low, high) edges.
TRC_BINS = (("degraded", 0.0, 0.5), ("marginal", 0.5, 0.7), ("clean", 0.7, 1.01))


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── decode + NMS ─────────────────────────────────────────────────────────────
@torch.no_grad()
def decode_predictions(
    decoded: torch.Tensor,
    cfg: dict,
    keep_classes: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """NMS the eval-mode head output into per-image detection tensors.

    Args:
        decoded:      [B, 4+nc, N] — center-xywh (pixel, resized frame) + class
                      scores, straight from the Detect head in eval mode.
        cfg:          Full config (``detection.confidence_threshold`` / ``iou_threshold``).
        keep_classes: COCO head class indices to keep (Option A: the mapped
                      classes only). None → all classes.

    Returns:
        List of B tensors [n, 6] — (x1, y1, x2, y2, conf, cls) in the
        resized frame.
    """
    from ultralytics.utils.nms import non_max_suppression

    dcfg = cfg.get("detection", {})
    return non_max_suppression(
        decoded,
        conf_thres=float(dcfg.get("confidence_threshold", 0.25)),
        iou_thres=float(dcfg.get("iou_threshold", 0.45)),
        classes=keep_classes,
        nc=decoded.shape[1] - 4,
    )


def detections_to_coco(
    dets: List[torch.Tensor],
    targets: List[dict],
    img_hw,
    coco_from_head: Dict[int, int],
) -> List[dict]:
    """Convert NMS'd detections to COCO result dicts, scaled to orig_size.

    Args:
        dets:           Per-image [n,6] (xyxy, conf, cls) in the resized frame.
        targets:        Per-sample target dicts ('image_id', 'orig_size' [H,W]).
        img_hw:         (H, W) of the batched tensors (the resized frame).
        coco_from_head: Head class index → unified category id (inverse of
                        ``detection.class_id_map``) — GT JSONs use unified ids.
    """
    h, w = float(img_hw[0]), float(img_hw[1])
    results = []
    for det, t in zip(dets, targets):
        if det is None or not len(det):
            continue
        image_id = int(t["image_id"])
        oh, ow = (int(v) for v in t["orig_size"])
        sx, sy = ow / w, oh / h
        for x1, y1, x2, y2, conf, cls in det.cpu().tolist():
            cat = coco_from_head.get(int(cls))
            if cat is None:
                continue
            results.append({
                "image_id": image_id,
                "category_id": cat,
                "bbox": [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy],
                "score": float(conf),
            })
    return results


# ── COCOeval scoring ─────────────────────────────────────────────────────────
def _coco_map(coco_gt, results: List[dict], img_ids: List[int],
              verbose: bool = False) -> Dict[str, float]:
    """Score detection results with COCOeval on the given image ids.

    Returns {'mAP50': float, 'mAP50_95': float, 'mAP_small/medium/large': ...};
    all -1.0 (COCO's "no data" marker → reported as nan) when results/ids are empty.
    """
    from pycocotools.cocoeval import COCOeval

    if not results or not img_ids:
        return {k: float("nan") for k in
                ("mAP50", "mAP50_95", "mAP_small", "mAP_medium", "mAP_large")}

    # loadRes prints; keep output clean unless verbose.
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with sink:
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = sorted(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats

    def stat(i: int) -> float:
        # COCOeval uses -1 as its "no ground truth in this category/area" marker.
        v = float(s[i])
        return float("nan") if v < 0 else v

    return {
        "mAP50_95": stat(0), "mAP50": stat(1),
        "mAP_small": stat(3), "mAP_medium": stat(4), "mAP_large": stat(5),
    }


# ── stress-corruption hook (5.7) ─────────────────────────────────────────────
class StressThermalLoader:
    """Wraps ``DualModalDataset._load_thermal`` to inject the Phase-3 blur /
    saturate degradations BEFORE preprocessing, so TRC drops accordingly.

    Deterministic per file (byte-sum of the name picks blur vs saturate) —
    every variant in the ablation sees the *same* corrupted test set.

    A module-level class, not a closure: the dataset is pickled to DataLoader
    worker processes (spawn on Windows), and a local function would not
    survive that.
    """

    KINDS = ("blur", "saturate")

    def __init__(self, dataset):
        # Bind the *unbound* function + dataset separately so this stays
        # picklable regardless of how the dataset was constructed.
        self.dataset = dataset
        self.inner = type(dataset)._load_thermal

    def __call__(self, fname: str):
        from utils.visualize_preproc import _degrade
        kind = self.KINDS[sum(fname.encode()) % len(self.KINDS)]
        return _degrade(self.inner(self.dataset, fname), kind)


def install_stress_corruption(loader_or_dataset) -> None:
    """Install :class:`StressThermalLoader` on a dataset (or a DataLoader's
    dataset). Idempotent.

    When handed a ``DataLoader``, its cached persistent-worker iterator is
    dropped: ``create_dataloaders`` sets ``persistent_workers=True``, so worker
    processes hold a *copy* of the dataset made when iteration first started —
    a patch installed afterwards would silently never reach them.
    """
    from torch.utils.data import DataLoader

    loader = loader_or_dataset if isinstance(loader_or_dataset, DataLoader) else None
    dataset = loader.dataset if loader is not None else loader_or_dataset

    if isinstance(getattr(dataset, "_load_thermal", None), StressThermalLoader):
        print("  [OK] stress corruption already installed; left as is.")
        return
    dataset._load_thermal = StressThermalLoader(dataset)
    if loader is not None:
        loader._iterator = None  # force persistent workers to respawn
    print("  [OK] stress corruption installed (blur/saturate, deterministic per file).")


# ── main evaluation loop ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: str = "cuda",
             trc_bins: bool = True, verbose: bool = False) -> Dict[str, float]:
    """Run the full COCO evaluation of ``model`` on ``loader``'s split.

    Args:
        model:    A ``FusionDetector``-compatible module —
                  ``model(visible, thermal, trc)`` returning eval-mode
                  ``(decoded [B,4+nc,N], raw)`` output.
        loader:   A Phase-2 dataloader (its ``dataset.coco`` is the GT).
        cfg:      Full parsed config.
        device:   Inference device.
        trc_bins: Also compute the TRC-binned per-bin mAP (5.7).

    Returns:
        Metrics dict: overall mAPs + ``mAP50(<bin>)`` / ``n(<bin>)`` per TRC bin.
    """
    from models.detector.fusion_detector import FusionDetector

    class_map = get_class_id_map(cfg)                 # unified → head index
    head_classes = sorted(set(class_map.values()))
    coco_from_head = {v: k for k, v in class_map.items()}

    model.eval()
    results: List[dict] = []
    trc_by_img: Dict[int, float] = {}
    img_hw = None

    for visible, thermal, targets in loader:
        trc = FusionDetector.trc_from_targets(targets, device=device)
        img_hw = tuple(visible.shape[-2:])
        out = model(visible.to(device), thermal.to(device), trc)
        decoded = out[0] if isinstance(out, (list, tuple)) else out
        dets = decode_predictions(decoded, cfg, keep_classes=head_classes)
        results += detections_to_coco(dets, targets, img_hw, coco_from_head)
        for t in targets:
            trc_by_img[int(t["image_id"])] = float(t.get("trc", 1.0))

    coco_gt = loader.dataset.coco
    all_ids = list(trc_by_img)
    metrics = _coco_map(coco_gt, results, all_ids, verbose=verbose)
    metrics["n_images"] = len(all_ids)
    metrics["n_dets"] = len(results)
    metrics["trc_mean"] = float(np.mean(list(trc_by_img.values()))) if trc_by_img else float("nan")

    if trc_bins:
        for name, lo, hi in TRC_BINS:
            ids = [i for i, v in trc_by_img.items() if lo <= v < hi]
            m = _coco_map(coco_gt, results, ids)
            metrics[f"mAP50({name})"] = m["mAP50"]
            metrics[f"mAP50_95({name})"] = m["mAP50_95"]
            metrics[f"n({name})"] = len(ids)
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Human-readable metric block for logs/reports."""
    lines = [f"  images: {metrics.get('n_images', 0)}   "
             f"detections: {metrics.get('n_dets', 0)}   "
             f"TRC mean: {metrics.get('trc_mean', float('nan')):.3f}",
             f"  mAP@0.5      = {metrics['mAP50']:.4f}",
             f"  mAP@0.5:0.95 = {metrics['mAP50_95']:.4f}",
             f"  mAP small/med/large = {metrics['mAP_small']:.4f} / "
             f"{metrics['mAP_medium']:.4f} / {metrics['mAP_large']:.4f}"]
    for name, _, _ in TRC_BINS:
        k = f"mAP50({name})"
        if k in metrics:
            lines.append(f"  {k:<18s} = {metrics[k]:.4f}   (n={metrics[f'n({name})']})")
    return "\n".join(lines)


# ── checkpoint loading ───────────────────────────────────────────────────────
def load_checkpoint_model(ckpt_path: str, cfg: dict, device: str):
    """Rebuild the detector recorded in a Phase-5 checkpoint and load weights.

    The checkpoint's ``variant`` key selects the architecture: 'fusion' /
    'fusion_no_trc' → ``FusionDetector``; 'rgb' / 'thermal' → the Phase-5
    single-modality baselines.
    """
    from models.detector.fusion_detector import FusionDetector

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg_snapshot", cfg)
    variant = ckpt.get("variant", "fusion")
    if variant in ("fusion", "fusion_no_trc"):
        model = FusionDetector(cfg)
    else:
        from training.baselines import SingleModalDetector
        model = SingleModalDetector(cfg, modality=("visible" if variant == "rgb"
                                                   else "thermal"))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  [OK] loaded {variant!r} checkpoint (epoch {ckpt.get('epoch')}, "
          f"stage {ckpt.get('stage')}, best mAP50 {ckpt.get('best_map', float('nan')):.4f})")
    return model, cfg


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 COCO evaluation")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--checkpoint", required=True, help="Path to a Phase-5 .pt checkpoint.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--stress", action="store_true",
                    help="Inject blur/saturate corruptions into the thermal stream (5.7).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg["datasets"].get("seed", 42)))

    model, cfg = load_checkpoint_model(args.checkpoint, cfg, args.device)

    from datasets.dataloader import create_dataloaders
    loaders = dict(zip(("train", "val", "test"),
                       create_dataloaders(cfg, batch_size=args.batch_size)))
    loader = loaders.get(args.split)
    if loader is None:
        print(f"  [ERROR] no annotations for split '{args.split}'.")
        return 1
    if args.stress:
        install_stress_corruption(loader)

    tag = f"{args.split}{' +stress' if args.stress else ''}"
    print(f"\n=== evaluate ({tag}) ===")
    metrics = evaluate(model, loader, cfg, device=args.device, verbose=True)
    print(format_metrics(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
