#!/usr/bin/env python3
"""
TriNetra-AMRF — Preprocessing & TRC Visualization (Phase 3.7)
============================================================

Sanity-check the Layer-2 pipeline visually and statistically:

  * ``visualize_preprocessing`` — a per-frame panel:
        raw thermal | denoised | normalized | trc_map | RGB overlay
    Confirms filters preserve edges and that TRC responds to degradation.

  * ``trc_sanity_report`` — TRC distribution over a subset of a split, flagging
    low-reliability frames (``trc < threshold``) for review.

  * ``stress_test`` — inject blur / saturation into thermal and confirm the TRC
    score drops relative to the clean frame (the core novelty check).

Figures are written to files (Agg backend) so this runs on headless machines.

Usage:
    python utils/visualize_preproc.py --split val --num-samples 6
    python utils/visualize_preproc.py --report --split val --num 200
    python utils/visualize_preproc.py --stress --split val --num-samples 4
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
from PIL import Image

# Windows consoles default to cp1252 and choke on box-drawing/emoji; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Allow `python utils/visualize_preproc.py` to import top-level packages.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.denoise import denoise                 # noqa: E402
from preprocessing.normalize import normalize             # noqa: E402
from preprocessing.trc import compute_trc                 # noqa: E402
from preprocessing.pipeline import PreprocessPipeline     # noqa: E402


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ── loading helpers ──────────────────────────────────────────────────────────
def _load_pair(vis_path: Path, thr_path: Path):
    visible = np.array(Image.open(vis_path).convert("RGB"))
    thermal = np.array(Image.open(thr_path).convert("L"))[:, :, None]
    return visible, thermal


def _list_split_files(cfg: dict, split: str, n: int) -> List[str]:
    ann_json = resolve(cfg["datasets"]["annotations_dir"]) / f"{split}.json"
    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return [im["file_name"] for im in coco["images"][:n]]


# ── 5-panel per-frame visualization ──────────────────────────────────────────
def visualize_preprocessing(
    visible: np.ndarray,
    thermal: np.ndarray,
    cfg: dict,
    output_path: Optional[Path] = None,
    title: str = "",
):
    """Render raw thermal | denoised | normalized | trc_map | overlay.

    ``visible``/``thermal`` are uint8 arrays (HxWx3 and HxWx1/HxW).
    """
    thr_raw = thermal[:, :, 0] if thermal.ndim == 3 else thermal

    thr_den = denoise(thermal, "thermal", cfg, guide=visible)
    thr_norm = normalize(thr_den, "thermal", cfg)
    thr_den2 = thr_den[:, :, 0] if thr_den.ndim == 3 else thr_den
    thr_norm2 = thr_norm[:, :, 0] if thr_norm.ndim == 3 else thr_norm

    # Force a spatial map for the panel regardless of the global config flag.
    map_cfg = json.loads(json.dumps(cfg))  # cheap deep copy of plain dict
    map_cfg.setdefault("preprocessing", {}).setdefault("trc", {})["spatial_map"] = True
    trc_out = compute_trc(thr_norm, visible=visible, align_error=0.0, cfg=map_cfg)
    trc = trc_out["trc"]
    trc_map = trc_out["trc_map"]
    comps = trc_out["components"]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(thr_raw, cmap="inferno"); axes[0].set_title("raw thermal")
    axes[1].imshow(thr_den2, cmap="inferno"); axes[1].set_title("denoised")
    axes[2].imshow(thr_norm2, cmap="inferno"); axes[2].set_title("normalized")

    im = axes[3].imshow(trc_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[3].set_title(f"trc_map (frame trc={trc:.2f})")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    thr_rgb = plt.get_cmap("inferno")(thr_norm2 / 255.0)[:, :, :3]
    overlay = np.clip(0.55 * (visible / 255.0) + 0.45 * thr_rgb, 0, 1)
    axes[4].imshow(overlay); axes[4].set_title("overlay (RGB+IR)")

    for ax in axes:
        ax.axis("off")

    comp_str = "  ".join(f"{k}={v:.2f}" for k, v in comps.items())
    fig.suptitle(f"{title}   TRC={trc:.3f}   [{comp_str}]", fontsize=11)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path
    plt.close(fig)
    return None


def visualize_from_split(cfg: dict, split: str, num_samples: int, out_dir: Path) -> None:
    files = _list_split_files(cfg, split, num_samples)
    vis_dir = resolve(cfg["datasets"]["visible_dir"]) / split
    thr_dir = resolve(cfg["datasets"]["thermal_dir"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        visible, thermal = _load_pair(vis_dir / fname, thr_dir / fname)
        out = out_dir / f"preproc_{split}_{Path(fname).stem}.png"
        visualize_preprocessing(visible, thermal, cfg, output_path=out, title=fname)
        print(f"  wrote {out}")


# ── TRC distribution report ──────────────────────────────────────────────────
def trc_sanity_report(cfg: dict, split: str, n: int, out_path: Path,
                      threshold: float = 0.3) -> Dict[str, float]:
    """Compute TRC over the first ``n`` frames of a split; plot a histogram and
    flag frames below ``threshold``.
    """
    files = _list_split_files(cfg, split, n)
    vis_dir = resolve(cfg["datasets"]["visible_dir"]) / split
    thr_dir = resolve(cfg["datasets"]["thermal_dir"]) / split
    pre = PreprocessPipeline(cfg)

    scores: List[float] = []
    flagged: List[str] = []
    for fname in files:
        visible, thermal = _load_pair(vis_dir / fname, thr_dir / fname)
        _, _, extras = pre(visible, thermal)
        trc = float(extras["trc"])
        scores.append(trc)
        if trc < threshold:
            flagged.append(fname)

    scores_arr = np.asarray(scores, dtype=np.float32)
    stats = {
        "n": int(scores_arr.size),
        "mean": float(scores_arr.mean()) if scores_arr.size else 0.0,
        "std": float(scores_arr.std()) if scores_arr.size else 0.0,
        "min": float(scores_arr.min()) if scores_arr.size else 0.0,
        "max": float(scores_arr.max()) if scores_arr.size else 0.0,
        "n_flagged": len(flagged),
        "threshold": threshold,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_arr, bins=25, range=(0, 1), color="#00897B", edgecolor="black")
    ax.axvline(threshold, color="#D81B60", linestyle="--",
               label=f"threshold={threshold}")
    ax.axvline(stats["mean"], color="#1E88E5", linestyle="-",
               label=f"mean={stats['mean']:.3f}")
    ax.set_xlabel("Thermal Reliability Confidence (trc)")
    ax.set_ylabel("frame count")
    ax.set_title(f"TRC distribution — split '{split}' (n={stats['n']}, "
                 f"{stats['n_flagged']} flagged < {threshold})")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"  TRC stats over {stats['n']} frames of '{split}':")
    print(f"    mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")
    print(f"    {stats['n_flagged']} frame(s) flagged below {threshold}.")
    if flagged[:10]:
        print(f"    e.g. {flagged[:10]}")
    print(f"  wrote {out_path}")
    return stats


# ── synthetic degradation stress test ────────────────────────────────────────
def _degrade(thermal: np.ndarray, kind: str) -> np.ndarray:
    """Inject a controlled degradation into a thermal frame (uint8 HxWx1/HxW).

    The models mimic real IR failure modes:
      * blur     — defocus / motion (destroys high-frequency detail).
      * saturate — sensor overexposure: a strong gain drives a large fraction of
                   pixels to the ceiling, and a light bloom blur erases the
                   detail behind the clipped region. (A naive hard-threshold
                   would instead *add* artificial edges on dark night frames.)
      * flat     — dead / no-contrast frame (NUC failure, lens cap).
    """
    import cv2
    t = thermal[:, :, 0] if thermal.ndim == 3 else thermal
    if kind == "blur":
        out = cv2.GaussianBlur(t, (21, 21), 0)
    elif kind == "saturate":
        gained = np.clip(t.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        out = cv2.GaussianBlur(gained, (9, 9), 0)  # overexposure bloom
    elif kind == "flat":
        out = np.full_like(t, int(t.mean()))  # dead / no-contrast frame
    else:
        out = t
    return out[:, :, None] if thermal.ndim == 3 else out


def stress_test(cfg: dict, split: str, num_samples: int, out_dir: Path) -> None:
    """Confirm TRC drops when thermal is blurred / saturated / flattened."""
    files = _list_split_files(cfg, split, num_samples)
    vis_dir = resolve(cfg["datasets"]["visible_dir"]) / split
    thr_dir = resolve(cfg["datasets"]["thermal_dir"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["clean", "blur", "saturate", "flat"]
    print(f"  TRC stress test on {len(files)} frame(s) of '{split}':")
    all_ok = True
    for fname in files:
        visible, thermal = _load_pair(vis_dir / fname, thr_dir / fname)
        row = {}
        for kind in kinds:
            thr = thermal if kind == "clean" else _degrade(thermal, kind)
            out = compute_trc(thr, visible=visible, align_error=0.0, cfg=cfg)
            row[kind] = float(out["trc"])
        clean = row["clean"]
        drops = {k: row[k] for k in kinds if k != "clean"}
        ok = all(v <= clean + 1e-6 for v in drops.values())
        all_ok &= ok
        mark = "OK" if ok else "!! degraded TRC not lower"
        print(f"    {fname}: clean={clean:.3f}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in drops.items())
              + f"   [{mark}]")

    # Bar chart of the last frame's degradation response for a visual artifact.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(list(row.keys()), list(row.values()),
           color=["#43A047", "#FB8C00", "#E53935", "#8E24AA"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("trc")
    ax.set_title(f"TRC vs. injected thermal degradation ({fname})")
    fig.tight_layout()
    out = out_dir / f"trc_stress_{split}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  overall: {'PASS' if all_ok else 'FAIL'} — wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize Phase 3 preprocessing + TRC.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--out-dir", default="runs/visualizations")
    parser.add_argument("--report", action="store_true", help="TRC distribution report.")
    parser.add_argument("--num", type=int, default=200, help="Frames for --report.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Flag trc below this.")
    parser.add_argument("--stress", action="store_true", help="Synthetic degradation test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve(args.out_dir)

    if args.report:
        trc_sanity_report(cfg, args.split, args.num,
                          out_dir / f"trc_report_{args.split}.png", args.threshold)
    elif args.stress:
        stress_test(cfg, args.split, args.num_samples, out_dir)
    else:
        visualize_from_split(cfg, args.split, args.num_samples, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
