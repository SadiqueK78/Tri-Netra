#!/usr/bin/env python3
"""
TriNetra-AMRF — Four-Way TRC Ablation Runner (Phase 5.6)
========================================================

Purpose (Phase 5 — the experiment that earns the thesis claim):
    Train and evaluate four variants under an identical data / schedule /
    seed budget (PHASE5_IMPLEMENTATION.md §7):

        1. rgb           — RGB-only YOLOv8 (visible floor)
        2. thermal       — thermal-only YOLOv8 + ThermalStem (thermal floor)
        3. fusion_no_trc — FusionDetector with the TRC gate disabled (gate ≡ 1)
        4. fusion_trc    — FusionDetector, default config (the novelty)

    Each variant is scored on the LLVIP test set: overall mAP@0.5 /
    mAP@0.5:0.95, the TRC-binned degraded column (trc < 0.5), and the
    stress-corrupted set (blur/saturate injected into thermal). Results land
    in ``runs/ablation/results.md``, alongside trainable-param counts and the
    learned fusion α values (α staying near 1.0 is itself evidence the
    network finds TRC useful).

    Success criterion: row 4 ≥ row 3 overall, and row 4 > row 3 by a clear
    margin in the trc<0.5 / stress columns. On clean LLVIP night frames the
    two may tie within noise — expected, report honestly (§7 note).

Usage:
    python training/run_ablation.py                      # all four variants
    python training/run_ablation.py --variants fusion_trc fusion_no_trc
    python training/run_ablation.py --skip-train         # score existing best.pt
    python training/run_ablation.py --epochs 2           # quick smoke pass
    python training/run_ablation.py --subset 1000 --epochs 5   # fast comparative
                                                                 # pass on a real
                                                                 # data slice, full
                                                                 # test-set scoring
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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

from training.evaluate import (                                  # noqa: E402
    evaluate,
    install_stress_corruption,
    load_checkpoint_model,
)
from training.train_fusion import build_subset_loaders, fit, resolve  # noqa: E402

VARIANTS = ("rgb", "thermal", "fusion_no_trc", "fusion_trc")
ABLATION_DIR = "weights/ablation"


def variant_cfg(name: str, cfg: dict) -> dict:
    """Per-variant config delta (§7) — everything else stays identical."""
    vcfg = copy.deepcopy(cfg)
    if name == "fusion_no_trc":
        vcfg.setdefault("fusion", {}).setdefault("trc_gate", {})["enabled"] = False
    return vcfg


def apply_variant_freeze(name: str, model, cfg: dict) -> None:
    """Apply the variant's trainable/frozen schedule.

    Fusion variants get the Phase-4 Stage-1 freeze; the baselines fix theirs in
    ``SingleModalDetector.__init__`` (RGB full fine-tune, thermal stem+backbone)
    and need nothing here.
    """
    if name in ("fusion_trc", "fusion_no_trc"):
        from models.fusion.freeze import apply_stage
        apply_stage(model, cfg, stage=1)


def build_variant(name: str, cfg: dict, device: str):
    """Instantiate one ablation variant (fresh weights, shared seed)."""
    torch.manual_seed(int(cfg["datasets"].get("seed", 42)))
    if name in ("fusion_trc", "fusion_no_trc"):
        from models.detector.fusion_detector import FusionDetector
        model = FusionDetector(cfg).to(device)
    else:
        from training.baselines import SingleModalDetector
        modality = "visible" if name == "rgb" else "thermal"
        model = SingleModalDetector(cfg, modality=modality).to(device)
    apply_variant_freeze(name, model, cfg)
    print(model.param_summary())
    return model


def fusion_alphas(model) -> Optional[str]:
    """The learned per-scale gate α values, e.g. 'P3 1.02 · P4 0.97 · P5 1.11'."""
    neck = getattr(model, "fusion_neck", None)
    if neck is None:
        return None
    return " · ".join(f"{s} {float(getattr(neck, n).alpha):.3f}"
                      for s, n in zip(("P3", "P4", "P5"), ("fuse3", "fuse4", "fuse5")))


def run_variant(name: str, cfg: dict, args) -> Dict[str, object]:
    """Train (or load) one variant, then score it on test + stress test."""
    from datasets.dataloader import create_dataloaders

    print(f"\n{'=' * 70}\n=== ablation variant: {name}\n{'=' * 70}")
    vcfg = variant_cfg(name, cfg)
    out_dir = resolve(ABLATION_DIR) / name
    best_pt = out_dir / "best.pt"
    device = args.device

    row: Dict[str, object] = {"variant": name}
    bs = args.batch_size or int(vcfg["training"].get("batch_size", 16))
    if args.subset:
        train_loader, val_loader = build_subset_loaders(
            vcfg, args.subset, args.val_subset or max(20, args.subset // 4), bs)
        out_dir = resolve(ABLATION_DIR) / "poc" / name
        best_pt = out_dir / "best.pt"
        # Test/stress scoring still uses the real, full test split — only
        # training is shrunk. A small slice would make the comparison noise,
        # not signal.
        _, _, test_loader = create_dataloaders(vcfg, batch_size=bs)
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            vcfg, batch_size=args.batch_size)
    assert test_loader is not None, "test dataloader unavailable."

    if args.skip_train and best_pt.is_file():
        print(f"  [OK] --skip-train: scoring existing {best_pt}")
        model, vcfg = load_checkpoint_model(str(best_pt), vcfg, device)
        # The freeze state is not stored in the checkpoint, so re-apply the
        # variant's schedule before counting — otherwise every parameter reads
        # as trainable and the budget column would be wrong.
        apply_variant_freeze(name, model, vcfg)
        row["trainable_params"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = build_variant(name, vcfg, device)
        assert train_loader is not None and val_loader is not None, (
            "train/val dataloaders unavailable — run the Phase-2 dataset scripts first.")
        tb_dir = resolve(vcfg["training"].get("log_dir", "runs/")) / "ablation" / (
            name + "-" + time.strftime("%Y%m%d-%H%M%S"))
        sched = vcfg.get("fusion", {}).get("train_schedule", {})
        summary = fit(model, vcfg, train_loader, val_loader,
                      epochs=args.epochs or int(sched.get("stage1_epochs", 20)),
                      lr=float(vcfg["training"].get("learning_rate", 1e-3)),
                      device=device, out_dir=out_dir, tb_dir=tb_dir, variant=name)
        row["trainable_params"] = summary["trainable_params"]
        # Score the BEST epoch, not the last one.
        del model
        model, vcfg = load_checkpoint_model(summary["best_path"], vcfg, device)

    row["alphas"] = fusion_alphas(model)

    # Clean test set: overall + TRC-binned mAP.
    print(f"\n--- {name}: clean test set ---")
    m = evaluate(model, test_loader, vcfg, device=device)
    row.update(mAP50=m["mAP50"], mAP50_95=m["mAP50_95"],
               mAP50_degraded=m.get("mAP50(degraded)", float("nan")),
               n_degraded=m.get("n(degraded)", 0), trc_mean=m["trc_mean"])

    # Stress-corrupted test set (blur/saturate injected before preprocessing,
    # so TRC drops accordingly) — where the adaptive gate must show its margin.
    # A *fresh* loader: the clean pass above already started this one's
    # persistent workers, which hold a pre-patch copy of the dataset.
    _, _, stress_loader = create_dataloaders(vcfg, batch_size=args.batch_size)
    install_stress_corruption(stress_loader)
    print(f"--- {name}: stress-corrupted test set ---")
    st = evaluate(model, stress_loader, vcfg, device=device)
    row.update(mAP50_stress=st["mAP50"],
               mAP50_stress_degraded=st.get("mAP50(degraded)", float("nan")),
               n_stress_degraded=st.get("n(degraded)", 0),
               trc_mean_stress=st["trc_mean"])

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return row


LABELS = {"rgb": "RGB-only", "thermal": "Thermal-only",
          "fusion_no_trc": "Fusion (no TRC)", "fusion_trc": "Fusion (TRC)"}


def write_results(rows, out_path: Path) -> None:
    """Write the §7 results table (plus provenance columns) as markdown.

    Two mAP blocks per variant: the clean LLVIP test set and the stress-
    corrupted one. The `trc<0.5` column is reported for both — on clean LLVIP
    that bin is empty (measured TRC range 0.70-0.77, all "clean" bin), so the
    degraded evidence lives entirely in the stress block. That is stated in the
    notes rather than hidden behind an em dash.
    """
    def f(v):
        return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.4f}"

    def n(v):
        return f"{v:,}" if isinstance(v, int) else str(v)

    lines = [
        "# Phase 5 — Four-Way TRC Ablation Results",
        "",
        f"Generated by `training/run_ablation.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "## Clean LLVIP test set",
        "",
        "| Variant | mAP@0.5 | mAP@0.5:0.95 | mAP@0.5 (trc<0.5) | trainable params | fusion α (P3·P4·P5) |",
        "|---------|---------|--------------|--------------------|------------------|----------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {LABELS.get(r['variant'], r['variant'])} "
            f"| {f(r.get('mAP50'))} | {f(r.get('mAP50_95'))} "
            f"| {f(r.get('mAP50_degraded'))} "
            f"| {n(r.get('trainable_params', '—'))} "
            f"| {r.get('alphas') or '—'} |")

    lines += [
        "",
        "## Stress-corrupted test set (blur / saturate injected into thermal)",
        "",
        "| Variant | mAP@0.5 | mAP@0.5 (trc<0.5) | degraded frames |",
        "|---------|---------|--------------------|-----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {LABELS.get(r['variant'], r['variant'])} "
            f"| {f(r.get('mAP50_stress'))} "
            f"| {f(r.get('mAP50_stress_degraded'))} "
            f"| {r.get('n_stress_degraded', 0)} |")

    trc_clean = rows[0].get("trc_mean", float("nan")) if rows else float("nan")
    trc_stress = rows[0].get("trc_mean_stress", float("nan")) if rows else float("nan")
    lines += [
        "",
        "## Notes",
        "",
        f"- Mean TRC: {trc_clean:.3f} clean → {trc_stress:.3f} stressed "
        f"(identical corrupted set for every variant — the corruption is keyed",
        "  deterministically off the file name).",
        f"- Degraded frames (trc<0.5) on the CLEAN set: "
        f"{', '.join(str(r.get('n_degraded', 0)) for r in rows)} per row. LLVIP is a",
        "  night dataset whose thermal is uniformly reliable (measured TRC range",
        "  0.70–0.77), so this bin is expected to be empty — the degraded evidence",
        "  comes from the stress block above.",
        "- Success criterion: Fusion (TRC) ≥ Fusion (no TRC) overall, with a clear",
        "  margin in the stress / trc<0.5 columns. Near-ties on the clean columns",
        "  are expected (§7 note) and are reported as measured.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] results table → {out_path}")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 four-way TRC ablation")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS),
                    choices=list(VARIANTS), help="Subset of variants to run.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override the per-variant epoch budget (default: stage1_epochs).")
    ap.add_argument("--subset", type=int, default=None, metavar="N",
                    help="Train each variant on N real train-split images "
                         "(held-out val subset too) instead of the full set — "
                         "for a fast comparative accuracy pass across variants "
                         "before committing to the full run. Test/stress scoring "
                         "still uses the full test split.")
    ap.add_argument("--val-subset", type=int, default=None, metavar="N",
                    help="Val image count for --subset (default: max(20, N//4)).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training when a variant's best.pt already exists.")
    ap.add_argument("--out", default="runs/ablation/results.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = [run_variant(name, cfg, args) for name in args.variants]
    write_results(rows, resolve(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
