#!/usr/bin/env python3
"""
TriNetra-AMRF — Single-Modality Baselines (Phase 5.5)
=====================================================

Purpose (Phase 5 — the ablation floors):
    The two single-modality baselines the four-way ablation compares against
    (PHASE5_IMPLEMENTATION.md §6):

      * **RGB-only**     — the stock COCO-pretrained YOLOv8 run on the visible
        stream and fine-tuned on LLVIP (full model trainable by default).
      * **Thermal-only** — the same model with its 3-ch stem swapped for the
        Phase-4 ``ThermalStem`` (mean3ch warm start — identical to the fusion
        detector's thermal branch); stem + backbone train, neck + head stay
        frozen to mirror the fusion run's Stage-1 budget.

    Both wrap a plain ultralytics ``DetectionModel`` in the *same* call
    signature as ``FusionDetector`` (``model(visible, thermal, trc)``), so the
    Phase-5 ``fit()`` loop, loss adapter, and evaluator run unchanged —
    same data, same schedule, same budget: apples-to-apples.

Usage:
    python training/baselines.py --modality rgb
    python training/baselines.py --modality thermal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
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

from models.fusion.backbone import BACKBONE_END, load_yolo_model  # noqa: E402
from models.fusion.freeze import param_counts, refreeze_bn        # noqa: E402
from models.fusion.thermal_stem import ThermalStem                # noqa: E402


class SingleModalDetector(nn.Module):
    """One-stream YOLOv8 with the ``FusionDetector`` call signature.

    ``modality='visible'`` feeds the RGB tensor to the stock pretrained model;
    ``modality='thermal'`` swaps the stem for a 1-channel ``ThermalStem``
    (mean3ch warm start) and feeds the thermal tensor. The unused modality and
    ``trc`` are accepted and ignored so the shared training/eval code needs no
    branching.
    """

    def __init__(self, cfg: dict, modality: str = "visible",
                 freeze: Optional[Iterable[str]] = None):
        """
        Args:
            cfg:      Full parsed config (reads ``fusion.backbone.base_model``
                      and ``fusion.backbone.thermal_stem_init``).
            modality: 'visible' | 'thermal'.
            freeze:   Module groups to freeze: any of {'backbone', 'neck',
                      'head'}. Default: none for visible (full fine-tune),
                      ('neck', 'head') for thermal (mirror the fusion run's
                      Stage-1 budget).
        """
        super().__init__()
        if modality not in ("visible", "thermal"):
            raise ValueError(f"Unknown modality: {modality!r}")
        self.modality = modality
        fcfg = cfg.get("fusion", {}).get("backbone", {})

        self.det = load_yolo_model(fcfg.get("base_model", "weights/yolov8m.pt"))
        if modality == "thermal":
            stem_init = str(fcfg.get("thermal_stem_init", "mean3ch"))
            original = self.det.model[0]
            stem = ThermalStem(original, init=stem_init)
            # Unlike the Phase-4 DualBackbone (which walks layers by index), the
            # stock DetectionModel.forward reads the from-index wiring attributes
            # off every layer, so the replacement has to carry them.
            for attr in ("i", "f", "type", "np"):
                if hasattr(original, attr):
                    setattr(stem, attr, getattr(original, attr))
            self.det.model[0] = stem

        # Loss/eval plumbing: the loss shim reads `.head`; keep names/stride
        # like FusionDetector does.
        self.head = self.det.model[-1]
        self.names = getattr(self.det, "names", None)
        self.stride = getattr(self.det, "stride", None)

        # `YOLO(...).model` comes off the checkpoint with requires_grad=False on
        # every parameter (ultralytics re-enables grads inside its own trainer).
        # Start from all-trainable so the freeze set below is the only thing
        # that decides the budget — otherwise the RGB baseline, whose default
        # freeze set is empty, would train nothing at all.
        for prm in self.det.parameters():
            prm.requires_grad = True

        if freeze is None:
            freeze = () if modality == "visible" else ("neck", "head")
        self._freeze_groups(freeze)

    # ── freeze handling ──────────────────────────────────────────────────────
    def _group_modules(self, group: str):
        full = self.det.model
        if group == "backbone":
            return [full[i] for i in range(BACKBONE_END)]
        if group == "neck":
            return [full[i] for i in range(BACKBONE_END, len(full) - 1)]
        if group == "head":
            return [full[-1]]
        raise KeyError(f"Unknown freeze group {group!r} (backbone|neck|head)")

    def _freeze_groups(self, groups: Iterable[str]) -> None:
        self.frozen_groups = tuple(groups)
        for g in self.frozen_groups:
            for mod in self._group_modules(g):
                for p in mod.parameters():
                    p.requires_grad = False
                # Mark for refreeze_bn so frozen BN stats never drift.
                mod._trinetra_frozen = True  # type: ignore[attr-defined]
                for m in mod.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.eval()

    def train(self, mode: bool = True):
        """Standard ``train()`` but keeps frozen-module BatchNorms in eval."""
        super().train(mode)
        if mode:
            refreeze_bn(self)
        return self

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, visible: torch.Tensor, thermal: Optional[torch.Tensor] = None,
                trc=None):
        """Run the selected stream; the other modality and ``trc`` are ignored.

        Returns the stock DetectionModel output — training mode: the head's
        ``{boxes, scores, feats}`` dict; eval mode: ``(decoded, raw)``.
        """
        x = visible if self.modality == "visible" else thermal
        assert x is not None, f"{self.modality} tensor missing from the batch"
        return self.det(x)

    def param_summary(self) -> str:
        trainable, frozen = param_counts(self)
        total = trainable + frozen
        return (f"SingleModalDetector({self.modality}) parameters: {total:,} total — "
                f"{trainable:,} trainable ({100 * trainable / total:.1f}%), "
                f"{frozen:,} frozen (frozen groups: {self.frozen_groups or 'none'})")


# ── training entry (reuses the Phase-5 fit loop) ─────────────────────────────
def train_baseline(cfg: dict, modality: str, epochs: Optional[int] = None,
                   batch_size: Optional[int] = None, device: str = "cuda",
                   freeze: Optional[Iterable[str]] = None,
                   out_dir: Optional[Path] = None) -> dict:
    """Train one baseline under the exact fusion Stage-1 schedule (§6)."""
    from datasets.dataloader import create_dataloaders
    from training.train_fusion import fit, resolve

    tcfg = cfg.get("training", {})
    sched_cfg = cfg.get("fusion", {}).get("train_schedule", {})
    variant = "rgb" if modality == "visible" else "thermal"
    out_dir = out_dir or resolve("weights/baselines") / variant
    tb_dir = resolve(tcfg.get("log_dir", "runs/")) / "baselines" / (
        variant + "-" + time.strftime("%Y%m%d-%H%M%S"))

    torch.manual_seed(int(cfg["datasets"].get("seed", 42)))
    model = SingleModalDetector(cfg, modality=modality, freeze=freeze).to(device)
    print(model.param_summary())

    train_loader, val_loader, _ = create_dataloaders(cfg, batch_size=batch_size)
    assert train_loader is not None and val_loader is not None, (
        "train/val dataloaders unavailable — run the Phase-2 dataset scripts first.")

    return fit(model, cfg, train_loader, val_loader,
               epochs=epochs or int(sched_cfg.get("stage1_epochs", 20)),
               lr=float(tcfg.get("learning_rate", 1e-3)),
               device=device, out_dir=out_dir, tb_dir=tb_dir, variant=variant)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 single-modality baselines")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--modality", required=True, choices=["rgb", "thermal"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--freeze", nargs="*", default=None,
                    choices=["backbone", "neck", "head"],
                    help="Override the default freeze set for this modality.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    modality = "visible" if args.modality == "rgb" else "thermal"
    summary = train_baseline(cfg, modality, epochs=args.epochs,
                             batch_size=args.batch_size, device=args.device,
                             freeze=args.freeze)
    print(f"\n[DONE] variant={summary['variant']} best mAP50="
          f"{summary['best_map50']:.4f} → {summary['best_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
