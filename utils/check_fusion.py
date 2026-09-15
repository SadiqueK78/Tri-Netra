#!/usr/bin/env python3
"""
TriNetra-AMRF — Phase 4 Fusion Validation & Tests (Phase 4.8)
=============================================================

Sanity-check the TRC-gated fusion detector before Phase 5 training:

  * ``check_shapes``     — one dummy batch; assert the fused P3/P4/P5 shapes
                           exactly match the visible backbone taps (what the
                           frozen pretrained neck expects), and that the Detect
                           head produces per-scale outputs.
  * ``check_trc_effect`` — run the SAME input with trc=1 vs trc=0; the outputs
                           MUST differ (thermal suppressed at 0). If they are
                           identical the gate isn't wired — the novelty test.
  * ``check_grad_flow``  — one backward pass in Stage-1 freeze state; assert
                           gradients reach ONLY the intended modules (thermal
                           stream + fusion) and never the frozen pretrained
                           visible backbone / neck / head.
  * ``count_params``     — print trainable vs frozen parameter counts.
  * ``check_dataloader`` — pull one real batch from the Phase 3 val/test loader
                           and run it end-to-end through the detector.
  * ``check_loss``       — Phase 5 smoke check: wire ``v8DetectionLoss`` via
                           training/loss_adapter, run one batch (real if
                           ``--split`` given, else synthetic), assert the loss
                           is finite and the Stage-1 freeze contract still
                           holds after loss wiring.

Usage:
    python utils/check_fusion.py                       # all model-only tests
    python utils/check_fusion.py --test trc_effect
    python utils/check_fusion.py --test grad_flow
    python utils/check_fusion.py --test loss --split val       # Phase 5 §10.2
    python utils/check_fusion.py --split val --batch-size 2   # + real batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Windows consoles default to cp1252 and choke on box-drawing/emoji; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detector.fusion_detector import FusionDetector      # noqa: E402
from models.fusion.freeze import param_counts                   # noqa: E402


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dummy_batch(cfg: dict, batch_size: int = 2, device: str = "cpu"):
    """Random (visible, thermal, trc) batch at the configured input size."""
    w, h = cfg["datasets"].get("image_size", [640, 512])
    g = torch.Generator().manual_seed(int(cfg["datasets"].get("seed", 42)))
    visible = torch.rand(batch_size, 3, h, w, generator=g).to(device)
    thermal = torch.rand(batch_size, 1, h, w, generator=g).to(device)
    trc = torch.full((batch_size,), 0.8).to(device)
    return visible, thermal, trc


def _head_tensors(out):
    """Normalize the Detect head output to the list of per-scale tensors.

    ultralytics 8.4.x: training mode -> dict {boxes, scores, feats};
    eval mode -> (decoded [B,84,N], that same dict). Older versions return
    a plain [P3, P4, P5] list (train) / (decoded, list) tuple (eval).
    """
    if isinstance(out, (list, tuple)) and not torch.is_tensor(out):
        if all(torch.is_tensor(t) for t in out):
            return list(out)                   # legacy training mode
        out = out[1]                           # eval mode: (decoded, raw)
    if isinstance(out, dict):
        return list(out["feats"])              # 8.4.x raw per-scale features
    return list(out)


# ── 1. shapes ────────────────────────────────────────────────────────────────
def check_shapes(cfg: dict, model: FusionDetector, device: str) -> None:
    print("\n=== check_shapes ===")
    visible, thermal, trc = _dummy_batch(cfg, 2, device)

    model.eval()
    with torch.no_grad():
        vis_feats, thr_feats = model.backbone(visible, thermal)
        fused = model.fusion_neck(vis_feats, thr_feats, trc)
        # Fused features must be drop-in replacements for the visible taps.
        for k, (v, t, f) in enumerate(zip(vis_feats, thr_feats, fused)):
            scale = ("P3", "P4", "P5")[k]
            assert v.shape == t.shape == f.shape, (
                f"{scale}: vis {tuple(v.shape)} / thr {tuple(t.shape)} / "
                f"fused {tuple(f.shape)} mismatch")
            print(f"  [OK] {scale}: vis == thr == fused == {tuple(f.shape)}")

        out = model(visible, thermal, trc)
    per_scale = _head_tensors(out)
    assert len(per_scale) == 3, f"expected 3 head scales, got {len(per_scale)}"
    for scale, t in zip(("P3", "P4", "P5"), per_scale):
        print(f"  [OK] head {scale} output: {tuple(t.shape)}")
    print("  [PASS] fused features match the pretrained neck's expected shapes.")


# ── 2. TRC effect (the novelty test) ─────────────────────────────────────────
def check_trc_effect(cfg: dict, model: FusionDetector, device: str) -> None:
    print("\n=== check_trc_effect ===")
    visible, thermal, _ = _dummy_batch(cfg, 2, device)
    model.eval()
    with torch.no_grad():
        out_full = _head_tensors(model(visible, thermal, torch.ones(2, device=device)))
        out_zero = _head_tensors(model(visible, thermal, torch.zeros(2, device=device)))
    diffs = [(a - b).abs().max().item() for a, b in zip(out_full, out_zero)]
    print(f"  max |trc=1 − trc=0| per scale: "
          f"{', '.join(f'{d:.6f}' for d in diffs)}")
    assert any(d > 1e-6 for d in diffs), (
        "Outputs identical for trc=1 vs trc=0 — the TRC gate is NOT wired!")
    print("  [PASS] TRC gate changes the output (thermal suppressed at trc=0).")

    # Bonus: intermediate features must also differ at every scale.
    with torch.no_grad():
        vis_feats, thr_feats = model.backbone(visible, thermal)
        f1 = model.fusion_neck(vis_feats, thr_feats, torch.ones(2, device=device))
        f0 = model.fusion_neck(vis_feats, thr_feats, torch.zeros(2, device=device))
    for scale, a, b in zip(("P3", "P4", "P5"), f1, f0):
        d = (a - b).abs().max().item()
        assert d > 1e-6, f"fused {scale} identical for trc=1 vs trc=0"
        print(f"  [OK] fused {scale} responds to trc (max diff {d:.6f})")


# ── 3. gradient flow (Stage-1 freeze contract) ──────────────────────────────
def check_grad_flow(cfg: dict, model: FusionDetector, device: str) -> None:
    print("\n=== check_grad_flow ===")
    model.freeze_pretrained()      # Stage 1
    model.train()
    visible, thermal, trc = _dummy_batch(cfg, 2, device)

    model.zero_grad(set_to_none=True)
    out = model(visible, thermal, trc)
    loss = sum(t.float().pow(2).mean() for t in _head_tensors(out))
    loss.backward()

    groups = {
        "visible_backbone (frozen)": (model.backbone.visible, False),
        "neck (frozen)":             (model.neck, False),
        "head (frozen)":             (model.head, False),
        "thermal_backbone (train)":  (model.backbone.thermal, True),
        "fusion (train)":            (model.fusion_neck, True),
    }
    for name, (mod, expect_grad) in groups.items():
        got = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in mod.parameters())
        # Frozen params must not even receive a .grad tensor.
        if not expect_grad:
            assert all(p.grad is None for p in mod.parameters()), \
                f"{name}: frozen params received gradients!"
        else:
            assert got, f"{name}: expected gradients but none arrived!"
        print(f"  [OK] {name:<28s} grads={'yes' if got else 'no':<3s} "
              f"(expected {'yes' if expect_grad else 'no'})")
    print("  [PASS] Stage-1 gradient flow matches the freeze schedule.")


# ── 4. parameter counts ──────────────────────────────────────────────────────
def count_params(cfg: dict, model: FusionDetector) -> None:
    print("\n=== count_params (Stage 1) ===")
    model.freeze_pretrained()
    print(model.param_summary())
    trainable, frozen = param_counts(model)
    assert frozen > trainable, (
        "Stage 1 should freeze most parameters (frozen ≫ trainable).")
    print("  [PASS] frozen ≫ trainable in Stage 1.")


# ── 5. Phase 5 loss smoke check (one batch → finite loss, grads flow) ────────
def check_loss(cfg: dict, model: FusionDetector, device: str,
               split: str | None, batch_size: int) -> None:
    print(f"\n=== check_loss (split={split or 'synthetic'}) ===")
    from training.loss_adapter import build_loss, get_class_id_map, targets_to_batch, unpack_loss_items

    if split:
        from datasets.dataloader import create_dataloaders
        loaders = dict(zip(("train", "val", "test"),
                           create_dataloaders(cfg, batch_size=batch_size,
                                              num_workers=0)))
        loader = loaders.get(split)
        assert loader is not None, f"no annotations for split '{split}'"
        visible, thermal, targets = next(iter(loader))
        trc = FusionDetector.trc_from_targets(targets, device=device)
        visible, thermal = visible.to(device), thermal.to(device)
    else:
        # Synthetic batch: random images + a couple of plausible person boxes.
        visible, thermal, trc = _dummy_batch(cfg, batch_size, device)
        h, w = visible.shape[-2:]
        g = torch.Generator().manual_seed(int(cfg["datasets"].get("seed", 42)))
        targets = []
        for _ in range(visible.shape[0]):
            xy = torch.rand(3, 2, generator=g) * torch.tensor([w * 0.7, h * 0.7])
            wh = 20 + torch.rand(3, 2, generator=g) * torch.tensor([w * 0.2, h * 0.2])
            targets.append({"boxes": torch.cat([xy, wh], dim=1),
                            "labels": torch.ones(3, dtype=torch.int64)})

    model.freeze_pretrained()      # Stage 1
    model.train()
    loss_fn = build_loss(model, cfg)
    batch = targets_to_batch(targets, visible.shape[-2:], get_class_id_map(cfg))
    print(f"  batch: {visible.shape[0]} image(s), {batch['bboxes'].shape[0]} box(es)")

    model.zero_grad(set_to_none=True)
    preds = model(visible, thermal, trc)
    loss, items = loss_fn(preds, batch)
    total = loss.sum()
    assert torch.isfinite(total), f"non-finite loss: {items}"
    box, cls, dfl = unpack_loss_items(items)
    print(f"  [OK] loss finite: box={box:.4f} cls={cls:.4f} "
          f"dfl={dfl:.4f} (total {total.item():.4f})")

    total.backward()
    for name, (mod, expect_grad) in {
        "visible_backbone (frozen)": (model.backbone.visible, False),
        "neck (frozen)":             (model.neck, False),
        "head (frozen)":             (model.head, False),
        "thermal_backbone (train)":  (model.backbone.thermal, True),
        "fusion (train)":            (model.fusion_neck, True),
    }.items():
        if not expect_grad:
            assert all(p.grad is None for p in mod.parameters()), \
                f"{name}: frozen params received gradients through the loss!"
        else:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in mod.parameters()), \
                f"{name}: expected gradients but none arrived!"
        print(f"  [OK] {name:<28s} grad contract holds")
    model.zero_grad(set_to_none=True)
    print("  [PASS] v8DetectionLoss wiring: finite loss, freeze contract intact.")


# ── 6. one real batch from the Phase 3 dataloader ────────────────────────────
def check_dataloader(cfg: dict, model: FusionDetector, device: str,
                     split: str, batch_size: int) -> None:
    print(f"\n=== check_dataloader (split={split}) ===")
    from datasets.dataloader import create_dataloaders

    loaders = dict(zip(("train", "val", "test"),
                       create_dataloaders(cfg, batch_size=batch_size, num_workers=0)))
    loader = loaders.get(split)
    if loader is None:
        print(f"  [SKIP] no annotations for split '{split}'.")
        return

    visible, thermal, targets = next(iter(loader))
    trc = FusionDetector.trc_from_targets(targets, device=device)
    print(f"  batch: visible {tuple(visible.shape)}, thermal {tuple(thermal.shape)}, "
          f"trc {tuple(trc.shape)} (mean {trc.mean().item():.3f})")

    model.eval()
    with torch.no_grad():
        out = model(visible.to(device), thermal.to(device), trc)
    for scale, t in zip(("P3", "P4", "P5"), _head_tensors(out)):
        print(f"  [OK] head {scale} output: {tuple(t.shape)}")
    print("  [PASS] a real Phase-3 batch flows end-to-end through the detector.")


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 fusion validation")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--test",
                    choices=["shapes", "trc_effect", "grad_flow", "params", "loss", "all"],
                    default="all")
    ap.add_argument("--split", default=None,
                    help="Also run one real batch from this split (val/test/train).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg["datasets"].get("seed", 42)))

    print(f"Building FusionDetector "
          f"(base={cfg['fusion']['backbone']['base_model']}, "
          f"strategy={cfg['fusion']['strategy']}, device={args.device}) ...")
    model = FusionDetector(cfg).to(args.device)

    if args.test in ("shapes", "all"):
        check_shapes(cfg, model, args.device)
    if args.test in ("trc_effect", "all"):
        check_trc_effect(cfg, model, args.device)
    if args.test in ("grad_flow", "all"):
        check_grad_flow(cfg, model, args.device)
    if args.test in ("params", "all"):
        count_params(cfg, model)
    if args.test == "loss":
        # Phase 5 (§10.2) — loss/grad smoke check; uses a real batch when
        # --split is given, else a synthetic one. Not part of "all".
        check_loss(cfg, model, args.device, args.split, args.batch_size)
    elif args.split:
        check_dataloader(cfg, model, args.device, args.split, args.batch_size)

    print("\nAll requested checks passed. ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
