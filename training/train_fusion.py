#!/usr/bin/env python3
"""
TriNetra-AMRF — Fusion Detector Training (Phase 5.3)
====================================================

Purpose (Phase 5 — Layer 4: Detection Integration & Training):
    Two-stage training loop for the Phase-4 ``FusionDetector``:

      Stage 1 — freeze pretrained (visible backbone, neck, head); train the
                thermal stem/backbone + TRC-gated fusion blocks.
      Stage 2 — (optional) unfreeze neck + head at low LR with linear warmup,
                resuming from the best Stage-1 checkpoint.

    Engineering: AMP (fp16 autocast + GradScaler), gradient clipping,
    trainable-params-only AdamW (rebuilt at the stage boundary), cosine LR,
    per-epoch COCO mAP validation, early stopping, last/best checkpointing,
    TensorBoard logging (box/cls/dfl, val mAP, LR, and the learned fusion α
    values — is the network obeying TRC?).

    The generic ``fit()`` loop is architecture-agnostic — the Phase-5
    baselines (``training/baselines.py``) and the ablation runner reuse it so
    every variant trains under the identical schedule.

Usage:
    python training/train_fusion.py --stage 1                  # Stage 1
    python training/train_fusion.py --stage 2                  # fine-tune from best.pt
    python training/train_fusion.py --overfit 32               # sanity: 32-image overfit
    python training/train_fusion.py --stage 1 --epochs 2       # short smoke run
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

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

from models.detector.fusion_detector import FusionDetector      # noqa: E402
from models.fusion.freeze import apply_stage, param_counts      # noqa: E402
from training.evaluate import evaluate                          # noqa: E402
from training.loss_adapter import (                             # noqa: E402
    build_loss,
    get_class_id_map,
    targets_to_batch,
)


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def git_commit() -> Optional[str]:
    """Current git commit hash (for checkpoint provenance), or None."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def trainable_params(model: torch.nn.Module):
    """Only params with requires_grad — never rely on zero grads (§4.3)."""
    return [p for p in model.parameters() if p.requires_grad]


# ── startup self-checks (§4.3) ───────────────────────────────────────────────
def startup_checks(model, loss_fn, loader, cfg: dict, device: str) -> None:
    """Re-assert the freeze contract *after* loss/optimizer wiring, and that
    one real batch produces a finite loss with grads reaching only trainable
    params (the Phase-4 ``check_grad_flow`` contract, generalized to any stage).
    """
    print("\n=== startup self-checks ===")
    class_map = get_class_id_map(cfg)
    visible, thermal, targets = next(iter(loader))
    trc = FusionDetector.trc_from_targets(targets, device=device)
    batch = targets_to_batch(targets, visible.shape[-2:], class_map)

    model.train()
    model.zero_grad(set_to_none=True)
    preds = model(visible.to(device), thermal.to(device), trc)
    loss, items = loss_fn(preds, batch)
    total = loss.sum()
    assert torch.isfinite(total), f"non-finite loss at startup: {items}"
    print(f"  [OK] one-batch loss finite: box={items[0]:.4f} "
          f"cls={items[1]:.4f} dfl={items[2]:.4f}")

    total.backward()
    frozen_hit = [n for n, p in model.named_parameters()
                  if not p.requires_grad and p.grad is not None]
    assert not frozen_hit, f"frozen params received gradients: {frozen_hit[:5]}"
    got = any(p.grad is not None and p.grad.abs().sum() > 0
              for p in trainable_params(model))
    assert got, "no gradients reached any trainable parameter!"
    model.zero_grad(set_to_none=True)
    t, f = param_counts(model)
    print(f"  [OK] freeze contract holds: {t:,} trainable / {f:,} frozen")
    print("  [PASS] startup self-checks passed.")


# ── checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(path: Path, model, opt, epoch: int, stage: int,
                    best_map: float, cfg: dict, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "best_map": best_map,
        "cfg_snapshot": cfg,
        "variant": variant,
        "seed": int(cfg.get("datasets", {}).get("seed", 42)),
        "git_commit": git_commit(),
    }, path)


# ── the generic training loop ────────────────────────────────────────────────
def fit(
    model: torch.nn.Module,
    cfg: dict,
    train_loader,
    val_loader,
    *,
    epochs: int,
    lr: float,
    device: str,
    out_dir: Path,
    tb_dir: Optional[Path] = None,
    variant: str = "fusion",
    stage: int = 1,
    warmup_epochs: int = 0,
    resume: Optional[str] = None,
    early_stop_patience: Optional[int] = None,
    log_every: int = 20,
) -> dict:
    """Train ``model`` for one stage and return the best-checkpoint summary.

    The loop is shared by all Phase-5 variants (fusion + baselines): AdamW on
    trainable params only, cosine LR (optional linear warmup), AMP, grad
    clipping, per-epoch val mAP via COCOeval, early stopping on mAP@0.5,
    ``last.pt``/``best.pt`` checkpoints and TensorBoard scalars.
    """
    tcfg = cfg.get("training", {})
    class_map = get_class_id_map(cfg)
    amp = bool(tcfg.get("mixed_precision", True)) and device.startswith("cuda")
    clip = float(tcfg.get("grad_clip_norm", 10.0))
    patience = (int(tcfg.get("early_stopping_patience", 15))
                if early_stop_patience is None else early_stop_patience)

    loss_fn = build_loss(model, cfg)
    startup_checks(model, loss_fn, train_loader, cfg, device)

    opt = torch.optim.AdamW(trainable_params(model), lr=lr,
                            weight_decay=float(tcfg.get("weight_decay", 5e-4)))
    if warmup_epochs > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=warmup_epochs),
             torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=max(1, epochs - warmup_epochs))],
            milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(enabled=amp)

    start_epoch, best_map = 0, -1.0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_map = float(ckpt.get("best_map", -1.0))
        # The LR schedule is a pure function of the epoch, so replay it rather
        # than storing it — otherwise a resumed run restarts the cosine at its
        # peak LR and undoes the decay already earned.
        for _ in range(start_epoch):
            sched.step()
        print(f"  [OK] resumed {resume} at epoch {start_epoch} "
              f"(best mAP50 {best_map:.4f}, lr {opt.param_groups[0]['lr']:.2e})")

    writer = None
    if tb_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(tb_dir))
        except Exception as e:  # pragma: no cover
            print(f"  [WARN] TensorBoard unavailable ({e}); scalar logging disabled.")

    n_train, n_frozen = param_counts(model)
    print(f"\n=== fit: variant={variant} stage={stage} epochs={epochs} lr={lr:g} "
          f"amp={amp} ({n_train:,} trainable / {n_frozen:,} frozen) ===")

    bad_epochs, step = 0, start_epoch * len(train_loader)
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, running = time.time(), torch.zeros(3)
        for it, (visible, thermal, targets) in enumerate(train_loader):
            trc = FusionDetector.trc_from_targets(targets, device=device)
            batch = targets_to_batch(targets, visible.shape[-2:], class_map)
            with torch.amp.autocast("cuda", enabled=amp):
                preds = model(visible.to(device, non_blocking=True),
                              thermal.to(device, non_blocking=True), trc)
                loss, items = loss_fn(preds, batch)
            scaler.scale(loss.sum()).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(trainable_params(model), clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            running += items.cpu()
            step += 1
            if writer and step % log_every == 0:
                for k, v in zip(("box", "cls", "dfl"), items.tolist()):
                    writer.add_scalar(f"train/loss_{k}", v, step)
            if it % max(1, len(train_loader) // 5) == 0:
                print(f"  e{epoch:03d} it{it:04d}/{len(train_loader)}  "
                      f"box={items[0]:.4f} cls={items[1]:.4f} dfl={items[2]:.4f}")

        sched.step()
        mean = (running / max(1, len(train_loader))).tolist()
        cur_lr = opt.param_groups[0]["lr"]

        metrics = evaluate(model, val_loader, cfg, device=device, trc_bins=False)
        map50, map5095 = metrics["mAP50"], metrics["mAP50_95"]
        if math.isnan(map50):
            map50 = 0.0
        print(f"  epoch {epoch:03d} done in {time.time() - t0:.1f}s — "
              f"loss(box/cls/dfl)={mean[0]:.4f}/{mean[1]:.4f}/{mean[2]:.4f}  "
              f"val mAP50={map50:.4f} mAP50-95={map5095:.4f}  lr={cur_lr:.2e}")

        if writer:
            writer.add_scalar("val/mAP50", map50, epoch)
            writer.add_scalar("val/mAP50_95", map5095, epoch)
            writer.add_scalar("train/lr", cur_lr, epoch)
            # Interpretability: the learned α of each fusion block (is the
            # network obeying TRC?). Baselines have no fusion_neck — skip.
            neck = getattr(model, "fusion_neck", None)
            if neck is not None:
                for name in ("fuse3", "fuse4", "fuse5"):
                    writer.add_scalar(f"fusion/alpha_{name}",
                                      float(getattr(neck, name).alpha), epoch)

        save_checkpoint(last_path, model, opt, epoch, stage, best_map, cfg, variant)
        if map50 > best_map:
            best_map, bad_epochs = map50, 0
            save_checkpoint(best_path, model, opt, epoch, stage, best_map, cfg, variant)
            print(f"  [OK] new best mAP50={best_map:.4f} → {best_path}")
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"  [STOP] early stopping: no val mAP50 improvement "
                      f"in {patience} epochs.")
                break

    if writer:
        writer.close()
    return {"best_path": str(best_path), "last_path": str(last_path),
            "best_map50": best_map, "variant": variant, "stage": stage,
            "trainable_params": n_train}


# ── overfit-sanity loaders (§10 check 3) ─────────────────────────────────────
def build_overfit_loaders(cfg: dict, n_images: int, batch_size: int):
    """(train, val) loaders over the SAME ``n_images`` training images, with
    the deterministic val transform (no augmentation) so loss can actually
    reach ~0 and mAP ~1 on the memorized set.
    """
    from datasets.augmentations import build_transforms
    from datasets.dataloader import (
        build_preprocess_pipeline,
        dual_modal_collate_fn,
    )
    from datasets.dual_modal_dataset import DualModalDataset
    from torch.utils.data import DataLoader

    ann = resolve(cfg["datasets"]["annotations_dir"]) / "train.json"
    ds = DualModalDataset(
        visible_dir=str(resolve(cfg["datasets"]["visible_dir"]) / "train"),
        thermal_dir=str(resolve(cfg["datasets"]["thermal_dir"]) / "train"),
        annotations_json=str(ann),
        transform=build_transforms(cfg, "val"),   # deterministic resize+normalize
        preprocess=build_preprocess_pipeline(cfg),
    )
    ds.image_ids = ds.image_ids[:n_images]
    print(f"  [OK] overfit set: {len(ds)} images from train split.")
    mk = lambda shuf: DataLoader(ds, batch_size=min(batch_size, n_images),
                                 shuffle=shuf, num_workers=0,
                                 collate_fn=dual_modal_collate_fn)
    return mk(True), mk(False)


# ── stage orchestration (§4.2) ───────────────────────────────────────────────
def train(cfg: dict, stage: int = 1, epochs: Optional[int] = None,
          resume: Optional[str] = None, overfit: Optional[int] = None,
          batch_size: Optional[int] = None, device: str = "cuda") -> dict:
    """Run one training stage of the fusion detector (or the overfit sanity)."""
    tcfg = cfg.get("training", {})
    sched_cfg = cfg.get("fusion", {}).get("train_schedule", {})
    out_dir = resolve(tcfg.get("checkpoint_dir", "weights/fusion/"))
    tb_dir = resolve(tcfg.get("log_dir", "runs/")) / "fusion" / time.strftime("%Y%m%d-%H%M%S")

    torch.manual_seed(int(cfg["datasets"].get("seed", 42)))

    model = FusionDetector(cfg).to(device)
    apply_stage(model, cfg, stage=1)          # Stage-1 freeze is the baseline state
    if stage == 2:
        # Stage 2 resumes from the best Stage-1 weights, then unfreezes
        # neck/head per fusion.train_schedule.stage2_unfreeze.
        init = resume or str(out_dir / "best.pt")
        ckpt = torch.load(init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  [OK] Stage 2 init from {init} (mAP50 {ckpt.get('best_map', float('nan')):.4f})")
        apply_stage(model, cfg, stage=2)
        resume = None                          # fresh optimizer for the new trainable set
    print(model.param_summary())

    if overfit:
        train_loader, val_loader = build_overfit_loaders(
            cfg, overfit, batch_size or int(tcfg.get("batch_size", 16)))
        n_epochs = epochs or 50
        return fit(model, cfg, train_loader, val_loader,
                   epochs=n_epochs, lr=float(tcfg.get("learning_rate", 1e-3)),
                   device=device, out_dir=out_dir / "overfit",
                   tb_dir=tb_dir, variant="fusion", stage=stage,
                   early_stop_patience=0)      # never stop early when memorizing

    from datasets.dataloader import create_dataloaders
    train_loader, val_loader, _ = create_dataloaders(cfg, batch_size=batch_size)
    assert train_loader is not None and val_loader is not None, (
        "train/val dataloaders unavailable — run the Phase-2 dataset scripts first.")

    if stage == 1:
        lr = float(tcfg.get("learning_rate", 1e-3))
        n_epochs = epochs or int(sched_cfg.get("stage1_epochs", 20))
        warmup = 0
    else:
        lr = float(sched_cfg.get("stage2_lr", 1e-4))
        n_epochs = epochs or int(tcfg.get("epochs", 100))
        warmup = int(tcfg.get("warmup_epochs", 3))

    return fit(model, cfg, train_loader, val_loader,
               epochs=n_epochs, lr=lr, device=device,
               out_dir=out_dir, tb_dir=tb_dir, variant="fusion",
               stage=stage, warmup_epochs=warmup, resume=resume)


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 fusion detector training")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override the stage's epoch count.")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to resume (stage 1) or init from (stage 2).")
    ap.add_argument("--overfit", type=int, default=None, metavar="N",
                    help="Sanity mode: memorize N train images (default 50 epochs).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = train(cfg, stage=args.stage, epochs=args.epochs, resume=args.resume,
                    overfit=args.overfit, batch_size=args.batch_size,
                    device=args.device)
    print(f"\n[DONE] variant={summary['variant']} stage={summary['stage']} "
          f"best mAP50={summary['best_map50']:.4f} → {summary['best_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
