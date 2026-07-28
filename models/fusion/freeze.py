#!/usr/bin/env python3
"""
TriNetra-AMRF — Freeze/Unfreeze Schedule (Phase 4.7)
====================================================

Purpose (Phase 4 — transfer-learning schedule):
    Drive the two-stage training strategy from PHASE4_IMPLEMENTATION.md §2:

        Stage 1 — freeze pretrained (visible backbone, neck, head), train the
                  new machinery (thermal stem/backbone, fusion blocks + TRC gate).
        Stage 2 — optionally unfreeze neck/head (or everything) at low LR.

    Important: frozen BatchNorm layers are put in ``.eval()`` so their running
    stats don't drift while the rest of the model trains (``freeze_modules``
    installs a hook that re-applies this after every ``model.train()`` call).

Module names understood by the helpers (attributes of ``FusionDetector``):
    visible_backbone · thermal_backbone · fusion · neck · head
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch.nn as nn

# name -> FusionDetector attribute path
_MODULE_MAP = {
    "visible_backbone": "backbone.visible",
    "thermal_backbone": "backbone.thermal",
    "fusion": "fusion_neck",
    "neck": "neck",
    "head": "head",
}


def _resolve(model: nn.Module, name: str) -> nn.Module:
    """Look up a named sub-module of the detector (see ``_MODULE_MAP``)."""
    path = _MODULE_MAP.get(name)
    if path is None:
        raise KeyError(f"Unknown module name {name!r}; expected one of {sorted(_MODULE_MAP)}")
    mod = model
    for attr in path.split("."):
        mod = getattr(mod, attr)
    return mod


def _set_bn_eval(module: nn.Module) -> None:
    """Put every BatchNorm in ``module`` into eval mode (freeze running stats)."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def freeze_modules(model: nn.Module, names: Iterable[str]) -> None:
    """Freeze the named sub-modules: ``requires_grad=False`` + BN in eval.

    Marks each module with ``_trinetra_frozen`` so ``refreeze_bn`` can
    re-apply BN eval after a global ``model.train()``.
    """
    for name in names:
        mod = _resolve(model, name)
        for p in mod.parameters():
            p.requires_grad = False
        mod._trinetra_frozen = True  # type: ignore[attr-defined]
        _set_bn_eval(mod)


def unfreeze_modules(model: nn.Module, names: Iterable[str]) -> None:
    """Unfreeze the named sub-modules (grad on; BN returns to train-mode flow)."""
    for name in names:
        mod = _resolve(model, name)
        for p in mod.parameters():
            p.requires_grad = True
        mod._trinetra_frozen = False  # type: ignore[attr-defined]


def refreeze_bn(model: nn.Module) -> None:
    """Re-apply BN eval on frozen modules. Call after every ``model.train()``."""
    for mod in model.modules():
        if getattr(mod, "_trinetra_frozen", False):
            _set_bn_eval(mod)


def apply_stage(model: nn.Module, cfg: dict, stage: int) -> None:
    """Drive the 2-stage schedule from ``fusion.train_schedule``.

    Stage 1: freeze ``stage1_freeze`` (everything else trains).
    Stage 2: additionally unfreeze ``stage2_unfreeze`` ([] = skip stage 2).
    """
    sched = cfg.get("fusion", {}).get("train_schedule", {})
    stage1_freeze = list(sched.get("stage1_freeze",
                                   ["visible_backbone", "neck", "head"]))
    if stage == 1:
        # Start from all-trainable, then freeze the pretrained parts.
        unfreeze_modules(model, _MODULE_MAP.keys())
        freeze_modules(model, stage1_freeze)
    elif stage == 2:
        unfreeze_modules(model, sched.get("stage2_unfreeze", []))
    else:
        raise ValueError(f"Unknown stage: {stage} (expected 1 or 2)")


def param_counts(model: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable, frozen)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def describe_freeze_state(model: nn.Module) -> List[str]:
    """Human-readable per-module trainable/frozen summary (for logging)."""
    lines = []
    for name in _MODULE_MAP:
        mod = _resolve(model, name)
        t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        f = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
        state = "trainable" if t and not f else ("frozen" if f and not t else "mixed")
        lines.append(f"{name:<17s} {state:>9s}  ({t:,} trainable / {f:,} frozen)")
    return lines
