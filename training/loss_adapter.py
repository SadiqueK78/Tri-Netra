#!/usr/bin/env python3
"""
TriNetra-AMRF — Loss Adapter (Phase 5.2)
========================================

Purpose (Phase 5 — Layer 4: Detection Integration & Training):
    Wire ultralytics' stock ``v8DetectionLoss`` (box CIoU + BCE cls + DFL) to
    the Phase-4 ``FusionDetector`` and the Phase-2 dataloader targets. Verified
    on ultralytics 8.4.89: the Detect head's train-mode output dict
    ``{boxes, scores, feats}`` is exactly what ``v8DetectionLoss`` consumes,
    so no loss math is reimplemented — only two thin adapters:

      * ``targets_to_batch`` — per-sample target dicts (COCO **top-left**
        [x,y,w,h] pixel boxes, unified category ids) → the flat batch dict the
        loss expects (``batch_idx`` / ``cls`` / ``bboxes`` as normalized
        **center**-xywh in [0,1]).
      * ``LossModelShim`` — duck-types the three attributes
        ``v8DetectionLoss.__init__`` reads off a ``DetectionModel``
        (``.args`` hyp gains, ``.model[-1]`` Detect module, ``.parameters()``).

    Class handling is Option A (PHASE5_IMPLEMENTATION.md §2): the pretrained
    80-class COCO head stays frozen; unified ids map to COCO indices via
    ``detection.class_id_map`` (person 1 → COCO 0).

Usage:
    loss_fn = build_loss(model, cfg)                  # v8DetectionLoss, ready
    batch   = targets_to_batch(targets, visible.shape[-2:], class_map)
    loss, items = loss_fn(preds, batch)               # (box, cls, dfl)

Self-test (no dataset needed):
    python -m training.loss_adapter --self-test
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

# Windows consoles default to cp1252 and choke on arrows/emoji; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# Ultralytics loss-gain defaults (mirrored in configs/default.yaml).
DEFAULT_LOSS_GAINS = {"box": 7.5, "cls": 0.5, "dfl": 1.5}


# ── class-id mapping ─────────────────────────────────────────────────────────
def get_class_id_map(cfg: dict) -> Dict[int, int]:
    """Read ``detection.class_id_map`` (unified id → COCO head index).

    Falls back to ``{1: 0}`` (person → COCO person), the LLVIP default.
    """
    m = cfg.get("detection", {}).get("class_id_map") or {1: 0}
    return {int(k): int(v) for k, v in m.items()}


# ── loss-items compatibility shim ────────────────────────────────────────────
def unpack_loss_items(items) -> Tuple[float, float, float]:
    """Return ``(box, cls, dfl)`` floats from a ``v8DetectionLoss`` call.

    Ultralytics 8.4.89 (this project's original pin) returns ``loss_items`` as
    a plain ``[box, cls, dfl]`` tensor; 8.4.150+ returns
    ``dict(zip(self.loss_names, ...))`` with keys ``box_loss``/``cls_loss``/
    ``dfl_loss`` instead. Handle both so callers don't care which is running.
    """
    if isinstance(items, dict):
        def _get(*keys):
            for k in keys:
                if k in items:
                    return float(items[k])
            raise KeyError(f"none of {keys} found in loss items dict {list(items)}")

        return _get("box_loss", "box"), _get("cls_loss", "cls"), _get("dfl_loss", "dfl", "l1_loss")
    return float(items[0]), float(items[1]), float(items[2])


# ── target adapter ───────────────────────────────────────────────────────────
def targets_to_batch(
    targets: List[dict],
    img_hw: Tuple[int, int],
    class_id_map: Dict[int, int],
    device=None,
) -> Dict[str, torch.Tensor]:
    """Convert the dataloader's per-sample target dicts into the flat batch
    dict ``v8DetectionLoss`` expects.

    Per sample i, for each of its N_i boxes:
        batch_idx ← i                            (which image in the batch)
        cls       ← class_id_map[labels[j]]      (unified id → COCO index;
                                                  unmapped labels are dropped)
        bboxes    ← COCO top-left [x,y,w,h] px → normalized CENTER-xywh:
                    cx=(x+w/2)/W  cy=(y+h/2)/H  w/=W  h/=H

    Args:
        targets:      List of per-sample dicts with 'boxes' [N,4] / 'labels' [N].
        img_hw:       (H, W) of the *batched tensors* — the boxes were already
                      resized by the Phase-2 transform, so normalize by the
                      tensor shape, never by ``orig_size``.
        class_id_map: Unified category id → COCO head class index.
        device:       Optional device for the returned tensors.

    Returns:
        {'batch_idx': [M], 'cls': [M], 'bboxes': [M,4]} float tensors
        (M = total kept boxes across the batch; all-empty batches give M=0).
    """
    h, w = float(img_hw[0]), float(img_hw[1])
    idx_out, cls_out, box_out = [], [], []
    for i, t in enumerate(targets):
        boxes = torch.as_tensor(t["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(t["labels"], dtype=torch.int64).reshape(-1)
        for box, lab in zip(boxes, labels):
            mapped = class_id_map.get(int(lab))
            if mapped is None:
                continue  # class not in the head's mapped set
            x, y, bw, bh = box.tolist()
            idx_out.append(i)
            cls_out.append(mapped)
            box_out.append([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])

    batch = {
        "batch_idx": torch.tensor(idx_out, dtype=torch.float32),
        "cls": torch.tensor(cls_out, dtype=torch.float32),
        "bboxes": (torch.tensor(box_out, dtype=torch.float32)
                   if box_out else torch.zeros((0, 4), dtype=torch.float32)),
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


# ── model shim ───────────────────────────────────────────────────────────────
class LossModelShim:
    """Duck-types the 3 attributes ``v8DetectionLoss.__init__`` reads off a
    ``DetectionModel``: ``.args`` (hyp gains), ``.model[-1]`` (Detect module),
    and ``.parameters()`` (device probe). Keeps Phase-4 code untouched.
    """

    def __init__(self, detector, hyp: SimpleNamespace):
        self.args = hyp
        # Only index [-1] is read; pad so the Detect module lands there.
        self.model = [None] * 22 + [detector.head]
        self._params = detector.parameters

    def parameters(self):
        return self._params()


def build_loss(detector, cfg: dict):
    """Build a ``v8DetectionLoss`` bound to the detector's frozen Detect head.

    Reads the box/cls/dfl gains from ``training.loss_gains`` (ultralytics
    defaults otherwise) and asserts the head's stride tensor is the pretrained
    [8,16,32] (PHASE5_IMPLEMENTATION.md §3.4 — a fresh head would need a dummy
    forward that our dual-input model can't run automatically).
    """
    from ultralytics.utils.loss import v8DetectionLoss

    stride = detector.head.stride
    expected = torch.tensor([8.0, 16.0, 32.0])
    assert stride is not None and torch.equal(stride.cpu().float(), expected), (
        f"Detect head stride {stride} != {expected.tolist()} — head strides "
        "not initialized from the pretrained checkpoint?")

    gains = {**DEFAULT_LOSS_GAINS,
             **(cfg.get("training", {}).get("loss_gains") or {})}
    hyp = SimpleNamespace(**gains)
    return v8DetectionLoss(LossModelShim(detector, hyp))


# ── self-test ────────────────────────────────────────────────────────────────
def _self_test() -> int:
    """Unit checks for the two classic bugs (§3.2): corner→center shift,
    tensor-shape normalization — plus class-map filtering and empty batches.
    """
    print("=== loss_adapter self-test ===")
    H, W = 512, 640
    cmap = {1: 0}

    # 1. corner→center + normalization: a 100×50 box at top-left (10, 20).
    targets = [{"boxes": torch.tensor([[10.0, 20.0, 100.0, 50.0]]),
                "labels": torch.tensor([1])}]
    b = targets_to_batch(targets, (H, W), cmap)
    cx, cy, bw, bh = b["bboxes"][0].tolist()
    assert abs(cx - (10 + 50) / W) < 1e-6, f"cx {cx} — corner→center shift missing?"
    assert abs(cy - (20 + 25) / H) < 1e-6, f"cy {cy} — corner→center shift missing?"
    assert abs(bw - 100 / W) < 1e-6 and abs(bh - 50 / H) < 1e-6
    assert b["cls"][0].item() == 0.0 and b["batch_idx"][0].item() == 0.0
    print("  [OK] corner-xywh px → normalized center-xywh (100×50 @ (10,20))")

    # 2. normalization uses the TENSOR shape (H≠W catches an H/W swap).
    assert bw != bh, "test box must be non-square to catch an H/W swap"
    b2 = targets_to_batch(targets, (W, H), cmap)  # deliberately swapped
    assert not torch.allclose(b["bboxes"], b2["bboxes"]), \
        "H/W swap undetected — normalization not using (H, W) order?"
    print("  [OK] normalization is (H, W)-order sensitive")

    # 3. batch_idx assignment + unmapped-class filtering.
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0],
                                [1.0, 1.0, 2.0, 2.0]]),
         "labels": torch.tensor([2, 1])},  # class 2 unmapped → dropped
    ]
    b = targets_to_batch(targets, (H, W), cmap)
    assert b["batch_idx"].tolist() == [0.0, 1.0], b["batch_idx"]
    assert b["cls"].tolist() == [0.0, 0.0]
    print("  [OK] batch_idx per sample; unmapped class ids dropped")

    # 4. empty batch → M=0 tensors with the right shapes.
    b = targets_to_batch([{"boxes": torch.zeros((0, 4)),
                           "labels": torch.zeros((0,), dtype=torch.int64)}],
                         (H, W), cmap)
    assert b["bboxes"].shape == (0, 4) and b["cls"].shape == (0,)
    print("  [OK] empty targets → zero-row batch dict")

    # 5. multi-class map extension (KAIST/M3FD future: car 2 → COCO 2).
    b = targets_to_batch(
        [{"boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]), "labels": torch.tensor([2])}],
        (H, W), {1: 0, 2: 2})
    assert b["cls"].tolist() == [2.0]
    print("  [OK] class_id_map extends to multi-class sources")

    print("  [PASS] all loss_adapter self-tests passed.")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase 5 loss adapter")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    ap.print_help()
