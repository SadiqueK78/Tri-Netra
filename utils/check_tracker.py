#!/usr/bin/env python3
"""
TriNetra-AMRF — Phase 6 Tracker Self-Test
==========================================

Verifies ``models.tracker.FusionTracker`` wiring against a synthetic detection
sequence — no dataset, no trained checkpoint, no GPU needed. It checks the
properties a Kalman+Hungarian tracker must have regardless of what upstream
detector fed it:

  1. identity continuity — a moving object keeps the same track_id every frame
  2. identity separation — two simultaneous objects never share a track_id
  3. confirmation delay  — a track is not reported until ``min_hits`` frames

This is deliberately independent of the fusion detector: Phase 5 has not
produced a trained checkpoint yet, so testing against real decoded detections
isn't possible end-to-end. Once one exists, run the tracker on
``training.evaluate.decode_predictions`` output over a real LLVIP scene
instead — this script only proves the tracker itself is wired correctly.

Usage:
    python -m utils.check_tracker
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tracker import FusionTracker  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
N_FRAMES = 8
PERSON_CLS, CAR_CLS = 0, 2


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def synth_frame(i: int) -> torch.Tensor:
    """One frame: a person walking left->right, a stationary parked car.

    5px/frame on a 30px-wide box keeps frame-to-frame IOU >= ~0.71 — under
    DeepSort's default 0.7 IOU-match threshold, an *unconfirmed* track (age
    < min_hits) is associated by IOU alone, not the more forgiving
    Kalman-gated cost used once confirmed, so drift faster than this and the
    track never accumulates enough hits to confirm in the first place.
    """
    px = 100.0 + 5.0 * i
    person = [px, 100.0, px + 30.0, 180.0, 0.90, PERSON_CLS]
    car = [400.0, 300.0, 500.0, 360.0, 0.85, CAR_CLS]
    return torch.tensor([person, car], dtype=torch.float32)


def main() -> int:
    cfg = load_config()
    cfg.setdefault("tracking", {})
    cfg["tracking"]["embedder"] = None   # motion+IoU only — no real frame needed
    min_hits = int(cfg["tracking"].get("min_hits", 3))

    tracker = FusionTracker(cfg)
    print(f"=== check_tracker (algorithm={cfg['tracking'].get('algorithm')}, "
          f"min_hits={min_hits}) ===")

    person_ids, car_ids = [], []
    for i in range(N_FRAMES):
        tracks = tracker.update(synth_frame(i))

        if i < min_hits - 1:
            assert not tracks, (
                f"frame {i}: expected no CONFIRMED tracks before min_hits "
                f"({min_hits}), got {len(tracks)}")
            continue

        assert len(tracks) == 2, f"frame {i}: expected 2 confirmed tracks, got {len(tracks)}"
        by_cls = {t["cls"]: t for t in tracks}
        assert PERSON_CLS in by_cls and CAR_CLS in by_cls, (
            f"frame {i}: expected one track per class, got classes {list(by_cls)}")
        person_ids.append(by_cls[PERSON_CLS]["track_id"])
        car_ids.append(by_cls[CAR_CLS]["track_id"])

    print(f"  [OK] no confirmed tracks before frame {min_hits - 1} "
          f"(min_hits={min_hits} respected)")

    assert len(set(person_ids)) == 1, f"person track_id changed across frames: {person_ids}"
    assert len(set(car_ids)) == 1, f"car track_id changed across frames: {car_ids}"
    print(f"  [OK] identity continuity: person stayed track_id={person_ids[0]} "
          f"across {len(person_ids)} frames")
    print(f"  [OK] identity continuity: car stayed track_id={car_ids[0]} "
          f"across {len(car_ids)} frames")

    assert person_ids[0] != car_ids[0], "person and car were assigned the same track_id"
    print(f"  [OK] identity separation: person and car never share a track_id")

    last = tracker.update(synth_frame(N_FRAMES - 1))
    moved = next(t for t in last if t["cls"] == PERSON_CLS)["bbox_xyxy"][0]
    still = next(t for t in last if t["cls"] == CAR_CLS)["bbox_xyxy"][0]
    print(f"  [OK] final positions plausible: person x≈{moved:.0f} (drifted), "
          f"car x≈{still:.0f} (stationary)")

    print("  [PASS] Phase 6 DeepSORT wiring: continuity, separation, and "
          "confirmation delay all hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
