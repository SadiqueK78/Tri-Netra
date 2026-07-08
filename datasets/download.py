#!/usr/bin/env python3
"""
TriNetra-AMRF — Dataset Download & Verification
===============================================

Purpose (Phase 2.1/2.2 — Data Acquisition):
    Verify that a raw dataset is present and internally consistent, document
    the (registration-gated) manual download steps, and emit the calibration
    artifacts other phases expect.

Most surveillance RGB-T datasets (LLVIP, KAIST, FLIR ADAS) require accepting a
license / registering before download, so this script does NOT attempt to
scrape them. Instead it:

    1. Checks that the expected raw directory tree exists.
    2. Verifies visible/thermal pairing (identical filenames) and that every
       image has a matching annotation.
    3. Reports counts per official split.
    4. Writes an identity homography to datasets/calibration/homography.npy
       (LLVIP is already pixel-aligned; Phase 3 can overwrite this).

Usage:
    python datasets/download.py --source llvip            # verify
    python datasets/download.py --source llvip --instructions   # print how-to
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Windows consoles/pipes default to cp1252 and choke on box-drawing/emoji
# output; force UTF-8 so scripts never crash on a print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover — older interpreters / redirected streams
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

_INSTRUCTIONS = {
    "llvip": """
LLVIP — Low-Light Visible-Infrared Paired dataset
-------------------------------------------------
  Homepage : https://bupt-ai-cz.github.io/LLVIP/
  GitHub   : https://github.com/bupt-ai-cz/LLVIP

  1. Request access on the homepage (a Google-Drive / Baidu link is emailed).
  2. Download and extract so the tree looks like:

       datasets/LLVIP/
       ├── Annotations/         (15,488 VOC .xml files)
       ├── visible/
       │   ├── train/           (12,025 .jpg)
       │   └── test/            (3,463 .jpg)
       └── infrared/
           ├── train/           (12,025 .jpg)
           └── test/            (3,463 .jpg)

  3. Re-run:  python datasets/download.py --source llvip
""",
}


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _list_images(directory: Path) -> set:
    """Return the set of image *stems* in a directory (non-recursive)."""
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS}


def verify_llvip(cfg: dict) -> bool:
    """Verify the LLVIP raw tree; return True if everything checks out."""
    src = cfg["datasets"]["sources"]["llvip"]
    root = resolve(src["root"])
    vis_root = root / src["visible_subdir"]
    thr_root = root / src["thermal_subdir"]
    ann_dir = root / src["annotations_subdir"]

    print("=" * 64)
    print("  LLVIP — Dataset Verification")
    print("=" * 64)
    print(f"  Root: {root}")

    if not root.is_dir():
        print(f"  [FAIL] Dataset root not found: {root}")
        print("         Run with --instructions for download steps.")
        return False

    ok = True
    total_pairs = 0
    for split in (src["official_train_dir"], src["official_test_dir"]):
        vis = _list_images(vis_root / split)
        thr = _list_images(thr_root / split)

        missing_thermal = vis - thr
        missing_visible = thr - vis
        paired = vis & thr
        total_pairs += len(paired)

        print(f"\n  ── split '{split}' ──")
        print(f"     visible images : {len(vis)}")
        print(f"     thermal images : {len(thr)}")
        print(f"     matched pairs  : {len(paired)}")
        if missing_thermal:
            print(f"     [WARN] {len(missing_thermal)} visible image(s) lack a thermal pair "
                  f"(e.g. {sorted(missing_thermal)[:3]})")
            ok = False
        if missing_visible:
            print(f"     [WARN] {len(missing_visible)} thermal image(s) lack a visible pair "
                  f"(e.g. {sorted(missing_visible)[:3]})")
            ok = False

    # Annotation coverage (flat dir spanning both splits).
    ann_stems = {p.stem for p in ann_dir.glob("*.xml")} if ann_dir.is_dir() else set()
    all_vis = _list_images(vis_root / src["official_train_dir"]) | _list_images(
        vis_root / src["official_test_dir"]
    )
    print(f"\n  ── annotations ──")
    print(f"     xml files      : {len(ann_stems)}")
    missing_ann = all_vis - ann_stems
    if missing_ann:
        print(f"     [WARN] {len(missing_ann)} image(s) have no annotation "
              f"(e.g. {sorted(missing_ann)[:3]})")
        ok = False
    else:
        print("     [OK] Every image has a matching annotation.")

    print(f"\n  Total matched visible/thermal pairs: {total_pairs}")
    return ok


def write_identity_calibration(cfg: dict) -> None:
    """
    Write a 3x3 identity homography to the calibration dir.

    LLVIP visible/thermal frames are already spatially aligned, so the
    identity matrix is correct. Phase 3 (alignment) may replace this with a
    calibrated homography for other sensors.
    """
    calib_dir = resolve(cfg["datasets"]["calibration_dir"])
    calib_dir.mkdir(parents=True, exist_ok=True)
    out = calib_dir / "homography.npy"
    np.save(out, np.eye(3, dtype=np.float64))
    print(f"  Wrote identity homography -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify/prepare a raw RGB-T dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source", default=None, help="Defaults to config active_source.")
    parser.add_argument("--instructions", action="store_true",
                        help="Print manual download instructions and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = (args.source or cfg["datasets"]["active_source"]).lower()

    if args.instructions:
        print(_INSTRUCTIONS.get(source, f"No instructions on file for '{source}'."))
        return 0

    if source == "llvip":
        ok = verify_llvip(cfg)
    else:
        print(f"[ERROR] Verification for source '{source}' is not implemented.")
        return 2

    if ok:
        write_identity_calibration(cfg)
        print("\n  [OK] Dataset verified. Next: python datasets/convert_annotations.py "
              f"--source {source} --validate")
        return 0
    else:
        print("\n  [FAIL] Dataset verification found problems (see warnings above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
