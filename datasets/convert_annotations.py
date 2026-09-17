#!/usr/bin/env python3
"""
TriNetra-AMRF — Annotation Conversion to Unified COCO JSON
==========================================================

Purpose (Phase 2.3 — Data Acquisition & Dataset Preparation):
    Convert per-source detection annotations into a single, unified COCO
    JSON file (``instances_all.json``). Every downstream component — the
    splitter, the PyTorch ``Dataset``, and the evaluation code — reads this
    canonical format, so all source-specific quirks are absorbed here.

Supported source formats:
    - LLVIP  : Pascal VOC XML  (person only)               [implemented]
    - M3FD   : Pascal VOC XML  (6 classes, "Lamp" dropped)  [implemented]
    - KAIST  : Caltech-style TXT                            [stub]
    - FLIR   : COCO JSON (already unified)                  [not implemented]

Key design choices:
    * The master JSON is produced BEFORE the train/val/test split. Each image
      record carries an extra ``split`` field ("train"/"test") reflecting the
      source's *official* split when one exists (LLVIP does). ``split_dataset.py``
      uses this to preserve the official test set.
    * ``file_name`` is the SHARED basename (identical for visible & thermal).
    * Category IDs come from the unified ``datasets.categories`` map in the
      config, NOT from the source's own class indices.

Usage:
    python datasets/convert_annotations.py --source llvip
    python datasets/convert_annotations.py --source llvip --validate
    python datasets/convert_annotations.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Windows consoles/pipes default to cp1252 and choke on box-drawing/emoji
# output; force UTF-8 so scripts never crash on a print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# ── Path setup ───────────────────────────────────────────────────────────────
# Resolve the project root (…/Tri-Netra) so the script works regardless of the
# directory it is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


# ── Config helpers ───────────────────────────────────────────────────────────
def load_config(config_path: str | os.PathLike) -> dict:
    """Load the YAML config file into a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def build_category_index(cfg: dict) -> Dict[str, int]:
    """Return {category_name -> category_id} from the unified category list."""
    return {c["name"]: int(c["id"]) for c in cfg["datasets"]["categories"]}


# ── VOC → COCO ───────────────────────────────────────────────────────────────
def _parse_voc_xml(xml_path: Path, trust_filename: bool = True,
                    fallback_ext: str = ".jpg") -> Optional[dict]:
    """
    Parse a single Pascal VOC XML file.

    Args:
        trust_filename: When True (LLVIP), use the XML's own ``<filename>``
            tag. When False (M3FD), ignore it and always use the XML file's
            own stem + ``fallback_ext`` — M3FD's ``<filename>`` tags are
            unreliable (measured: 2,284/4,200 don't match their own file's
            stem, 399 distinct values are claimed by more than one XML), so
            the only trustworthy correspondence is same-named XML/image
            pairs on disk.
        fallback_ext: Extension to pair with the XML's own stem, either as
            the true fallback (``<filename>`` missing) or, when
            ``trust_filename=False``, unconditionally.

    Returns a dict with keys ``filename``, ``width``, ``height`` and a list of
    ``objects`` (each ``{name, xmin, ymin, xmax, ymax}``), or ``None`` if the
    file is malformed.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"  [WARN] Skipping unparseable XML {xml_path.name}: {exc}")
        return None

    root = tree.getroot()

    filename = root.findtext("filename") if trust_filename else None
    if not filename:
        filename = xml_path.stem + fallback_ext

    size = root.find("size")
    width = int(float(size.findtext("width"))) if size is not None else 0
    height = int(float(size.findtext("height"))) if size is not None else 0

    objects: List[dict] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        # VOC boxes are 1-indexed inclusive corners; we keep raw pixel corners
        # and convert to COCO [x, y, w, h] below.
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        difficult = int(obj.findtext("difficult") or 0)
        objects.append(
            {
                "name": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "difficult": difficult,
            }
        )

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def convert_voc_to_coco(
    voc_dir: Path,
    category_index: Dict[str, int],
    categories: List[dict],
    split_of: Optional[Dict[str, str]] = None,
    default_size: Optional[tuple] = None,
    class_alias: Optional[Dict[str, str]] = None,
    trust_filename: bool = True,
    image_ext: str = ".jpg",
) -> dict:
    """
    Convert a directory of Pascal VOC XML files to a COCO dict.

    Args:
        voc_dir:        Directory containing ``.xml`` annotation files.
        category_index: {unified_name -> unified_id} mapping.
        categories:     The unified category list (copied verbatim into output).
        split_of:       Optional {filename -> "train"/"test"} tagging map.
        default_size:   (width, height) fallback if an XML omits <size>.
        class_alias:    Optional {source_class_name -> unified_name} remap
                        (e.g. LLVIP already uses "person", so identity).
        trust_filename: Use each XML's own ``<filename>`` tag (LLVIP) vs.
                        always deriving it from the XML's own stem (M3FD,
                        whose ``<filename>`` tags are unreliable — see
                        ``_parse_voc_xml``).
        image_ext:      Extension to pair with the XML's own stem when
                        ``trust_filename=False`` or the tag is missing.

    Returns:
        A COCO-format dict: {info, categories, images, annotations}.
    """
    class_alias = class_alias or {}
    xml_files = sorted(voc_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found in {voc_dir}")

    images: List[dict] = []
    annotations: List[dict] = []
    image_id = 0
    ann_id = 0
    skipped_classes: Dict[str, int] = {}
    n_neg = 0

    for xml_path in xml_files:
        parsed = _parse_voc_xml(xml_path, trust_filename=trust_filename,
                                 fallback_ext=image_ext)
        if parsed is None:
            continue

        filename = parsed["filename"]
        width = parsed["width"] or (default_size[0] if default_size else 0)
        height = parsed["height"] or (default_size[1] if default_size else 0)

        image_id += 1
        img_record = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        }
        if split_of is not None:
            # Tag with the official split; key on the stem so ".jpg"/".png"
            # mismatches don't matter.
            img_record["split"] = split_of.get(Path(filename).stem, "train")
        images.append(img_record)

        if not parsed["objects"]:
            n_neg += 1

        for obj in parsed["objects"]:
            raw_name = obj["name"]
            unified_name = class_alias.get(raw_name, raw_name)
            if unified_name not in category_index:
                skipped_classes[raw_name] = skipped_classes.get(raw_name, 0) + 1
                continue

            x = obj["xmin"]
            y = obj["ymin"]
            w = obj["xmax"] - obj["xmin"]
            h = obj["ymax"] - obj["ymin"]
            # Guard against degenerate / inverted boxes.
            if w <= 0 or h <= 0:
                continue

            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_index[unified_name],
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "difficult": obj["difficult"],
                }
            )

    if skipped_classes:
        print(f"  [INFO] Skipped classes not in unified map: {skipped_classes}")
    print(f"  [INFO] {n_neg} image(s) have no annotations (negatives).")

    return {
        "info": {
            "description": "TriNetra Dual-Modality Dataset",
            "version": "1.0",
            "year": 2026,
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


# ── LLVIP driver ─────────────────────────────────────────────────────────────
def _build_llvip_split_map(src_cfg: dict, source_root: Path) -> Dict[str, str]:
    """
    Build {image_stem -> "train"/"test"} from LLVIP's official folder layout.

    LLVIP stores images under visible/{train,test}; the flat Annotations/
    directory covers both. We look at which visible subfolder a stem lives in.
    """
    vis_root = source_root / src_cfg["visible_subdir"]
    split_map: Dict[str, str] = {}
    for split_name, sub in (
        ("train", src_cfg["official_train_dir"]),
        ("test", src_cfg["official_test_dir"]),
    ):
        sub_dir = vis_root / sub
        if not sub_dir.is_dir():
            print(f"  [WARN] Expected LLVIP folder missing: {sub_dir}")
            continue
        for img in sub_dir.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                split_map[img.stem] = split_name
    return split_map


def convert_llvip(cfg: dict) -> dict:
    """Convert the LLVIP source into the unified COCO dict."""
    src_cfg = cfg["datasets"]["sources"]["llvip"]
    source_root = resolve(src_cfg["root"])
    ann_dir = source_root / src_cfg["annotations_subdir"]
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"LLVIP annotations dir not found: {ann_dir}\n"
            f"Expected the dataset at {source_root} — see datasets/download.py."
        )

    category_index = build_category_index(cfg)
    categories = cfg["datasets"]["categories"]
    native = tuple(cfg["datasets"].get("native_size", (1280, 1024)))
    # LLVIP class names already match the unified vocabulary (identity alias),
    # but we honor the source-level class_map if present.
    class_alias = {k: k for k in src_cfg.get("class_map", {})}

    print(f"  Parsing VOC XML from: {ann_dir}")
    split_map = _build_llvip_split_map(src_cfg, source_root)
    n_train = sum(v == "train" for v in split_map.values())
    n_test = sum(v == "test" for v in split_map.values())
    print(f"  Official split detected: {n_train} train / {n_test} test image(s).")

    coco = convert_voc_to_coco(
        voc_dir=ann_dir,
        category_index=category_index,
        categories=categories,
        split_of=split_map,
        default_size=native,
        class_alias=class_alias,
    )
    return coco


# ── M3FD driver ──────────────────────────────────────────────────────────────
def convert_m3fd(cfg: dict) -> dict:
    """Convert the M3FD source into the unified COCO dict.

    Unlike LLVIP, M3FD ships no official train/test split — the whole 4,200
    pairs are converted un-tagged (``split_of=None``) and
    ``split_dataset.py`` (with ``respect_official_test: false``) does a
    fresh stratified split by ``split_ratios``.

    M3FD's raw classes are People/Car/Bus/Motorcycle/Lamp/Truck. "Lamp" has
    no unified equivalent and is intentionally left out of the source's
    ``class_map`` — objects with an unmapped name are dropped (and counted
    in the "[INFO] Skipped classes" log line) by ``convert_voc_to_coco``.
    """
    src_cfg = cfg["datasets"]["sources"]["m3fd"]
    source_root = resolve(src_cfg["root"])
    ann_dir = source_root / src_cfg["annotations_subdir"]
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"M3FD annotations dir not found: {ann_dir}\n"
            f"Expected the dataset at {source_root}."
        )

    category_index = build_category_index(cfg)
    categories = cfg["datasets"]["categories"]
    native = tuple(cfg["datasets"].get("native_size", (1024, 768)))

    # The config's m3fd.class_map is {raw_lowercase_name: unified_id};
    # convert_voc_to_coco wants {raw_lowercase_name: unified_name} (it looks
    # the id up itself via category_index), so translate id -> name here.
    id_to_name = {c["id"]: c["name"] for c in categories}
    class_alias = {
        raw: id_to_name[uid]
        for raw, uid in src_cfg.get("class_map", {}).items()
        if uid in id_to_name
    }

    print(f"  Parsing VOC XML from: {ann_dir}")
    coco = convert_voc_to_coco(
        voc_dir=ann_dir,
        category_index=category_index,
        categories=categories,
        split_of=None,
        default_size=native,
        class_alias=class_alias,
        trust_filename=False,   # M3FD's <filename> tags are unreliable
        image_ext=".png",
    )
    return coco


# ── KAIST stub ───────────────────────────────────────────────────────────────
def convert_kaist_to_coco(*_args, **_kwargs) -> dict:  # pragma: no cover
    """Convert KAIST Caltech-style TXT annotations to COCO (not yet needed)."""
    raise NotImplementedError(
        "KAIST conversion is not implemented yet. Only LLVIP is wired up for "
        "Phase 2. Add the TXT parser here when KAIST is downloaded."
    )


# ── Validation ───────────────────────────────────────────────────────────────
def validate_coco_json(json_path: str | os.PathLike) -> bool:
    """
    Validate a COCO JSON file for internal consistency.

    Checks:
        - images / annotations / categories keys present
        - no duplicate image, annotation, or category IDs
        - every annotation.image_id references an existing image
        - every annotation.category_id references an existing category
        - all bboxes have positive width & height

    Returns True if valid; prints a report and returns False otherwise.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ok = True
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            print(f"  [FAIL] Missing top-level key: '{key}'")
            ok = False
    if not ok:
        return False

    image_ids = [img["id"] for img in coco["images"]]
    cat_ids = [c["id"] for c in coco["categories"]]
    ann_ids = [a["id"] for a in coco["annotations"]]

    if len(image_ids) != len(set(image_ids)):
        print("  [FAIL] Duplicate image IDs detected.")
        ok = False
    if len(ann_ids) != len(set(ann_ids)):
        print("  [FAIL] Duplicate annotation IDs detected.")
        ok = False
    if len(cat_ids) != len(set(cat_ids)):
        print("  [FAIL] Duplicate category IDs detected.")
        ok = False

    image_id_set = set(image_ids)
    cat_id_set = set(cat_ids)
    bad_ref = 0
    bad_cat = 0
    bad_box = 0
    for a in coco["annotations"]:
        if a["image_id"] not in image_id_set:
            bad_ref += 1
        if a["category_id"] not in cat_id_set:
            bad_cat += 1
        _, _, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            bad_box += 1
    if bad_ref:
        print(f"  [FAIL] {bad_ref} annotation(s) reference a missing image_id.")
        ok = False
    if bad_cat:
        print(f"  [FAIL] {bad_cat} annotation(s) reference a missing category_id.")
        ok = False
    if bad_box:
        print(f"  [FAIL] {bad_box} annotation(s) have non-positive width/height.")
        ok = False

    n_with_ann = len({a["image_id"] for a in coco["annotations"]})
    print(
        f"  Images: {len(image_ids)} | Annotations: {len(ann_ids)} | "
        f"Categories: {len(cat_ids)} | Images w/ ≥1 box: {n_with_ann}"
    )
    print("  [OK] COCO JSON is valid." if ok else "  [FAIL] COCO JSON has errors.")
    return ok


# ── CLI ──────────────────────────────────────────────────────────────────────
_CONVERTERS = {
    "llvip": convert_llvip,
    "m3fd": convert_m3fd,
    "kaist": convert_kaist_to_coco,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert source annotations to unified COCO JSON.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument(
        "--source",
        default=None,
        help="Dataset source key (llvip, kaist, ...). Defaults to config's active_source.",
    )
    parser.add_argument("--output", default=None, help="Output JSON path (overrides config).")
    parser.add_argument("--validate", action="store_true", help="Validate the written JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source or cfg["datasets"]["active_source"]
    source = source.lower()

    if source not in _CONVERTERS:
        print(f"[ERROR] Unknown/unsupported source '{source}'. "
              f"Available: {sorted(_CONVERTERS)}")
        return 2

    print("=" * 64)
    print(f"  Converting source: {source.upper()}")
    print("=" * 64)

    coco = _CONVERTERS[source](cfg)

    out_path = resolve(args.output) if args.output else resolve(cfg["datasets"]["master_annotations"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"  Wrote master COCO JSON -> {out_path}")

    if args.validate:
        print("-" * 64)
        print("  Validating output...")
        ok = validate_coco_json(out_path)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
