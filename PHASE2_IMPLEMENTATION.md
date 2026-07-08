# Phase 2 — Data Acquisition & Dataset Preparation — **IMPLEMENTED**

> Status: ✅ **Complete & verified end-to-end on LLVIP** (2026-07-06)
> This document records exactly what was built, how it works, how it was
> verified, and how to re-run it. It is the "as-built" companion to the
> planning doc [`implementation_plan.md`](implementation_plan.md).

---

## 1. Summary

Phase 2 turns the raw LLVIP download into a working PyTorch data pipeline that
yields **aligned visible + thermal image pairs with COCO annotations**. The
entire pipeline is config-driven from [`configs/default.yaml`](configs/default.yaml).

| Metric | Value |
|--------|-------|
| Source dataset | LLVIP (visible-infrared paired, low-light surveillance) |
| Total pairs | 15,488 (each = 1 visible + 1 thermal, identical filename) |
| Native resolution | 1280 × 1024 |
| Model input size | 640 × 512 (W × H) |
| Classes | `person` (mapped into a unified 6-class vocabulary) |
| Total annotations | 42,432 boxes |
| **Train / Val / Test split** | **9,903 / 2,122 / 3,463** pairs |
| Split policy | Official LLVIP test set preserved; val carved from train |

---

## 2. Files delivered

### Data pipeline (`datasets/`)

| File | Lines of responsibility |
|------|-------------------------|
| `datasets/__init__.py` | Package exports (`DualModalDataset`, `create_dataloaders`, transforms) |
| `datasets/download.py` | Verify raw LLVIP tree, check pairing & annotation coverage, write identity calibration matrix, print manual-download instructions |
| `datasets/convert_annotations.py` | Parse Pascal VOC XML → unified **COCO JSON**; tag each image with its official split; `validate_coco_json()` |
| `datasets/split_dataset.py` | Respect official test set, carve stratified val from train, copy pairs into canonical dirs, write per-split COCO JSON + statistics |
| `datasets/dual_modal_dataset.py` | `DualModalDataset` — paired visible+thermal loading with COCO targets |
| `datasets/augmentations.py` | `DualModalTransform` — **synchronized** spatial aug + per-modality photometric aug (Albumentations 2.x) |
| `datasets/dataloader.py` | `create_dataloaders()` factory + `dual_modal_collate_fn` |

### Utilities (`utils/`)

| File | Purpose |
|------|---------|
| `utils/data_stats.py` | Class distribution, COCO size buckets, negatives, aspect ratios |
| `utils/visualize.py` | Side-by-side pair viz (RGB \| thermal \| overlay) + augmented-batch grid |

### Config

| File | Change |
|------|--------|
| `configs/default.yaml` | Added `active_source`, per-source `sources` block, unified `categories`, `image_size`, `augmentation`, `normalize`, `seed`, `respect_official_test`; set `detection.num_classes: 6` |

---

## 3. Canonical data layout produced

```
datasets/
├── LLVIP/                        # raw download (untouched, git-ignored)
│   ├── Annotations/*.xml         # 15,488 VOC files
│   ├── visible/{train,test}/     # 12,025 + 3,463 RGB jpg
│   └── infrared/{train,test}/    # 12,025 + 3,463 thermal jpg
│
├── visible/                      # ← GENERATED (canonical)
│   ├── train/  (9,903 jpg)
│   ├── val/    (2,122 jpg)
│   └── test/   (3,463 jpg)
├── thermal/                      # ← GENERATED — same filenames as visible/
│   ├── train/  val/  test/
├── annotations/                  # ← GENERATED
│   ├── instances_all.json        # master COCO (pre-split, split-tagged)
│   ├── train.json  val.json  test.json
└── calibration/
    └── homography.npy            # 3×3 identity (LLVIP is pre-aligned)
```

> **Pairing convention:** `visible/<split>/000001.jpg` pairs with
> `thermal/<split>/000001.jpg` — identical filename = same physical scene.

---

## 4. How each stage works

### 4.1 `download.py` — verification
- Scans `visible/{train,test}` and `infrared/{train,test}`, computes the set of
  matched stems, and reports any image missing its thermal/visible twin.
- Confirms every image has a matching `.xml` annotation.
- Writes `datasets/calibration/homography.npy = I₃` because LLVIP frames are
  already spatially registered (Phase 3 alignment can overwrite this).

### 4.2 `convert_annotations.py` — VOC → COCO
- Walks `Annotations/*.xml`, extracts `filename`, image size, and each
  `<object>`'s class + `<bndbox>`.
- Converts VOC corner boxes `[xmin, ymin, xmax, ymax]` → COCO
  `[x, y, w, h]`, computes `area`, drops degenerate boxes.
- Maps source class names → **unified category IDs** (`person → 1`).
- Tags each image record with `split: "train"|"test"` from LLVIP's official
  folder layout, so the splitter can preserve the benchmark test set.
- `validate_coco_json()` checks: no duplicate IDs, every `image_id`/`category_id`
  resolves, all boxes have positive width/height.

**Output:** `datasets/annotations/instances_all.json` (15,488 imgs, 42,432 anns).

### 4.3 `split_dataset.py` — train/val/test
- `respect_official_test: true` → the 3,463 official-test images go straight to
  `test`; the 12,025 official-train images are split into train/val.
- Val fraction = `val / (train + val)` = `0.15 / 0.85` ≈ **17.6 %** of train.
- **Stratified** on a coarse positive/negative key (has-objects vs. empty) with
  a fixed `seed: 42` → deterministic, reproducible splits.
- Copies each pair by filename into the canonical dirs (copy is Windows-safe;
  `--symlink` is available where Developer Mode/admin allows it).
- Writes standalone `train/val/test.json` and prints a per-split, per-class
  object-count table.

### 4.4 `dual_modal_dataset.py` — the core `Dataset`
Each `__getitem__(idx)` returns `(visible, thermal, targets)`:

| Field | Type / shape | Notes |
|-------|--------------|-------|
| `visible` | `FloatTensor [3, H, W]` | RGB, ImageNet-normalized |
| `thermal` | `FloatTensor [1, H, W]` | LLVIP's 3-ch JPG collapsed to 1 channel, normalized |
| `targets["boxes"]` | `FloatTensor [N, 4]` | COCO `[x, y, w, h]` in the **resized** frame |
| `targets["labels"]` | `LongTensor [N]` | unified category IDs |
| `targets["image_id"]` | `LongTensor []` | scalar COCO id |
| `targets["orig_size"]` | `LongTensor [2]` | `(height, width)` before transform |

Backed by `pycocotools.COCO`; skips (with a warning) any image whose
visible/thermal file is missing.

### 4.5 `augmentations.py` — synchronized dual-modal augmentation
The single most important correctness property in Phase 2:

> If you flip/crop/rotate the visible image, the thermal image **must** receive
> the identical spatial transform, or pixel-level correspondence breaks.

Implemented as a 3-stage `DualModalTransform`:

| Stage | Applies to | Transforms |
|-------|-----------|------------|
| **A — shared spatial** | visible + thermal + boxes together | `RandomResizedCrop`, `HorizontalFlip`, `Affine(rotate)` — one `A.Compose` with `bbox_params=coco` and `additional_targets={'thermal':'image'}` |
| **B1 — visible pixels** | visible only | `ColorJitter`, `RandomGamma`, `Normalize(ImageNet)`, `ToTensorV2` |
| **B2 — thermal pixels** | thermal only | `GaussNoise`, `RandomBrightnessContrast`, `Normalize(0.5,0.5)`, `ToTensorV2` |

`get_val_transforms()` uses a deterministic `Resize` for A and no photometric aug.

### 4.6 `dataloader.py` — batching
- `create_dataloaders(cfg)` → `(train, val, test)` loaders (any split with no
  JSON yields `None`).
- `dual_modal_collate_fn` stacks the fixed-shape image tensors and keeps
  `targets` as a **list of dicts** (standard for variable-object detection).
- `shuffle`/`drop_last` on train only; `pin_memory` auto-enabled with CUDA.

---

## 5. Verification performed (all passed ✅)

| Check | Result |
|-------|--------|
| `download.py` pairing | 15,488 / 15,488 pairs matched; every image annotated |
| `convert_annotations.py --validate` | 15,488 imgs / 42,432 anns / 6 cats — **COCO valid** |
| Split totals | 9,903 + 2,122 + 3,463 = 15,488 ✅ (on-disk file counts match JSON) |
| Annotation conservation | 28,282 + 5,848 + 8,302 = 42,432 ✅ (no leakage/loss) |
| DataLoader shapes | `visible [B,3,512,640]`, `thermal [B,1,512,640]`, targets list ✅ |
| Normalization ranges | visible ≈ `[-2.1, 2.6]`, thermal ≈ `[-1.0, 1.0]` ✅ |
| Pair alignment (visual) | box aligns across RGB/thermal/overlay ✅ |
| Synchronized aug (visual) | identical crop/flip/rotate on both modalities ✅ |
| `import datasets` | package + all exports import cleanly ✅ |

Artifacts written to `runs/visualizations/` (3 pair PNGs + 1 augmented batch grid).

---

## 6. How to re-run

```bash
# 0. (optional) verify the raw dataset is present & consistent
python datasets/download.py --source llvip

# 1. VOC XML → unified master COCO JSON (+ validate)
python datasets/convert_annotations.py --source llvip --validate

# 2. Split into train/val/test and copy into canonical dirs
python datasets/split_dataset.py --config configs/default.yaml
#    add --dry-run to preview counts without copying
#    add --symlink to link instead of copy (needs Win Developer Mode/admin)

# 3. Inspect statistics
python utils/data_stats.py --all

# 4. Programmatic shape check
python -c "import yaml; from datasets.dataloader import create_dataloaders; \
cfg=yaml.safe_load(open('configs/default.yaml')); \
tl,vl,_=create_dataloaders(cfg, batch_size=4, num_workers=0); \
v,t,y=next(iter(tl)); print(v.shape, t.shape, len(y))"

# 5. Visual sanity checks
python utils/visualize.py --split val --num-samples 3        # raw pairs
python utils/visualize.py --batch --split train --num-samples 6   # augmented batch
```

---

## 7. Key design decisions (and why)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Split policy | Preserve official LLVIP test | Results stay comparable to published LLVIP benchmarks; no test leakage |
| File placement | Copy (not symlink) | Windows-safe by default; canonical dirs are self-contained |
| Thermal channels | Collapse LLVIP's 3-ch JPG → 1 ch at load | Framework treats thermal as single-channel; smaller, correct semantics |
| Category vocabulary | Unified 6-class map, LLVIP uses only `person` | Lets M3FD/KAIST/FLIR slot into the same IDs later without re-plumbing |
| Aug architecture | 3-stage (shared spatial + per-modality pixel) | Guarantees geometric sync while allowing modality-specific photometrics |
| Output box format | COCO `[x,y,w,h]` in resized frame | Matches annotations; detector head conversion happens in the model phase |

---

## 8. Known limitations / follow-ups

- **`.gitignore`** currently ignores only `datasets/LLVIP/`. The generated
  `datasets/visible/`, `datasets/thermal/`, `datasets/annotations/` (~4 GB)
  should be added so they aren't committed.
- Only **LLVIP** is wired up. `convert_annotations.py` has a VOC path reusable
  for **M3FD**; **KAIST** TXT parsing is a stub (`convert_kaist_to_coco`).
- `homography.npy` is identity — real cross-sensor calibration is a **Phase 3**
  concern (see [`PHASE3_IMPLEMENTATION.md`](PHASE3_IMPLEMENTATION.md)).
- Mosaic augmentation is intentionally deferred (`augmentation.mosaic: false`).
```
