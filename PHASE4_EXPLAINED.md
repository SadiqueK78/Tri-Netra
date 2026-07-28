# Phase 4 — Explained in Detail (What Was Implemented & How)

> A detailed companion to [`PHASE4_IMPLEMENTATION.md`](PHASE4_IMPLEMENTATION.md).
> That file was the *plan*; **this file documents what was actually built, how
> each module works internally, the engineering decisions made along the way,
> and what the validation tests proved** — written for viva preparation and
> for anyone reading the code later.
>
> Status: ✅ Implemented and validated (all `utils/check_fusion.py` tests pass
> on CUDA, including a real LLVIP val batch end-to-end).

---

## 1. The One-Line Answer (memorize this)

> **Phase 4 builds the adaptive fusion layer: two parallel YOLOv8 backbones
> (one RGB, one thermal) whose feature maps are combined at three scales by a
> TRC-gated fusion block — `fused = P_vis + block(α·trc · P_thr)` — and fed
> into the original COCO-pretrained YOLO neck and head, which stay frozen.
> Only the new thermal stream and fusion blocks are trained.**

---

## 2. Where Phase 4 Sits (The Big Picture)

```
Phase 2 (Layer 1)     Phase 3 (Layer 2)          Phase 4 (Layer 3)             Phase 5 (Layer 4)
Data & DataLoader  →  Preprocessing & TRC  →  Adaptive Cross-Modal Fusion  →  Detection training
(paired tensors)      (clean pair + trc)      (one fused feature set)          (loss + mAP)
```

Phase 4 consumes exactly what the Phase 2/3 dataloader emits per batch:

| Input | Shape | Source |
|---|---|---|
| `visible` | `[B, 3, 512, 640]` float | Phase 2 transform (ImageNet-normalized) |
| `thermal` | `[B, 1, 512, 640]` float | Phase 2 transform (0.5/0.5-normalized) |
| `targets[i]['trc']` | scalar ∈ [0, 1] | Phase 3 TRC (LLVIP val mean ≈ 0.74) |

and produces the multi-scale detection features / decoded predictions that
Phase 5 will train and evaluate.

---

## 3. The Architecture That Was Built

```
                 ┌────────── Visible backbone (YOLOv8m layers 0–9, COCO, FROZEN) ─────────┐
 visible ───────▶│  layer 4 → P3 [B,192,64,80]   layer 6 → P4 [B,384,32,40]   layer 9 → P5 [B,576,16,20] │──┐
 [B,3,512,640]   └──────────────────────────────────────────────────────────────────────┘  │
                                                                                            ▼
                                                                          ┌──────────────────────────────┐
                                                                          │  FusionNeck: 3 × TRCGatedFusion │
                                                                          │  fused_k = P_vis_k + proj(α·trc·P_thr_k) │
                                                                          └──────────────────────────────┘
                 ┌────────── Thermal backbone (copy of 0–9, warm-started, TRAINED) ──────┐  ▲
 thermal ───────▶│  ThermalStem(1-ch) → layers 1–8 → SPPF                                │──┘
 [B,1,512,640]   │  taps: P3' [B,192,64,80]  P4' [B,384,32,40]  P5' [B,576,16,20]        │
                 └──────────────────────────────────────────────────────────────────────┘
                                                                                            │
                              fused (P3, P4, P5) — same shapes as visible taps ─────────────┘
                                            │
                                            ▼
                        YOLO neck (layers 10–21, PAN-FPN, FROZEN)
                                            │
                                            ▼
                        Detect head (layer 22, FROZEN) → predictions
```

Verified layer map (ultralytics **8.4.89**, `yolov8m.pt`, 23 modules):
layers **0–9** backbone (taps at **4 / 6 / 9** = P3/P4/P5, strides 8/16/32),
**10–21** neck, **22** `Detect` head with from-indices `[15, 18, 21]`.

---

## 4. What Was Implemented, File by File

### 4.1 `models/fusion/thermal_stem.py` — `ThermalStem`

**Problem:** YOLO's layer 0 is `Conv(3 → 48)` expecting RGB. Thermal is 1-channel.

**How it was implemented:**
- A new `nn.Conv2d(1, C, ...)` is built with **identical kernel size, stride,
  and padding** as the pretrained stem, so the downsampling geometry (H/2, W/2)
  is preserved and every later layer sees the expected spatial size.
- **`mean3ch` warm start (the default):** the pretrained stem weight is a
  tensor `[C, 3, k, k]`. We take `weight.mean(dim=1, keepdim=True)` →
  `[C, 1, k, k]` and copy it in. Intuition: the stem's filters are low-level
  edge/blob detectors; averaging the RGB planes gives a *grayscale version of
  the same filters*, which is a far better starting point for a grayscale-like
  thermal image than random noise.
- The stem's **BatchNorm and SiLU are deep-copied** from the pretrained stem
  (copied, not shared — so training the thermal stem never mutates the frozen
  visible stem).
- A `random` init mode also exists (config `thermal_stem_init: random`) for
  the warm-start-vs-scratch ablation.

### 4.2 `models/fusion/backbone.py` — `DualBackbone`

**How the split works:**
- `load_yolo_model()` loads `weights/yolov8m.pt` via ultralytics and returns
  the inner `DetectionModel`. Its `.model` attribute is the 23-module list.
- **Visible stream:** `nn.ModuleList(full[0:10])` — these are the *same module
  objects* as in the loaded model (shared by reference, deliberately — see
  §4.5: the neck/head wrapper uses the same load, so everything stays in sync
  and no memory is duplicated).
- **Thermal stream:** `copy.deepcopy` of the visible layers. The deep copy
  *carries the pretrained weights over*, which **is** the warm start
  (`thermal_warm_start: true`). If warm start is disabled, every `Conv2d` /
  `BatchNorm2d` in the copy is re-randomized with `reset_parameters()`.
  Then slot 0 is replaced with the `ThermalStem`.
- **Tap extraction:** each stream is wrapped in `_BackboneStream`, which runs
  layers 0–9 sequentially and collects the outputs of layers **4, 6, 9** into
  a `(P3, P4, P5)` tuple. (No forward hooks needed — the YOLOv8 backbone is a
  pure sequential chain, so slicing + collecting is simpler and hook-free.)
- **Channel probing, not hard-coding:** at build time, a dummy
  `zeros(1,3,64,64)` is pushed through the visible stream under `no_grad`,
  and the channel counts are read off the outputs → `(192, 384, 576)` for
  yolov8m. This makes the code width-multiple aware: swapping `yolov8s.pt` or
  `yolov8l.pt` in the config "just works" with different channel counts.

### 4.3 `models/fusion/fusion_block.py` — `TRCGatedFusion` ← **the novelty**

The core equation implemented:

```
gate  = α · trc                       (α: learnable scalar, init 1.0)
fused = P_vis + proj( gate · P_thr )  ('gated' strategy, residual mode)
```

**How the gate works internally (`_gate` + `_broadcast_trc`):**
1. `trc` arrives as a `[B]` tensor (one scalar per image, from Phase 3).
   The helper reshapes it to `[B,1,1,1]` so it broadcasts over `[B,C,h,w]`.
2. It also accepts a **spatial map** `[B,1,H,W]` (Phase 3's `trc_map`,
   off by default): the map is bilinearly resized to each scale's `h×w`,
   giving *region-adaptive* gating with zero code changes. Floats and 0-dim
   tensors are accepted too (convenience for tests/inference).
3. The thermal feature map is multiplied by `α · trc`. At `trc=0` the thermal
   contribution is exactly zero → fused output falls back to visible-only.
   At `trc=1` the thermal branch contributes fully.
4. `α` is an `nn.Parameter` (init 1.0). During training the network can learn
   to obey TRC more (<1 never happens — α scales, the *shape* of the gating
   is fixed) or amplify the thermal branch (α > 1). With
   `learnable_alpha: false` it stays a constant 1.0.

**Three strategies (config `fusion.strategy`) — all implemented:**

| Strategy | What `proj` is | Fusion formula |
|---|---|---|
| `gated` (default) | Conv3×3 → BN → SiLU → Conv1×1 | `P_vis + proj(gate·P_thr)` |
| `concatenation` | Conv1×1(2C→C) → BN → SiLU | `P_vis + proj([P_vis ‖ gate·P_thr])` |
| `attention` | GroupNorm'd MultiheadAttention (q=vis, k/v=gated thr) + Conv1×1 | `P_vis + proj(attn(vis, gate·thr))` |

**Two safety invariants that make the frozen neck workable:**
1. **Channel identity:** output channels == visible input channels at every
   scale, so the frozen pretrained neck receives tensors of *exactly* the
   shape it was trained on.
2. **Near-identity init:** the final projection conv of the `gated` and
   `attention` paths is initialized at **small magnitude (std=1e-2, zero
   bias)**, so at step 0 `fused ≈ P_vis`. The untrained fusion block cannot
   flood the frozen head with garbage — the detector starts out behaving like
   a plain RGB YOLO and *learns* to mix in thermal.

   > Engineering note: the plan said "zero-init". During implementation this
   > was deliberately relaxed to small-normal init, because an exactly-zero
   > conv makes `trc=1` vs `trc=0` produce *identical* outputs — the
   > TRC-effect validation test (rightly) failed against it. Small-normal
   > keeps the graceful-degradation property **and** keeps the gate
   > observably wired from step 0. This was caught by the test suite —
   > a good example of the checks doing their job.
3. **`trc_enabled: false`** bypasses the gate entirely (`trc ≡ 1`) — that is
   the "fusion **without** TRC" ablation baseline Phase 5 needs.

### 4.4 `models/fusion/fusion_neck.py` — `FusionNeck`

Thin, deliberate wiring: holds **three independent `TRCGatedFusion` blocks**
(`fuse3/fuse4/fuse5`), one per scale, each constructed with that scale's
channel count from `DualBackbone.channels`. The same `trc` tensor gates all
three scales (a frame that's unreliable at P3 is unreliable at P5). Blocks are
*not* shared across scales because channel counts differ and each scale should
learn its own mixing.

### 4.5 `models/detector/fusion_detector.py` — `FusionDetector`

The assembly. The clever part is **reusing ultralytics' own forward wiring**
instead of reimplementing the neck:

- Every ultralytics layer carries two attributes: `f` (from-index: which
  earlier layer(s) it reads) and `i` (its own index). `DetectionModel.forward`
  keeps a table `y` of saved outputs and resolves `f` against it.
- `FusionDetector.forward` replicates exactly that loop for layers 10–22, but
  **seeds the table with the fused features standing in for the backbone
  taps**: `y = {4: f3, 6: f4, 9: f5}`, and starts `x = f5` (layer 9's slot).
- When the neck's `Concat` layers ask for layer 4 / 6 / 9 output (their `f`
  lists are `[-1, 6]`, `[-1, 4]`, `[-1, 9]`), they transparently receive the
  **fused** features. The pretrained neck never knows anything changed.

This means: zero modification of pretrained modules, zero re-implementation of
PAN-FPN, and the head's `f = [15, 18, 21]` wiring keeps working untouched.

Other pieces:
- `trc_from_targets(targets)` — static helper that stacks the per-sample
  `targets[i]['trc']` scalars from the Phase 2/3 dataloader into a `[B]`
  tensor (and auto-prefers `trc_map` → `[B,1,H,W]` when every sample has one).
  This is the one-line bridge between the dataloader and the model.
- `freeze_pretrained()` / `unfreeze(...)` — drive the staged schedule (below).
- `train()` **override** — calls `refreeze_bn` after every `model.train()` so
  frozen BatchNorms stay in eval mode (see 4.6).
- `param_summary()` — prints the trainable/frozen breakdown per module.
- `names` / `stride` copied from the ultralytics model for Phase 5 decoding.

### 4.6 `models/fusion/freeze.py` — the staged transfer-learning schedule

Maps five logical names to detector sub-modules:

```
visible_backbone → backbone.visible      thermal_backbone → backbone.thermal
fusion           → fusion_neck           neck → neck        head → head
```

- `freeze_modules`: sets `requires_grad=False` on all params **and puts every
  BatchNorm into `.eval()`**. Why the BN part matters: a BN layer in train
  mode updates its running mean/var from whatever passes through — even with
  frozen weights. Since Stage 1 feeds the frozen neck *fused* features (a new
  distribution), train-mode BNs would silently drift away from their COCO
  statistics and degrade the frozen head. Freezing means freezing the stats
  too.
- The subtlety: `model.train()` (called every epoch by any standard loop)
  flips **all** children back to train mode, silently undoing BN-eval. Fix:
  frozen modules are tagged with `_trinetra_frozen = True`, and
  `FusionDetector.train()` re-applies BN-eval to tagged modules after every
  `train()` call. The freeze survives the training loop.
- `apply_stage(model, cfg, 1)`: unfreeze everything, then freeze
  `stage1_freeze` from the config (`visible_backbone`, `neck`, `head`).
  `apply_stage(model, cfg, 2)`: additionally unfreeze `stage2_unfreeze`.
- `param_counts` / `describe_freeze_state`: introspection used by the tests
  and the param summary.

### 4.7 `configs/default.yaml` — the new `fusion` block

Replaced the old placeholder (`strategy: attention`, `feature_dim: 256`) with
the full Phase 4 block: strategy (`gated`), TRC gate options (enabled /
learnable α / spatial / residual), backbone options (`base_model:
weights/yolov8m.pt`, warm start, stem init), and the two-stage train schedule
(freeze lists, epochs, stage-2 LR). Every module reads its options from here —
nothing is hard-coded.

### 4.8 `utils/check_fusion.py` — the validation suite

Five checks (all runnable individually via `--test`):

| Check | What it proves |
|---|---|
| `check_shapes` | fused P3/P4/P5 shapes == visible tap shapes == what the frozen neck expects; head runs |
| `check_trc_effect` | **the novelty test** — same input, `trc=1` vs `trc=0` → outputs MUST differ |
| `check_grad_flow` | one backward pass in Stage 1 → gradients reach ONLY thermal+fusion; frozen params get *no grad tensor at all* |
| `count_params` | trainable ≪ frozen in Stage 1 |
| `check_dataloader` | one real Phase-3 val batch flows end-to-end |

---

## 5. What the Tests Actually Showed (real numbers)

Run on CUDA, yolov8m, 640×512 input:

```
=== check_shapes ===
P3: vis == thr == fused == (2, 192, 64, 80)
P4: vis == thr == fused == (2, 384, 32, 40)
P5: vis == thr == fused == (2, 576, 16, 20)          → frozen neck accepts them

=== check_trc_effect ===
max |trc=1 − trc=0| per head scale: 0.358, 0.199, 0.153   → gate IS wired
fused-feature diffs per scale:      0.065, 0.090, 0.099

=== check_grad_flow (Stage 1) ===
visible_backbone  grads=no   ✓        thermal_backbone  grads=yes  ✓
neck              grads=no   ✓        fusion            grads=yes  ✓
head              grads=no   ✓

=== count_params (Stage 1) ===
Total 42,922,051 — trainable 17,019,411 (39.7%) / frozen 25,902,640
  visible_backbone  frozen     11,855,856
  thermal_backbone  trainable  11,854,992
  fusion            trainable   5,164,419
  neck              frozen     10,224,768
  head              frozen      3,822,016

=== check_dataloader (val split, real LLVIP batch) ===
visible (2,3,512,640) · thermal (2,1,512,640) · trc (2,) mean 0.736
head outputs (2,192,64,80) / (2,384,32,40) / (2,576,16,20)   → end-to-end OK
```

Additional strategy sanity check (standalone block, C=64): all three
strategies (`gated` 0.059, `concatenation` 0.608, `attention` 0.014 max diff)
respond to the gate, all accept a spatial `trc_map`, and `trc_enabled: false`
produces a **bit-exact zero** difference — confirming the ablation baseline
truly bypasses TRC.

**Interpretation of 39.7% trainable:** most of that is the thermal backbone
(11.85M) — warm-started, so although *trainable*, it starts from good weights
and mostly fine-tunes. The genuinely-from-scratch parameters are only the
~5.2M fusion blocks (~12% of the model). The pretrained detector knowledge
(25.9M params) is fully preserved.

---

## 6. Engineering Decisions & Deviations From the Plan

| Decision | What was chosen | Why |
|---|---|---|
| Tap mechanism | Slice layers 0–9 + collect in the loop | YOLOv8 backbone is purely sequential; hooks add complexity for nothing |
| Visible stream sharing | *Shared by reference* with the neck/head source model | one weight load, no duplication, everything stays consistent |
| Thermal warm start | `deepcopy` of visible layers = free warm start | copying weights IS the warm start; one mechanism, no extra code |
| Fusion init | **small-normal (std 1e-2), not zero-init** | zero-init made trc=1 ≡ trc=0 (caught by the TRC-effect test); small-normal keeps near-identity *and* an observable gate |
| Neck reuse | Replay ultralytics' `f`/`i` from-index wiring, seed taps with fused features | zero re-implementation of PAN-FPN; pretrained modules untouched |
| BN freeze persistence | `_trinetra_frozen` tag + `train()` override | `model.train()` would otherwise silently un-freeze BN stats every epoch |
| Head output format | Handle **dict** `{boxes, scores, feats}` | ultralytics 8.4.89's `Detect` returns a dict, not the older list — discovered during validation; the plan's assumption was outdated. Phase 5 loss wiring must consume this dict |
| Channel counts | Probed at build time with a dummy forward | width-multiple aware — model variant is swappable from config |

---

## 7. Q&A Cheat Sheet (viva)

- **What exactly is trained in Phase 4?** The thermal stem + thermal backbone
  (11.85M, warm-started) and the three fusion blocks (5.16M, new). The visible
  backbone, neck, and head (25.9M) stay frozen with COCO weights.
- **Why can't everything be frozen?** The fusion blocks and thermal stem are
  new — random weights would feed garbage to the frozen head.
- **Where does fusion happen?** Mid-fusion: after the backbone, at P3/P4/P5
  (strides 8/16/32), before the neck.
- **What does TRC do at runtime?** It multiplies the thermal feature maps
  (`α·trc·P_thr`) before fusion. trc=0 → thermal contribution is exactly zero
  and the model degrades gracefully to visible-only.
- **How do you know the gate is really connected?** `check_trc_effect`: the
  same input with trc=1 vs trc=0 changes the head outputs by up to 0.36.
- **How do you know the freeze is airtight?** `check_grad_flow`: after a
  backward pass, frozen modules have *no gradient tensors at all* (not just
  zero gradients), and frozen BNs stay in eval even after `model.train()`.
- **Why residual `P_vis + block(...)`?** Safety: an untrained block ≈ 0, so
  the model starts as a working RGB detector rather than a broken one.
- **Is "better mAP" proven?** No — Phase 4 delivers a *working, adaptive
  module*; the TRC-vs-no-TRC mAP ablation is Phase 5's job.

---

## 8. Hand-Off to Phase 5

Phase 4 exposes exactly what Phase 5 needs:

```python
model = FusionDetector(cfg)              # builds everything from config
model.freeze_pretrained()                # Stage 1
trc   = FusionDetector.trc_from_targets(targets, device)   # [B] from dataloader
out   = model(visible, thermal, trc)     # train: dict{boxes,scores,feats}
                                         # eval : (decoded [B,84,N], dict)
model.unfreeze(["neck", "head"])         # optional Stage 2
```

Open items deliberately left for Phase 5 (see `PHASE5_IMPLEMENTATION.md`):
loss wiring against the 8.4.89 head-output dict, the 80-class-head vs
LLVIP-classes decision, the training loop itself, mAP evaluation, and the
four-way ablation (RGB-only / thermal-only / fusion-no-TRC / fusion-with-TRC).
