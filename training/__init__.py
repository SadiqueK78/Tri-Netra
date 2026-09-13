"""
TriNetra-AMRF — Training Package
==================================

Phase 5 (Layer 4 — Detection Integration & Training) is implemented here:

  * ``loss_adapter``  — targets → ultralytics batch dict + the ``v8DetectionLoss``
                        model shim (§5.2). ``python -m training.loss_adapter
                        --self-test`` runs its unit checks.
  * ``train_fusion``  — the two-stage training loop for the Phase-4
                        ``FusionDetector`` (§5.3). Its ``fit()`` is generic and
                        is reused by every Phase-5 variant.
  * ``evaluate``      — decode → NMS → COCOeval mAP, plus the TRC-binned and
                        stress-corrupted analyses (§5.4 / §5.7).
  * ``baselines``     — the RGB-only and thermal-only single-modality floors
                        (§5.5), trained under the identical schedule.
  * ``run_ablation``  — the four-way TRC ablation and its results table (§5.6).

Still placeholders for later phases:

  * ``train_detector`` — standalone multi-scale detector training.
  * ``train_risk``     — hierarchical threat reasoning (Layer 6).
"""
