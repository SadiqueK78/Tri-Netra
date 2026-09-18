#!/usr/bin/env python3
"""
TriNetra-AMRF — Live Detection Demo UI
========================================

Purpose:
    A small local web UI for project reviews / demos: upload a visible
    (RGB) image and its paired thermal image, run them through a trained
    FusionDetector checkpoint, and see the annotated detection result
    right in the browser. Reuses inference/predict_image.py's tested
    predict() function — no duplicated model logic.

    Runs on CPU by default and deliberately does NOT touch the GPU: this
    is meant to be usable *while a training run is still active* on the
    same machine, without risking that run's GPU memory.

    Reloads the checkpoint fresh on every submission (not cached), so if
    a training run is actively improving weights/fusion/best.pt, the demo
    reflects the latest checkpoint on each click, not a stale snapshot
    from when the UI was launched.

Usage:
    python app/demo_ui.py
    python app/demo_ui.py --checkpoint weights/ablation/poc/fusion_trc/best.pt
    python app/demo_ui.py --device cuda   # only if the GPU is actually free
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "local_6gb.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict_image import predict  # noqa: E402


def checkpoint_info(checkpoint_path: str) -> str:
    """A one-line summary of the checkpoint's training provenance, if available."""
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        best_map = ckpt.get("best_map", float("nan"))
        variant = ckpt.get("variant", "?")
        return f"Checkpoint: `{Path(checkpoint_path).name}` — variant={variant}, epoch={epoch}, best val mAP50={best_map:.4f}"
    except Exception as e:  # pragma: no cover
        return f"Checkpoint: `{Path(checkpoint_path).name}` (metadata unavailable: {e})"


def build_app(checkpoint: str, config: str, device: str):
    def run(visible_img, thermal_img, conf_threshold):
        if visible_img is None or thermal_img is None:
            return None, "Please provide both a visible and a thermal image."

        with tempfile.TemporaryDirectory() as tmp:
            vis_path = Path(tmp) / "visible.png"
            thr_path = Path(tmp) / "thermal.png"
            out_path = Path(tmp) / "output.jpg"
            visible_img.save(vis_path)
            thermal_img.save(thr_path)

            try:
                _, n_det = predict(
                    checkpoint=checkpoint,
                    visible_path=str(vis_path),
                    thermal_path=str(thr_path),
                    config_path=config,
                    output_path=str(out_path),
                    conf_override=float(conf_threshold),
                    device=device,
                )
            except Exception as e:
                return None, f"Inference failed: {e}"

            from PIL import Image
            result = Image.open(out_path).convert("RGB")
            info = f"{n_det} detection(s) at confidence >= {conf_threshold:.2f}.  {checkpoint_info(checkpoint)}"
            return result, info

    example_vis = PROJECT_ROOT / "datasets" / "visible" / "test" / "00147.png"
    example_thr = PROJECT_ROOT / "datasets" / "thermal" / "test" / "00147.png"
    examples = [[str(example_vis), str(example_thr), 0.15]] if example_vis.exists() else None

    with gr.Blocks(title="TriNetra-AMRF — Detection Demo") as demo:
        gr.Markdown(
            "# TriNetra-AMRF — Live Detection Demo\n"
            "Upload a **visible (RGB)** image and its paired **thermal** image of the "
            "same scene, and the TRC-gated fusion detector will draw its detections "
            "on the output. Running on **CPU** so it doesn't compete with an "
            "in-progress training run's GPU usage — checkpoint is reloaded fresh on "
            "every run, so results reflect the latest saved training progress."
        )
        with gr.Row():
            visible_in = gr.Image(type="pil", label="Visible (RGB) image")
            thermal_in = gr.Image(type="pil", label="Thermal image (same scene)")
        conf_slider = gr.Slider(0.05, 0.9, value=0.15, step=0.05,
                                label="Confidence threshold (model is still mid-training — "
                                      "lower this if you see few/no detections)")
        run_btn = gr.Button("Run Detection", variant="primary")
        output_img = gr.Image(type="pil", label="Detections")
        info_box = gr.Markdown()

        run_btn.click(run, inputs=[visible_in, thermal_in, conf_slider],
                      outputs=[output_img, info_box])
        if examples:
            gr.Examples(examples=examples,
                       inputs=[visible_in, thermal_in, conf_slider])

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="TriNetra live detection demo UI")
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "weights" / "fusion" / "best.pt"))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cpu",
                    help="Default cpu — set to cuda only if the GPU is free (no training running).")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo = build_app(args.checkpoint, args.config, args.device)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
