#!/usr/bin/env python3
"""
TriNetra-AMRF — Four-Way Model Comparison Demo UI
====================================================

Purpose:
    Upload ONE visible+thermal image pair and see all four Phase-5
    ablation variants — RGB-only, Thermal-only, Fusion (no TRC), and
    Fusion (TRC) — run on the SAME image side by side. This is the
    interactive counterpart to the ablation results table: instead of
    reading mAP numbers, a reviewer can watch the same scene detected
    differently (or identically) by each variant in real time.

    These four checkpoints were trained on LLVIP (the Phase-5 comparative
    ablation pass), not M3FD — so example/test images should come from
    datasets/LLVIP/, not the M3FD split the live Fusion+TRC training run
    uses. Each checkpoint carries its own config snapshot from training
    time, so it's scored under the exact settings it was trained with
    regardless of what configs/*.yaml currently looks like.

    Runs on CPU by default — deliberately isolated from the GPU so it can
    run alongside an active training job with zero memory contention.

Usage:
    python app/compare_ui.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict_image import predict  # noqa: E402

# Known clean-test-set numbers from the Phase-5 ablation (for on-screen context —
# not recomputed live, just the headline figures already measured).
MODELS = {
    "RGB-only":        ("weights/ablation/poc/rgb/best.pt",           "mAP@0.5=68.5  |  R only"),
    "Thermal-only":    ("weights/ablation/poc/thermal/best.pt",       "mAP@0.5=60.3  |  T only"),
    "Fusion (no TRC)": ("weights/ablation/poc/fusion_no_trc/best.pt", "mAP@0.5=79.6  |  T+R, gate disabled"),
    "Fusion (TRC)":    ("weights/ablation/poc/fusion_trc/best.pt",    "mAP@0.5=79.6  |  T+R, adaptive gate"),
}


def build_app(config: str, device: str):
    def run(visible_img, thermal_img, conf_threshold):
        if visible_img is None or thermal_img is None:
            return [None] * len(MODELS) + ["Please provide both a visible and a thermal image."]

        outputs, notes = [], []
        with tempfile.TemporaryDirectory() as tmp:
            vis_path = Path(tmp) / "visible.png"
            thr_path = Path(tmp) / "thermal.png"
            visible_img.save(vis_path)
            thermal_img.save(thr_path)

            from PIL import Image
            for name, (ckpt_rel, _) in MODELS.items():
                ckpt = str(PROJECT_ROOT / ckpt_rel)
                out_path = Path(tmp) / f"{name}.jpg"
                try:
                    _, n_det = predict(
                        checkpoint=ckpt, visible_path=str(vis_path),
                        thermal_path=str(thr_path), config_path=config,
                        output_path=str(out_path), conf_override=float(conf_threshold),
                        device=device,
                    )
                    outputs.append(Image.open(out_path).convert("RGB"))
                    notes.append(f"**{name}**: {n_det} detection(s)")
                except Exception as e:
                    outputs.append(None)
                    notes.append(f"**{name}**: failed ({e})")

        return outputs + ["  \n".join(notes)]

    example_vis = PROJECT_ROOT / "datasets" / "LLVIP" / "visible" / "test" / "190001.jpg"
    example_thr = PROJECT_ROOT / "datasets" / "LLVIP" / "infrared" / "test" / "190001.jpg"
    examples = [[str(example_vis), str(example_thr), 0.25]] if example_vis.exists() else None

    with gr.Blocks(title="TriNetra-AMRF — Four-Way Comparison") as demo:
        gr.Markdown(
            "# TriNetra-AMRF — Four-Way Model Comparison\n"
            "Upload one **visible+thermal** LLVIP-style pair (these four checkpoints "
            "were trained on LLVIP, not M3FD) and see all four Phase-5 ablation "
            "variants detect on the exact same image, side by side. Running on "
            "**CPU** so it doesn't compete with an active GPU training run."
        )
        with gr.Row():
            visible_in = gr.Image(type="pil", label="Visible (RGB) image")
            thermal_in = gr.Image(type="pil", label="Thermal image (same scene)")
        conf_slider = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
        run_btn = gr.Button("Run All Four Models", variant="primary")

        out_imgs = []
        with gr.Row():
            for name, (_, subtitle) in MODELS.items():
                with gr.Column():
                    gr.Markdown(f"**{name}**  \n*{subtitle}*")
                    out_imgs.append(gr.Image(type="pil", label=name, show_label=False))
        info_box = gr.Markdown()

        run_btn.click(run, inputs=[visible_in, thermal_in, conf_slider],
                      outputs=out_imgs + [info_box])
        if examples:
            gr.Examples(examples=examples, inputs=[visible_in, thermal_in, conf_slider])

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="TriNetra four-way model comparison demo UI")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="Fallback config; each checkpoint's own training-time "
                         "config snapshot takes priority automatically.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    demo = build_app(args.config, args.device)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
