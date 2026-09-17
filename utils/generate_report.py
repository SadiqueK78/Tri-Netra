#!/usr/bin/env python3
"""
TriNetra-AMRF — Per-Epoch Training Report Generator
====================================================

Purpose:
    Reads a training run's ``training_log.csv`` (written incrementally,
    one row per epoch, by ``training.train_fusion.fit()``) and renders a
    human-readable markdown report alongside it — a per-epoch table plus a
    running summary (best epoch so far, elapsed time, ETA). Intended to be
    run as a companion process alongside an already-running training job:
    it only reads the CSV, never touches the training process, so it can
    be started/stopped/restarted at any time without any risk to training.

Usage:
    # One-shot: regenerate the report once from the current CSV state.
    python utils/generate_report.py --csv weights/fusion/training_log.csv

    # Watch mode: regenerate every 60s until the CSV stops growing for
    # `--idle-exit` seconds (i.e. training finished or stalled), or forever
    # with --idle-exit 0.
    python utils/generate_report.py --csv weights/fusion/training_log.csv --watch
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Dict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


def read_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except ValueError:
                continue  # a row being written mid-flush; skip, pick it up next pass
    return rows


def render_report(rows: List[Dict[str, float]], title: str, total_epochs: int) -> str:
    if not rows:
        return f"# {title}\n\nNo epochs logged yet.\n"

    best_i = max(range(len(rows)), key=lambda i: rows[i]["val_mAP50"])
    best = rows[best_i]
    last = rows[-1]
    total_time = sum(r["epoch_time_s"] for r in rows)
    avg_time = total_time / len(rows)
    remaining = max(0, total_epochs - len(rows))
    eta_s = remaining * avg_time

    def fmt_hms(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    lines = [
        f"# {title}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Epochs completed:** {len(rows)} / {total_epochs}",
        f"- **Best epoch so far:** {int(best['epoch'])} "
        f"(val mAP@0.5 = {best['val_mAP50']:.4f}, mAP@0.5:0.95 = {best['val_mAP50_95']:.4f})",
        f"- **Latest epoch:** {int(last['epoch'])} "
        f"(val mAP@0.5 = {last['val_mAP50']:.4f}, lr = {last['lr']:.2e})",
        f"- **Elapsed training time:** {fmt_hms(total_time)} "
        f"(avg {avg_time:.1f}s/epoch)",
        f"- **Estimated time remaining:** {fmt_hms(eta_s)} "
        f"({remaining} epochs left, assuming no early stop and constant epoch time)",
        "",
        "## Per-Epoch Log",
        "",
        "| Epoch | Box Loss | Cls Loss | DFL Loss | Val mAP@0.5 | Val mAP@0.5:0.95 | LR | Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        marker = " **★ best**" if int(r["epoch"]) == int(best["epoch"]) else ""
        lines.append(
            f"| {int(r['epoch'])} | {r['box_loss']:.4f} | {r['cls_loss']:.4f} | "
            f"{r['dfl_loss']:.4f} | {r['val_mAP50']:.4f} | {r['val_mAP50_95']:.4f} | "
            f"{r['lr']:.2e} | {r['epoch_time_s']:.1f}{marker} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a per-epoch training report from a CSV log.")
    ap.add_argument("--csv", required=True, help="Path to training_log.csv")
    ap.add_argument("--out", default=None, help="Output .md path (default: alongside the CSV)")
    ap.add_argument("--title", default="TriNetra Fusion+TRC — Training Report")
    ap.add_argument("--total-epochs", type=int, default=150)
    ap.add_argument("--watch", action="store_true", help="Regenerate every --interval seconds.")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--idle-exit", type=int, default=1800,
                     help="Exit watch mode after this many seconds with no new epoch "
                          "rows (0 = never exit on its own).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_name("training_report.md")

    last_n, idle_for = -1, 0
    while True:
        if csv_path.exists():
            rows = read_rows(csv_path)
            report = render_report(rows, args.title, args.total_epochs)
            out_path.write_text(report, encoding="utf-8")
            if len(rows) != last_n:
                print(f"[OK] report updated ({len(rows)} epochs) -> {out_path}", flush=True)
                last_n, idle_for = len(rows), 0
            else:
                idle_for += args.interval
        else:
            print(f"[..] waiting for {csv_path} to appear", flush=True)

        if not args.watch:
            break
        if args.idle_exit and idle_for >= args.idle_exit:
            print(f"[DONE] no new epochs for {idle_for}s — exiting watch mode.", flush=True)
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
