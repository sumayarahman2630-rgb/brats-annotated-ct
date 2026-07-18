"""Plots validation foreground PSNR (dB) against training step, from the
regression pipeline's own training log CSV.

Input: the CSV written by training/train_stage1_regression.py's
init_log_file/append_log_row (columns: step, split, l1_loss, psnr_fg_db,
lr, elapsed_sec). Only rows with split == "val" and a non-empty psnr_fg_db
are plotted -- val rows only appear when quick_validation successfully
completes (see DEVELOPMENT_LOG.md: a validation OOM at a given step is now logged
as a warning and skipped, not a crash, so a gap in the curve at a given
step means that step's validation was skipped, not that training stopped).

Output: a single PNG line plot (step vs. PSNR), saved to --output.

Run as:
    python -m analysis.plot_validation_psnr_curve \
        --log_file /kaggle/working/logs/stage1_regression_log.csv \
        --output /kaggle/working/analysis_plots/validation_psnr_curve.png

Or, to read the log path out of the training config instead of typing it:
    python -m analysis.plot_validation_psnr_curve --config configs/stage1_regression.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import matplotlib.pyplot as plt
import yaml

log = logging.getLogger("plot_validation_psnr_curve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    """--log_file directly, or --config to read training.log_file from a
    stage1_regression.yaml-style config -- --log_file wins if both given."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_file", type=str, default=None, help="Path to the training CSV log.")
    parser.add_argument("--config", type=str, default="configs/stage1_regression.yaml",
                         help="Used to resolve --log_file's default (training.log_file) if --log_file isn't given.")
    parser.add_argument("--output", type=str, default="/kaggle/working/analysis_plots/validation_psnr_curve.png")
    return parser.parse_args()


def resolve_log_file(args) -> str:
    """--log_file if given, else training.log_file from --config."""
    if args.log_file:
        return args.log_file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config["training"]["log_file"]


def read_val_psnr_series(log_file: str) -> tuple[list[int], list[float]]:
    """Parse the training CSV log and return (steps, psnr_fg_db) for every
    val row that has a real (non-empty) PSNR value."""
    steps, psnr = [], []
    with open(log_file, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            if not row["psnr_fg_db"]:
                continue
            steps.append(int(row["step"]))
            psnr.append(float(row["psnr_fg_db"]))
    return steps, psnr


def main():
    args = parse_args()
    log_file = resolve_log_file(args)
    if not os.path.exists(log_file):
        raise SystemExit(f"Training log not found: {log_file!r}. Has training run at least once?")

    steps, psnr = read_val_psnr_series(log_file)
    if not steps:
        raise SystemExit(f"No 'val' rows with a PSNR value found in {log_file!r} -- nothing to plot.")
    log.info("Read %d validation PSNR points from %s (step %d -> %d)", len(steps), log_file, steps[0], steps[-1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, psnr, marker="o", markersize=4, linewidth=1.5, color="#2b6cb0")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Foreground-only validation PSNR (dB)")
    ax.set_title("Regression U-Net: validation PSNR over training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info("Saved %s (final PSNR: %.2f dB at step %d)", args.output, psnr[-1], steps[-1])


if __name__ == "__main__":
    main()
