"""Box plots of the per-patient PSNR and SSIM distributions across the
validation split, each with individual patient points overlaid (jittered
scatter) so the box plot's summary statistics can be checked against the
actual spread, not just trusted blindly.

Input: --metrics_csv, the CSV written by generate_val_metrics_table.py
(columns: patient_id, l1_norm, psnr_full, ssim_full, psnr_fg). Uses
foreground-only PSNR (psnr_fg) as "the" PSNR metric throughout this
project -- whole-volume PSNR is inflated by trivially-easy background
agreement (see PROJECT_NOTES.md's round-8 audit) -- and whole-volume SSIM
(ssim_full), the only SSIM this project computes.

Output: a single PNG with two side-by-side box plots, saved to --output.

Run as:
    python -m analysis.plot_psnr_ssim_distribution \
        --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv \
        --output /kaggle/working/analysis_plots/psnr_ssim_distribution.png
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger("plot_psnr_ssim_distribution")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_csv", type=str, default="/kaggle/working/analysis_plots/val_metrics.csv")
    parser.add_argument("--output", type=str, default="/kaggle/working/analysis_plots/psnr_ssim_distribution.png")
    return parser.parse_args()


def read_metrics(metrics_csv: str) -> tuple[list[float], list[float]]:
    """Read (psnr_fg, ssim_full) pairs from the metrics CSV, dropping any
    row with a NaN psnr_fg (patients with an empty brain mask, if any)."""
    psnr, ssim = [], []
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            p = float(row["psnr_fg"])
            if p != p:  # NaN check without importing math
                continue
            psnr.append(p)
            ssim.append(float(row["ssim_full"]))
    return psnr, ssim


def _boxplot_with_points(ax, values: list[float], label: str, color: str) -> None:
    """One box plot with individual points jittered slightly on the x-axis
    so overlapping values are still visible."""
    ax.boxplot(values, positions=[1], widths=0.5, showfliers=False,
               boxprops=dict(color=color), medianprops=dict(color=color, linewidth=2))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(values))
    ax.scatter(np.full(len(values), 1.0) + jitter, values, color=color, alpha=0.6, s=25, zorder=3)
    ax.set_xticks([1])
    ax.set_xticklabels([label])


def main():
    args = parse_args()
    if not os.path.exists(args.metrics_csv):
        raise SystemExit(f"{args.metrics_csv!r} not found -- run analysis.generate_val_metrics_table first.")

    psnr, ssim = read_metrics(args.metrics_csv)
    if not psnr:
        raise SystemExit(f"No usable rows in {args.metrics_csv!r}.")
    log.info("Plotting distribution over %d val patients", len(psnr))

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    _boxplot_with_points(axes[0], psnr, "foreground PSNR (dB)", "#2b6cb0")
    axes[0].set_ylabel("dB")
    axes[0].set_title("PSNR")
    axes[0].grid(True, alpha=0.3, axis="y")

    _boxplot_with_points(axes[1], ssim, "whole-volume SSIM", "#c05621")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Validation metric distribution (n={len(psnr)} patients)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
