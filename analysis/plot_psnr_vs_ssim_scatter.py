"""Scatter plot of per-patient SSIM (x) vs. foreground PSNR (y), points
color-coded by PSNR value, to visualize whether the two metrics agree on
which validation patients the model does best/worst on.

Input: --metrics_csv, the CSV written by generate_val_metrics_table.py.

Output: a single PNG scatter plot with a colorbar, saved to --output.

Run as:
    python -m analysis.plot_psnr_vs_ssim_scatter \
        --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv \
        --output /kaggle/working/analysis_plots/psnr_vs_ssim_scatter.png
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import matplotlib.pyplot as plt

log = logging.getLogger("plot_psnr_vs_ssim_scatter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_csv", type=str, default="/kaggle/working/analysis_plots/val_metrics.csv")
    parser.add_argument("--output", type=str, default="/kaggle/working/analysis_plots/psnr_vs_ssim_scatter.png")
    return parser.parse_args()


def read_metrics(metrics_csv: str) -> tuple[list[str], list[float], list[float]]:
    """Read (patient_id, ssim_full, psnr_fg) triples from the metrics CSV,
    dropping any row with a NaN psnr_fg."""
    ids, ssim, psnr = [], [], []
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            p = float(row["psnr_fg"])
            if p != p:  # NaN check without importing math
                continue
            ids.append(row["patient_id"])
            ssim.append(float(row["ssim_full"]))
            psnr.append(p)
    return ids, ssim, psnr


def main():
    args = parse_args()
    if not os.path.exists(args.metrics_csv):
        raise SystemExit(f"{args.metrics_csv!r} not found -- run analysis.generate_val_metrics_table first.")

    ids, ssim, psnr = read_metrics(args.metrics_csv)
    if not ids:
        raise SystemExit(f"No usable rows in {args.metrics_csv!r}.")
    log.info("Plotting %d val patients", len(ids))

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(ssim, psnr, c=psnr, cmap="viridis", s=60, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Whole-volume SSIM")
    ax.set_ylabel("Foreground PSNR (dB)")
    ax.set_title(f"PSNR vs. SSIM per validation patient (n={len(ids)})")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Foreground PSNR (dB)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
