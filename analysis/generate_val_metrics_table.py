"""Computes per-patient PSNR/SSIM/L1 over the regression model's held-out
validation split and writes them to a CSV -- the data source for
plot_psnr_ssim_distribution.py and plot_psnr_vs_ssim_scatter.py (both take
--metrics_csv rather than recomputing this themselves).

Input: a trained regression checkpoint (found the same way every other
script in this project finds one -- checkpoint.working_dir /
checkpoint.extra_resume_dirs in the given --config), plus the SynthRAD val
split that config resolves to (same patient-level split algorithm as
training, so this only ever scores patients the model never trained on).

Output: --output_csv (one row per val patient: patient_id, l1_norm,
psnr_full, ssim_full, psnr_fg) and a summary table (mean/std/min/max per
metric) printed to the console.

Run as:
    python -m analysis.generate_val_metrics_table --config configs/stage1_regression.yaml \
        --output_csv /kaggle/working/analysis_plots/val_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import statistics

import torch
import torch.nn.functional as F
import yaml

from data.preprocessing import denormalize_ct
from models.unet3d_regression import build_regression_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.train_stage1_regression import build_regression_dataloaders

log = logging.getLogger("generate_val_metrics_table")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CSV_FIELDS = ["patient_id", "l1_norm", "psnr_full", "ssim_full", "psnr_fg"]


def parse_args():
    """--config resolves the checkpoint and val split; --checkpoint_path
    overrides which exact checkpoint to evaluate."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage1_regression.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Use this exact checkpoint instead of auto-finding the latest one.")
    parser.add_argument("--num_patients", type=int, default=None, help="Defaults to the entire val split.")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/analysis_plots/val_metrics.csv")
    return parser.parse_args()


def compute_val_metrics(config: dict, checkpoint_path: str | None, num_patients: int | None) -> list[dict]:
    """Load the checkpoint (raw weights, same anti-EMA-contamination pattern
    as every other evaluation script in this project) and run every
    requested val patient through sliding-window inference, returning one
    metrics dict per patient."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_regression_model(config).to(device)

    if checkpoint_path is None:
        search_dirs = [config["checkpoint"]["working_dir"]] + list(config["checkpoint"].get("extra_resume_dirs", []))
        checkpoint_path = find_latest_checkpoint(search_dirs)
        if checkpoint_path is None:
            raise RuntimeError(f"No regression checkpoint found in {search_dirs}.")
    step, _extra = load_checkpoint(checkpoint_path, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded checkpoint %s (step %d) -- RAW weights", checkpoint_path, step)
    model.eval()

    _train_loader, val_loader = build_regression_dataloaders(config, seed=config.get("seed", 0))
    n_val = len(val_loader.dataset)
    num_patients = num_patients or n_val
    log.info("Validation set: %d patients, scoring up to %d", n_val, num_patients)

    ct_clip_range = tuple(config["data"].get("ct_clip_range", (-1000.0, 3000.0)))
    data_range_hu = ct_clip_range[1] - ct_clip_range[0]
    patch_size = config["data"].get("patch_size")
    if not patch_size:
        raise RuntimeError("data.patch_size must be set -- reused as the sliding-window size.")
    patch_size = tuple(patch_size)

    rows = []
    for i, batch in enumerate(val_loader):
        if i >= num_patients:
            break
        patient_id = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        mri = batch["mri"].to(device)
        real_ct_norm = batch["ct"].squeeze(0).squeeze(0).numpy()
        mask_arr = batch["mask"].squeeze(0).squeeze(0).numpy().astype(bool)

        pred_norm = model.predict_full_volume(mri, patch_size=patch_size)
        l1_norm = F.l1_loss(pred_norm, batch["ct"].to(device)).item()
        pred_norm = pred_norm.squeeze(0).squeeze(0).float().cpu().numpy()

        real_hu = denormalize_ct(real_ct_norm, *ct_clip_range)
        synth_hu = denormalize_ct(pred_norm, *ct_clip_range)

        psnr_full = peak_signal_noise_ratio(real_hu, synth_hu, data_range=data_range_hu)
        ssim_full = structural_similarity(real_hu, synth_hu, data_range=data_range_hu)
        psnr_fg = (
            peak_signal_noise_ratio(real_hu[mask_arr], synth_hu[mask_arr], data_range=data_range_hu)
            if mask_arr.any() else float("nan")
        )

        log.info("%s: L1=%.4f PSNR=%.2fdB SSIM=%.4f fg-PSNR=%.2fdB", patient_id, l1_norm, psnr_full, ssim_full, psnr_fg)
        rows.append({
            "patient_id": patient_id, "l1_norm": l1_norm,
            "psnr_full": psnr_full, "ssim_full": ssim_full, "psnr_fg": psnr_fg,
        })
    return rows


def write_csv(rows: list[dict], output_csv: str) -> None:
    """Write the per-patient metrics rows to output_csv."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Wrote %d rows to %s", len(rows), output_csv)


def print_summary_table(rows: list[dict]) -> None:
    """Print a mean/std/min/max summary per metric to the console."""
    metrics = ["l1_norm", "psnr_full", "ssim_full", "psnr_fg"]
    header = f"{'metric':<12}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    for metric in metrics:
        values = [r[metric] for r in rows if r[metric] == r[metric]]  # drop NaN
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(f"{metric:<12}{mean:>10.3f}{std:>10.3f}{min(values):>10.3f}{max(values):>10.3f}")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rows = compute_val_metrics(config, args.checkpoint_path, args.num_patients)
    if not rows:
        raise SystemExit("No val patients were scored -- check the config's val split isn't empty.")

    write_csv(rows, args.output_csv)
    print()
    print_summary_table(rows)


if __name__ == "__main__":
    main()
