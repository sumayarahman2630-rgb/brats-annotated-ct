"""Quantitative, full-volume Dice/IoU evaluation of a Stage 3 checkpoint on
the SYNTHETIC validation split (the patient-level held-out split
build_synthetic_ct_dataloaders produces, e.g. 37 patients out of 368 under
data.train_val_split=0.9) -- the "official" segmentation quality number
for this dataset, as distinct from two other, DIFFERENT numbers this
project produces that must not be reported interchangeably with it:

1. training/train_stage3_segmentation.py's periodic quick_validation
   check, computed on a center-cropped (not tumor-centered) PATCH of each
   val volume, for cheap in-training monitoring only.
2. inference/validate_jordan_segmentation.py's Dice/IoU, computed on the
   Jordan EXTERNAL dataset (real hospital CT, 8-bit windowed RGB, pseudo-3D
   slice replication) -- a different, out-of-distribution population.

This script runs the SAME full-volume sliding-window inference
(models/unet3d_segmentation.py's predict_full_volume) that
inference/visualize_predictions.py uses for its qualitative panels, but
over every validation patient rather than a handful, and computes/logs the
quantitative Dice and IoU each time rather than only rendering an image.
See methodology_draft.md's Section 6.2 for how this number should be
described.

Run as:
    python -m inference.validate_synthetic_segmentation --config configs/stage3_ct_segmentation.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import torch
import yaml

from inference.validate_jordan_segmentation import dice_iou
from models.unet3d_segmentation import build_segmentation_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.train_stage3_segmentation import build_synthetic_ct_dataloaders

log = logging.getLogger("validate_synthetic_segmentation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CSV_FIELDS = ["patient_id", "dice", "iou"]


def parse_args():
    """--config resolves the checkpoint and dataset paths; the rest are overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Use this exact checkpoint instead of auto-finding the latest one.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for the binary prediction.")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/stage3_synthetic_val_metrics.csv")
    return parser.parse_args()


def evaluate_synthetic_val(model, device, val_loader, patch_size: tuple[int, int, int], threshold: float) -> list[dict]:
    """Full sliding-window inference + thresholded Dice/IoU for every val
    patient -- the real, full-volume number this script exists to produce,
    as opposed to the periodic in-training patch-level check."""
    rows = []
    for batch in val_loader:
        patient_id = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        ct_vol = batch["ct"].to(device)
        mask_vol = batch["mask"].squeeze(0).squeeze(0).numpy()

        with torch.no_grad():
            pred_vol = model.predict_full_volume(ct_vol, patch_size=patch_size)
        pred_vol = pred_vol.squeeze(0).squeeze(0).float().cpu().numpy()
        pred_bin = (pred_vol > threshold).astype(np.float32)

        dice, iou = dice_iou(pred_bin, mask_vol)
        log.info("%s: dice=%.4f iou=%.4f", patient_id, dice, iou)
        rows.append({"patient_id": patient_id, "dice": dice, "iou": iou})
    return rows


def write_csv(rows: list[dict], output_csv: str) -> None:
    """Write per-patient full-volume Dice/IoU to output_csv."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Wrote %d rows to %s", len(rows), output_csv)


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = build_segmentation_model(config).to(device)
    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"--checkpoint_path {ckpt_path!r} does not exist.")
    else:
        search_dirs = [config["checkpoint"]["working_dir"]] + list(config["checkpoint"].get("extra_resume_dirs", []))
        ckpt_path = find_latest_checkpoint(search_dirs)
        if ckpt_path is None:
            raise RuntimeError(f"No Stage 3 checkpoint found in {search_dirs}.")
    # ema=None: raw weights only, same anti-EMA-contamination pattern as every
    # other evaluation script in this project.
    step, _extra = load_checkpoint(ckpt_path, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded checkpoint %s (step %d) -- RAW weights, no EMA involved", ckpt_path, step)
    model.eval()

    _train_loader, val_loader = build_synthetic_ct_dataloaders(config, seed=config.get("seed", 0))
    log.info("Synthetic validation set: %d patients (full-volume sliding-window inference)", len(val_loader.dataset))
    patch_size = tuple(config["data"]["patch_size"])

    rows = evaluate_synthetic_val(model, device, val_loader, patch_size, args.threshold)
    write_csv(rows, args.output_csv)

    dices = [r["dice"] for r in rows]
    ious = [r["iou"] for r in rows]
    log.info(
        "Synthetic validation (full-volume, sliding-window): %d patients, mean dice=%.4f (std=%.4f), mean iou=%.4f (std=%.4f)",
        len(rows), float(np.mean(dices)), float(np.std(dices)), float(np.mean(ious)), float(np.std(ious)),
    )
    log.info("Done. Per-patient results at %s.", args.output_csv)


if __name__ == "__main__":
    main()
