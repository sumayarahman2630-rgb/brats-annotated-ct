"""Stage 3 -- official internal validation; output: per-patient Dice/IoU (CSV) plus mean/std on the synthetic CT validation split."""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import torch
import yaml

from inference.postprocessing import find_optimal_threshold, keep_largest_connected_component
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
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for the binary prediction (ignored if --auto_threshold is set).")
    parser.add_argument("--auto_threshold", action="store_true", help="Search for the global threshold maximizing mean Dice on this val set, instead of using --threshold.")
    parser.add_argument("--use_largest_component", action="store_true", help="Keep only the largest connected component of each thresholded prediction.")
    parser.add_argument("--min_size_ratio", type=float, default=0.0, help="With --use_largest_component: also keep any other component at least this fraction of the largest one's size (default 0.0 -- strict, only the single largest). Use e.g. 0.5 to preserve genuine bilateral/multi-focal disease.")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/stage3_synthetic_val_metrics.csv")
    return parser.parse_args()


def compute_synthetic_predictions(model, device, val_loader, patch_size: tuple[int, int, int]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Full sliding-window inference for every val patient -- returns
    (patient_id, probability_volume, mask_volume) triples WITHOUT
    thresholding, so the same predictions can be scored at multiple
    thresholds / post-processing settings without re-running the
    (expensive) sliding-window forward pass more than once."""
    results = []
    for batch in val_loader:
        patient_id = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        ct_vol = batch["ct"].to(device)
        mask_vol = batch["mask"].squeeze(0).squeeze(0).numpy()
        with torch.no_grad():
            pred_vol = model.predict_full_volume(ct_vol, patch_size=patch_size)
        pred_vol = pred_vol.squeeze(0).squeeze(0).float().cpu().numpy()
        results.append((patient_id, pred_vol, mask_vol))
    return results


def score_predictions(
    predictions: list[tuple[str, np.ndarray, np.ndarray]],
    threshold: float,
    use_largest_component: bool = False,
    min_size_ratio: float = 0.0,
) -> list[dict]:
    """Threshold + (optionally) largest-connected-component filter +
    Dice/IoU for a set of already-computed (patient_id, prob_vol,
    mask_vol) predictions. `min_size_ratio` is passed straight through to
    keep_largest_connected_component (default 0.0 -- strict, single
    largest only) -- must be threaded through consistently wherever
    use_largest_component is also passed to a visualization/display
    function scoring the SAME predictions, or the reported number and the
    displayed mask can disagree (real bug found and fixed 2026-07-25 in
    inference/generate_full_report.py, before min_size_ratio existed)."""
    rows = []
    for patient_id, prob_vol, mask_vol in predictions:
        pred_bin = (prob_vol > threshold).astype(np.float32)
        if use_largest_component:
            pred_bin = keep_largest_connected_component(pred_bin, min_size_ratio=min_size_ratio)
        dice, iou = dice_iou(pred_bin, mask_vol)
        log.info("%s: dice=%.4f iou=%.4f", patient_id, dice, iou)
        rows.append({"patient_id": patient_id, "dice": dice, "iou": iou})
    return rows


def evaluate_synthetic_val(model, device, val_loader, patch_size: tuple[int, int, int], threshold: float) -> list[dict]:
    """Backward-compatible convenience wrapper: compute + score at a fixed
    threshold, no post-processing. Equivalent to
    score_predictions(compute_synthetic_predictions(...), threshold)."""
    predictions = compute_synthetic_predictions(model, device, val_loader, patch_size)
    return score_predictions(predictions, threshold, use_largest_component=False)


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
    """Load the checkpoint (raw weights), run full-volume inference on
    every synthetic val patient, optionally search a threshold and/or apply
    largest-component filtering, then write per-patient Dice/IoU to CSV."""
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

    predictions = compute_synthetic_predictions(model, device, val_loader, patch_size)

    threshold = args.threshold
    if args.auto_threshold:
        prob_target_pairs = [(prob, mask) for _pid, prob, mask in predictions]
        threshold, searched_mean_dice = find_optimal_threshold(prob_target_pairs, dice_fn=lambda p, t: dice_iou(p, t)[0])
        log.info("--auto_threshold: selected threshold=%.2f (mean dice=%.4f during search, no post-processing)", threshold, searched_mean_dice)
        threshold_path = os.path.join(os.path.dirname(args.output_csv) or ".", "best_threshold.txt")
        os.makedirs(os.path.dirname(threshold_path) or ".", exist_ok=True)
        with open(threshold_path, "w") as f:
            f.write(f"{threshold:.2f}\n")
        log.info("Wrote selected threshold to %s -- reuse this as validate_jordan_segmentation.py's --threshold "
                  "(never search a threshold on Jordan directly).", threshold_path)

    rows = score_predictions(predictions, threshold, use_largest_component=args.use_largest_component, min_size_ratio=args.min_size_ratio)
    write_csv(rows, args.output_csv)

    dices = [r["dice"] for r in rows]
    ious = [r["iou"] for r in rows]
    log.info(
        "Synthetic validation (full-volume, sliding-window, threshold=%.2f, largest_component=%s): "
        "%d patients, mean dice=%.4f (std=%.4f), mean iou=%.4f (std=%.4f)",
        threshold, args.use_largest_component, len(rows),
        float(np.mean(dices)), float(np.std(dices)), float(np.mean(ious)), float(np.std(ious)),
    )
    log.info("Done. Per-patient results at %s.", args.output_csv)


if __name__ == "__main__":
    main()
