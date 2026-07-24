"""Single entry point that produces the full post-training Stage 3 report
in one run, so nothing needs to be requested/run individually after a
training session finishes:

1. Training loss/Dice curve (train + val, the whole run), read from
   training.log_file.
2. Internal (synthetic validation) full-volume Dice/IoU: mean/std,
   per-patient CSV + summary log (same computation as
   inference/validate_synthetic_segmentation.py).
3. External (Jordan) full-volume Dice/IoU: mean/std, per-slice CSV +
   summary log (same computation as
   inference/validate_jordan_segmentation.py), using the SAME threshold as
   the internal evaluation (never independently tuned on Jordan -- see
   inference/postprocessing.py's docstring).
4. A comparison bar chart against a literature baseline, ONLY if its
   numbers are supplied explicitly via --comparison_label /
   --comparison_internal_dice / --comparison_external_dice -- this script
   never fabricates or guesses a literature comparison value; if none are
   given, the chart is skipped with a clear log message explaining why.
5. 3-5 example prediction visualizations per source (CT / real mask /
   predicted mask, via inference/visualize_predictions.py's save_panel),
   sampled evenly across the sorted best-to-worst Dice range so the
   images show the actual spread of quality, not just the best cases.

Everything is saved to --output_dir; nothing is only printed to the console.

Run as (after training finishes):
    python -m inference.generate_full_report --config configs/stage3_ct_segmentation.yaml \\
        --comparison_label "Wang et al. (2024)" --comparison_internal_dice 0.71 --comparison_external_dice 0.58
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import torch
import yaml

from data.loaders_jordan_ct import discover_jordan_slices, JordanCTSegDataset
from inference.postprocessing import find_optimal_threshold, keep_largest_connected_component
from inference.validate_jordan_segmentation import _build_pseudo_volume, dice_iou, evaluate_jordan
from inference.validate_synthetic_segmentation import compute_synthetic_predictions, score_predictions, write_csv
from inference.visualize_predictions import _best_slice_index, save_panel
from models.unet3d_segmentation import build_segmentation_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.train_stage3_segmentation import build_synthetic_ct_dataloaders

log = logging.getLogger("generate_full_report")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Use this exact checkpoint instead of auto-finding the latest one.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/stage3_full_report")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for the binary prediction (ignored if --auto_threshold is set).")
    parser.add_argument("--auto_threshold", action="store_true", help="Search for the global threshold maximizing mean internal Dice, and reuse it for the external evaluation too.")
    parser.add_argument("--use_largest_component", action="store_true", help="Keep only the largest connected component of each thresholded prediction, both sources.")
    parser.add_argument("--replication_depth", type=int, default=16, help="Jordan pseudo-3D slice replication depth -- see data/loaders_jordan_ct.py's module docstring.")
    parser.add_argument("--num_visualizations", type=int, default=5, help="Example prediction panels per source, sampled evenly across the sorted best-to-worst Dice range.")
    parser.add_argument("--comparison_label", type=str, default=None, help="Literature baseline name for the comparison chart, e.g. 'Wang et al. (2024)'. No chart is produced unless this AND at least one comparison dice value are supplied.")
    parser.add_argument("--comparison_internal_dice", type=float, default=None)
    parser.add_argument("--comparison_external_dice", type=float, default=None)
    return parser.parse_args()


def plot_training_curves(log_file: str, output_dir: str) -> None:
    """Loss and Dice, train + val, across the whole run -- read directly
    from training.log_file (train_stage3_segmentation.py's own CSV), no
    re-computation needed."""
    import matplotlib.pyplot as plt

    if not os.path.exists(log_file):
        log.warning("Training log %s not found -- skipping the training curve plot.", log_file)
        return

    rows = {"train": {"step": [], "loss": [], "dice": []}, "val": {"step": [], "loss": [], "dice": []}}
    with open(log_file) as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in rows:
                continue
            rows[split]["step"].append(int(row["step"]))
            rows[split]["loss"].append(float(row["loss"]))
            rows[split]["dice"].append(float(row["dice_score"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(rows["train"]["step"], rows["train"]["loss"], label="train", alpha=0.6, linewidth=0.8)
    axes[0].plot(rows["val"]["step"], rows["val"]["loss"], label="val", marker="o", markersize=3)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(rows["train"]["step"], rows["train"]["dice"], label="train", alpha=0.6, linewidth=0.8)
    axes[1].plot(rows["val"]["step"], rows["val"]["dice"], label="val", marker="o", markersize=3)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("dice"); axes[1].set_title("Dice (periodic patch-level check -- NOT the full-volume metric)"); axes[1].legend()

    plt.suptitle("Stage 3 training run")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info("Saved %s", out_path)


def _select_evenly_spread(rows: list[dict], n: int) -> list[dict]:
    """Sort by dice descending, then pick n entries evenly spaced across
    the sorted list -- shows the actual best-to-worst spread rather than
    just the top n (which would only ever show the best cases)."""
    sorted_rows = sorted(rows, key=lambda r: r["dice"], reverse=True)
    if len(sorted_rows) <= n:
        return sorted_rows
    indices = np.linspace(0, len(sorted_rows) - 1, n).round().astype(int)
    seen = set()
    out = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(sorted_rows[i])
    return out


def save_synthetic_visualizations_with_ct(
    predictions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    scored_rows: list[dict], threshold: float, output_dir: str, n: int,
) -> None:
    """predictions: (patient_id, ct_vol, prob_vol, mask_vol) tuples, already
    computed -- no extra inference needed for the visualizations."""
    selected = _select_evenly_spread(scored_rows, n)
    pred_by_patient = {pid: (ct, prob, mask) for pid, ct, prob, mask in predictions}
    out_dir = os.path.join(output_dir, "examples_synthetic")

    for rank, row in enumerate(selected, start=1):
        ct_vol, prob_vol, mask_vol = pred_by_patient[row["patient_id"]]
        slice_idx = _best_slice_index(mask_vol)
        out_path = os.path.join(out_dir, f"{rank:02d}_dice{row['dice']:.3f}_{row['patient_id']}.png")
        save_panel(
            ct_vol[slice_idx], mask_vol[slice_idx], prob_vol[slice_idx], threshold,
            f"synthetic -- {row['patient_id']}, dice={row['dice']:.3f}, slice {slice_idx}", out_path,
        )


def save_jordan_visualizations(model, device, jordan_ct_root: str, jordan_mask_root: str, jordan_rows: list[dict], replication_depth: int, spatial_multiple: int, threshold: float, output_dir: str, n: int) -> None:
    """Re-runs pseudo-3D inference only for the small number of selected
    slices (cheap -- Jordan has at most a few dozen slices total), rather
    than threading prediction volumes through evaluate_jordan's existing,
    unmodified return type."""
    selected = _select_evenly_spread(jordan_rows, n)
    slices = discover_jordan_slices(jordan_ct_root, jordan_mask_root)
    slice_by_key = {(s.patient_id, s.slice_num): s for s in slices}
    out_dir = os.path.join(output_dir, "examples_jordan")

    for rank, row in enumerate(selected, start=1):
        key = (row["patient_id"], row["slice_num"])
        s = slice_by_key.get(key)
        if s is None:
            continue
        item = JordanCTSegDataset([s])[0]
        ct_2d = item["ct"].squeeze(0).numpy()
        mask_2d = item["mask"].squeeze(0).numpy()
        pseudo_volume, center_index = _build_pseudo_volume(ct_2d, replication_depth, spatial_multiple)
        ct_tensor = torch.from_numpy(pseudo_volume).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_volume = model.predict_full_volume(ct_tensor, patch_size=pseudo_volume.shape)
        pred_center = pred_volume.squeeze(0).squeeze(0).float().cpu().numpy()[center_index][: mask_2d.shape[0], : mask_2d.shape[1]]
        out_path = os.path.join(out_dir, f"{rank:02d}_dice{row['dice']:.3f}_{row['patient_id']}_slice{row['slice_num']}.png")
        save_panel(ct_2d, mask_2d, pred_center, threshold, f"Jordan -- {row['patient_id']} slice {row['slice_num']}, dice={row['dice']:.3f}", out_path)


def save_comparison_chart(internal_dice: float, external_dice: float, comparison_label: str, comparison_internal_dice: float | None, comparison_external_dice: float | None, output_dir: str) -> None:
    """Grouped bar chart: this work vs. the supplied literature baseline.
    Only called when the caller confirmed at least one comparison value
    was actually supplied -- never fabricates a baseline number itself."""
    import matplotlib.pyplot as plt

    categories, this_work, other_work = [], [], []
    if comparison_internal_dice is not None:
        categories.append("Internal (synthetic val)")
        this_work.append(internal_dice)
        other_work.append(comparison_internal_dice)
    if comparison_external_dice is not None:
        categories.append("External (Jordan)")
        this_work.append(external_dice)
        other_work.append(comparison_external_dice)

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, this_work, width, label="This work")
    ax.bar(x + width / 2, other_work, width, label=comparison_label)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.0)
    ax.set_title("Dice comparison")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison_chart.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info("Saved %s", out_path)


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

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
    step, _extra = load_checkpoint(ckpt_path, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded checkpoint %s (step %d) -- RAW weights, no EMA involved", ckpt_path, step)
    model.eval()

    # 1. Training curves
    plot_training_curves(config["training"]["log_file"], args.output_dir)

    # 2. Internal (synthetic) full-volume Dice/IoU
    _train_loader, val_loader = build_synthetic_ct_dataloaders(config, seed=config.get("seed", 0))
    patch_size = tuple(config["data"]["patch_size"])
    predictions = compute_synthetic_predictions(model, device, val_loader, patch_size)

    threshold = args.threshold
    if args.auto_threshold:
        prob_target_pairs = [(prob, mask) for _pid, prob, mask in predictions]
        threshold, searched_mean_dice = find_optimal_threshold(prob_target_pairs, dice_fn=lambda p, t: dice_iou(p, t)[0])
        log.info("--auto_threshold: selected threshold=%.2f (search mean dice=%.4f)", threshold, searched_mean_dice)

    synthetic_rows = score_predictions(predictions, threshold, use_largest_component=args.use_largest_component)
    write_csv(synthetic_rows, os.path.join(args.output_dir, "internal_synthetic_metrics.csv"))
    internal_dices = [r["dice"] for r in synthetic_rows]
    internal_mean, internal_std = float(np.mean(internal_dices)), float(np.std(internal_dices))
    log.info("INTERNAL (synthetic val, full-volume): %d patients, mean dice=%.4f (std=%.4f)", len(synthetic_rows), internal_mean, internal_std)

    # 3. External (Jordan) full-volume Dice/IoU -- same threshold as internal, never independently tuned
    data_cfg = config["data"]
    jordan_rows = evaluate_jordan(
        model, device, data_cfg["jordan_ct_root"], data_cfg["jordan_mask_root"], args.replication_depth,
        data_cfg.get("spatial_multiple", 16), threshold, use_largest_component=args.use_largest_component,
    )
    jordan_csv_fields = ["patient_id", "slice_num", "dice", "iou"]
    jordan_csv_path = os.path.join(args.output_dir, "external_jordan_metrics.csv")
    with open(jordan_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=jordan_csv_fields)
        writer.writeheader()
        for row in jordan_rows:
            writer.writerow(row)
    external_dices = [r["dice"] for r in jordan_rows]
    external_mean, external_std = float(np.mean(external_dices)), float(np.std(external_dices))
    log.info("EXTERNAL (Jordan, full-volume, threshold=%.2f reused from internal): %d slices, mean dice=%.4f (std=%.4f)", threshold, len(jordan_rows), external_mean, external_std)

    # 4. Comparison chart -- only if the caller supplied real numbers
    if args.comparison_label and (args.comparison_internal_dice is not None or args.comparison_external_dice is not None):
        save_comparison_chart(internal_mean, external_mean, args.comparison_label, args.comparison_internal_dice, args.comparison_external_dice, args.output_dir)
    else:
        log.info("Skipping the comparison chart -- no --comparison_label/--comparison_*_dice values were supplied. "
                 "This script does not fabricate literature numbers; pass the real reported values explicitly to get the chart.")

    # 5. Example visualizations, sorted best-to-worst, both sources
    predictions_with_ct = []
    for (patient_id, prob_vol, mask_vol), batch in zip(predictions, val_loader):
        ct_vol = batch["ct"].squeeze(0).squeeze(0).numpy()
        predictions_with_ct.append((patient_id, ct_vol, prob_vol, mask_vol))
    save_synthetic_visualizations_with_ct(predictions_with_ct, synthetic_rows, threshold, args.output_dir, args.num_visualizations)
    save_jordan_visualizations(
        model, device, data_cfg["jordan_ct_root"], data_cfg["jordan_mask_root"], jordan_rows,
        args.replication_depth, data_cfg.get("spatial_multiple", 16), threshold, args.output_dir, args.num_visualizations,
    )

    log.info("Done. Full report saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
