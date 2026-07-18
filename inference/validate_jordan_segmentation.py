"""External validation of the Stage 3 segmentation checkpoint against the
Jordan University Hospital dataset -- real CT, real tumor annotations,
never seen during training. Also produces a quick visualization comparing
one synthetic (training-distribution) example against one Jordan
(external) example side by side.

Read PROJECT_NOTES.md's Stage 3 section before trusting these numbers as
more than a rough signal -- three of this script's own design choices are
direct, load-bearing workarounds for known dataset limitations, not
incidental implementation details:

1. **No real 3D context.** Jordan only has isolated tumor-containing
   slices (1-6 per patient), not full volumes -- there is no real
   neighboring-slice information for a 3D model to use. This script
   fakes a thin 3D "slab" by replicating the single 2D slice
   `--replication_depth` times along Z (see `_build_pseudo_volume`) and
   reads back the CENTER slice of the model's output. This gives the
   model *something* 3D-shaped to run on, but every neighboring slice it
   sees is a copy of the same slice, not real anatomy -- treat this as
   "can the model do something reasonable given a single real slice",
   not "3D segmentation quality on Jordan data".
2. **No shared intensity scale.** Jordan CT is an 8-bit windowed
   secondary-capture image (0-255, RGB), not raw HU -- data/loaders_jordan_ct.py
   can only min-max normalize each slice to itself. The model was trained
   on real-HU-normalized synthetic CT. Any Dice/IoU number here reflects
   whether the model's predicted SHAPE overlaps the real tumor outline,
   not whether its HU-based reasoning transfers -- it structurally cannot
   be asked to transfer that, given the input format.
3. **Filename-based slice matching is unverified beyond the naming
   convention.** See data/loaders_jordan_ct.py's discover_jordan_slices --
   any CT/mask pair used here already passed that matching, but a
   silently wrong pairing (if the naming convention itself has an
   exception this script's regex didn't anticipate) would still look like
   a valid pair and quietly corrupt the reported metrics.

Run as:
    python -m inference.validate_jordan_segmentation --config configs/stage3_ct_segmentation.yaml
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
from data.preprocessing import pad_to_multiple
from models.unet3d_segmentation import build_segmentation_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.train_stage3_segmentation import build_synthetic_ct_dataloaders

log = logging.getLogger("validate_jordan_segmentation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CSV_FIELDS = ["patient_id", "slice_num", "dice", "iou"]


def parse_args():
    """--config resolves the checkpoint and dataset paths; the rest are overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Use this exact checkpoint instead of auto-finding the latest one.")
    parser.add_argument("--jordan_ct_root", type=str, default=None, help="Override data.jordan_ct_root from --config.")
    parser.add_argument("--jordan_mask_root", type=str, default=None, help="Override data.jordan_mask_root from --config.")
    parser.add_argument("--replication_depth", type=int, default=16, help="How many times to replicate each 2D slice along Z to fake a thin 3D volume -- see module docstring's limitation #1.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for the binary prediction.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/jordan_validation")
    return parser.parse_args()


def _build_pseudo_volume(slice_2d: np.ndarray, depth: int, spatial_multiple: int) -> tuple[np.ndarray, int]:
    """Replicate a 2D (H, W) slice `depth` times along a new leading Z axis
    and pad every axis to spatial_multiple, so the model (which expects a
    3D input divisible by 2**(num_downsamples)) can run on it. Returns the
    padded pseudo-volume and the index of the ORIGINAL slice within the
    (possibly padded) Z axis, so the caller reads back the right one."""
    replicated = np.repeat(slice_2d[np.newaxis, :, :], depth, axis=0)  # (depth, H, W)
    padded = pad_to_multiple(replicated, spatial_multiple, pad_value=-1.0)
    center_index = padded.shape[0] // 2
    return padded, center_index


def dice_iou(pred_bin: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> tuple[float, float]:
    """2D Dice and IoU between a binary prediction and a binary target."""
    intersection = float((pred_bin * target).sum())
    pred_sum, target_sum = float(pred_bin.sum()), float(target.sum())
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def evaluate_jordan(model, device, jordan_ct_root: str, jordan_mask_root: str, replication_depth: int, spatial_multiple: int, threshold: float) -> list[dict]:
    """Run every matched Jordan slice through the pseudo-3D workaround and
    return per-slice Dice/IoU."""
    slices = discover_jordan_slices(jordan_ct_root, jordan_mask_root)
    if not slices:
        raise RuntimeError(f"No matched Jordan slices found (ct_root={jordan_ct_root!r}, mask_root={jordan_mask_root!r}).")
    dataset = JordanCTSegDataset(slices)

    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        ct_2d = item["ct"].squeeze(0).numpy()      # (H, W)
        mask_2d = item["mask"].squeeze(0).numpy()  # (H, W)

        pseudo_volume, center_index = _build_pseudo_volume(ct_2d, replication_depth, spatial_multiple)
        ct_tensor = torch.from_numpy(pseudo_volume).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_volume = model.predict_full_volume(ct_tensor, patch_size=pseudo_volume.shape)
        pred_center = pred_volume.squeeze(0).squeeze(0).cpu().numpy()[center_index]
        pred_center = pred_center[: mask_2d.shape[0], : mask_2d.shape[1]]  # undo any H/W padding
        pred_bin = (pred_center > threshold).astype(np.float32)

        dice, iou = dice_iou(pred_bin, mask_2d)
        log.info("%s slice %d: dice=%.4f iou=%.4f", item["patient_id"], item["slice_num"], dice, iou)
        rows.append({"patient_id": item["patient_id"], "slice_num": item["slice_num"], "dice": dice, "iou": iou})
    return rows


def write_csv(rows: list[dict], output_csv: str) -> None:
    """Write per-slice Jordan Dice/IoU results to output_csv."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Wrote %d rows to %s", len(rows), output_csv)


def save_visualization(model, device, val_loader, jordan_ct_root: str, jordan_mask_root: str, replication_depth: int, spatial_multiple: int, threshold: float, patch_size: tuple[int, int, int], output_path: str) -> None:
    """One synthetic (val, unseen-in-training) example + one Jordan
    (external) example, each as a CT / real mask / predicted mask row, so
    the two very different data sources can be checked by eye side by
    side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # Row 1: synthetic CT val example (full 3D volume, mid-slice view)
    batch = next(iter(val_loader))
    ct_vol = batch["ct"].to(device)
    with torch.no_grad():
        pred_vol = model.predict_full_volume(ct_vol, patch_size=patch_size)
    mid = ct_vol.shape[2] // 2
    ct_np = ct_vol.squeeze(0).squeeze(0).cpu().numpy()[mid]
    mask_np = batch["mask"].squeeze(0).squeeze(0).numpy()[mid]
    pred_np = pred_vol.squeeze(0).squeeze(0).cpu().numpy()[mid]
    axes[0, 0].imshow(ct_np, cmap="gray"); axes[0, 0].set_title("synthetic CT (val)"); axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_np, cmap="gray"); axes[0, 1].set_title("real tumor mask"); axes[0, 1].axis("off")
    axes[0, 2].imshow(pred_np > threshold, cmap="gray"); axes[0, 2].set_title("predicted mask"); axes[0, 2].axis("off")

    # Row 2: one Jordan external example (pseudo-3D workaround, see module docstring)
    slices = discover_jordan_slices(jordan_ct_root, jordan_mask_root)
    if slices:
        item = JordanCTSegDataset(slices[:1])[0]
        ct_2d = item["ct"].squeeze(0).numpy()
        mask_2d = item["mask"].squeeze(0).numpy()
        pseudo_volume, center_index = _build_pseudo_volume(ct_2d, replication_depth, spatial_multiple)
        ct_tensor = torch.from_numpy(pseudo_volume).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_volume = model.predict_full_volume(ct_tensor, patch_size=pseudo_volume.shape)
        pred_center = pred_volume.squeeze(0).squeeze(0).cpu().numpy()[center_index][: mask_2d.shape[0], : mask_2d.shape[1]]
        axes[1, 0].imshow(ct_2d, cmap="gray"); axes[1, 0].set_title(f"Jordan CT ({item['patient_id']})"); axes[1, 0].axis("off")
        axes[1, 1].imshow(mask_2d, cmap="gray"); axes[1, 1].set_title("real tumor mask"); axes[1, 1].axis("off")
        axes[1, 2].imshow(pred_center > threshold, cmap="gray"); axes[1, 2].set_title("predicted mask"); axes[1, 2].axis("off")
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "no Jordan slices found", ha="center", va="center"); ax.axis("off")

    plt.suptitle("Synthetic (top, in-distribution) vs. Jordan (bottom, external) -- see module docstring for caveats")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    log.info("Saved %s", output_path)


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

    data_cfg = config["data"]
    jordan_ct_root = args.jordan_ct_root or data_cfg["jordan_ct_root"]
    jordan_mask_root = args.jordan_mask_root or data_cfg["jordan_mask_root"]
    spatial_multiple = data_cfg.get("spatial_multiple", 16)
    patch_size = tuple(data_cfg["patch_size"])

    rows = evaluate_jordan(model, device, jordan_ct_root, jordan_mask_root, args.replication_depth, spatial_multiple, args.threshold)
    write_csv(rows, os.path.join(args.output_dir, "jordan_metrics.csv"))

    dices = [r["dice"] for r in rows]
    ious = [r["iou"] for r in rows]
    log.info("Jordan external validation: %d slices, mean dice=%.4f (std=%.4f), mean iou=%.4f (std=%.4f)",
              len(rows), np.mean(dices), np.std(dices), np.mean(ious), np.std(ious))

    _train_loader, val_loader = build_synthetic_ct_dataloaders(config, seed=config.get("seed", 0))
    save_visualization(
        model, device, val_loader, jordan_ct_root, jordan_mask_root,
        args.replication_depth, spatial_multiple, args.threshold, patch_size,
        os.path.join(args.output_dir, "synthetic_vs_jordan_example.png"),
    )

    log.info("Done. See %s for the full per-slice record and visualization.", args.output_dir)


if __name__ == "__main__":
    main()
