"""Pipeline role: reusable, per-patient qualitative visualization of a Stage
3 segmentation checkpoint's predictions, on BOTH available data sources --
synthetic CT validation patients (real 3D volumes with a real tumor mask,
in-distribution, unseen during training) and Jordan external CT slices (real
hospital data, out-of-distribution -- see data/loaders_jordan_ct.py's module
docstring for its known format/dimensionality limitations).

Both sources funnel through the SAME sliding-window inference
(models/unet3d_segmentation.py's predict_full_volume) and the SAME 3-panel
save_panel() renderer -- only the data-loading path differs per source, so
a synthetic example and a Jordan example are visually comparable side by
side despite coming from very different pipelines.

Feeding a full volume directly to the model (rather than through
predict_full_volume) crashes with a torch.cat skip-connection shape
mismatch unless the volume's spatial dims happen to already be exact
multiples of 2**(num_levels-1) -- predict_full_volume avoids this by
padding every individual tile up to the exact trained patch_size (itself
validated as divisible by that factor) before each forward() call, so this
script never calls the model directly on an arbitrarily-shaped volume.

Run as:
    python -m inference.visualize_predictions --config configs/stage3_ct_segmentation.yaml --source both --num_patients 5
"""
from __future__ import annotations

import argparse
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

log = logging.getLogger("visualize_predictions")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    """CLI flags -- config path plus overrides (checkpoint, which source(s)
    to visualize, how many patients/slices per source, output location)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Use this exact checkpoint file instead of auto-finding the latest one.")
    parser.add_argument("--source", type=str, default="both", choices=["synthetic", "jordan", "both"])
    parser.add_argument("--num_patients", type=int, default=5, help="How many patients (synthetic) / slices (Jordan) to visualize per source.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for the binary predicted-mask overlay.")
    parser.add_argument("--replication_depth", type=int, default=16, help="Jordan-only: how many times to replicate each 2D slice along Z -- see data/loaders_jordan_ct.py's module docstring, limitation #2 (no real 3D context).")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/stage3_prediction_visualizations")
    return parser.parse_args()


def _build_pseudo_volume(slice_2d: np.ndarray, depth: int, spatial_multiple: int) -> tuple[np.ndarray, int]:
    """Same workaround as inference/validate_jordan_segmentation.py's helper
    of the same name (duplicated here, not imported -- it's a tiny, Jordan-
    specific 4-line helper, and this script is meant to stand alone). See
    that module's docstring for the full reasoning: Jordan has no real 3D
    neighboring-slice data, so this fakes a thin volume by replicating the
    single real slice `depth` times along Z, then pads every axis to
    spatial_multiple so predict_full_volume's divisibility requirement is
    met. Returns the padded pseudo-volume and the index of the ORIGINAL
    slice within it, so the caller reads back the real one, not a
    replicated copy."""
    replicated = np.repeat(slice_2d[np.newaxis, :, :], depth, axis=0)
    padded = pad_to_multiple(replicated, spatial_multiple, pad_value=-1.0)
    center_index = padded.shape[0] // 2
    return padded, center_index


def _best_slice_index(mask_3d: np.ndarray) -> int:
    """Pick the Z slice with the most true-tumor pixels, not just the
    geometric mid-slice -- a full brain volume's middle slice very often
    shows zero tumor at all for a small lesion, which would visualize
    nothing informative about segmentation quality. Falls back to the
    volume's mid-slice if there is no tumor anywhere in it (a genuinely
    tumor-free patient)."""
    per_slice_area = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    if per_slice_area.max() == 0:
        return mask_3d.shape[0] // 2
    return int(per_slice_area.argmax())


def save_panel(ct_2d: np.ndarray, real_mask_2d: np.ndarray, pred_prob_2d: np.ndarray, threshold: float, title: str, out_path: str) -> None:
    """CT / CT+real-mask-overlay / CT+predicted-mask-overlay, 3 panels, one
    PNG -- the common renderer both visualize_synthetic and visualize_jordan
    funnel into. `pred_prob_2d` is a probability map (already sigmoided by
    predict_full_volume, NOT raw logits) -- thresholded here, not by the
    caller, so every call site applies the same --threshold consistently."""
    import matplotlib.pyplot as plt

    pred_bin_2d = (pred_prob_2d > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(ct_2d, cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")

    axes[1].imshow(ct_2d, cmap="gray")
    axes[1].imshow(np.ma.masked_where(real_mask_2d == 0, real_mask_2d), cmap="spring", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("real tumor mask (overlay)")
    axes[1].axis("off")

    axes[2].imshow(ct_2d, cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred_bin_2d == 0, pred_bin_2d), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(f"predicted mask (overlay, threshold={threshold})")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info("Saved %s", out_path)


def visualize_synthetic(model, device, config: dict, num_patients: int, threshold: float, output_dir: str) -> int:
    """Full-volume sliding-window inference on real synthetic-CT VALIDATION
    patients -- reuses build_synthetic_ct_dataloaders, the exact same
    patient-level split algorithm and seed training itself used, so this is
    guaranteed in-distribution but held-out (never trained on) data with a
    real tumor mask available for comparison."""
    _train_loader, val_loader = build_synthetic_ct_dataloaders(config, seed=config.get("seed", 0))
    patch_size = tuple(config["data"]["patch_size"])

    n = 0
    for batch in val_loader:
        if n >= num_patients:
            break
        patient_id = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        ct_vol = batch["ct"].to(device)
        mask_vol = batch["mask"].squeeze(0).squeeze(0).numpy()

        pred_vol = model.predict_full_volume(ct_vol, patch_size=patch_size)
        pred_vol = pred_vol.squeeze(0).squeeze(0).float().cpu().numpy()

        slice_idx = _best_slice_index(mask_vol)
        ct_2d = ct_vol.squeeze(0).squeeze(0).cpu().numpy()[slice_idx]
        real_mask_2d = mask_vol[slice_idx]
        pred_2d = pred_vol[slice_idx]

        out_path = os.path.join(output_dir, "synthetic", f"{patient_id}.png")
        save_panel(ct_2d, real_mask_2d, pred_2d, threshold, f"synthetic (val, in-distribution) -- {patient_id}, slice {slice_idx}", out_path)
        n += 1
    log.info("Visualized %d synthetic validation patients.", n)
    return n


def visualize_jordan(model, device, jordan_ct_root: str, jordan_mask_root: str, replication_depth: int, spatial_multiple: int, threshold: float, num_patients: int, output_dir: str) -> int:
    """Pseudo-3D sliding-window inference on Jordan external CT slices --
    see data/loaders_jordan_ct.py's module docstring for why this is a
    workaround (no real 3D context, no shared intensity scale with
    training data) rather than genuine 3D segmentation on this source, and
    inference/validate_jordan_segmentation.py for the same technique used
    for its quantitative Dice/IoU metrics."""
    slices = discover_jordan_slices(jordan_ct_root, jordan_mask_root)
    if not slices:
        log.warning("No Jordan slices found (ct_root=%s mask_root=%s) -- skipping.", jordan_ct_root, jordan_mask_root)
        return 0
    dataset = JordanCTSegDataset(slices[:num_patients])

    for i in range(len(dataset)):
        item = dataset[i]
        ct_2d = item["ct"].squeeze(0).numpy()
        mask_2d = item["mask"].squeeze(0).numpy()

        pseudo_volume, center_index = _build_pseudo_volume(ct_2d, replication_depth, spatial_multiple)
        ct_tensor = torch.from_numpy(pseudo_volume).unsqueeze(0).unsqueeze(0).float().to(device)
        pred_volume = model.predict_full_volume(ct_tensor, patch_size=pseudo_volume.shape)
        pred_center = pred_volume.squeeze(0).squeeze(0).float().cpu().numpy()[center_index]
        pred_center = pred_center[: mask_2d.shape[0], : mask_2d.shape[1]]  # undo H/W padding

        out_path = os.path.join(output_dir, "jordan", f"{item['patient_id']}_slice{item['slice_num']}.png")
        save_panel(ct_2d, mask_2d, pred_center, threshold, f"Jordan (external) -- {item['patient_id']} slice {item['slice_num']}", out_path)
    log.info("Visualized %d Jordan slices.", len(dataset))
    return len(dataset)


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
    n_saved = 0
    if args.source in ("synthetic", "both"):
        n_saved += visualize_synthetic(model, device, config, args.num_patients, args.threshold, args.output_dir)
    if args.source in ("jordan", "both"):
        n_saved += visualize_jordan(
            model, device, data_cfg["jordan_ct_root"], data_cfg["jordan_mask_root"],
            args.replication_depth, data_cfg.get("spatial_multiple", 16), args.threshold,
            args.num_patients, args.output_dir,
        )
    log.info("Done. %d images saved under %s", n_saved, args.output_dir)


if __name__ == "__main__":
    main()
