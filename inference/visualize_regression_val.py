"""Pipeline role: the qualitative + quantitative evidence for the regression
model's result (28.21 dB foreground PSNR at step 20000) -- real vs.
synthetic CT on the FULL held-out validation split, with an error map.
Reuses the exact same patient-level split algorithm as
training/train_stage1_regression.py (same config, same seed) so this is
guaranteed to score only patients the model never trained on.

Deliberately uses RAW checkpoint weights only, same anti-EMA-contamination
pattern as compare_synthrad_val.py (see PROJECT_NOTES.md round 6): no EMA object
is constructed here at all.

Inference runs on each val patient's FULL cropped volume via sliding-window
prediction (models/unet3d_regression.py's predict_full_volume) -- a direct
single-forward-pass full-volume call OOM'd on a real Kaggle T4 (2026-07-16,
during training's own periodic validation), so this always tiles the volume
into patch_size windows (data.patch_size) with 50% overlap and blends the
result, same bounded memory footprint training already proved safe.

Run as:
    python -m inference.visualize_regression_val --config configs/stage1_regression.yaml --num_patients 5
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.preprocessing import denormalize_ct
from models.unet3d_regression import build_regression_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.train_stage1_regression import build_regression_dataloaders

log = logging.getLogger("visualize_regression_val")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    """CLI flags -- config path plus optional overrides (patient count,
    output dir, an exact checkpoint to evaluate instead of the latest)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage1_regression.yaml")
    parser.add_argument("--num_patients", type=int, default=None, help="Defaults to the entire val split.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/regression_val_comparison")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                         help="Use this exact checkpoint file instead of auto-finding the latest one.")
    return parser.parse_args()


def main():
    """Load the checkpoint (raw weights), run every val patient through
    sliding-window inference, compute whole-volume + foreground PSNR/SSIM,
    save a 4-panel comparison image per patient, and print the averages."""
    args = parse_args()

    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as e:
        raise SystemExit(
            "scikit-image is required for PSNR/SSIM (usually preinstalled on Kaggle). If missing: !pip install scikit-image"
        ) from e
    import matplotlib.pyplot as plt

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = build_regression_model(config).to(device)

    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"--checkpoint_path {ckpt_path!r} does not exist.")
    else:
        search_dirs = [config["checkpoint"]["working_dir"]] + list(config["checkpoint"].get("extra_resume_dirs", []))
        ckpt_path = find_latest_checkpoint(search_dirs)
        if ckpt_path is None:
            raise RuntimeError(f"No regression checkpoint found in {search_dirs}.")
    # ema=None, same as compare_synthrad_val.py: raw weights only, structurally can't
    # reintroduce the EMA-contamination bug from PROJECT_NOTES.md's "Known bugs fixed" (round 6).
    step, _extra = load_checkpoint(ckpt_path, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded checkpoint %s (step %d) -- RAW weights, no EMA involved", ckpt_path, step)
    model.eval()

    _train_loader, val_loader = build_regression_dataloaders(config, seed=config.get("seed", 0))
    log.info("Validation set: %d patients (same patient-level split training used)", len(val_loader.dataset))

    ct_clip_range = tuple(config["data"].get("ct_clip_range", (-1000.0, 3000.0)))
    data_range_hu = ct_clip_range[1] - ct_clip_range[0]
    num_patients = args.num_patients or len(val_loader.dataset)
    patch_size = config["data"].get("patch_size")
    patch_size = tuple(patch_size) if patch_size else None
    if patch_size is None:
        raise RuntimeError("data.patch_size must be set in the config -- it's reused as the sliding-window size for full-volume inference.")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for i, batch in enumerate(val_loader):
        if i >= num_patients:
            break
        patient_id = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        mri = batch["mri"].to(device)
        real_ct_norm = batch["ct"].squeeze(0).squeeze(0).numpy()
        mask_arr = batch["mask"].squeeze(0).squeeze(0).numpy().astype(bool)

        # Full-volume, not a single forward() call: a full brain-crop volume OOM'd on a
        # real Kaggle T4 even under no_grad + AMP (see models/unet3d_regression.py's
        # predict_full_volume docstring) -- sliding-window inference keeps peak memory
        # bounded to patch_size regardless of how large the actual volume is.
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

        log.info(
            "%s: L1(norm)=%.4f | whole-volume PSNR=%.2f dB SSIM=%.4f | foreground-only PSNR=%.2f dB",
            patient_id, l1_norm, psnr_full, ssim_full, psnr_fg,
        )
        results.append((patient_id, psnr_full, ssim_full, psnr_fg))

        error_map = np.abs(real_hu - synth_hu)

        mid = real_hu.shape[0] // 2
        mri_arr = mri.squeeze(0).squeeze(0).cpu().numpy()
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        axes[0].imshow(mri_arr[mid], cmap="gray")
        axes[0].set_title("input MRI")
        axes[0].axis("off")
        axes[1].imshow(real_hu[mid], cmap="gray", vmin=ct_clip_range[0], vmax=800)
        axes[1].set_title("REAL CT (ground truth)")
        axes[1].axis("off")
        axes[2].imshow(synth_hu[mid], cmap="gray", vmin=ct_clip_range[0], vmax=800)
        axes[2].set_title(f"synthetic CT\nPSNR={psnr_full:.1f}dB SSIM={ssim_full:.3f}\nfg-only PSNR={psnr_fg:.1f}dB")
        axes[2].axis("off")
        err_im = axes[3].imshow(error_map[mid], cmap="hot", vmin=0, vmax=500)
        axes[3].set_title("|real - synthetic| (HU)")
        axes[3].axis("off")
        plt.colorbar(err_im, ax=axes[3], fraction=0.046)
        plt.suptitle(f"{patient_id} (val, unseen during training)")
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"compare_{patient_id}.png")
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        log.info("Saved %s", out_path)

    if results:
        avg_psnr = np.mean([r[1] for r in results])
        avg_ssim = np.mean([r[2] for r in results])
        avg_psnr_fg = np.nanmean([r[3] for r in results])
        log.info(
            "Average over %d val patients: whole-volume PSNR=%.2f dB SSIM=%.4f | foreground-only PSNR=%.2f dB",
            len(results), avg_psnr, avg_ssim, avg_psnr_fg,
        )
    log.info("Done. Comparison images saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
