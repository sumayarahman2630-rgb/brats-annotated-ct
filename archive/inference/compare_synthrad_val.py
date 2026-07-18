"""Diagnostic: real vs. synthetic CT, side by side, on the SynthRAD
VALIDATION patients Stage 1 training actually held out -- this is a much
stronger sanity check than BraTS output alone, because there's ground
truth to compare against directly (BraTS has no real CT to compare to,
just visual plausibility).

Reuses data.loaders_synthrad.build_synthrad_dataloaders with the exact same
config and seed training used, so the validation split is guaranteed
identical -- no risk of accidentally scoring against patients the model
trained on.

Deliberately uses RAW checkpoint weights only: no EMA object is even
constructed here, so there is no `ema.copy_to(model)` call anywhere in this
script for a stray flag to accidentally re-enable -- structurally can't
reintroduce the EMA-contamination bug documented in DEVELOPMENT_LOG.md's "Known bugs
fixed" (round 6).

Run as:
    python -m inference.compare_synthrad_val --config configs/stage1_synthrad.yaml --num_patients 3
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import yaml

from data.loaders_synthrad import build_synthrad_dataloaders
from data.preprocessing import denormalize_ct
from archive.models.stage1_mri2ct_ddpm import build_stage1_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint

log = logging.getLogger("compare_synthrad_val")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage1_synthrad.yaml")
    parser.add_argument("--num_patients", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=None, help="DDIM steps; defaults to diffusion.ddim_steps in the config.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/synthrad_val_comparison")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                         help="Use this exact checkpoint file instead of auto-finding the latest one -- "
                              "e.g. to A/B a specific earlier step against the current one.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as e:
        raise SystemExit(
            "scikit-image is required for PSNR/SSIM (usually preinstalled on Kaggle). "
            "If missing: !pip install scikit-image"
        ) from e
    import matplotlib.pyplot as plt

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = build_stage1_model(config).to(device)

    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"--checkpoint_path {ckpt_path!r} does not exist.")
    else:
        search_dirs = [config["checkpoint"]["working_dir"]] + list(config["checkpoint"].get("extra_resume_dirs", []))
        ckpt_path = find_latest_checkpoint(search_dirs)
        if ckpt_path is None:
            raise RuntimeError(f"No Stage 1 checkpoint found in {search_dirs}.")
    # ema=None: no EMA object is created at all, so raw trained weights stay in
    # `model` with no possibility of a later ema.copy_to(model) call overwriting them.
    step, _extra = load_checkpoint(ckpt_path, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded checkpoint %s (step %d) -- RAW weights, no EMA involved", ckpt_path, step)
    model.eval()

    _train_loader, val_loader = build_synthrad_dataloaders(config, seed=config.get("seed", 0))
    log.info("Validation set: %d patients (same split training used)", len(val_loader.dataset))

    ct_clip_range = tuple(config["data"].get("ct_clip_range", (-1000.0, 3000.0)))
    data_range_hu = ct_clip_range[1] - ct_clip_range[0]
    num_steps = args.num_steps or config["diffusion"].get("ddim_steps", 100)

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for i, batch in enumerate(val_loader):
        if i >= args.num_patients:
            break
        patient_id = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        mri = batch["mri"].to(device)
        real_ct_norm = batch["ct"].squeeze(0).squeeze(0).numpy()
        mask_arr = batch["mask"].squeeze(0).squeeze(0).numpy().astype(bool)

        with torch.no_grad():
            synth_ct_norm = model.sample(mri, num_steps=num_steps)
        synth_ct_norm = synth_ct_norm.squeeze(0).squeeze(0).cpu().numpy()

        real_hu = denormalize_ct(real_ct_norm, *ct_clip_range)
        synth_hu = denormalize_ct(synth_ct_norm, *ct_clip_range)

        psnr_full = peak_signal_noise_ratio(real_hu, synth_hu, data_range=data_range_hu)
        ssim_full = structural_similarity(real_hu, synth_hu, data_range=data_range_hu)
        # Whole-volume PSNR/SSIM include a lot of background (-1000 HU outside the
        # brain in both real and synthetic, if the model has learned that much) --
        # that agreement alone inflates the score. Foreground-only PSNR is the more
        # honest "does it get the actual brain tissue right" number.
        psnr_fg = (
            peak_signal_noise_ratio(real_hu[mask_arr], synth_hu[mask_arr], data_range=data_range_hu)
            if mask_arr.any() else float("nan")
        )

        log.info(
            "%s: whole-volume PSNR=%.2f dB SSIM=%.4f | foreground-only PSNR=%.2f dB",
            patient_id, psnr_full, ssim_full, psnr_fg,
        )
        results.append((patient_id, psnr_full, ssim_full, psnr_fg))

        # Background-vs-foreground separation check: does the model actually distinguish
        # "outside brain" from "inside brain" at all? A working model should show synthetic
        # background close to -1000 (matching real) and synthetic foreground noticeably
        # higher/more varied than synthetic background -- if synthetic background and
        # foreground means are close to each other, the model isn't localizing the brain
        # region at all, which would explain PSNR worse than a trivial flat-background guess.
        bg_mask = ~mask_arr
        log.info(
            "  real:      background mean=%7.1f std=%6.1f | foreground mean=%7.1f std=%6.1f",
            real_hu[bg_mask].mean() if bg_mask.any() else float("nan"), real_hu[bg_mask].std() if bg_mask.any() else float("nan"),
            real_hu[mask_arr].mean(), real_hu[mask_arr].std(),
        )
        log.info(
            "  synthetic: background mean=%7.1f std=%6.1f | foreground mean=%7.1f std=%6.1f",
            synth_hu[bg_mask].mean() if bg_mask.any() else float("nan"), synth_hu[bg_mask].std() if bg_mask.any() else float("nan"),
            synth_hu[mask_arr].mean(), synth_hu[mask_arr].std(),
        )

        mid = real_hu.shape[0] // 2
        mri_arr = mri.squeeze(0).squeeze(0).cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        axes[0].imshow(mri_arr[mid], cmap="gray")
        axes[0].set_title("input MRI")
        axes[0].axis("off")
        axes[1].imshow(real_hu[mid], cmap="gray", vmin=ct_clip_range[0], vmax=800)
        axes[1].set_title("REAL CT (ground truth)")
        axes[1].axis("off")
        axes[2].imshow(synth_hu[mid], cmap="gray", vmin=ct_clip_range[0], vmax=800)
        axes[2].set_title(f"synthetic CT\nPSNR={psnr_full:.1f}dB SSIM={ssim_full:.3f}\nfg-only PSNR={psnr_fg:.1f}dB")
        axes[2].axis("off")
        plt.suptitle(patient_id)
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
            "Average over %d patients: whole-volume PSNR=%.2f dB SSIM=%.4f | foreground-only PSNR=%.2f dB",
            len(results), avg_psnr, avg_ssim, avg_psnr_fg,
        )
    log.info("Done. Comparison images saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
