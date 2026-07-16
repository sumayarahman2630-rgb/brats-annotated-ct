"""Stage 2: generate a synthetic CT for every BraTS T1 volume using the
Stage 1 model, paired with its tumor mask. This is the deliverable dataset.

Run as:
    python -m inference.run_stage2_brats --config configs/stage2_inference_brats.yaml

All settings (brats_root, output_dir, stage1_config, etc.) come from that
config file by default -- CLI flags below override individual values for
one-off runs (e.g. `--limit 5` to smoke-test before committing to the full
cohort) without editing the file.

Resumable by design: before generating a patient, checks whether its output
CT already exists in output_dir and skips it if so (pass --overwrite to
force regeneration). If interrupted partway through the cohort -- Kaggle
session timeout, OOM on one bad volume, anything -- rerunning the same
command picks up exactly where it left off. One patient failing (caught and
logged) never stops the rest of the cohort; see manifest.csv in output_dir
for a full per-patient success/failure record.

Loads whichever Stage 1 checkpoint currently exists (via the same
find_latest_checkpoint search used by training/train_stage1.py), so this
can run against a still-training model -- goal 2 explicitly does not
require goal 1 to be finished first.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.loaders_brats import BraTSVolumeDataset, discover_brats_patients
from data.preprocessing import denormalize_ct, pad_or_crop_to_shape, resample_to_spacing
from models.stage1_mri2ct_ddpm import build_stage1_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.ema import EMA

log = logging.getLogger("run_stage2_brats")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MANIFEST_FIELDS = [
    "patient_id", "status", "ct_path", "mask_path", "error", "elapsed_sec",
    "checkpoint_step", "ddim_steps", "use_ema", "generated_at",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage2_inference_brats.yaml", help="Stage 2 config; see that file for all settings.")
    parser.add_argument("--stage1_config", type=str, default=None, help="Override stage1_config from --config.")
    parser.add_argument("--brats_root", type=str, default=None, help="Override brats_root from --config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir from --config.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint search dir(s); defaults to the stage1 config's checkpoint section.")
    parser.add_argument("--num_steps", type=int, default=None, help="DDIM sampling steps; defaults to diffusion.ddim_steps in the config.")
    parser.add_argument("--use_ema", action="store_true", default=None)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N patients (for a quick test run).")
    return parser.parse_args()


def resolve_settings(args) -> dict:
    """CLI flags override the --config file's values; the config file
    supplies everything not passed on the command line."""
    stage2_cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            stage2_cfg = yaml.safe_load(f) or {}
    else:
        log.warning("Stage 2 config %s not found -- relying entirely on CLI flags.", args.config)

    def pick(cli_val, key, default=None):
        return cli_val if cli_val is not None else stage2_cfg.get(key, default)

    settings = {
        "stage1_config": pick(args.stage1_config, "stage1_config", "configs/stage1_synthrad.yaml"),
        "brats_root": pick(args.brats_root, "brats_root"),
        "output_dir": pick(args.output_dir, "output_dir"),
        "checkpoint_dir": pick(args.checkpoint_dir, "checkpoint_dir"),
        "num_steps": pick(args.num_steps, "num_steps"),
        "use_ema": pick(args.use_ema, "use_ema", True),
        "overwrite": pick(args.overwrite, "overwrite", False),
        "limit": pick(args.limit, "limit"),
    }
    if not settings["brats_root"] or not settings["output_dir"]:
        raise RuntimeError(
            "brats_root and output_dir must be set either in --config "
            f"({args.config}) or passed as --brats_root / --output_dir."
        )
    return settings


def init_manifest(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()


def append_manifest_row(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writerow(row)


def already_done(output_dir: str, patient_id: str) -> bool:
    return os.path.exists(os.path.join(output_dir, patient_id, "synthetic_ct.nii.gz"))


def write_hu_image(hu_array: np.ndarray, reference_image: sitk.Image, path: str) -> None:
    img = sitk.GetImageFromArray(hu_array.astype(np.int16))
    img.CopyInformation(reference_image)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path)


def process_seg_mask(seg_path: str, target_spacing, full_shape, reference_image: sitk.Image, out_path: str) -> None:
    seg_img = resample_to_spacing(sitk.ReadImage(seg_path), target_spacing, is_mask=True, default_value=0.0)
    seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.uint8)
    seg_full = pad_or_crop_to_shape(seg_arr, full_shape, pad_value=0)
    out_img = sitk.GetImageFromArray(seg_full.astype(np.uint8))
    out_img.CopyInformation(reference_image)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(out_img, out_path)


def place_in_full_canvas(cropped_array: np.ndarray, crop_box, full_shape, fill_value: float) -> np.ndarray:
    """Inverse of the bounding-box crop BraTSVolumeDataset applies: paste the
    model's (cropped-resolution) output back into a canvas the size of the
    full resampled T1 grid, so it lines up voxel-for-voxel with the original
    T1 and its tumor mask for pairing. Everything outside the crop box --
    where the model never generated anything -- gets `fill_value` (pass the
    normalized background, -1.0, before denormalizing the whole canvas so it
    comes out as exactly CT_BACKGROUND_HU, matching Stage 1's convention)."""
    canvas = np.full(full_shape, fill_value, dtype=cropped_array.dtype)
    (x0, x1), (y0, y1), (z0, z1) = crop_box
    canvas[x0:x1, y0:y1, z0:z1] = cropped_array
    return canvas


def _git_commit_hash() -> str | None:
    """Best-effort: record which commit generated this dataset, for
    reproducibility. None if git isn't available or this isn't a repo
    checkout (e.g. a zipped download) -- not fatal either way."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:  # noqa: BLE001 -- purely informational, never worth failing a generation run over
        return None


def write_dataset_metadata(output_dir: str, run_info: dict) -> None:
    """Machine-readable dataset-level provenance, refreshed after every run
    (including resumed ones) to reflect the most recent generation session's
    settings. manifest.csv remains the source of truth for exactly which
    checkpoint/settings produced each INDIVIDUAL patient, since a dataset
    built across multiple resumed sessions could have used different
    checkpoints for different patients."""
    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(run_info, f, indent=2)
    log.info("Wrote dataset metadata: %s", path)


def write_dataset_card(output_dir: str, run_info: dict, counts: dict) -> None:
    """Human-readable dataset card (README.md) for the generated dataset,
    populated with this run's actual numbers -- not a template the user
    has to fill in by hand. Regenerated (overwritten) after every run, so
    it always reflects the current state of output_dir."""
    total = sum(counts.values())
    lines = [
        "# Synthetic CT dataset generated from BraTS T1 MRI",
        "",
        "Each `<patient_id>/` folder contains a synthetic CT volume generated from that "
        "patient's real BraTS T1 MRI, paired with their original BraTS tumor segmentation "
        "mask. This pairing is the point of the dataset: annotated CT for tumor-region work "
        "where real annotated CT is scarce.",
        "",
        "## Generation method",
        "",
        f"- **Model**: wavelet-domain conditional diffusion (3D), see the source repository's "
        f"CLAUDE.md for the full architecture decision record.",
        f"- **Checkpoint**: `{run_info['checkpoint_path']}` (training step {run_info['checkpoint_step']})",
        f"- **Weights used**: {'EMA' if run_info['use_ema'] else 'raw (non-EMA)'}",
        f"- **Sampling**: DDIM, {run_info['ddim_steps']} steps",
        f"- **Generated**: {run_info['generated_at']}",
        f"- **Source commit**: `{run_info['git_commit'] or 'unknown (git unavailable at generation time)'}`",
        "",
        "## Contents",
        "",
        f"- Total BraTS patients considered: {total}",
        f"- Successfully generated (CT + mask pair): {counts.get('success', 0)}",
        f"- Skipped, already generated in a previous run: {counts.get('skipped', 0)}",
        f"- Skipped, no tumor mask available to pair with: {counts.get('no_mask', 0)}",
        f"- Failed (see manifest.csv for the error): {counts.get('failed', 0)}",
        "",
        "Per-patient file layout:",
        "```",
        "<patient_id>/",
        "    synthetic_ct.nii.gz   -- int16 HU, same voxel grid/orientation as the source T1",
        "    tumor_mask.nii.gz     -- uint8 label map, same grid, copied from the original BraTS seg.nii",
        "```",
        "`manifest.csv` (in this directory) has the full per-patient record: status, output paths, "
        "the exact checkpoint step and DDIM step count used for that specific patient, and error "
        "messages for any failures. If this dataset was built across multiple resumed sessions with "
        "different checkpoints, manifest.csv -- not this file -- is the source of truth for which "
        "checkpoint produced any given patient.",
        "",
        "## Known limitations -- read before using this data",
        "",
        "- **This is a research artifact, not a clinical or radiotherapy-planning CT.** The "
        "non-brain region (skull, scalp, face) is deliberately zeroed to -1000 HU by design -- "
        "the deliverable is tumor-region CT, not full-head CT.",
        "- Synthetic CT quality is only as good as the Stage 1 checkpoint used (see checkpoint step "
        "above) -- a checkpoint trained for few steps will produce correspondingly undertrained "
        "output. Check the source repository's CLAUDE.md \"Known bugs fixed\" / status sections for "
        "what was verified about this specific checkpoint's quality (e.g. PSNR/SSIM against real "
        "SynthRAD validation CT) before treating this dataset as ground truth for downstream tasks.",
        "- Tumor masks are the ORIGINAL BraTS annotations, unmodified -- their quality/consistency is "
        "whatever the BraTS2020 challenge's own annotations provide, this pipeline doesn't alter them.",
        "",
        "## Citation / provenance",
        "",
        "Generated by `inference/run_stage2_brats.py` in the brats-annotated-ct project. See that "
        "repository's CLAUDE.md for the full method, architecture decisions, and reference repos "
        "this approach was inspired by (not copied from).",
        "",
    ]
    path = os.path.join(output_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    log.info("Wrote dataset card: %s", path)


def main():
    args = parse_args()
    settings = resolve_settings(args)

    with open(settings["stage1_config"]) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = build_stage1_model(config).to(device)
    ema = EMA(model, decay=config["training"].get("ema_decay", 0.9999))

    search_dirs = [settings["checkpoint_dir"]] if settings["checkpoint_dir"] else (
        [config["checkpoint"]["working_dir"]] + list(config["checkpoint"].get("extra_resume_dirs", []))
    )
    ckpt_path = find_latest_checkpoint(search_dirs)
    if ckpt_path is None:
        raise RuntimeError(
            f"No Stage 1 checkpoint found in {search_dirs}. Train Stage 1 first (training/train_stage1.py), "
            "or pass --checkpoint_dir pointing at a directory with ckpt_step*.pt files."
        )
    step, _extra = load_checkpoint(ckpt_path, model, ema, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded Stage 1 checkpoint %s (step %d)", ckpt_path, step)

    if settings["use_ema"]:
        ema.copy_to(model)
        log.info("Using EMA weights for generation.")
    else:
        log.info("Using raw (non-EMA) weights for generation.")
    model.eval()

    data_cfg = config["data"]
    target_spacing = tuple(data_cfg.get("target_spacing", (1.0, 1.0, 1.0)))
    spatial_multiple = data_cfg.get("spatial_multiple", 16)
    crop_margin = data_cfg.get("crop_margin", 10)
    ct_clip_range = tuple(data_cfg.get("ct_clip_range", (-1000.0, 3000.0)))
    num_steps = settings["num_steps"] or config["diffusion"].get("ddim_steps", 100)

    patients = discover_brats_patients(settings["brats_root"])
    if settings["limit"]:
        patients = patients[: settings["limit"]]
    if not patients:
        raise RuntimeError(f"No BraTS patients discovered under {settings['brats_root']!r}.")

    dataset = BraTSVolumeDataset(patients, target_spacing=target_spacing, spatial_multiple=spatial_multiple, crop_margin=crop_margin)

    output_dir = settings["output_dir"]
    manifest_path = os.path.join(output_dir, "manifest.csv")
    init_manifest(manifest_path)

    n_success, n_failed, n_skipped, n_no_mask = 0, 0, 0, 0
    for i, patient in enumerate(patients):
        if not settings["overwrite"] and already_done(output_dir, patient.patient_id):
            log.info("[%d/%d] %s already generated, skipping", i + 1, len(patients), patient.patient_id)
            n_skipped += 1
            continue

        if patient.seg_path is None:
            # The deliverable is a CT+mask pair -- generating a CT with no mask to pair
            # it with would just waste GPU time on an unusable output, so skip outright
            # rather than silently producing an unpaired volume.
            log.info("[%d/%d] %s has no seg.nii (tumor mask) -- skipping, nothing to pair the synthetic CT with", i + 1, len(patients), patient.patient_id)
            append_manifest_row(manifest_path, {
                "patient_id": patient.patient_id, "status": "skipped_no_mask",
                "ct_path": "", "mask_path": "", "error": "", "elapsed_sec": "0.0",
                "checkpoint_step": "", "ddim_steps": "", "use_ema": "", "generated_at": "",
            })
            n_no_mask += 1
            continue

        t0 = time.time()
        try:
            item = dataset[i]
            mri = item["mri"].unsqueeze(0).to(device)

            with torch.no_grad():
                ct_pred_norm = model.sample(mri, num_steps=num_steps)

            ct_pred_norm = ct_pred_norm.squeeze(0).squeeze(0).cpu().numpy()
            # undo pad_to_multiple (back to the cropped brain-region shape the model actually saw)...
            ct_pred_norm = pad_or_crop_to_shape(ct_pred_norm, item["cropped_shape"], pad_value=-1.0)
            # ...then undo the bounding-box crop (paste back into the full T1 grid, so the
            # synthetic CT lines up voxel-for-voxel with the original T1 and its tumor mask)
            ct_pred_full = place_in_full_canvas(ct_pred_norm, item["crop_box"], item["full_shape"], fill_value=-1.0)
            ct_hu = denormalize_ct(ct_pred_full, *ct_clip_range)

            ct_out_path = os.path.join(output_dir, patient.patient_id, "synthetic_ct.nii.gz")
            write_hu_image(ct_hu, item["reference_image"], ct_out_path)

            # patient.seg_path is guaranteed non-None here -- patients without one were skipped above
            mask_out_path = os.path.join(output_dir, patient.patient_id, "tumor_mask.nii.gz")
            process_seg_mask(patient.seg_path, target_spacing, item["full_shape"], item["reference_image"], mask_out_path)

            elapsed = time.time() - t0
            log.info("[%d/%d] %s -> %s (%.1fs)", i + 1, len(patients), patient.patient_id, ct_out_path, elapsed)
            append_manifest_row(manifest_path, {
                "patient_id": patient.patient_id, "status": "success",
                "ct_path": ct_out_path, "mask_path": mask_out_path, "error": "", "elapsed_sec": f"{elapsed:.1f}",
                "checkpoint_step": step, "ddim_steps": num_steps, "use_ema": settings["use_ema"],
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            n_success += 1

        except Exception as e:  # noqa: BLE001 -- one bad patient must not kill the cohort run
            elapsed = time.time() - t0
            log.exception("[%d/%d] %s FAILED", i + 1, len(patients), patient.patient_id)
            append_manifest_row(manifest_path, {
                "patient_id": patient.patient_id, "status": "failed",
                "ct_path": "", "mask_path": "", "error": str(e), "elapsed_sec": f"{elapsed:.1f}",
                "checkpoint_step": step, "ddim_steps": num_steps, "use_ema": settings["use_ema"],
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            n_failed += 1

    log.info("Done. success=%d failed=%d skipped=%d no_mask=%d (total=%d). See %s for the full record.",
              n_success, n_failed, n_skipped, n_no_mask, len(patients), manifest_path)

    run_info = {
        "checkpoint_path": ckpt_path,
        "checkpoint_step": step,
        "use_ema": settings["use_ema"],
        "ddim_steps": num_steps,
        "brats_root": settings["brats_root"],
        "stage1_config": settings["stage1_config"],
        "git_commit": _git_commit_hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    counts = {"success": n_success, "failed": n_failed, "skipped": n_skipped, "no_mask": n_no_mask}
    write_dataset_metadata(output_dir, run_info)
    write_dataset_card(output_dir, run_info, counts)


if __name__ == "__main__":
    main()
