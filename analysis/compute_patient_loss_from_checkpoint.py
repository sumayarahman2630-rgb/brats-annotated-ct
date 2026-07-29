"""Stage 3 -- per-patient loss from an ALREADY-TRAINED checkpoint, without
retraining.

Input: an existing Stage 3 checkpoint (found the same way every other
evaluation script in this project finds one) and the training pool it was
trained on. Output: a patient_loss_log.csv-compatible file (same "step,
patient_id, loss" schema training.patient_loss_log writes) that
analysis/build_exclude_list.py can consume directly.

Why this exists: training/train_stage3_segmentation.py only started
logging per-patient loss once that feature was added -- an older
checkpoint's training run predates it and has no such log, and there's no
way to reconstruct one after the fact from the checkpoint alone (it only
stores final weights, not the loss history). Retraining from scratch would
work, but throws away a checkpoint that's already known to give a good
result, and costs a full training budget over again just to get a
bookkeeping file.

This script does neither. It loads the checkpoint's RAW weights (same
anti-EMA-contamination convention as every evaluation script here), puts
the model in eval() mode, and runs ONLY forward passes -- no backward(),
no optimizer.step(), no scheduler.step(), nothing that could move a single
weight. --num_passes independent passes over the training pool (each pass
resamples a fresh random patch per patient, same foreground-biased
cropping training itself used, augmentation forced OFF -- see below) give
several loss readings per patient to average, the same way real training
naturally samples a patient many times over many steps. The "step" column
in the output is just the pass index, not a real training step; since the
model is frozen the whole time, there's no "early vs late" distinction to
make, so pass build_exclude_list.py --loss_tail_fraction 1.0 when using
this file's output (its default 0.25 would incorrectly discard 75% of
these -- entirely valid, all-equally-late -- readings).

Augmentation is forced off regardless of data.augment in the config: the
point here is to measure how well the ACTUAL model handles each patient's
REAL data, not a randomly flipped/rotated/jittered variant of it -- a
patient shouldn't look artificially easy or hard because of which random
augmentation a given pass happened to apply.

Run as:
    python -m analysis.compute_patient_loss_from_checkpoint --config configs/stage3_ct_segmentation.yaml \
        --output_csv /kaggle/working/logs/stage3_patient_loss_log.csv
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
import yaml

from data.loaders_synthetic_ct import build_synthetic_ct_dataloaders
from models.unet3d_segmentation import build_segmentation_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint
from training.train_stage3_segmentation import (
    PATIENT_LOSS_LOG_FIELDS,
    append_patient_loss_rows,
    init_patient_loss_log,
    per_sample_loss,
)

log = logging.getLogger("compute_patient_loss_from_checkpoint")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    """--config resolves the checkpoint search dirs and the training pool; the rest control how many passes to average and where to write."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Use this exact checkpoint instead of auto-finding the latest one.")
    parser.add_argument("--num_passes", type=int, default=20, help="Independent resampling passes over the training pool -- more passes = a more stable per-patient average, at the cost of more forward-pass time (no backward, so still much cheaper than a training step).")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/logs/stage3_patient_loss_log.csv")
    return parser.parse_args()


def compute_patient_loss_from_checkpoint(config: dict, checkpoint_path: str | None, num_passes: int) -> list[tuple[int, str, float]]:
    """Loads the checkpoint (raw weights) and runs num_passes forward-only
    epochs over the TRAIN split, returning (pass_index, patient_id, loss)
    rows in the exact schema training.patient_loss_log uses. Never calls
    .backward() or touches the optimizer -- the checkpoint's weights are
    provably unchanged by this function (verified directly in this
    module's test: a state_dict comparison before/after)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    model = build_segmentation_model(config).to(device)
    if checkpoint_path is None:
        search_dirs = [config["checkpoint"]["working_dir"]] + list(config["checkpoint"].get("extra_resume_dirs", []))
        checkpoint_path = find_latest_checkpoint(search_dirs)
        if checkpoint_path is None:
            raise RuntimeError(f"No Stage 3 checkpoint found in {search_dirs}.")
    step, _extra = load_checkpoint(checkpoint_path, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
    log.info("Loaded checkpoint %s (trained to step %d) -- RAW weights, no EMA involved", checkpoint_path, step)
    model.eval()

    train_cfg = config["training"]
    loss_kwargs = {
        "bce_weight": train_cfg.get("bce_weight", 1.0),
        "loss_type": train_cfg.get("loss_type", "dice_bce"),
        "tversky_alpha": train_cfg.get("tversky_alpha", 0.3),
        "tversky_beta": train_cfg.get("tversky_beta", 0.7),
        "focal_gamma": train_cfg.get("focal_gamma", 2.0),
        "focal_alpha": train_cfg.get("focal_alpha", 0.25),
    }
    log.info("Scoring with the checkpoint's own training.loss_type=%s (loss_kwargs=%s)", loss_kwargs["loss_type"], loss_kwargs)

    if config["data"].get("augment"):
        log.info("data.augment=true in the config, but forced OFF for this measurement -- see module docstring for why.")
    config = dict(config)
    config["data"] = dict(config["data"])
    config["data"]["augment"] = False

    train_loader, _val_loader = build_synthetic_ct_dataloaders(config, seed=config.get("seed", 0))
    log.info("Training pool: %d patients, %d pass(es) each = up to %d loss readings per patient", len(train_loader.dataset), num_passes, num_passes)

    rows: list[tuple[int, str, float]] = []
    for pass_idx in range(1, num_passes + 1):
        for batch in train_loader:
            ct, mask = batch["ct"].to(device), batch["mask"].to(device)
            with torch.no_grad():
                logits = model(ct)
                sample_losses = per_sample_loss(logits.float(), mask.float(), **loss_kwargs)
            patient_ids = batch["patient_id"] if isinstance(batch["patient_id"], list) else [batch["patient_id"]]
            rows.extend(zip([pass_idx] * len(patient_ids), patient_ids, sample_losses.tolist()))
        log.info("Pass %d/%d complete (%d rows so far)", pass_idx, num_passes, len(rows))

    return rows


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rows = compute_patient_loss_from_checkpoint(config, args.checkpoint_path, args.num_passes)
    if not rows:
        raise SystemExit("No rows produced -- check that data.synthetic_ct_root resolves to a non-empty training pool.")

    init_patient_loss_log(args.output_csv, resuming=False)
    append_patient_loss_rows(args.output_csv, rows)
    log.info("Wrote %d rows (%s) to %s", len(rows), PATIENT_LOSS_LOG_FIELDS, args.output_csv)
    print(
        f"\nDone. {len(rows)} rows written to {args.output_csv}.\n"
        "This file was NOT produced by real training steps -- when feeding it to "
        "analysis.build_exclude_list, pass --loss_tail_fraction 1.0 (the frozen "
        "model has no 'early vs late' distinction, so the default tail-trimming "
        "would discard 75% of otherwise-valid readings for no reason)."
    )


if __name__ == "__main__":
    main()
