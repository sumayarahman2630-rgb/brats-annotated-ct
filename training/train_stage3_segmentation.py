"""Stage 3 training entry point: binary tumor segmentation from CT, trained
ONLY on the synthetic CT + tumor mask pairs Stage 2 generated. Run as:

    python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml

The Jordan hospital dataset is NEVER read here -- it exists purely for
external validation (inference/validate_jordan_segmentation.py), and using
it in training would defeat the point of holding it out. See
PROJECT_NOTES.md's Stage 3 section for the full reasoning and known
limitations of that external comparison.

Same resumability design as training/train_stage1_regression.py (numbered
checkpoints, highest-step-wins resume, checkpoint-saved-before-validation
ordering so a validation OOM can only cost one skipped readout, never
training progress -- that ordering was a real bug found and fixed in Stage
1, applied here from the start rather than waiting to hit it again).
Deliberately duplicated (own CycleLoader, own lr-lambda helper) rather than
imported from train_stage1_regression.py, same pipeline-isolation
reasoning used everywhere else in this project.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.loaders_synthetic_ct import build_synthetic_ct_dataloaders
from data.preprocessing import pad_or_crop_to_shape
from models.unet3d_segmentation import build_segmentation_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.ema import EMA

log = logging.getLogger("train_stage3_segmentation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int) -> None:
    """Seed every RNG a training run touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_steps: int, total_steps: int, schedule: str):
    """Returns a step -> LR-multiplier function for LambdaLR: linear warmup
    then either constant or cosine decay to 0 by total_steps."""
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


class CycleLoader:
    """Wraps a DataLoader so it can be pulled from indefinitely, re-shuffling
    each time it's exhausted -- training here is step-based (total_steps),
    not epoch-based."""

    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def next(self) -> dict:
        """Return the next batch, transparently starting a new epoch (with
        a fresh shuffle) if the current one is exhausted."""
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss (differentiable): 1 - 2*|pred∩target| / (|pred|+|target|),
    computed per-sample over the whole volume/patch then averaged over the
    batch. `smooth` avoids a 0/0 NaN when both pred and target are entirely
    empty (a real possibility for a random -- non-foreground-biased --
    background patch)."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Hard (thresholded) Dice coefficient, for human-readable logging --
    unlike dice_loss above, this is not differentiable and not what's
    optimized; it's what gets reported."""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, bce_weight: float) -> torch.Tensor:
    """Dice + BCE, the standard combination for segmentation with severe
    foreground/background imbalance: Dice directly rewards overlap with
    the (small) tumor region regardless of its size; BCE gives a smooth,
    well-behaved per-voxel gradient everywhere, including on patches with
    no foreground at all (where dice_loss's smoothing term alone gives a
    weak signal). pred is already sigmoid-activated (see
    models/unet3d_segmentation.py's forward()), so this uses
    binary_cross_entropy, not the with_logits variant."""
    return dice_loss(pred, target) + bce_weight * F.binary_cross_entropy(pred, target)


def init_log_file(path: str, resuming: bool) -> None:
    """Create the CSV training log with a header row, unless resuming an
    existing run (in which case the existing file/header is kept and new
    rows are appended)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if resuming and os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "loss", "dice_score", "lr", "elapsed_sec"])


def append_log_row(path: str, step: int, split: str, loss: float, dice: float, lr: float, elapsed: float) -> None:
    """Append one row (train or val) to the CSV training log."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, split, f"{loss:.6f}", f"{dice:.4f}", f"{lr:.8f}", f"{elapsed:.1f}"])


def _center_crop_batch_to_patch(batch: dict, patch_size: tuple[int, int, int] | None) -> dict:
    """Same OOM-avoidance fix as Stage 1's quick_validation (see
    train_stage1_regression.py's _center_crop_batch_to_patch docstring for
    the full story): val_loader yields FULL-volume patients for
    validate_jordan_segmentation.py-style whole-brain metrics, but a full
    volume through this model's periodic in-training check would risk the
    same OOM Stage 1 hit on a real Kaggle T4. Center-crop down to the
    training patch size for this cheap periodic check instead."""
    if patch_size is None:
        return batch
    out = {}
    for key, pad_value in (("ct", -1.0), ("mask", 0.0)):
        arr = batch[key].squeeze(0).squeeze(0).numpy()
        cropped = pad_or_crop_to_shape(arr, patch_size, pad_value=pad_value)
        out[key] = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)
    out["patient_id"] = batch["patient_id"]
    return out


@torch.no_grad()
def quick_validation(
    model, val_loader, device, max_patients: int, amp_enabled: bool,
    bce_weight: float, patch_size: tuple[int, int, int] | None,
):
    """Combined loss + Dice score over up to max_patients val patients, on
    RAW (non-EMA) weights -- matches training loss's own weights, same
    anti-EMA-contamination reasoning as Stage 1. Center-crops each val
    volume to `patch_size` rather than running full-volume inference (see
    _center_crop_batch_to_patch) -- this periodic check is a patch-level
    signal, not a whole-brain one.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    n = 0
    for batch in val_loader:
        if n >= max_patients:
            break
        batch = _center_crop_batch_to_patch(batch, patch_size)
        ct, mask = batch["ct"].to(device), batch["mask"].to(device)
        with autocast_ctx:
            pred = model(ct)
        total_loss += combined_loss(pred.float(), mask.float(), bce_weight).item()
        total_dice += dice_score(pred.float(), mask.float())
        n += 1
    model.train()
    return total_loss / max(1, n), total_dice / max(1, n)


def main():
    """Parse args/config, build model+data+optimizer, resume from the
    latest checkpoint if one exists, then run the train loop: forward/
    backward/step each iteration, periodic logging, checkpoint-then-
    validate every `checkpoint_interval`/`val_interval` steps, until
    training.total_steps is reached."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Smoke-test override: run only this many steps instead of training.total_steps.")
    parser.add_argument("--max_patients", type=int, default=None,
                         help="Smoke-test override: use only this many discovered patients instead of data.max_patients.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_steps is not None:
        config["training"]["total_steps"] = args.max_steps
        log.info("--max_steps override: training.total_steps = %d", args.max_steps)
    if args.max_patients is not None:
        config["data"]["max_patients"] = args.max_patients
        log.info("--max_patients override: data.max_patients = %d", args.max_patients)

    total_steps = config["training"]["total_steps"]
    config["training"]["checkpoint_interval"] = min(config["training"]["checkpoint_interval"], total_steps)
    config["training"]["val_interval"] = min(config["training"].get("val_interval", total_steps), total_steps)

    set_seed(config.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    if device.type != "cuda":
        log.warning("No CUDA GPU detected -- training will be extremely slow. Run on a Kaggle GPU session for real training.")

    train_loader, val_loader = build_synthetic_ct_dataloaders(config, seed=config.get("seed", 0))
    train_cycle = CycleLoader(train_loader)

    model = build_segmentation_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameter count: %.2fM", n_params / 1e6)

    train_cfg = config["training"]
    bce_weight = train_cfg.get("bce_weight", 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(train_cfg.get("warmup_steps", 0), train_cfg["total_steps"], train_cfg.get("lr_schedule", "cosine")),
    )
    ema = EMA(model, decay=train_cfg.get("ema_decay", 0.999))

    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ckpt_cfg = config["checkpoint"]
    search_dirs = [ckpt_cfg["working_dir"]] + list(ckpt_cfg.get("extra_resume_dirs", []))
    latest_ckpt = find_latest_checkpoint(search_dirs)

    global_step = 0
    if latest_ckpt is not None:
        global_step, _extra = load_checkpoint(latest_ckpt, model, ema, optimizer, scheduler, map_location=device.type)
        ema.decay = train_cfg.get("ema_decay", 0.999)  # re-apply from config on resume, same fix as Stage 1 -- see PROJECT_NOTES.md
        log.info("Resumed from checkpoint %s at step %d", latest_ckpt, global_step)
    else:
        log.info("No checkpoint found in %s -- starting from scratch", search_dirs)

    log_file = train_cfg["log_file"]
    init_log_file(log_file, resuming=global_step > 0)

    checkpoint_interval = train_cfg["checkpoint_interval"]
    log_interval = train_cfg.get("log_interval", 25)
    val_interval = train_cfg.get("val_interval", checkpoint_interval)
    val_max_patients = train_cfg.get("val_max_patients", 5)
    keep_last_n = train_cfg.get("keep_last_n_checkpoints", 3)
    val_patch_size = config["data"].get("patch_size")
    val_patch_size = tuple(val_patch_size) if val_patch_size else None

    model.train()
    t_start = time.time()
    autocast_ctx = (lambda: torch.amp.autocast("cuda", enabled=amp_enabled)) if device.type == "cuda" else (lambda: nullcontext())

    while global_step < total_steps:
        batch = train_cycle.next()
        ct, mask = batch["ct"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            pred = model(ct)
            loss = combined_loss(pred, mask, bce_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip_norm", 1.0))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        global_step += 1

        if global_step % log_interval == 0 or global_step == total_steps:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            train_dice = dice_score(pred.detach().float(), mask.float())
            log.info("step %d/%d | loss=%.5f | dice=%.4f | lr=%.6f | elapsed=%.1fs", global_step, total_steps, loss.item(), train_dice, lr, elapsed)
            append_log_row(log_file, global_step, "train", loss.item(), train_dice, lr, elapsed)

        # Checkpoint BEFORE validation, deliberately -- see Stage 1's
        # train_stage1_regression.py for the real-Kaggle bug this ordering fixes.
        if global_step % checkpoint_interval == 0 or global_step == total_steps:
            path = save_checkpoint(ckpt_cfg["working_dir"], global_step, model, ema, optimizer, scheduler, keep_last_n=keep_last_n)
            log.info("Saved checkpoint: %s", path)

        if global_step % val_interval == 0 or global_step == total_steps:
            try:
                val_loss, val_dice = quick_validation(
                    model, val_loader, device, val_max_patients, amp_enabled, bce_weight, val_patch_size,
                )
                elapsed = time.time() - t_start
                log.info("step %d val: loss=%.5f dice=%.4f (over up to %d val patients)", global_step, val_loss, val_dice, val_max_patients)
                append_log_row(log_file, global_step, "val", val_loss, val_dice, scheduler.get_last_lr()[0], elapsed)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                log.warning(
                    "step %d: validation OOM (%s) -- skipping this validation pass, training continues. "
                    "Checkpoint for this step was already saved.",
                    global_step, str(e).splitlines()[0] if str(e) else type(e).__name__,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model.train()

    log.info("Training complete: %d steps.", global_step)


if __name__ == "__main__":
    main()
