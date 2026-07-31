"""Stage 3 -- trains the CT segmentation model; output: a resumable checkpoint and CSV training log (Jordan data is never touched here)."""
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


def _host_rss_mb() -> float | None:
    """Best-effort host (CPU) resident memory in MB, via /proc/self/status
    (Linux/Kaggle) -- returns None (not raises) on any platform/format this
    doesn't work on, e.g. Windows, so it never breaks a training run just
    for diagnostic logging."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        return None
    return None


def log_memory_usage(step: int, device: torch.device) -> None:
    """Added 2026-07-24 after two real runs both hit a severe, reproducible
    slowdown (10-20s -> 2000+s per 25 steps) around the same step range
    (~7000-7600) -- logs GPU allocated/reserved memory AND host RSS, since
    a host-side leak (e.g. in repeated per-sample file reads) would leave
    GPU memory looking completely normal and hide the real signal if only
    GPU stats were logged."""
    gpu_allocated_mb = gpu_reserved_mb = None
    if device.type == "cuda":
        gpu_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    host_rss_mb = _host_rss_mb()
    log.info(
        "step %d memory: gpu_allocated=%s gpu_reserved=%s host_rss=%s",
        step,
        f"{gpu_allocated_mb:.1f}MB" if gpu_allocated_mb is not None else "n/a",
        f"{gpu_reserved_mb:.1f}MB" if gpu_reserved_mb is not None else "n/a",
        f"{host_rss_mb:.1f}MB" if host_rss_mb is not None else "n/a",
    )


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
        """Wraps a DataLoader and grabs its first iterator."""
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
    background patch). `pred` must already be a probability in [0, 1]
    (apply sigmoid to the model's raw logits before calling this -- see
    combined_loss below, which does that itself)."""
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
    optimized; it's what gets reported. `pred` must already be a
    probability in [0, 1] -- every call site below applies torch.sigmoid
    to the model's raw logits first."""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> torch.Tensor:
    """Tversky loss: generalizes Dice by weighting false positives (alpha)
    and false negatives (beta) separately, instead of implicitly 1:1 --
    Tversky index = (TP+smooth) / (TP + alpha*FP + beta*FN + smooth), loss
    = 1 - index. alpha=beta=0.5 reduces exactly to Dice. Added 2026-07-19
    after a real Kaggle run showed Dice+BCE collapsing to predicting an
    essentially empty mask (external Jordan validation dice ~0.0005-0.013,
    uniformly near-zero) -- beta>alpha (the default here) penalizes missed
    tumor (false negatives) more than false alarms, which is the standard
    remedy for exactly this "model gives up and predicts background
    everywhere" failure mode under severe class imbalance. `pred` must
    already be a probability in [0, 1] (sigmoid applied by the caller, same
    convention as dice_loss)."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_index.mean()


def focal_tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0) -> torch.Tensor:
    """Focal Tversky loss: raises (1 - Tversky index) to the power gamma,
    which (for gamma < 1) further increases the gradient contribution from
    hard/already-mostly-wrong examples relative to easy ones -- an
    additional lever against the same collapse-to-empty failure mode
    tversky_loss addresses, usable together with it."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return ((1.0 - tversky_index) ** gamma).mean()


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Per-voxel focal loss (Lin et al. 2017), adapted for binary
    segmentation: down-weights voxels the model already classifies
    confidently and correctly (the vast, easy majority of background
    voxels under severe imbalance), concentrating gradient on hard/
    ambiguous ones -- a direct counter to BCE's tendency to be dominated by
    the easy-background majority.

    Takes RAW LOGITS, like combined_loss below -- NOT computed as
    `-alpha*(1-p)**gamma*log(p)` with `p` a directly-sigmoided probability,
    which would reintroduce the exact autocast-unsafe numerical pattern
    already found and fixed once in this file (2026-07-19, see
    combined_loss's docstring). Instead this uses the standard stable
    trick (also how torchvision's own sigmoid_focal_loss is implemented):
    get the per-voxel BCE via the already-safe binary_cross_entropy_with_logits,
    then recover p_t = exp(-bce) (mathematically exact, since bce = -log(p_t)
    for the correct class by construction), and apply the focal modulating
    factor on top of that -- no unsafe log(sigmoid(x)) call anywhere."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    loss_type: str = "dice_bce",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> torch.Tensor:
    """Dispatches to one of several loss combinations for segmentation
    under severe foreground/background imbalance, selected by
    `loss_type` (training.loss_type in the config):

    - "dice_bce" (default, unchanged since this file's original version):
      Dice directly rewards overlap regardless of the tumor's size; BCE
      gives a smooth, well-behaved per-voxel gradient everywhere,
      including on patches with no foreground at all. Found (2026-07-19,
      real Kaggle run) to be prone to collapsing toward predicting an
      empty mask under this project's actual class imbalance -- kept as
      the default for backward compatibility, not because it's
      recommended over the alternatives below.
    - "tversky": tversky_loss alone, beta>alpha penalizing missed tumor
      more than false alarms -- the standard fix for the collapse-to-empty
      failure mode above.
    - "focal_tversky": focal_tversky_loss alone -- an additional lever on
      top of the same idea.
    - "focal": focal_loss_with_logits alone -- addresses the same
      underlying problem from BCE's side instead of Dice's.

    Takes RAW LOGITS in every branch (models/unet3d_segmentation.py's
    forward() does not apply sigmoid itself) -- torch.nn.functional.
    binary_cross_entropy / BCELoss are explicitly unsafe under CUDA
    autocast (they require an already-in-[0,1] input, which fp16 casting
    can silently corrupt), and this crashed on a real Kaggle GPU
    (2026-07-19) before that fix. Every branch here either uses
    F.binary_cross_entropy_with_logits directly or only ever calls
    torch.sigmoid on logits explicitly (never combined with a manual log()
    the way the original bug did)."""
    probs = torch.sigmoid(logits)
    if loss_type == "dice_bce":
        return dice_loss(probs, target) + bce_weight * F.binary_cross_entropy_with_logits(logits, target)
    elif loss_type == "tversky":
        return tversky_loss(probs, target, alpha=tversky_alpha, beta=tversky_beta)
    elif loss_type == "focal_tversky":
        return focal_tversky_loss(probs, target, alpha=tversky_alpha, beta=tversky_beta)
    elif loss_type == "focal":
        return focal_loss_with_logits(logits, target, gamma=focal_gamma, alpha=focal_alpha)
    else:
        raise ValueError(f"Unknown training.loss_type {loss_type!r} -- expected one of: dice_bce, tversky, focal_tversky, focal.")


def per_sample_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    loss_type: str = "dice_bce",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> torch.Tensor:
    """Same math as combined_loss, but returns one loss value per sample in
    the batch (shape (batch,)) instead of the batch mean -- this is what
    the per-patient loss log below attributes to each patient_id, since
    training.batch_size can be > 1 and combined_loss's single scalar can't
    be split back out after the fact.

    Deliberately a SEPARATE function, not a refactor of combined_loss to
    optionally skip its final .mean(): duplicating the ~10 lines of math
    means a bug or future change here can never alter what's actually
    backpropagated, and vice versa -- the same reasoning this project
    already applies to every other piece of per-stage duplicated logic.
    Every call site passes detached tensors under torch.no_grad(), so this
    adds bookkeeping cost but never touches the training graph.

    Exactly equal to combined_loss(...).item() averaged back out per
    sample WHEN every sample in the batch has the same shape (always true
    here -- training patches all use data.patch_size, and
    build_synthetic_ct_dataloaders already refuses batch_size > 1 without
    a fixed patch_size). The equal-shape requirement matters because
    combined_loss's BCE/focal branches use PyTorch's default 'mean'
    reduction, which averages over every voxel across the WHOLE batch --
    that only equals the mean of each sample's own per-voxel mean when
    every sample contributes the same number of voxels.
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    if loss_type == "dice_bce":
        intersection = (probs_flat * target_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_per_sample = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
        bce_per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none").reshape(logits.shape[0], -1).mean(dim=1)
        return dice_per_sample + bce_weight * bce_per_sample
    elif loss_type in ("tversky", "focal_tversky"):
        tp = (probs_flat * target_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - target_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * target_flat).sum(dim=1)
        tversky_index = (tp + 1.0) / (tp + tversky_alpha * fp + tversky_beta * fn + 1.0)
        if loss_type == "tversky":
            return 1.0 - tversky_index
        return (1.0 - tversky_index) ** 0.75  # matches focal_tversky_loss's default gamma -- combined_loss never overrides it
    elif loss_type == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
        focal_weight = alpha_t * (1.0 - p_t) ** focal_gamma
        return (focal_weight * bce).reshape(logits.shape[0], -1).mean(dim=1)
    else:
        raise ValueError(f"Unknown training.loss_type {loss_type!r} -- expected one of: dice_bce, tversky, focal_tversky, focal.")


PATIENT_LOSS_LOG_FIELDS = ["step", "patient_id", "loss"]


def init_patient_loss_log(path: str, resuming: bool) -> None:
    """Same resumability convention as init_log_file: keep and append to an
    existing file on resume, otherwise start fresh with just a header."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if resuming and os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PATIENT_LOSS_LOG_FIELDS)


def append_patient_loss_rows(path: str, rows: list[tuple[int, str, float]]) -> None:
    """Appends a batch of (step, patient_id, loss) rows at once -- the
    training loop buffers these in memory and flushes at the same cadence
    as the main log_interval, rather than opening the file every single
    step (at batch_size=2 and total_steps=20000, that would be up to
    40000 separate file opens over a run)."""
    if not rows:
        return
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for step, patient_id, loss in rows:
            writer.writerow([step, patient_id, f"{loss:.6f}"])


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
    loss_kwargs: dict, patch_size: tuple[int, int, int] | None,
):
    """Combined loss + Dice score over up to max_patients val patients, on
    RAW (non-EMA) weights -- matches training loss's own weights, same
    anti-EMA-contamination reasoning as Stage 1. Center-crops each val
    volume to `patch_size` rather than running full-volume inference (see
    _center_crop_batch_to_patch) -- this periodic check is a patch-level
    signal, not a whole-brain one. `loss_kwargs` is passed straight through
    to combined_loss (bce_weight, loss_type, tversky_alpha, ...) so
    validation always scores with the exact same loss configuration
    training uses.
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
            logits = model(ct)
        total_loss += combined_loss(logits.float(), mask.float(), **loss_kwargs).item()
        total_dice += dice_score(torch.sigmoid(logits.float()), mask.float())
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
    parser.add_argument("--warm_start_checkpoint", type=str, default=None,
                         help="Load ONLY model weights from this exact checkpoint file, then start a "
                              "completely FRESH optimizer/scheduler/EMA/step-count (step 0) using the "
                              "current config's lr/warmup_steps/total_steps -- for switching loss_type "
                              "(or any other training-dynamics change) partway through a run, where a "
                              "normal resume would carry over the OLD optimizer momentum and a fully-"
                              "decayed LR schedule, leaving the new loss no room to actually move the "
                              "weights (found 2026-07-19 switching to Tversky/Focal mid-schedule).  "
                              "Bypasses checkpoint.working_dir's normal auto-resume search entirely -- "
                              "point checkpoint.working_dir at a NEW, different directory in the config "
                              "for this run, so its checkpoints (step 0, 25, 50, ...) don't share a "
                              "directory with the old run's (which reached step 20000+): otherwise a "
                              "later plain resume would find the OLD run's higher step count and revive "
                              "exactly the stale-LR problem this flag exists to avoid.")
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
    loss_kwargs = {
        "bce_weight": train_cfg.get("bce_weight", 1.0),
        "loss_type": train_cfg.get("loss_type", "dice_bce"),
        "tversky_alpha": train_cfg.get("tversky_alpha", 0.3),
        "tversky_beta": train_cfg.get("tversky_beta", 0.7),
        "focal_gamma": train_cfg.get("focal_gamma", 2.0),
        "focal_alpha": train_cfg.get("focal_alpha", 0.25),
    }
    log.info("Using loss_type=%s (loss_kwargs=%s)", loss_kwargs["loss_type"], loss_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(train_cfg.get("warmup_steps", 0), train_cfg["total_steps"], train_cfg.get("lr_schedule", "cosine")),
    )

    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ckpt_cfg = config["checkpoint"]
    search_dirs = [ckpt_cfg["working_dir"]] + list(ckpt_cfg.get("extra_resume_dirs", []))

    global_step = 0
    if args.warm_start_checkpoint:
        # Deliberately bypasses find_latest_checkpoint entirely: load ONLY the model's
        # weights, then build EMA fresh (from those now-loaded weights, not stale
        # random-init ones) -- optimizer and scheduler were already constructed fresh
        # above and are left untouched. global_step stays 0: this is a new schedule,
        # not a continuation of the old one's step count.
        warm_step, _extra = load_checkpoint(args.warm_start_checkpoint, model, ema=None, optimizer=None, scheduler=None, map_location=device.type)
        ema = EMA(model, decay=train_cfg.get("ema_decay", 0.999))
        existing = find_latest_checkpoint(search_dirs)
        if existing is not None:
            log.warning(
                "checkpoint.working_dir (%s) already has a checkpoint (%s) from a previous run -- "
                "warm-starting into the SAME directory risks a future plain (non-warm-start) resume "
                "finding that OLD, higher-step checkpoint instead of this fresh run's, reviving the "
                "exact stale-LR problem --warm_start_checkpoint exists to avoid. Point "
                "checkpoint.working_dir at a new directory for this run.",
                ckpt_cfg["working_dir"], existing,
            )
        if os.path.exists(train_cfg["log_file"]):
            log.warning(
                "training.log_file (%s) already exists from a previous run -- since this is a fresh "
                "(step 0) schedule, init_log_file below will TRUNCATE and overwrite it, destroying the "
                "old run's log. Point training.log_file at a new path for this run if you want to keep it.",
                train_cfg["log_file"],
            )
        log.info(
            "Warm-started model weights from %s (was step %d there, now discarded) -- "
            "optimizer, scheduler, EMA, and step count are all FRESH, starting at step 0 "
            "with lr=%.6g.", args.warm_start_checkpoint, warm_step, train_cfg["lr"],
        )
    else:
        ema = EMA(model, decay=train_cfg.get("ema_decay", 0.999))
        latest_ckpt = find_latest_checkpoint(search_dirs)
        if latest_ckpt is not None:
            global_step, _extra = load_checkpoint(latest_ckpt, model, ema, optimizer, scheduler, map_location=device.type)
            ema.decay = train_cfg.get("ema_decay", 0.999)  # re-apply from config on resume, same fix as Stage 1 -- see PROJECT_NOTES.md
            log.info("Resumed from checkpoint %s at step %d", latest_ckpt, global_step)
        else:
            log.info("No checkpoint found in %s -- starting from scratch", search_dirs)

    log_file = train_cfg["log_file"]
    init_log_file(log_file, resuming=global_step > 0)

    # Defaults to a sibling of log_file rather than requiring every existing config
    # to add this key -- see per_sample_loss's docstring for why this is tracked at
    # all: it's what analysis/build_exclude_list.py combines with
    # analysis/analyze_mask_quality.py's mask-geometry audit to flag patients whose
    # Stage 2 output is both statistically unusual AND consistently hard for the
    # model to learn from.
    patient_loss_log = train_cfg.get("patient_loss_log", os.path.join(os.path.dirname(log_file) or ".", "patient_loss_log.csv"))
    init_patient_loss_log(patient_loss_log, resuming=global_step > 0)
    pending_patient_loss_rows: list[tuple[int, str, float]] = []

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

    # Tracks consecutive non-finite-loss skips (reset to 0 on every successful step)
    # -- see the non-finite-loss branch below for why this exists.
    consecutive_non_finite_skips = 0
    max_consecutive_non_finite_skips = train_cfg.get("max_consecutive_non_finite_skips", 20)

    while global_step < total_steps:
        batch = train_cycle.next()
        ct, mask = batch["ct"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(ct)
            # combined_loss's own reductions (dice/tversky's .sum(dim=1) over
            # a whole 96x96x64 patch, focal's .mean() over the full batch)
            # are plain elementwise/reduction ops, not on autocast's forced-
            # fp32 op list -- left in fp16 under autocast, their running sum
            # over hundreds of thousands of voxels can silently exceed fp16's
            # ~65504 max and overflow to inf/nan, independent of (and NOT
            # caught by) GradScaler, which only guards the backward/gradient
            # stage. Found 2026-07-22: a real run hit loss=nan at step ~8000
            # and stayed nan every step after -- GradScaler correctly skipped
            # every subsequent optimizer.step() once that started (protecting
            # the weights, but freezing training silently). Fix: compute the
            # loss itself in float32, matching quick_validation's existing
            # (already-correct) combined_loss(logits.float(), mask.float(), ...)
            # call above -- this was the one place still missing that cast.
            loss = combined_loss(logits.float(), mask.float(), **loss_kwargs)

        if not torch.isfinite(loss):
            # GradScaler already refuses to apply a non-finite gradient (it
            # would silently skip scaler.step() below and just shrink its
            # scale factor), so weights can't actually get corrupted by
            # this -- but without this check, a persistently non-finite loss
            # (e.g. the fp16-overflow bug above, or any future numerical
            # edge case) logs "loss=nan" every step forever with no
            # indication training has effectively frozen, exactly what
            # happened in the real run that surfaced this 2026-07-22.
            # Skip the step entirely (don't even spend a wasted backward())
            # rather than relying on that silent skip.
            log.warning(
                "step %d: non-finite loss (%s) on patient(s) %s -- skipping this step, weights left unchanged. "
                "If this recurs, inspect that patient's synthetic_ct.nii(.gz)/tumor_mask.nii(.gz) directly for a "
                "NaN/Inf voxel (a bad Stage 2 output would explain a persistently-triggering patient).",
                global_step, loss.item(), batch["patient_id"],
            )
            consecutive_non_finite_skips += 1
            if consecutive_non_finite_skips >= max_consecutive_non_finite_skips:
                # A handful of consecutive skips could plausibly be a run of individually
                # bad patients (e.g. a corrupted Stage 2 output the exclude-list tooling
                # hasn't caught yet). max_consecutive_non_finite_skips (default 20)
                # different, essentially-random patient batches ALL producing a
                # non-finite loss in a row is a different situation: the model's
                # weights themselves are almost certainly in a state where the forward
                # pass overflows for ANY input, not any specific patient's data (real
                # case seen 2026-07-31: 20000-step run stuck at one step number for
                # dozens of consecutive skips across entirely different patients, never
                # recovering -- previously this would have silently run forever,
                # logging "skipping this step" while making zero real progress, until
                # something external noticed). Raising here turns that silent freeze
                # into a fast, visible failure -- and specifically into a NONZERO EXIT,
                # so scripts/train_stage3_watchdog.py detects it as a crash and
                # restarts from the last checkpoint within seconds, not the 15 minutes
                # its stall-timeout would otherwise take to notice zero new log rows.
                raise RuntimeError(
                    f"step {global_step}: {consecutive_non_finite_skips} consecutive non-finite-loss "
                    "skips across different patients -- the model's weights are very likely no longer "
                    "numerically usable (not a specific bad patient; see the comment above this raise). "
                    "Stopping so a restart/resume can recover from the last good checkpoint instead of "
                    "burning compute on a permanently broken model."
                )
            continue

        consecutive_non_finite_skips = 0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip_norm", 1.0))
        scaler.step(optimizer)
        scaler.update()

        # Belt-and-suspenders check added 2026-07-22 alongside the non-finite-loss guard above: GradScaler.step()
        # already refuses to apply a non-finite gradient, so the parameters below should never actually go
        # non-finite -- if this ever fires, GradScaler's own protection failed somehow, which is a much more
        # serious problem than a single bad batch and not something a skip-and-continue can recover from (the
        # weights are already corrupted at that point). Hard-stop rather than silently training garbage for the
        # rest of the run.
        for p in model.parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"step {global_step}: model parameters are non-finite despite GradScaler's protection -- "
                    "this should not be possible and indicates a bug beyond the loss-level non-finite guard. "
                    "Stopping rather than continuing to train a corrupted model."
                )

        scheduler.step()
        ema.update(model)

        global_step += 1

        # Per-patient loss bookkeeping (see per_sample_loss's docstring): computed
        # every step on detached tensors, so this can never influence gradients --
        # it's captured every step (not just at log_interval) so a patient sampled
        # only a handful of times over a run still gets a real per-patient average
        # rather than being undersampled down to almost nothing. Only the actual
        # file WRITE is batched to log_interval, same I/O cadence as the main log.
        with torch.no_grad():
            sample_losses = per_sample_loss(logits.detach().float(), mask.float(), **loss_kwargs)
        patient_ids_this_step = batch["patient_id"] if isinstance(batch["patient_id"], list) else [batch["patient_id"]]
        pending_patient_loss_rows.extend(
            (global_step, pid, sample_loss) for pid, sample_loss in zip(patient_ids_this_step, sample_losses.tolist())
        )

        if global_step % log_interval == 0 or global_step == total_steps:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            train_dice = dice_score(torch.sigmoid(logits.detach().float()), mask.float())
            log.info("step %d/%d | loss=%.5f | dice=%.4f | lr=%.6f | elapsed=%.1fs", global_step, total_steps, loss.item(), train_dice, lr, elapsed)
            append_log_row(log_file, global_step, "train", loss.item(), train_dice, lr, elapsed)
            append_patient_loss_rows(patient_loss_log, pending_patient_loss_rows)
            pending_patient_loss_rows.clear()

        # Checkpoint BEFORE validation, deliberately -- see Stage 1's
        # train_stage1_regression.py for the real-Kaggle bug this ordering fixes.
        if global_step % checkpoint_interval == 0 or global_step == total_steps:
            path = save_checkpoint(ckpt_cfg["working_dir"], global_step, model, ema, optimizer, scheduler, keep_last_n=keep_last_n)
            log.info("Saved checkpoint: %s", path)
            log_memory_usage(global_step, device)

        if global_step % val_interval == 0 or global_step == total_steps:
            try:
                val_loss, val_dice = quick_validation(
                    model, val_loader, device, val_max_patients, amp_enabled, loss_kwargs, val_patch_size,
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
