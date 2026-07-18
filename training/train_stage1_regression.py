"""Stage 1 training entry point for the simple regression U-Net alternative
to the wavelet diffusion model. Run as:

    python -m training.train_stage1_regression --config configs/stage1_regression.yaml

Added 2026-07-16 after a user-trained prototype of this same simple
architecture (L1 loss, skip connections, 96x96x64 patches) got train PSNR
32-34 dB on real Kaggle data, vastly ahead of the diffusion model -- but
the prototype had no patient-level train/val split and no brain-masking.
This script fixes both by reusing data/loaders_synthrad.py's
SynthRADBrainDataset directly (same class the diffusion pipeline uses,
so match_brats_domain/crop/normalize behave identically) instead of
writing new loading logic. Train patients are patch-cropped (data.patch_size);
val patients are kept at full-volume resolution (patch_size=None) so
validation PSNR/SSIM reflect the whole brain, not a random 96x96x64 sub-
region that would vary between checks.

Same resumability design as training/train_stage1.py and
train_stage1_2d.py (numbered checkpoints, highest-step-wins resume across
checkpoint.working_dir + checkpoint.extra_resume_dirs) -- deliberately
duplicated here (own CycleLoader, own lr-lambda helper) rather than
importing from those scripts, for the same pipeline-isolation reasoning
as the 2D pipeline: a bug in one training script can't affect another.
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

from data.loaders_synthrad import SynthRADBrainDataset, discover_synthrad_patients
from data.preprocessing import denormalize_ct, pad_or_crop_to_shape
from models.unet3d_regression import build_regression_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.ema import EMA

log = logging.getLogger("train_stage1_regression")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    peak_signal_noise_ratio = None


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


def init_log_file(path: str, resuming: bool) -> None:
    """Create the CSV training log with a header row, unless resuming an
    existing run (in which case the existing file/header is kept and new
    rows are appended)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if resuming and os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "l1_loss", "psnr_fg_db", "lr", "elapsed_sec"])


def append_log_row(path: str, step: int, split: str, l1_loss: float, psnr_fg, lr: float, elapsed: float) -> None:
    """Append one row (train or val) to the CSV training log -- this is the
    file analysis/plot_validation_psnr_curve.py reads."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        psnr_str = f"{psnr_fg:.3f}" if psnr_fg is not None and not math.isnan(psnr_fg) else ""
        writer.writerow([step, split, f"{l1_loss:.6f}", psnr_str, f"{lr:.8f}", f"{elapsed:.1f}"])


def build_regression_dataloaders(config: dict, seed: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Patient-level split (same algorithm as data/loaders_synthrad.py's
    build_synthrad_dataloaders) but, unlike that function, train and val
    Datasets get DIFFERENT patch_size: train is patched for memory/batching,
    val is full-volume (patch_size=None) so validation metrics reflect the
    whole brain instead of one random patch per check."""
    data_cfg = config["data"]
    patients = discover_synthrad_patients(data_cfg["synthrad_root"], region=data_cfg.get("region"))
    if not patients:
        raise RuntimeError(f"No SynthRAD patients discovered under {data_cfg['synthrad_root']!r}.")

    max_patients = data_cfg.get("max_patients")
    if max_patients:
        patients = patients[:max_patients]
        log.info("data.max_patients=%d set -- using only %d patients (smoke-test mode)", max_patients, len(patients))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    split = int(len(patients) * data_cfg.get("train_val_split", 0.9))
    train_idx, val_idx = order[:split], order[split:]
    if len(val_idx) == 0 and len(patients) > 1:
        train_idx, val_idx = order[:-1], order[-1:]

    train_patients = [patients[i] for i in train_idx]
    val_patients = [patients[i] for i in val_idx]
    log.info("SynthRAD split (patient-level, no leakage): %d train / %d val patients", len(train_patients), len(val_patients))

    common_kwargs = dict(
        target_spacing=tuple(data_cfg.get("target_spacing", (1.0, 1.0, 1.0))),
        ct_clip_range=tuple(data_cfg.get("ct_clip_range", (-1000.0, 3000.0))),
        match_brats_domain=data_cfg.get("match_brats_domain", True),
        crop_margin=data_cfg.get("crop_margin", 10),
        spatial_multiple=data_cfg.get("spatial_multiple", 16),
        cache_dir=data_cfg.get("cache_dir"),
        seed=seed,
    )
    patch_size = data_cfg.get("patch_size")
    patch_size = tuple(patch_size) if patch_size else None

    train_ds = SynthRADBrainDataset(train_patients, patch_size=patch_size, **common_kwargs)
    val_ds = SynthRADBrainDataset(val_patients, patch_size=None, **common_kwargs)  # full volume, deliberately different from train

    batch_size = config["training"].get("batch_size", 1)
    if batch_size > 1 and patch_size is None:
        raise ValueError(
            "training.batch_size > 1 requires data.patch_size to be set (train patients crop to "
            "different bounding-box shapes without it and can't be stacked into a batch)."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg.get("num_workers", 4), drop_last=True, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, data_cfg.get("num_workers", 4) // 2), pin_memory=True,
    )
    return train_loader, val_loader


def _center_crop_batch_to_patch(batch: dict, patch_size: tuple[int, int, int] | None) -> dict:
    """quick_validation's val_loader yields FULL-volume patients (see
    build_regression_dataloaders's docstring -- deliberate, for the
    standalone visualize_regression_val.py's whole-brain metrics). A real
    SynthRAD full volume is far bigger than the training patch, and running
    it through the model here OOM'd on a real Kaggle T4 (2026-07-16): a
    single GroupNorm activation at full-volume resolution needed 1.62 GiB.
    Center-crop (deterministic, so trends are comparable across checks) down
    to the same patch_size training already uses -- reuses pad_or_crop_to_shape
    rather than new cropping logic, same helper data/preprocessing.py already
    provides and tests cover."""
    if patch_size is None:
        return batch
    out = {}
    for key, pad_value in (("mri", -1.0), ("ct", -1.0), ("mask", 0.0)):
        arr = batch[key].squeeze(0).squeeze(0).numpy()
        cropped = pad_or_crop_to_shape(arr, patch_size, pad_value=pad_value)
        out[key] = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0)
    out["patient_id"] = batch["patient_id"]
    return out


@torch.no_grad()
def quick_validation(
    model, val_loader, device, max_patients: int, amp_enabled: bool,
    ct_clip_range: tuple[float, float], patch_size: tuple[int, int, int] | None,
):
    """L1 + foreground PSNR over up to max_patients val patients, on RAW
    (non-EMA) weights -- matches training loss's own weights, and avoids the
    EMA-cold-start pitfall documented in DEVELOPMENT_LOG.md (round 6): this validation
    exists to track raw-weight training progress, not to judge EMA quality.

    Center-crops each val volume to `patch_size` (same as training) rather
    than running full-volume inference -- see _center_crop_batch_to_patch's
    docstring for why. This means this periodic check is a patch-level signal,
    not a whole-brain one; inference/visualize_regression_val.py still does
    full-volume evaluation for the real post-training numbers.
    """
    model.eval()
    total_l1 = 0.0
    psnr_values = []
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    n = 0
    for batch in val_loader:
        if n >= max_patients:
            break
        batch = _center_crop_batch_to_patch(batch, patch_size)
        mri, ct = batch["mri"].to(device), batch["ct"].to(device)
        mask = batch["mask"].squeeze(0).squeeze(0).numpy().astype(bool)
        with autocast_ctx:
            pred = model(mri)
        total_l1 += F.l1_loss(pred.float(), ct.float()).item()

        if peak_signal_noise_ratio is not None and mask.any():
            real_hu = denormalize_ct(ct.squeeze(0).squeeze(0).float().cpu().numpy(), *ct_clip_range)
            pred_hu = denormalize_ct(pred.squeeze(0).squeeze(0).float().cpu().numpy(), *ct_clip_range)
            data_range = ct_clip_range[1] - ct_clip_range[0]
            psnr_values.append(peak_signal_noise_ratio(real_hu[mask], pred_hu[mask], data_range=data_range))
        n += 1
    model.train()
    avg_l1 = total_l1 / max(1, n)
    avg_psnr = float(np.mean(psnr_values)) if psnr_values else float("nan")
    return avg_l1, avg_psnr


def main():
    """Parse args/config, build model+data+optimizer, resume from the
    latest checkpoint if one exists, then run the train loop: forward/
    backward/step each iteration, periodic logging, checkpoint-then-
    validate every `checkpoint_interval`/`val_interval` steps (in that
    order -- see the inline comment at the checkpoint-save call below for
    why), until training.total_steps is reached."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/stage1_regression.yaml")
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

    train_loader, val_loader = build_regression_dataloaders(config, seed=config.get("seed", 0))
    train_cycle = CycleLoader(train_loader)

    model = build_regression_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameter count: %.2fM", n_params / 1e6)

    train_cfg = config["training"]
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
        ema.decay = train_cfg.get("ema_decay", 0.999)  # same re-apply-from-config fix as train_stage1.py (round 8) -- see DEVELOPMENT_LOG.md
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
    ct_clip_range = tuple(config["data"].get("ct_clip_range", (-1000.0, 3000.0)))
    val_patch_size = config["data"].get("patch_size")
    val_patch_size = tuple(val_patch_size) if val_patch_size else None

    model.train()
    t_start = time.time()
    autocast_ctx = (lambda: torch.amp.autocast("cuda", enabled=amp_enabled)) if device.type == "cuda" else (lambda: nullcontext())

    while global_step < total_steps:
        batch = train_cycle.next()
        mri, ct = batch["mri"].to(device, non_blocking=True), batch["ct"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            pred = model(mri)
            loss = F.l1_loss(pred, ct)

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
            log.info("step %d/%d | l1_loss=%.5f | lr=%.6f | elapsed=%.1fs", global_step, total_steps, loss.item(), lr, elapsed)
            append_log_row(log_file, global_step, "train", loss.item(), None, lr, elapsed)

        # Checkpoint BEFORE validation, deliberately: validation is the riskier of the
        # two (real-data OOM hit here on 2026-07-16, see quick_validation's docstring),
        # so this step's checkpoint must already be safely on disk before validation
        # runs -- otherwise a validation crash costs this step's progress too, not just
        # the validation itself. (Originally the other way around; that ordering meant
        # the very first checkpoint was never written when validation OOM'd at step 1000.)
        if global_step % checkpoint_interval == 0 or global_step == total_steps:
            path = save_checkpoint(ckpt_cfg["working_dir"], global_step, model, ema, optimizer, scheduler, keep_last_n=keep_last_n)
            log.info("Saved checkpoint: %s", path)

        if global_step % val_interval == 0 or global_step == total_steps:
            try:
                val_l1, val_psnr = quick_validation(
                    model, val_loader, device, val_max_patients, amp_enabled, ct_clip_range, val_patch_size,
                )
                elapsed = time.time() - t_start
                log.info("step %d val: l1_loss=%.5f foreground_psnr=%.2f dB (over up to %d val patients)", global_step, val_l1, val_psnr, val_max_patients)
                append_log_row(log_file, global_step, "val", val_l1, val_psnr, scheduler.get_last_lr()[0], elapsed)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                # This step's checkpoint is already safely saved above -- a validation
                # OOM here costs one skipped validation readout, not training progress.
                log.warning(
                    "step %d: validation OOM (%s) -- skipping this validation pass, training continues. "
                    "Checkpoint for this step was already saved.",
                    global_step, str(e).splitlines()[0] if str(e) else type(e).__name__,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model.train()  # quick_validation calls model.eval() before the crash; restore train mode

    log.info("Training complete: %d steps.", global_step)


if __name__ == "__main__":
    main()
