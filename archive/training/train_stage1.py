"""Stage 1 training entry point: MRI -> CT wavelet diffusion on the full
SynthRAD2023 brain cohort. Run as:

    python -m training.train_stage1 --config configs/stage1_synthrad.yaml

Resumability (see DEVELOPMENT_LOG.md for the full explanation): on startup this
script searches checkpoint.working_dir plus every directory listed in
checkpoint.extra_resume_dirs for the checkpoint with the highest step
count, and resumes from it automatically -- exact same code path whether
that's "the same Kaggle session after an interruption" or "a brand new
Kaggle session with a previous session's Output mounted as an input."
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
import yaml

from data.loaders_synthrad import build_synthrad_dataloaders
from archive.models.stage1_mri2ct_ddpm import SUBBAND_NAMES, build_stage1_model
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.ema import EMA

log = logging.getLogger("train_stage1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_steps: int, total_steps: int, schedule: str):
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
    not epoch-based, since a full SynthRAD epoch may already be long."""

    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def next(self) -> dict:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def init_log_file(path: str, resuming: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if resuming and os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "split", "loss", "lr", "elapsed_sec", "data_sec", "fwd_sec", "bwd_sec", "opt_sec"]
            + [f"mse_{n}" for n in SUBBAND_NAMES]
        )


def append_log_row(
    path: str, step: int, split: str, loss: float, lr: float, elapsed: float,
    per_subband, data_sec: float = 0.0, fwd_sec: float = 0.0, bwd_sec: float = 0.0, opt_sec: float = 0.0,
) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [step, split, f"{loss:.6f}", f"{lr:.8f}", f"{elapsed:.1f}",
             f"{data_sec:.3f}", f"{fwd_sec:.3f}", f"{bwd_sec:.3f}", f"{opt_sec:.3f}"]
            + [f"{v:.6f}" for v in per_subband.tolist()]
        )


def _sync(device: torch.device) -> None:
    """CUDA ops queue asynchronously -- without this, time.time() around a GPU op
    mostly measures how fast the CPU can enqueue work, not how long the GPU took,
    and the real cost just shows up misattributed to whatever's timed next."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def quick_validation_loss(model, val_cycle: CycleLoader, device, num_batches: int, amp_enabled: bool):
    model.eval()
    total_loss = 0.0
    total_subband = torch.zeros(8)
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    for _ in range(num_batches):
        batch = val_cycle.next()
        mri, ct = batch["mri"].to(device), batch["ct"].to(device)
        with autocast_ctx:
            losses = model.training_losses(mri, ct)
        total_loss += losses["loss"].item()
        total_subband += losses["per_subband_mse"].float().cpu()
    model.train()
    return total_loss / num_batches, total_subband / num_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/stage1_synthrad.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Smoke-test override: run only this many steps instead of training.total_steps "
                              "from the config, without editing the file.")
    parser.add_argument("--max_patients", type=int, default=None,
                         help="Smoke-test override: use only this many discovered patients instead of "
                              "data.max_patients from the config, without editing the file. "
                              "e.g. --max_steps 100 --max_patients 3 for a quick GPU/OOM check "
                              "before a full run on the full cohort.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_steps is not None:
        config["training"]["total_steps"] = args.max_steps
        log.info("--max_steps override: training.total_steps = %d", args.max_steps)
    if args.max_patients is not None:
        config["data"]["max_patients"] = args.max_patients
        log.info("--max_patients override: data.max_patients = %d", args.max_patients)

    # A smoke-test total_steps can be far smaller than the configured checkpoint/val
    # intervals (e.g. total_steps=100 but checkpoint_interval=1000) -- clamp both down
    # so a short run still actually exercises a checkpoint save and a validation pass,
    # which is the whole point of running it.
    total_steps = config["training"]["total_steps"]
    config["training"]["checkpoint_interval"] = min(config["training"]["checkpoint_interval"], total_steps)
    config["training"]["val_interval"] = min(config["training"].get("val_interval", total_steps), total_steps)

    set_seed(config.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    if device.type != "cuda":
        log.warning("No CUDA GPU detected -- training will be extremely slow. "
                    "This is expected on a local machine; run on a Kaggle GPU session for real training.")

    train_loader, val_loader = build_synthrad_dataloaders(config, seed=config.get("seed", 0))
    train_cycle = CycleLoader(train_loader)
    val_cycle = CycleLoader(val_loader)

    model = build_stage1_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameter count: %.1fM", n_params / 1e6)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(train_cfg.get("warmup_steps", 0), train_cfg["total_steps"], train_cfg.get("lr_schedule", "cosine")),
    )
    ema = EMA(model, decay=train_cfg.get("ema_decay", 0.9999))

    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ckpt_cfg = config["checkpoint"]
    search_dirs = [ckpt_cfg["working_dir"]] + list(ckpt_cfg.get("extra_resume_dirs", []))
    latest_ckpt = find_latest_checkpoint(search_dirs)

    global_step = 0
    if latest_ckpt is not None:
        global_step, _extra = load_checkpoint(latest_ckpt, model, ema, optimizer, scheduler, map_location=device.type)
        # EMA.load_state_dict overwrites ema.decay with whatever the checkpoint stored --
        # without this, changing training.ema_decay in the config would be silently
        # ignored on resume. Let the current config win, so continued-training runs can
        # actually adjust it (e.g. lowering it so EMA converges faster on a short
        # continuation, instead of staying stuck at whatever decay the original run used).
        ema.decay = train_cfg.get("ema_decay", 0.9999)
        # Same class of bug, second instance: subband_loss_weights is a registered buffer
        # on the model (models/stage1_mri2ct_ddpm.py), so model.load_state_dict() just
        # silently restored it from the checkpoint too -- unlike the diffusion-schedule
        # buffers (betas etc.), this one is always shape (8,) regardless of its actual
        # values, so there's no shape-mismatch error to catch the problem; it just quietly
        # keeps whatever weights were checkpointed. Concretely: the round-6 LLL-weighted
        # config change ([3,1,1,1,1,1,1,1]) never actually took effect during the
        # 5500->5750 continuation -- the model kept training with the original equal
        # weights the whole time, silently. Re-apply from config, same fix as ema.decay.
        diff_cfg = config["diffusion"]
        new_weights = diff_cfg.get("subband_loss_weights")
        if new_weights is not None:
            model.subband_loss_weights.copy_(
                torch.tensor(new_weights, dtype=model.subband_loss_weights.dtype, device=model.subband_loss_weights.device)
            )
        log.info(
            "Resumed from checkpoint %s at step %d (ema.decay=%.4f, subband_loss_weights=%s)",
            latest_ckpt, global_step, ema.decay, model.subband_loss_weights.tolist(),
        )
    else:
        log.info("No checkpoint found in %s -- starting from scratch", search_dirs)

    log_file = train_cfg["log_file"]
    init_log_file(log_file, resuming=global_step > 0)

    total_steps = train_cfg["total_steps"]
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    checkpoint_interval = train_cfg["checkpoint_interval"]
    log_interval = train_cfg.get("log_interval", 50)
    val_interval = train_cfg.get("val_interval", checkpoint_interval)
    val_batches = train_cfg.get("val_batches", 5)
    keep_last_n = train_cfg.get("keep_last_n_checkpoints", 3)

    model.train()
    t_start = time.time()
    autocast_ctx = (lambda: torch.amp.autocast("cuda", enabled=amp_enabled)) if device.type == "cuda" else (lambda: nullcontext())

    while global_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_subband = torch.zeros(8)
        data_sec = fwd_sec = bwd_sec = opt_sec = 0.0

        for _ in range(grad_accum_steps):
            t0 = time.time()
            batch = train_cycle.next()
            mri, ct = batch["mri"].to(device, non_blocking=True), batch["ct"].to(device, non_blocking=True)
            _sync(device)
            data_sec += time.time() - t0

            t0 = time.time()
            with autocast_ctx():
                losses = model.training_losses(mri, ct)
                loss = losses["loss"] / grad_accum_steps
            _sync(device)
            fwd_sec += time.time() - t0

            t0 = time.time()
            scaler.scale(loss).backward()
            _sync(device)
            bwd_sec += time.time() - t0

            accumulated_loss += losses["loss"].item() / grad_accum_steps
            accumulated_subband += losses["per_subband_mse"].float().cpu() / grad_accum_steps

        t0 = time.time()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        _sync(device)
        opt_sec += time.time() - t0

        global_step += 1

        if global_step % log_interval == 0:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            log.info(
                "step %d/%d  loss=%.5f  lr=%.2e  elapsed=%.0fs  |  data=%.2fs fwd=%.2fs bwd=%.2fs opt=%.2fs (this step)",
                global_step, total_steps, accumulated_loss, lr, elapsed, data_sec, fwd_sec, bwd_sec, opt_sec,
            )
            append_log_row(log_file, global_step, "train", accumulated_loss, lr, elapsed, accumulated_subband,
                            data_sec=data_sec, fwd_sec=fwd_sec, bwd_sec=bwd_sec, opt_sec=opt_sec)

        if global_step % val_interval == 0:
            val_loss, val_subband = quick_validation_loss(model, val_cycle, device, val_batches, amp_enabled)
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            log.info("step %d  val_loss=%.5f", global_step, val_loss)
            append_log_row(log_file, global_step, "val", val_loss, lr, elapsed, val_subband)

        if global_step % checkpoint_interval == 0 or global_step >= total_steps:
            path = save_checkpoint(
                ckpt_cfg["working_dir"], global_step, model, ema, optimizer, scheduler,
                keep_last_n=keep_last_n, extra={"config_path": args.config},
            )
            log.info("Saved checkpoint: %s", path)

    log.info("Training complete at step %d.", global_step)


if __name__ == "__main__":
    main()
