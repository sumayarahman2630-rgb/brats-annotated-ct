"""2D counterpart to training/train_stage1.py: trains the standalone 2D
slice-based MRI -> CT diffusion model. Run as:

    python -m training.train_stage1_2d --config configs/stage1_synthrad_2d.yaml

Same resumability design as the 3D script (see CLAUDE.md): auto-detects and
resumes from the highest-step checkpoint in checkpoint.working_dir plus any
checkpoint.extra_resume_dirs, works identically whether that's the same
Kaggle session after an interruption or a fresh session with a previous
checkpoint mounted as an input. Same --max_steps/--max_patients smoke-test
overrides, same per-component timing instrumentation, same config-value-
survives-resume fix for ema_decay as the 3D script (round 6 bug).
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

from data.loaders_synthrad_2d import build_synthrad_2d_dataloaders
from models.stage1_mri2ct_ddpm_2d import build_stage1_model_2d
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.ema import EMA

log = logging.getLogger("train_stage1_2d")
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
        writer.writerow(["step", "split", "loss", "lr", "elapsed_sec", "data_sec", "fwd_sec", "bwd_sec", "opt_sec"])


def append_log_row(
    path: str, step: int, split: str, loss: float, lr: float, elapsed: float,
    data_sec: float = 0.0, fwd_sec: float = 0.0, bwd_sec: float = 0.0, opt_sec: float = 0.0,
) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, split, f"{loss:.6f}", f"{lr:.8f}", f"{elapsed:.1f}",
                          f"{data_sec:.3f}", f"{fwd_sec:.3f}", f"{bwd_sec:.3f}", f"{opt_sec:.3f}"])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def quick_validation_loss(model, val_cycle: CycleLoader, device, num_batches: int, amp_enabled: bool) -> float:
    model.eval()
    total_loss = 0.0
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
    for _ in range(num_batches):
        batch = val_cycle.next()
        mri, ct = batch["mri"].to(device), batch["ct"].to(device)
        with autocast_ctx:
            losses = model.training_losses(mri, ct)
        total_loss += losses["loss"].item()
    model.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/stage1_synthrad_2d.yaml")
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
        log.warning("No CUDA GPU detected -- training will be slow. Expected on a local machine; use a Kaggle GPU for real training.")

    train_loader, val_loader = build_synthrad_2d_dataloaders(config, seed=config.get("seed", 0))
    train_cycle = CycleLoader(train_loader)
    val_cycle = CycleLoader(val_loader)

    model = build_stage1_model_2d(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameter count: %.1fM", n_params / 1e6)

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0))
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
        # Same class of bug as the 3D script's round-6 fix: ema.decay is silently
        # restored from the checkpoint by EMA.load_state_dict otherwise.
        ema.decay = train_cfg.get("ema_decay", 0.9999)
        log.info("Resumed from checkpoint %s at step %d (ema.decay=%.4f)", latest_ckpt, global_step, ema.decay)
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
            append_log_row(log_file, global_step, "train", accumulated_loss, lr, elapsed,
                            data_sec=data_sec, fwd_sec=fwd_sec, bwd_sec=bwd_sec, opt_sec=opt_sec)

        if global_step % val_interval == 0:
            val_loss = quick_validation_loss(model, val_cycle, device, val_batches, amp_enabled)
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            log.info("step %d  val_loss=%.5f", global_step, val_loss)
            append_log_row(log_file, global_step, "val", val_loss, lr, elapsed)

        if global_step % checkpoint_interval == 0 or global_step >= total_steps:
            path = save_checkpoint(
                ckpt_cfg["working_dir"], global_step, model, ema, optimizer, scheduler,
                keep_last_n=keep_last_n, extra={"config_path": args.config},
            )
            log.info("Saved checkpoint: %s", path)

    log.info("Training complete at step %d.", global_step)


if __name__ == "__main__":
    main()
