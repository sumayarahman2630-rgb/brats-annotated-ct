"""Checkpoint save/find/load, shared by training (resume) and inference
(load whatever Stage 1 checkpoint currently exists). See PROJECT_NOTES.md's
"Resumability strategy" section for why this is structured this way --
short version: numbered-by-step files are the source of truth for "which
checkpoint is newest" (both within one Kaggle session and across a fresh
session pointed at a mounted previous-session Output), a ckpt_latest.pt
copy exists for convenience, and writes are atomic so a checkpoint file is
never left truncated if the session is killed mid-save.
"""
from __future__ import annotations

import os
import re
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from training.ema import EMA

_CKPT_RE = re.compile(r"ckpt_step(\d+)\.pt$")


def checkpoint_path(directory: str, step: int) -> str:
    return os.path.join(directory, f"ckpt_step{step:08d}.pt")


def find_all_checkpoints(directory: str) -> list[tuple[int, str]]:
    if not os.path.isdir(directory):
        return []
    found = []
    for fname in os.listdir(directory):
        match = _CKPT_RE.search(fname)
        if match:
            found.append((int(match.group(1)), os.path.join(directory, fname)))
    return sorted(found)


def find_latest_checkpoint(search_dirs: list[str]) -> str | None:
    """Searches every directory in order and returns the checkpoint with the
    highest step count across ALL of them -- this is what makes resuming
    within the same Kaggle session and resuming from a freshly-mounted
    previous session's Output go through the same code path."""
    best: tuple[int, str] | None = None
    for directory in search_dirs:
        for step, path in find_all_checkpoints(directory):
            if best is None or step > best[0]:
                best = (step, path)
    return best[1] if best else None


def save_checkpoint(
    directory: str,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: optim.Optimizer,
    scheduler=None,
    keep_last_n: int = 3,
    extra: dict | None = None,
) -> str:
    os.makedirs(directory, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "extra": extra or {},
    }
    path = checkpoint_path(directory, step)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on both POSIX and Windows -- no truncated checkpoint if killed mid-write

    latest_path = os.path.join(directory, "ckpt_latest.pt")
    shutil.copyfile(path, latest_path)

    _prune_old_checkpoints(directory, keep_last_n)
    return path


def _prune_old_checkpoints(directory: str, keep_last_n: int) -> None:
    ckpts = find_all_checkpoints(directory)
    if len(ckpts) <= keep_last_n:
        return
    for _step, path in ckpts[:-keep_last_n]:
        try:
            os.remove(path)
        except OSError:
            pass


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA | None = None,
    optimizer: optim.Optimizer | None = None,
    scheduler=None,
    map_location: str = "cpu",
) -> tuple[int, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if ema is not None and payload.get("ema_state") is not None:
        ema.load_state_dict(payload["ema_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload["step"], payload.get("extra", {})
