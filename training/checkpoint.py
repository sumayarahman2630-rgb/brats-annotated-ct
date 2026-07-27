"""Shared training infrastructure, used by both Stage 1 and Stage 3's
training scripts and every evaluation/inference script that needs to
load one of their checkpoints. See PROJECT_NOTES.md's "Resumability
strategy" section for the reasoning behind the design: checkpoints are
named by step number so "which one is newest" can always be figured out
just by listing a directory (this matters both within one Kaggle session
and when a fresh session mounts a previous session's Output as input), a
ckpt_latest.pt copy exists purely for convenience, and writes go through
a temp file + atomic rename so a checkpoint is never left half-written if
the session gets killed mid-save.
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
    """The filename a checkpoint at this step would have -- zero-padded so plain alphabetical sort already puts them in step order."""
    return os.path.join(directory, f"ckpt_step{step:08d}.pt")


def find_all_checkpoints(directory: str) -> list[tuple[int, str]]:
    """Every checkpoint file in a directory as (step, path) pairs, sorted oldest to newest. Returns an empty list for a directory that doesn't exist yet, rather than raising."""
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
    """Writes model/EMA/optimizer/scheduler state to disk, updates the ckpt_latest.pt convenience copy, and prunes anything older than the keep_last_n most recent checkpoints."""
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
    """Deletes every checkpoint file except the keep_last_n most recent ones -- ignores errors deleting an individual file rather than letting a locked/already-gone file stop training."""
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
    """Loads a checkpoint file into whichever of model/ema/optimizer/scheduler are actually passed in -- pass None for anything you don't want touched (e.g. --warm_start_checkpoint loads only the model, leaving a fresh optimizer/scheduler in place). Returns the step it was saved at plus whatever else was stashed in "extra"."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if ema is not None and payload.get("ema_state") is not None:
        ema.load_state_dict(payload["ema_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload["step"], payload.get("extra", {})
