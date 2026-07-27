"""Shared training infrastructure, used by both Stage 1
(training/train_stage1_regression.py) and Stage 3
(training/train_stage3_segmentation.py) -- not specific to either.

Exponential moving average of model weights: a second, slowly-updating
copy of the parameters that smooths out the step-to-step noise of raw
SGD/Adam updates. Every checkpoint saves this alongside the real
weights, but it is NOT what any evaluation script in this project
actually uses for its reported numbers -- raw weights are, on purpose
(see PROJECT_NOTES.md's anti-EMA-contamination notes). EMA is reported
only as an extra, side-by-side comparison in
inference/generate_full_report.py, in case it turns out to score
noticeably better or worse than the raw weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Tracks one EMA shadow tensor per trainable parameter, keyed by name."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """Snapshots the model's current parameters as the starting shadow values."""
        self.decay = decay
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Nudges each shadow tensor toward the model's current weights -- call this once per training step, after the optimizer step."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrites a model's parameters in place with the current EMA shadow values -- used only for the side-by-side EMA evaluation, never for training."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.dtype))

    def state_dict(self) -> dict:
        """Everything needed to restore this EMA tracker later, for saving into a checkpoint."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores decay and shadow values from a previously saved state_dict()."""
        self.decay = state_dict["decay"]
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}
