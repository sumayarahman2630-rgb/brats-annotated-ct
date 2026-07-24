"""Shared Stage 3 segmentation post-processing utilities, used by both
inference/validate_synthetic_segmentation.py and
inference/validate_jordan_segmentation.py -- largest-connected-component
filtering and validation-set threshold search. Both are OPT-IN (disabled
by default in both evaluation scripts): existing default behavior
(threshold=0.5, no filtering) is unchanged unless explicitly requested via
the scripts' --auto_threshold / --use_largest_component flags.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage


def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """Zero out every connected component of `binary_mask` except the
    largest one (by voxel count) -- removes small, spurious false-positive
    blobs scattered elsewhere in the volume without touching the model's
    main predicted region. A no-op on an all-background mask (nothing to
    filter)."""
    labeled, num_components = ndimage.label(binary_mask)
    if num_components == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, index=range(1, num_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled == largest_label).astype(binary_mask.dtype)


def find_optimal_threshold(
    prob_and_target_pairs: list[tuple[np.ndarray, np.ndarray]],
    dice_fn: Callable[[np.ndarray, np.ndarray], float],
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweeps `thresholds` and returns the (threshold, mean_dice) pair that
    MAXIMIZES the MEAN Dice across every (probability_volume, target_mask)
    pair passed in -- a single GLOBAL threshold selected on this whole set,
    never a different threshold per sample (that would be per-sample
    cherry-picking, not a legitimate, reportable improvement).

    IMPORTANT for correct use across the two eval scripts: this must only
    ever be run on the SYNTHETIC validation set, never on Jordan -- Jordan
    exists as a true held-out external test of generalization, and tuning
    a threshold on it directly would compromise that. The threshold this
    function selects on the synthetic val set should be reused, fixed, as
    the Jordan script's --threshold argument, not independently searched
    for on Jordan itself.

    `prob_and_target_pairs` must already be COMPUTED probability volumes
    (not yet thresholded) -- this never re-runs inference, only re-scores
    at each candidate threshold, so sweeping many thresholds costs nothing
    beyond cheap array comparisons.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold, best_mean_dice = 0.5, -1.0
    for t in thresholds:
        dices = [dice_fn((prob > t).astype(np.float32), target) for prob, target in prob_and_target_pairs]
        mean_dice = float(np.mean(dices))
        if mean_dice > best_mean_dice:
            best_threshold, best_mean_dice = float(t), mean_dice
    return best_threshold, best_mean_dice
