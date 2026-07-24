"""Tests for inference/postprocessing.py: largest-connected-component
filtering and threshold search. CPU-only, pure numpy.
"""
import numpy as np

from inference.postprocessing import find_optimal_threshold, keep_largest_connected_component


def test_keep_largest_connected_component_removes_small_blobs():
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[2:4, 2:4, 2:4] = 1.0     # small blob, 8 voxels
    mask[10:16, 10:16, 10:16] = 1.0  # large blob, 216 voxels
    filtered = keep_largest_connected_component(mask)
    assert filtered.sum() == 216
    assert filtered[2:4, 2:4, 2:4].sum() == 0
    assert filtered[10:16, 10:16, 10:16].sum() == 216


def test_keep_largest_connected_component_is_a_no_op_on_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=np.float32)
    filtered = keep_largest_connected_component(mask)
    assert filtered.sum() == 0


def _dice(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def test_find_optimal_threshold_picks_the_threshold_maximizing_mean_dice():
    # Background is confidently (falsely) high everywhere except the true tumor region,
    # which is even higher -- only thresholds ABOVE the background level cleanly recover it;
    # thresholds at or below the background level include the whole volume as a false positive.
    target = np.zeros((10, 10, 10), dtype=np.float32)
    target[3:6, 3:6, 3:6] = 1.0
    prob = np.full((10, 10, 10), 0.6, dtype=np.float32)  # everywhere: over-confident false-positive background
    prob[3:6, 3:6, 3:6] = 0.9  # true tumor region: even higher confidence

    best_threshold, best_mean_dice = find_optimal_threshold(
        [(prob, target)], dice_fn=_dice, thresholds=np.array([0.1, 0.3, 0.5, 0.7, 0.85]),
    )
    assert best_threshold == 0.7  # the first candidate (lowest, of the tied 0.7/0.85) that excludes the false-positive background
    assert best_mean_dice > 0.9


def test_find_optimal_threshold_never_reruns_inference_just_rescoring():
    """Confirms the function only takes already-computed probability
    volumes -- no model/inference argument exists, by construction."""
    import inspect
    sig = inspect.signature(find_optimal_threshold)
    assert "model" not in sig.parameters
