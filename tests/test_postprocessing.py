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


def test_min_size_ratio_default_matches_original_strict_behavior():
    """min_size_ratio=0.0 (the default) must reproduce the exact original
    single-largest-only behavior -- this is a real backward-compatibility
    requirement, not just a nice-to-have."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[2:4, 2:4, 2:4] = 1.0
    mask[10:16, 10:16, 10:16] = 1.0
    assert keep_largest_connected_component(mask).sum() == keep_largest_connected_component(mask, min_size_ratio=0.0).sum() == 216


def test_min_size_ratio_keeps_comparably_sized_second_component():
    """Two separate blobs of comparable size (e.g. genuine bilateral
    disease) -- a positive min_size_ratio should keep both, unlike the
    strict single-largest default."""
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[2:6, 2:6, 2:6] = 1.0      # 64 voxels
    mask[12:16, 12:16, 12:16] = 1.0  # 64 voxels -- same size, separate component
    strict = keep_largest_connected_component(mask)
    assert strict.sum() == 64  # only one of the two, tie broken by argmax

    lenient = keep_largest_connected_component(mask, min_size_ratio=0.5)
    assert lenient.sum() == 128  # both kept, since the smaller is 100% >= 50% of the largest


def test_min_size_ratio_still_drops_a_truly_spurious_small_blob():
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[2:4, 2:4, 2:4] = 1.0        # 8 voxels -- tiny, spurious
    mask[10:16, 10:16, 10:16] = 1.0  # 216 voxels -- the real lesion
    filtered = keep_largest_connected_component(mask, min_size_ratio=0.5)
    assert filtered.sum() == 216  # 8 is nowhere near 50% of 216 -- correctly dropped


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
