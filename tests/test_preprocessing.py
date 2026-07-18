"""Preprocessing correctness checks. CPU-only, no GPU required.

Includes the exact numerical checks run ad-hoc during the round-8 PSNR
audit (DEVELOPMENT_LOG.md) that ruled out a normalization sign-flip/parameter-
mismatch bug -- formalized here so they run automatically instead of
being re-derived by hand next time this comes up.
"""
import numpy as np
import pytest
import SimpleITK as sitk

from data.preprocessing import (
    CT_CLIP_HIGH,
    CT_CLIP_LOW,
    NORMALIZED_BACKGROUND,
    apply_mask,
    bounding_box,
    crop_to_box,
    denormalize_ct,
    normalize_ct,
    normalize_mri,
    pad_or_crop_to_shape,
    pad_to_multiple,
    resample_to_spacing,
)


# --- normalize_ct / denormalize_ct ---

@pytest.mark.parametrize("hu", [-1000.0, -500.0, 0.0, 500.0, 1000.0, 3000.0])
def test_ct_normalize_denormalize_exact_inverse(hu):
    normed = normalize_ct(np.array([hu]))
    back = denormalize_ct(normed)
    assert back[0] == pytest.approx(hu, abs=1e-4)


def test_ct_background_maps_to_normalized_background():
    """CT_BACKGROUND_HU (-1000, the mask-fill value) must map exactly to
    NORMALIZED_BACKGROUND (-1.0) -- this is what lets MR and CT share the
    same background sentinel value after normalization."""
    normed = normalize_ct(np.array([-1000.0]))
    assert normed[0] == pytest.approx(NORMALIZED_BACKGROUND, abs=1e-6)


def test_ct_clip_range_clamps_outliers():
    out_of_range = np.array([-5000.0, 10000.0])
    normed = normalize_ct(out_of_range)
    assert normed[0] == pytest.approx(-1.0, abs=1e-6)
    assert normed[1] == pytest.approx(1.0, abs=1e-6)


def test_normalized_zero_maps_to_clip_range_midpoint():
    """Round-8 audit finding: denormalize_ct(0.0) is the clip range's
    midpoint, (CT_CLIP_LOW + CT_CLIP_HIGH) / 2 = 1000 HU -- NOT 0 HU, because
    the range is asymmetric. An undertrained model whose raw output clusters
    near normalized zero will look like it's outputting "positive HU" purely
    because of this, not because of a sign error. See DEVELOPMENT_LOG.md round 8."""
    assert denormalize_ct(np.array([0.0]))[0] == pytest.approx(1000.0, abs=1e-4)
    assert denormalize_ct(np.array([0.0]))[0] == pytest.approx((CT_CLIP_LOW + CT_CLIP_HIGH) / 2, abs=1e-4)


def test_sign_flip_hypothesis_rejected():
    """If some bug flipped the sign of a normalized value before
    denormalizing, real background (-1000 HU) would decode to +3000 HU.
    This test just pins that arithmetic fact so a future "is it a sign
    flip" question can be answered by running this file instead of
    re-deriving it by hand under time pressure."""
    normed_bg = normalize_ct(np.array([-1000.0]))[0]
    flipped = denormalize_ct(np.array([-normed_bg]))[0]
    assert flipped == pytest.approx(3000.0, abs=1e-4)


# --- normalize_mri ---

def test_mri_background_is_sentinel_not_rescaled():
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    fg = np.zeros((10, 10, 10), dtype=bool)
    fg[2:8, 2:8, 2:8] = True
    arr[fg] = np.random.default_rng(0).normal(300, 50, size=fg.sum())
    out = normalize_mri(arr, foreground_mask=fg)
    assert np.all(out[~fg] == NORMALIZED_BACKGROUND)
    assert np.all(out[fg] >= -1.0) and np.all(out[fg] <= 1.0)


def test_mri_empty_foreground_returns_all_background():
    arr = np.zeros((5, 5, 5), dtype=np.float32)
    out = normalize_mri(arr, foreground_mask=np.zeros((5, 5, 5), dtype=bool))
    assert np.all(out == NORMALIZED_BACKGROUND)


# --- apply_mask ---

def test_apply_mask_background_fill():
    arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    out = apply_mask(arr, mask, background_value=-1000.0)
    assert out[1, 1, 1] == arr[1, 1, 1]
    assert np.all(out[mask == False] == -1000.0)  # noqa: E712


# --- bounding_box / crop_to_box ---

def test_bounding_box_with_margin():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:10, 6:11, 7:12] = 1  # occupies indices 5-9, 6-10, 7-11
    box = bounding_box(mask, margin=2)
    assert box == ((3, 11), (4, 12), (5, 13))


def test_bounding_box_clips_to_array_bounds():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    box = bounding_box(mask, margin=5)
    (x0, x1), (y0, y1), (z0, z1) = box
    assert x0 == 0 and y0 == 0 and z0 == 0  # margin clipped, not negative


def test_crop_to_box_matches_bounding_box_shape():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:10, 6:11, 7:12] = 1
    box = bounding_box(mask, margin=2)
    cropped = crop_to_box(mask, box)
    expected_shape = tuple(hi - lo for lo, hi in box)
    assert cropped.shape == expected_shape
    assert cropped.sum() == mask.sum()  # no foreground voxels lost


# --- pad_to_multiple / pad_or_crop_to_shape ---

def test_pad_to_multiple_only_pads_never_crops():
    arr = np.zeros((17, 33, 50), dtype=np.float32)
    out = pad_to_multiple(arr, multiple=16)
    assert out.shape == (32, 48, 64)
    for dim in out.shape:
        assert dim % 16 == 0


def test_pad_to_multiple_noop_when_already_aligned():
    arr = np.zeros((16, 32, 48), dtype=np.float32)
    out = pad_to_multiple(arr, multiple=16)
    assert out.shape == arr.shape


def test_pad_to_multiple_works_on_2d_arrays():
    """pad_to_multiple/pad_or_crop_to_shape generalized (round 8) to any
    dimensionality so the 2D pipeline can reuse them instead of duplicating
    this logic -- must still work correctly for a plain 2D slice."""
    arr = np.zeros((17, 33), dtype=np.float32)
    out = pad_to_multiple(arr, multiple=16)
    assert out.shape == (32, 48)
    for dim in out.shape:
        assert dim % 16 == 0


def test_pad_or_crop_to_shape_roundtrip_center_alignment():
    """pad then crop back to the original shape should be an exact identity
    for the un-padded region -- this is the property run_stage2_brats.py's
    output reconstruction (pad_to_multiple then crop back) depends on."""
    rng = np.random.default_rng(0)
    original = rng.normal(size=(18, 26, 22)).astype(np.float32)
    padded = pad_to_multiple(original, multiple=16, pad_value=-1.0)
    cropped_back = pad_or_crop_to_shape(padded, original.shape, pad_value=-1.0)
    assert cropped_back.shape == original.shape
    assert np.allclose(cropped_back, original, atol=1e-6)


# --- resample_to_spacing ---

def test_resample_to_spacing_changes_size_consistently_with_spacing():
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))
    resampled = resample_to_spacing(img, target_spacing=(1.0, 1.0, 1.0), is_mask=False)
    # halving spacing should double voxel count per axis
    assert resampled.GetSize() == (40, 40, 40)


def test_resample_mask_uses_nearest_neighbor_stays_binary():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[3:7, 3:7, 3:7] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    resampled = resample_to_spacing(img, target_spacing=(0.5, 0.5, 0.5), is_mask=True)
    out_arr = sitk.GetArrayFromImage(resampled)
    unique_vals = set(np.unique(out_arr).tolist())
    assert unique_vals <= {0, 1}  # nearest-neighbor never introduces intermediate values
