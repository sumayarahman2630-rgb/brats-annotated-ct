"""Pipeline role: the single shared preprocessing module every data loader
(SynthRAD, BraTS) and both Stage 1/Stage 2 scripts import from -- resample,
HU clip/normalize, MRI percentile-normalize, brain-mask apply/crop/pad.
Having exactly one implementation of each of these is what guarantees
Stage 2's BraTS input is normalized identically to what Stage 1 trained on
(see PROJECT_NOTES.md's "domain gap" note for why that match matters here more
than in a typical pipeline). Everything here follows SynthRAD2023's
official preprocessing conventions (resample, clip, mask-based background
fill, bbox crop).
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

CT_BACKGROUND_HU = -1000.0
CT_CLIP_LOW = -1000.0
CT_CLIP_HIGH = 3000.0
NORMALIZED_BACKGROUND = -1.0


def read_image(path: str) -> sitk.Image:
    """Thin wrapper over sitk.ReadImage that accepts Path objects too."""
    return sitk.ReadImage(str(path))


def write_image(image: sitk.Image, path: str) -> None:
    """Thin wrapper over sitk.WriteImage that accepts Path objects too."""
    sitk.WriteImage(image, str(path))


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    is_mask: bool = False,
    default_value: float = 0.0,
) -> sitk.Image:
    """Matches SynthRAD2023/preprocessing's resample(): recompute size from
    the spacing ratio, linear interpolation for images, nearest-neighbor
    for masks/labels (to keep them binary/integer)."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = tuple(
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(orig_size, orig_spacing, target_spacing)
    )
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        default_value,
        sitk.sitkFloat32 if not is_mask else sitk.sitkUInt8,
    )


def clip_ct_hu(array: np.ndarray, low: float = CT_CLIP_LOW, high: float = CT_CLIP_HIGH) -> np.ndarray:
    """Clamp raw HU values to [low, high] before normalization."""
    return np.clip(array, low, high)


def normalize_ct(array: np.ndarray, low: float = CT_CLIP_LOW, high: float = CT_CLIP_HIGH) -> np.ndarray:
    """HU -> [-1, 1]. Air (-1000 HU, the mask-fill background value) maps
    exactly to -1, deliberately matching normalize_mri's background value
    so both modalities represent 'outside brain' identically."""
    clipped = clip_ct_hu(array, low, high)
    return ((clipped - low) / (high - low) * 2.0 - 1.0).astype(np.float32)


def denormalize_ct(array: np.ndarray, low: float = CT_CLIP_LOW, high: float = CT_CLIP_HIGH) -> np.ndarray:
    """[-1, 1] -> HU. Used on Stage 2 output so the saved synthetic CT is in
    real HU units, not the network's internal scale."""
    return (((array + 1.0) / 2.0) * (high - low) + low).astype(np.float32)


def normalize_mri(
    array: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    background_value: float = NORMALIZED_BACKGROUND,
) -> np.ndarray:
    """Percentile-clip and rescale to [-1, 1], computed over foreground
    voxels only (mask, or nonzero voxels if no mask given). Background
    voxels are set to a fixed sentinel rather than rescaled along with the
    foreground -- MR background is exactly 0 pre-normalization, and
    rescaling it alongside a foreground whose low percentile is > 0 would
    corrupt the background instead of leaving it as a clean flat value.
    """
    fg = foreground_mask.astype(bool) if foreground_mask is not None else (array != 0)
    if not np.any(fg):
        return np.full_like(array, background_value, dtype=np.float32)

    lo, hi = np.percentile(array[fg], [low_percentile, high_percentile])
    hi = max(hi, lo + 1e-6)
    clipped = np.clip(array, lo, hi)
    normed = (clipped - lo) / (hi - lo) * 2.0 - 1.0
    out = np.where(fg, normed, background_value)
    return out.astype(np.float32)


def apply_mask(
    array: np.ndarray,
    mask: np.ndarray,
    background_value: float,
) -> np.ndarray:
    """Zero out (set to background_value) every voxel outside `mask` -- the
    core operation behind data.match_brats_domain, since BraTS input is
    already skull-stripped and SynthRAD training data must match that."""
    return np.where(mask.astype(bool), array, background_value).astype(array.dtype)


def bounding_box(mask: np.ndarray, margin: int = 10) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Bounding box of nonzero mask voxels with a margin, clipped to array
    bounds. Mirrors SynthRAD2023/preprocessing's crop()."""
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        return (0, mask.shape[0]), (0, mask.shape[1]), (0, mask.shape[2])
    bounds = []
    for axis in range(3):
        lo = max(int(np.min(idx[axis])) - margin, 0)
        hi = min(int(np.max(idx[axis])) + margin, mask.shape[axis])
        bounds.append((lo, hi))
    return bounds[0], bounds[1], bounds[2]


def crop_to_box(array: np.ndarray, box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """Slice a 3D array down to the (x0,x1),(y0,y1),(z0,z1) box from bounding_box()."""
    (x0, x1), (y0, y1), (z0, z1) = box
    return array[x0:x1, y0:y1, z0:z1]


def pad_or_crop_to_shape(
    array: np.ndarray,
    target_shape: tuple[int, ...],
    pad_value: float = NORMALIZED_BACKGROUND,
) -> np.ndarray:
    """Center pad/crop each of the array's leading len(target_shape) axes
    independently to an exact target shape. Works for any dimensionality
    (3D volumes for Stage 1/2, or 2D slices for the 2D pipeline) -- ndim is
    read from target_shape's length, not hardcoded."""
    out = array
    ndim = len(target_shape)
    for axis in range(ndim):
        cur = out.shape[axis]
        target = target_shape[axis]
        if cur < target:
            total_pad = target - cur
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)
        elif cur > target:
            total_crop = cur - target
            crop_before = total_crop // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(crop_before, crop_before + target)
            out = out[tuple(sl)]
    return out


def pad_to_multiple(
    array: np.ndarray,
    multiple: int,
    pad_value: float = NORMALIZED_BACKGROUND,
    ndim: int | None = None,
) -> np.ndarray:
    """Pad spatial dims up to the next multiple of `multiple`. Required so
    a wavelet transform's downsampling and/or a U-Net's own downsampling
    stages divide the array exactly with no rounding/cropping mismatch --
    multiple should be (2 if using a wavelet transform, else 1) *
    2^(num_unet_downsamples). Works for 2D or 3D arrays; ndim defaults to
    the whole array's dimensionality (pass ndim=2 to only pad the first two
    axes of an array that has extra leading/trailing axes)."""
    ndim = ndim if ndim is not None else array.ndim
    target_shape = tuple(
        int(np.ceil(s / multiple) * multiple) for s in array.shape[:ndim]
    )
    return pad_or_crop_to_shape(array, target_shape, pad_value)


def random_patch_crop(
    arrays: list[np.ndarray],
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Crop the same random patch location out of several co-registered
    arrays (e.g. MR + CT + mask), for training-time memory efficiency."""
    shape = arrays[0].shape
    starts = []
    for axis in range(3):
        span = shape[axis] - patch_size[axis]
        starts.append(int(rng.integers(0, span + 1)) if span > 0 else 0)
    slices = tuple(
        slice(starts[a], starts[a] + patch_size[a]) for a in range(3)
    )
    return [arr[slices] for arr in arrays]


def foreground_biased_patch_crop(
    arrays: list[np.ndarray],
    foreground_mask: np.ndarray,
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
    foreground_prob: float = 0.5,
) -> list[np.ndarray]:
    """Added for Stage 3 (tumor segmentation): a tumor occupies a tiny
    fraction of a whole brain volume, so a purely uniform-random crop (see
    random_patch_crop above) would mostly land on empty background --
    weak or degenerate Dice gradient signal, since Dice over an all-empty
    patch's prediction/target pair carries little information. With
    probability `foreground_prob`, this instead centers the patch on a
    randomly chosen foreground (nonzero) voxel of `foreground_mask`
    (clamped so the patch stays in-bounds); otherwise -- or if the mask has
    no foreground voxels at all -- falls back to the exact same uniform
    random crop as random_patch_crop. Deliberately a new function rather
    than a random_patch_crop parameter: Stage 1/2's regression training
    reuses random_patch_crop unmodified and has no foreground-sparsity
    problem (the CT target is present everywhere), so this only applies
    where it's actually needed.
    """
    shape = arrays[0].shape
    use_foreground = foreground_prob > 0 and np.any(foreground_mask) and rng.random() < foreground_prob

    starts = []
    if use_foreground:
        fg_idx = np.nonzero(foreground_mask)
        pick = int(rng.integers(0, len(fg_idx[0])))
        for axis in range(3):
            center = int(fg_idx[axis][pick])
            span = shape[axis] - patch_size[axis]
            lo = max(0, center - patch_size[axis] // 2)
            lo = min(lo, span) if span > 0 else 0
            starts.append(lo)
    else:
        for axis in range(3):
            span = shape[axis] - patch_size[axis]
            starts.append(int(rng.integers(0, span + 1)) if span > 0 else 0)

    slices = tuple(slice(starts[a], starts[a] + patch_size[a]) for a in range(3))
    return [arr[slices] for arr in arrays]


def augment_ct_mask_patch(
    ct: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    flip_prob: float = 0.5,
    rot90_prob: float = 0.5,
    intensity_jitter_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Added 2026-07-24 for Stage 3 training-time augmentation, to improve
    generalization to the small (20-patient) Jordan external cohort and
    reduce overfitting to the 331 synthetic-CT training patients. Applies
    identically to `ct` and `mask` so they stay spatially aligned; only
    ever called on TRAIN patches (never validation).

    Per-axis random flips and a random 90-degree rotation about a random
    axis pair are used instead of arbitrary-angle rotation: both are exact
    (no interpolation needed, so the mask stays perfectly binary and the
    CT is not resampled/blurred), unlike a general-angle rotation which
    would require an interpolation choice for the mask (nearest-neighbor,
    to stay binary) and would blur the CT slightly -- not attempted here
    given the time constraint this was added under.

    Real bug found 2026-07-24, step ~25 of a real training run: patch_size
    (96, 96, 64) is NOT cubic, and a 90-degree rotation SWAPS the extents
    of its two rotated axes -- rotating in the (0,2) or (1,2) plane turns
    (96, 96, 64) into (96, 64, 96) or (64, 96, 96), changing that one
    sample's shape. Since rotation was applied independently per sample
    with a uniformly random axis pair, different samples in the same batch
    ended up with different shapes, and DataLoader's default collate
    crashed trying to stack them (`RuntimeError: stack expects each tensor
    to be equal size`). Fixed by restricting the rotation to only axis
    pairs whose sizes are already equal -- rotating in such a plane always
    preserves the overall shape exactly, for any k. For (96, 96, 64) this
    means only the (0, 1) plane is used; if a patch_size has no equal-size
    axis pair at all, rotation is skipped for that patch (flips still
    apply) rather than ever risking a shape change.

    `np.flip`/`np.rot90` return views with negative strides that
    `torch.from_numpy` cannot handle later in the pipeline -- `.copy()`
    after each is required, not optional.

    Intensity jitter is CT-only (a random scale+shift, applied only to the
    non-background region so the exact -1.0 background sentinel value stays
    a clean flat constant, matching normalize_ct's convention) and is
    skipped entirely if `intensity_jitter_std <= 0`.
    """
    for axis in range(3):
        if rng.random() < flip_prob:
            ct = np.flip(ct, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

    equal_axis_pairs = [(a, b) for a in range(3) for b in range(a + 1, 3) if ct.shape[a] == ct.shape[b]]
    if equal_axis_pairs and rng.random() < rot90_prob:
        axes = equal_axis_pairs[int(rng.integers(0, len(equal_axis_pairs)))]
        k = int(rng.integers(1, 4))
        ct = np.rot90(ct, k=k, axes=axes).copy()
        mask = np.rot90(mask, k=k, axes=axes).copy()

    if intensity_jitter_std > 0:
        scale = 1.0 + rng.normal(0.0, intensity_jitter_std)
        shift = rng.normal(0.0, intensity_jitter_std)
        foreground = ct != NORMALIZED_BACKGROUND
        ct = ct.copy()
        ct[foreground] = np.clip(ct[foreground] * scale + shift, -1.0, 1.0)

    return ct, mask
