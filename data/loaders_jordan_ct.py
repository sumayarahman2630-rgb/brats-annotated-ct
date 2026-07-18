"""Pipeline role: loads the Jordan University Hospital external validation
data (inference/validate_jordan_segmentation.py's only consumer) -- real CT
and real tumor mask, as individual 2D DICOM slices, matched by patient ID
and slice number extracted from filenames. NEVER used for training (see
PROJECT_NOTES.md's Stage 3 section for why: this dataset is small, 2D-only,
and a different acquisition/format from the synthetic training data --
exactly the kind of held-out set that should only ever measure
generalization, never influence it).

Known, load-bearing limitations of this dataset (see PROJECT_NOTES.md for
the full discussion -- summarized here since they directly shape this
loader's design):

1. **Format mismatch.** CT/mask files are RGB, 0-255, windowed
   ("secondary capture" DICOM -- an already-rendered image, not raw HU
   pixel data). There is no HU value to recover from an 8-bit windowed
   screenshot, so this loader can only min-max normalize each slice to
   itself (_normalize_jordan_ct below) -- structurally different from
   Stage 3's training data (real HU, clipped to a fixed physical range).
   Segmentation metrics computed against this data measure whether the
   model's predicted SHAPE agrees with the real tumor outline, not whether
   its intensity reasoning transfers -- it cannot be asked to, given the
   input format.
2. **Incomplete volumes.** Each patient has only the 1-6 tumor-containing
   slices, not a full 3D volume -- there is no real 3D neighborhood to feed
   a 3D model. See inference/validate_jordan_segmentation.py for how this
   is worked around (slice replication) and why that's a real, flagged
   approximation, not equivalent to genuine 3D context.
3. **Filename matching is not guaranteed correct.** discover_jordan_slices
   below matches CT and mask files purely by (patient_id, slice_num)
   parsed from filenames in two separate directories -- there is no
   cross-check that a matched pair actually depicts the same anatomical
   slice beyond the filename convention holding. Any CT or mask file that
   doesn't find a same-key partner is logged and excluded, never silently
   guessed at.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

_CT_RE = re.compile(r"^(.*)_CT_s(\d+)\.dcm$", re.IGNORECASE)
_MASK_RE = re.compile(r"^(.*)_CT_m(\d+)\.dcm$", re.IGNORECASE)


@dataclass
class JordanSlice:
    """One matched (CT, mask) DICOM pair -- both files confirmed to exist
    for this (patient_id, slice_num) key."""
    patient_id: str
    slice_num: int
    ct_path: str
    mask_path: str


def discover_jordan_slices(ct_root: str, mask_root: str) -> list[JordanSlice]:
    """Scans ct_root for `{patient_id}_CT_s{slice_num}.dcm` and mask_root for
    `{patient_id}_CT_m{slice_num}.dcm`, matches them by (patient_id,
    slice_num), and returns only the pairs found in BOTH directories.
    Any CT file with no matching mask (or vice versa) is logged as a
    warning with its exact filename, not silently dropped or guessed at --
    the matching depends entirely on the filename convention holding,
    which is exactly the "not guaranteed" risk flagged in the module
    docstring."""
    ct_root, mask_root = Path(ct_root), Path(mask_root)

    ct_by_key: dict[tuple[str, int], str] = {}
    for f in sorted(ct_root.iterdir()) if ct_root.is_dir() else []:
        m = _CT_RE.match(f.name)
        if m:
            ct_by_key[(m.group(1), int(m.group(2)))] = str(f)

    mask_by_key: dict[tuple[str, int], str] = {}
    for f in sorted(mask_root.iterdir()) if mask_root.is_dir() else []:
        m = _MASK_RE.match(f.name)
        if m:
            mask_by_key[(m.group(1), int(m.group(2)))] = str(f)

    if not ct_by_key:
        log.warning("discover_jordan_slices: no CT files matched under %s -- check ct_root and the _CT_s<n>.dcm naming.", ct_root)
    if not mask_by_key:
        log.warning("discover_jordan_slices: no mask files matched under %s -- check mask_root and the _CT_m<n>.dcm naming.", mask_root)

    matched_keys = ct_by_key.keys() & mask_by_key.keys()
    ct_only = ct_by_key.keys() - mask_by_key.keys()
    mask_only = mask_by_key.keys() - ct_by_key.keys()

    for key in sorted(ct_only):
        log.warning("discover_jordan_slices: CT slice %s has no matching mask -- excluded (file: %s)", key, ct_by_key[key])
    for key in sorted(mask_only):
        log.warning("discover_jordan_slices: mask slice %s has no matching CT -- excluded (file: %s)", key, mask_by_key[key])

    slices = [
        JordanSlice(patient_id=pid, slice_num=slice_num, ct_path=ct_by_key[(pid, slice_num)], mask_path=mask_by_key[(pid, slice_num)])
        for (pid, slice_num) in sorted(matched_keys)
    ]
    n_patients = len({s.patient_id for s in slices})
    log.info(
        "discover_jordan_slices: %d matched slices across %d patients (%d CT-only, %d mask-only excluded)",
        len(slices), n_patients, len(ct_only), len(mask_only),
    )
    return slices


def _read_dicom_grayscale(path: str) -> np.ndarray:
    """Read a DICOM file and return a single-channel float32 array. RGB
    (SamplesPerPixel == 3) is converted via the standard luminance formula;
    already-grayscale DICOM is returned as-is. Values are left in their
    original 0-255-ish range -- normalization happens separately (CT vs.
    mask need different treatment, see the two functions below)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr


def _normalize_jordan_ct(gray: np.ndarray) -> np.ndarray:
    """Min-max normalize a single Jordan CT slice to [-1, 1]. There is no
    HU value to recover from an 8-bit windowed image (see module
    docstring's format-mismatch limitation) -- this is the best available
    per-slice normalization, not a match to Stage 3's HU-based
    normalize_ct."""
    lo, hi = float(gray.min()), float(gray.max())
    hi = max(hi, lo + 1e-6)
    return ((gray - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def _binarize_jordan_mask(gray: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    """Threshold a (grayscale-converted) Jordan mask slice to binary. The
    mask is expected to already be near-binary (see module docstring), so
    a fixed midpoint threshold on the 0-255 range is a reasonable default
    rather than something that needs per-slice tuning."""
    return (gray > threshold).astype(np.float32)


class JordanCTSegDataset(Dataset):
    """One matched slice in, one (ct, binary_mask) 2D tensor pair out. No
    3D context, no train/val split -- see the module docstring for why
    this dataset exists only for external validation."""

    def __init__(self, slices: list[JordanSlice]):
        self.slices = slices

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> dict:
        """Read and normalize one matched CT/mask DICOM pair."""
        item = self.slices[idx]
        ct_gray = _read_dicom_grayscale(item.ct_path)
        mask_gray = _read_dicom_grayscale(item.mask_path)

        ct_norm = _normalize_jordan_ct(ct_gray)
        mask_bin = _binarize_jordan_mask(mask_gray)

        return {
            "ct": torch.from_numpy(ct_norm).unsqueeze(0).float(),   # (1, H, W)
            "mask": torch.from_numpy(mask_bin).unsqueeze(0).float(),  # (1, H, W)
            "patient_id": item.patient_id,
            "slice_num": item.slice_num,
        }


def build_jordan_dataloader(ct_root: str, mask_root: str) -> DataLoader:
    """Discover and wrap every matched Jordan slice in a DataLoader
    (batch_size=1 -- slices can have different H/W, so they can't be
    stacked into a batch)."""
    slices = discover_jordan_slices(ct_root, mask_root)
    if not slices:
        raise RuntimeError(f"No matched Jordan CT/mask slices found (ct_root={ct_root!r}, mask_root={mask_root!r}).")
    dataset = JordanCTSegDataset(slices)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
