"""Dataset for BraTS T1 MRI volumes + their tumor segmentation masks, used
by Stage 2 (synthetic CT generation).

Discovery groups files by patient ID *extracted from the filename itself*
(e.g. "BraTS20_Training_001" from "BraTS20_Training_001_t1.nii.gz"), not by
directory structure -- this works whether the Kaggle copy nests one folder
per patient (the standard MICCAI_BraTS2020_TrainingData layout) or has
everything in one flat folder, since BraTS filenames always embed the full
patient ID regardless of layout. This is why the approach differs from
loaders_synthrad.py's directory-based grouping: SynthRAD2023's official
files are just named ct.nii.gz/mr.nii.gz with no patient ID in the
filename, so directory grouping is the only option there.

Preprocessing reuses the exact same normalize_mri / resample_to_spacing /
pad_to_multiple functions Stage 1 training used on the MR channel, with the
same target_spacing, spatial_multiple, and crop_margin values -- pass in the
same values used for configs/stage1_synthrad.yaml's data section (see
inference/run_stage2_brats.py, which does this automatically by loading
the Stage 1 config).

Critically, this also reuses Stage 1's bounding-box crop step (see
SynthRADBrainDataset._load_and_preprocess in loaders_synthrad.py: mask ->
crop-to-bbox+margin -> normalize -> pad). BraTS T1 is already skull-stripped
so the masking step is implicit (background is already 0), but the CROP
step still has to happen explicitly -- without it, the model would see the
full ~240x240x155 BraTS grid (brain filling ~40-50% of the frame) instead of
the tightly brain-cropped volumes (brain filling ~80-90% of the frame) it
was actually trained on. That mismatch doesn't crash anything (the network
is fully convolutional) but is exactly the kind of silent distribution
shift that produces garbage output -- found and fixed 2026-07-15 while
Stage 1 training was still running, specifically by re-reading both
preprocessing paths side by side before Stage 2's first real run.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from data.preprocessing import bounding_box, crop_to_box, normalize_mri, pad_to_multiple, resample_to_spacing

log = logging.getLogger(__name__)

_T1_RE = re.compile(r"^(.*)_t1(\.nii(\.gz)?)$", re.IGNORECASE)
_SEG_RE = re.compile(r"^(.*)_seg(\.nii(\.gz)?)$", re.IGNORECASE)


@dataclass
class BraTSPatient:
    patient_id: str
    t1_path: str
    seg_path: str | None


def discover_brats_patients(root: str) -> list[BraTSPatient]:
    root = str(root)
    t1_by_id: dict[str, str] = {}
    seg_by_id: dict[str, str] = {}

    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            t1_match = _T1_RE.match(fname)
            if t1_match:
                patient_id = os.path.basename(t1_match.group(1))
                t1_by_id[patient_id] = os.path.join(dirpath, fname)
                continue
            seg_match = _SEG_RE.match(fname)
            if seg_match:
                patient_id = os.path.basename(seg_match.group(1))
                seg_by_id[patient_id] = os.path.join(dirpath, fname)

    patients = [
        BraTSPatient(patient_id=pid, t1_path=path, seg_path=seg_by_id.get(pid))
        for pid, path in t1_by_id.items()
    ]
    patients.sort(key=lambda p: p.patient_id)

    log.info("discover_brats_patients: found %d patients under %s (%d with a seg mask)", len(patients), root, sum(p.seg_path is not None for p in patients))
    if not patients:
        log.warning("No BraTS patients found under %s. Check the path against the actual Kaggle input layout.", root)
    n_missing_seg = sum(p.seg_path is None for p in patients)
    if n_missing_seg:
        log.warning("%d patients have a T1 volume but no matching _seg file -- they'll be generated without a paired tumor mask.", n_missing_seg)
    return patients


class BraTSVolumeDataset(Dataset):
    """Returns preprocessed T1 tensors ready for Stage 1 inference, plus
    everything needed to write the output back in the correct physical
    space: `full_shape` (the resampled, pre-crop T1 grid -- what the
    output canvas and tumor mask need to match) and `crop_box` (where the
    cropped/generated region belongs within that full grid), plus the
    resampled reference sitk.Image (for spacing/origin/direction)."""

    def __init__(
        self,
        patients: list[BraTSPatient],
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        spatial_multiple: int = 16,
        crop_margin: int = 10,
    ):
        self.patients = patients
        self.target_spacing = target_spacing
        self.spatial_multiple = spatial_multiple
        self.crop_margin = crop_margin

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]

        t1_img = resample_to_spacing(sitk.ReadImage(patient.t1_path), self.target_spacing, is_mask=False, default_value=0.0)
        t1_arr = sitk.GetArrayFromImage(t1_img).astype(np.float32)
        full_shape = t1_arr.shape

        foreground = t1_arr != 0  # BraTS T1 is skull-stripped: background is exactly 0
        if np.any(foreground):
            crop_box = bounding_box(foreground, margin=self.crop_margin)
        else:
            crop_box = ((0, full_shape[0]), (0, full_shape[1]), (0, full_shape[2]))

        t1_cropped = crop_to_box(t1_arr, crop_box)
        foreground_cropped = crop_to_box(foreground, crop_box)

        t1_norm = normalize_mri(t1_cropped, foreground_mask=foreground_cropped)
        t1_padded = pad_to_multiple(t1_norm, self.spatial_multiple, pad_value=-1.0)

        return {
            "mri": torch.from_numpy(t1_padded).unsqueeze(0).float(),
            "patient_id": patient.patient_id,
            "seg_path": patient.seg_path,
            "full_shape": full_shape,
            "crop_box": crop_box,
            "cropped_shape": t1_cropped.shape,
            "reference_image": t1_img,
        }
