"""Dataset for the full SynthRAD2023 brain MRI-CT cohort.

Discovery is pattern-based rather than a hardcoded path layout: the exact
nesting of the Kaggle-hosted copy (fd7akxj65n5yjxwds/synthrad-2023) hasn't
been inspected from this machine, so `discover_synthrad_patients` walks the
tree and matches per-patient leaf folders by filename pattern (the official
SynthRAD2023 release uses ct.nii.gz / mr.nii.gz / mask.nii.gz per patient
folder; patient-prefixed variants are also matched). It logs what it found
so a wrong `synthrad_root` fails loudly with zero patients instead of
silently training on nothing.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import (
    CT_BACKGROUND_HU,
    apply_mask,
    bounding_box,
    crop_to_box,
    normalize_ct,
    normalize_mri,
    pad_to_multiple,
    random_patch_crop,
    resample_to_spacing,
)

log = logging.getLogger(__name__)

_CT_RE = re.compile(r"(^|_)ct(\.nii(\.gz)?)$", re.IGNORECASE)
_MR_RE = re.compile(r"(^|_)mr(i)?(\.nii(\.gz)?)$", re.IGNORECASE)
_MASK_RE = re.compile(r"(^|_)mask(\.nii(\.gz)?)$", re.IGNORECASE)


@dataclass
class SynthRADPatient:
    patient_id: str
    ct_path: str
    mr_path: str
    mask_path: str | None


def discover_synthrad_patients(root: str, region: str | None = "brain") -> list[SynthRADPatient]:
    root = str(root)
    all_dirs = [dirpath for dirpath, _dirnames, _filenames in os.walk(root)]
    # Only filter by region if the dataset layout actually encodes it somewhere
    # (e.g. .../Task1/brain/...) -- for a flat/unlabeled layout, don't filter.
    region_is_meaningful = region is not None and any(region.lower() in d.lower() for d in all_dirs)

    patients: list[SynthRADPatient] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if region_is_meaningful and region.lower() not in dirpath.lower():
            continue

        ct_file = next((f for f in filenames if _CT_RE.search(f)), None)
        mr_file = next((f for f in filenames if _MR_RE.search(f)), None)
        mask_file = next((f for f in filenames if _MASK_RE.search(f)), None)

        if ct_file and mr_file:
            patient_id = os.path.basename(dirpath.rstrip("/\\")) or ct_file.split(".")[0]
            patients.append(
                SynthRADPatient(
                    patient_id=patient_id,
                    ct_path=os.path.join(dirpath, ct_file),
                    mr_path=os.path.join(dirpath, mr_file),
                    mask_path=os.path.join(dirpath, mask_file) if mask_file else None,
                )
            )

    log.info("discover_synthrad_patients: found %d patients under %s (region=%s)", len(patients), root, region)
    if not patients:
        log.warning(
            "No SynthRAD patients found under %s. Check data.synthrad_root and data.region in the config "
            "against the actual Kaggle input path before training.",
            root,
        )
    return sorted(patients, key=lambda p: p.patient_id)


class SynthRADBrainDataset(Dataset):
    def __init__(
        self,
        patients: list[SynthRADPatient],
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ct_clip_range: tuple[float, float] = (-1000.0, 3000.0),
        match_brats_domain: bool = True,
        crop_margin: int = 10,
        spatial_multiple: int = 16,
        patch_size: tuple[int, int, int] | None = None,
        cache_dir: str | None = None,
        seed: int = 0,
    ):
        self.patients = patients
        self.target_spacing = target_spacing
        self.ct_clip_range = ct_clip_range
        self.match_brats_domain = match_brats_domain
        self.crop_margin = crop_margin
        self.spatial_multiple = spatial_multiple
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.patients)

    def _load_and_preprocess(self, patient: SynthRADPatient) -> dict[str, np.ndarray]:
        cache_path = self.cache_dir / f"{patient.patient_id}.npz" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            return {"mri": data["mri"], "ct": data["ct"], "mask": data["mask"]}

        ct_img = resample_to_spacing(sitk.ReadImage(patient.ct_path), self.target_spacing, is_mask=False, default_value=CT_BACKGROUND_HU)
        mr_img = resample_to_spacing(sitk.ReadImage(patient.mr_path), self.target_spacing, is_mask=False, default_value=0.0)

        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mr_arr = sitk.GetArrayFromImage(mr_img).astype(np.float32)

        if patient.mask_path is not None:
            mask_img = resample_to_spacing(sitk.ReadImage(patient.mask_path), self.target_spacing, is_mask=True, default_value=0.0)
            mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
        else:
            mask_arr = (ct_arr > self.ct_clip_range[0] + 1.0).astype(np.uint8)

        if self.match_brats_domain and np.any(mask_arr):
            ct_arr = apply_mask(ct_arr, mask_arr, background_value=CT_BACKGROUND_HU)
            mr_arr = apply_mask(mr_arr, mask_arr, background_value=0.0)

        if np.any(mask_arr):
            box = bounding_box(mask_arr, margin=self.crop_margin)
            ct_arr = crop_to_box(ct_arr, box)
            mr_arr = crop_to_box(mr_arr, box)
            mask_arr = crop_to_box(mask_arr, box)

        ct_norm = normalize_ct(ct_arr, *self.ct_clip_range)
        mr_norm = normalize_mri(mr_arr, foreground_mask=mask_arr.astype(bool))

        ct_norm = pad_to_multiple(ct_norm, self.spatial_multiple, pad_value=-1.0)
        mr_norm = pad_to_multiple(mr_norm, self.spatial_multiple, pad_value=-1.0)
        mask_padded = pad_to_multiple(mask_arr.astype(np.float32), self.spatial_multiple, pad_value=0.0)

        out = {"mri": mr_norm, "ct": ct_norm, "mask": mask_padded}
        if cache_path is not None:
            np.savez_compressed(cache_path, **out)
        return out

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        arrays = self._load_and_preprocess(patient)
        mri, ct, mask = arrays["mri"], arrays["ct"], arrays["mask"]

        if self.patch_size is not None:
            mri, ct, mask = random_patch_crop([mri, ct, mask], self.patch_size, self._rng)

        return {
            "mri": torch.from_numpy(mri).unsqueeze(0).float(),
            "ct": torch.from_numpy(ct).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "patient_id": patient.patient_id,
        }


def build_synthrad_dataloaders(config: dict, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    patients = discover_synthrad_patients(data_cfg["synthrad_root"], region=data_cfg.get("region", "brain"))
    if not patients:
        raise RuntimeError(
            f"No SynthRAD patients discovered under {data_cfg['synthrad_root']!r}. "
            "Fix data.synthrad_root in the config before training."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    split = int(len(patients) * data_cfg.get("train_val_split", 0.9))
    train_idx, val_idx = order[:split], order[split:]
    if len(val_idx) == 0 and len(patients) > 1:
        train_idx, val_idx = order[:-1], order[-1:]

    train_patients = [patients[i] for i in train_idx]
    val_patients = [patients[i] for i in val_idx]
    log.info("SynthRAD split: %d train / %d val patients", len(train_patients), len(val_patients))

    common_kwargs = dict(
        target_spacing=tuple(data_cfg.get("target_spacing", (1.0, 1.0, 1.0))),
        ct_clip_range=tuple(data_cfg.get("ct_clip_range", (-1000.0, 3000.0))),
        match_brats_domain=data_cfg.get("match_brats_domain", True),
        crop_margin=data_cfg.get("crop_margin", 10),
        spatial_multiple=data_cfg.get("spatial_multiple", 16),
        cache_dir=data_cfg.get("cache_dir"),
        seed=seed,
    )
    patch_size = data_cfg.get("patch_size")
    patch_size = tuple(patch_size) if patch_size else None

    train_ds = SynthRADBrainDataset(train_patients, patch_size=patch_size, **common_kwargs)
    val_ds = SynthRADBrainDataset(val_patients, patch_size=patch_size, **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"].get("batch_size", 1),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, data_cfg.get("num_workers", 4) // 2),
        pin_memory=True,
    )
    return train_loader, val_loader
