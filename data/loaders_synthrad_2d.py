"""2D slice dataset for the standalone 2D pipeline. Wraps
SynthRADBrainDataset (data/loaders_synthrad.py) to reuse its exact
preprocessing (resample, brain-mask, crop, normalize) rather than
duplicating it -- the only new logic here is slicing a preprocessed 3D
volume down to 2D and building an index of which slices are worth training
on.

Background-slice skipping: a cropped brain volume still has some axial
slices near the top/bottom of the crop that are mostly empty (skull apex,
below the brain). Training heavily on near-empty slices wastes compute and
teaches the model little, so slices below `min_foreground_fraction` brain-
mask coverage are excluded from the index -- standard practice for
slice-based medical image training.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.loaders_synthrad import SynthRADPatient, discover_synthrad_patients
from data.preprocessing import pad_or_crop_to_shape

log = logging.getLogger(__name__)


class SynthRADSlice2DDataset(Dataset):
    def __init__(
        self,
        patients: list[SynthRADPatient],
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ct_clip_range: tuple[float, float] = (-1000.0, 3000.0),
        match_brats_domain: bool = True,
        crop_margin: int = 10,
        slice_size: tuple[int, int] = (256, 256),
        cache_dir: str | None = None,
        slice_axis: int = 0,
        min_foreground_fraction: float = 0.02,
        seed: int = 0,
    ):
        # Reuse the 3D dataset purely as a preprocessing engine (its own
        # spatial_multiple is irrelevant here -- each 2D slice gets padded/
        # cropped to a fixed slice_size independently, not the whole 3D volume).
        from data.loaders_synthrad import SynthRADBrainDataset

        self._volume_ds = SynthRADBrainDataset(
            patients,
            target_spacing=target_spacing,
            ct_clip_range=ct_clip_range,
            match_brats_domain=match_brats_domain,
            crop_margin=crop_margin,
            spatial_multiple=1,
            patch_size=None,
            cache_dir=cache_dir,
            seed=seed,
        )
        self.slice_axis = slice_axis
        self.slice_size = tuple(slice_size)
        self.min_foreground_fraction = min_foreground_fraction
        self.index: list[tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        n_skipped_slices = 0
        for patient_idx, patient in enumerate(self._volume_ds.patients):
            arrays = self._volume_ds._load_and_preprocess(patient)
            mask = arrays["mask"]
            n_slices = mask.shape[self.slice_axis]
            for s in range(n_slices):
                fg_fraction = float(np.take(mask, s, axis=self.slice_axis).mean())
                if fg_fraction >= self.min_foreground_fraction:
                    self.index.append((patient_idx, s))
                else:
                    n_skipped_slices += 1
        log.info(
            "SynthRADSlice2DDataset: %d usable slices across %d patients (%d slices skipped, below "
            "min_foreground_fraction=%.3f)",
            len(self.index), len(self._volume_ds.patients), n_skipped_slices, self.min_foreground_fraction,
        )
        if not self.index:
            log.warning("No usable slices found -- check min_foreground_fraction and crop_margin.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        patient_idx, slice_idx = self.index[idx]
        patient = self._volume_ds.patients[patient_idx]
        arrays = self._volume_ds._load_and_preprocess(patient)

        mri_slice = np.take(arrays["mri"], slice_idx, axis=self.slice_axis)
        ct_slice = np.take(arrays["ct"], slice_idx, axis=self.slice_axis)
        mask_slice = np.take(arrays["mask"], slice_idx, axis=self.slice_axis)

        if any(s > t for s, t in zip(mri_slice.shape, self.slice_size)):
            log.warning(
                "Patient %s slice %d has shape %s, larger than slice_size=%s -- will be CENTER-CROPPED, "
                "which can cut off brain content at the edges. Increase data.slice_size if this recurs.",
                patient.patient_id, slice_idx, mri_slice.shape, self.slice_size,
            )

        # Fixed target size (not just multiple-alignment) so every slice in a batch --
        # even from different patients with different natural crop sizes -- has
        # identical shape and can be stacked by the default DataLoader collate.
        mri_slice = pad_or_crop_to_shape(mri_slice, self.slice_size, pad_value=-1.0)
        ct_slice = pad_or_crop_to_shape(ct_slice, self.slice_size, pad_value=-1.0)
        mask_slice = pad_or_crop_to_shape(mask_slice.astype(np.float32), self.slice_size, pad_value=0.0)

        return {
            "mri": torch.from_numpy(mri_slice).unsqueeze(0).float(),
            "ct": torch.from_numpy(ct_slice).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_slice).unsqueeze(0).float(),
            "patient_id": patient.patient_id,
            "slice_idx": slice_idx,
        }


def build_synthrad_2d_dataloaders(config: dict, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    patients = discover_synthrad_patients(data_cfg["synthrad_root"], region=data_cfg.get("region"))
    if not patients:
        raise RuntimeError(
            f"No SynthRAD patients discovered under {data_cfg['synthrad_root']!r}. "
            "Fix data.synthrad_root in the config before training."
        )

    max_patients = data_cfg.get("max_patients")
    if max_patients:
        patients = patients[:max_patients]
        log.info("data.max_patients=%d set -- using only %d patients (smoke-test mode)", max_patients, len(patients))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    split = int(len(patients) * data_cfg.get("train_val_split", 0.9))
    train_idx, val_idx = order[:split], order[split:]
    if len(val_idx) == 0 and len(patients) > 1:
        train_idx, val_idx = order[:-1], order[-1:]

    train_patients = [patients[i] for i in train_idx]
    val_patients = [patients[i] for i in val_idx]
    log.info("SynthRAD 2D split: %d train / %d val patients", len(train_patients), len(val_patients))

    common_kwargs = dict(
        target_spacing=tuple(data_cfg.get("target_spacing", (1.0, 1.0, 1.0))),
        ct_clip_range=tuple(data_cfg.get("ct_clip_range", (-1000.0, 3000.0))),
        match_brats_domain=data_cfg.get("match_brats_domain", True),
        crop_margin=data_cfg.get("crop_margin", 10),
        slice_size=tuple(data_cfg.get("slice_size", (256, 256))),
        cache_dir=data_cfg.get("cache_dir"),
        slice_axis=data_cfg.get("slice_axis", 0),
        min_foreground_fraction=data_cfg.get("min_foreground_fraction", 0.02),
        seed=seed,
    )

    train_ds = SynthRADSlice2DDataset(train_patients, **common_kwargs)
    val_ds = SynthRADSlice2DDataset(val_patients, **common_kwargs)

    batch_size = config["training"].get("batch_size", 8)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg.get("num_workers", 4), drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, data_cfg.get("num_workers", 4) // 2), pin_memory=True,
    )
    return train_loader, val_loader
