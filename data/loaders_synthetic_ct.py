"""Stage 3 -- loads Stage 2's synthetic CT + tumor mask pairs; output: train/val DataLoaders of preprocessed (ct, binary mask) tensors."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import (
    CT_BACKGROUND_HU,
    augment_ct_mask_patch,
    bounding_box,
    crop_to_box,
    foreground_biased_patch_crop,
    normalize_ct,
    pad_to_multiple,
    random_patch_crop,
)

log = logging.getLogger(__name__)

_CT_RE = re.compile(r"synthetic_ct\.nii(\.gz)?$", re.IGNORECASE)
_MASK_RE = re.compile(r"tumor_mask\.nii(\.gz)?$", re.IGNORECASE)


@dataclass
class SyntheticCTPatient:
    """One discovered patient's file paths -- what discover_synthetic_ct_patients
    returns and what SyntheticCTSegDataset consumes."""
    patient_id: str
    ct_path: str
    mask_path: str


def load_exclude_list(path: str | None) -> set[str]:
    """Reads a plain-text exclude list -- one patient_id per line, blank
    lines and lines starting with '#' ignored -- and returns the set of
    IDs to drop. Returns an empty set if path is None (the default,
    meaning "exclude nobody"), matching how every other optional path in
    this project's configs behaves when left unset. Doesn't validate the
    IDs against any discovered patient list itself -- discover_synthetic_ct_patients
    below logs how many of these actually matched something, so a typo'd
    ID doesn't silently do nothing."""
    if not path:
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(line)
    return ids


def discover_synthetic_ct_patients(root: str, exclude_patient_ids: set[str] | None = None) -> list[SyntheticCTPatient]:
    """root's immediate children are Stage-2-output patient folders. Content-based
    validation (a folder qualifies only if both files are actually present),
    same robustness pattern as data/loaders_synthrad.py's discover_synthrad_patients
    -- any stray non-patient folder is excluded the same way, no name denylist.

    `exclude_patient_ids` (from load_exclude_list, typically built by
    analysis/build_exclude_list.py) drops matching patients from the pool
    entirely -- before the train/val split, so an excluded patient never
    ends up in either. That's deliberate: the premise for excluding a
    patient here is that its Stage 2 output itself is suspect, which would
    make it just as unreliable as a validation target as it is as a
    training example."""
    root_path = Path(root)
    if not root_path.is_dir():
        log.warning("discover_synthetic_ct_patients: %s is not a directory", root_path)
        return []

    patients: list[SyntheticCTPatient] = []
    n_skipped = 0
    for folder in sorted(root_path.iterdir()):
        if not folder.is_dir():
            continue
        filenames = [f.name for f in folder.iterdir() if f.is_file()]
        ct_file = next((f for f in filenames if _CT_RE.search(f)), None)
        mask_file = next((f for f in filenames if _MASK_RE.search(f)), None)

        if not (ct_file and mask_file):
            n_skipped += 1
            log.debug(
                "discover_synthetic_ct_patients: skipping %s -- incomplete (ct=%s mask=%s)",
                folder.name, bool(ct_file), bool(mask_file),
            )
            continue

        patients.append(
            SyntheticCTPatient(
                patient_id=folder.name,
                ct_path=str(folder / ct_file),
                mask_path=str(folder / mask_file),
            )
        )

    log.info(
        "discover_synthetic_ct_patients: found %d patients under %s (%d incomplete folders skipped)",
        len(patients), root_path, n_skipped,
    )
    if not patients:
        log.warning(
            "No synthetic CT patients found under %s. Check data.synthetic_ct_root "
            "in the config against the actual Kaggle input path.", root_path,
        )

    if exclude_patient_ids:
        before = len(patients)
        matched = {p.patient_id for p in patients} & exclude_patient_ids
        patients = [p for p in patients if p.patient_id not in exclude_patient_ids]
        log.info(
            "discover_synthetic_ct_patients: exclude list removed %d of %d patients (%d -> %d remaining)",
            len(matched), before, before, len(patients),
        )
        unmatched = exclude_patient_ids - matched
        if unmatched:
            log.warning(
                "discover_synthetic_ct_patients: %d ID(s) in the exclude list matched no discovered "
                "patient (typo, or a different synthetic_ct_root than the one the list was built "
                "from?): %s", len(unmatched), sorted(unmatched),
            )

    return patients


class SyntheticCTSegDataset(Dataset):
    """One patient in, one (ct, binary_mask) tensor pair out. Preprocessing:
    crop to the non-background bounding box (discards the large empty frame
    around Stage 2's brain-crop region) -> normalize CT to [-1, 1] (same
    normalize_ct as Stage 1, same ct_clip_range) -> binarize mask -> pad to
    a clean multiple. Train items get a patch crop (foreground-biased, see
    foreground_biased_patch_crop); val items (patch_size=None) get the full
    preprocessed volume."""

    def __init__(
        self,
        patients: list[SyntheticCTPatient],
        ct_clip_range: tuple[float, float] = (-1000.0, 3000.0),
        crop_margin: int = 10,
        spatial_multiple: int = 16,
        patch_size: tuple[int, int, int] | None = None,
        foreground_prob: float = 0.5,
        seed: int = 0,
        augment: bool = False,
        flip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        intensity_jitter_std: float = 0.05,
    ):
        """Stores preprocessing/augmentation settings for this split (train or val -- the caller decides via patch_size/foreground_prob/augment)."""
        self.patients = patients
        self.ct_clip_range = ct_clip_range
        self.crop_margin = crop_margin
        self.spatial_multiple = spatial_multiple
        self.patch_size = patch_size
        self.foreground_prob = foreground_prob
        # augment is only ever meaningful when patch_size is set (train items) -- val items
        # (patch_size=None) never augment regardless of this flag, enforced in __getitem__ below.
        self.augment = augment
        self.flip_prob = flip_prob
        self.rot90_prob = rot90_prob
        self.intensity_jitter_std = intensity_jitter_std
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of patients in this split."""
        return len(self.patients)

    def _load_and_preprocess(self, patient: SyntheticCTPatient) -> dict[str, np.ndarray]:
        """Read one patient's synthetic CT + tumor mask and run the full
        preprocessing chain: crop to non-background bbox -> normalize CT ->
        binarize mask -> pad."""
        ct_img = sitk.ReadImage(patient.ct_path)
        mask_img = sitk.ReadImage(patient.mask_path)
        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

        # BINARIZATION (see module docstring): BraTS labels 0/1/2/4 -> 0/1.
        # Deliberately done before any cropping/normalization below, so every
        # downstream step (bbox, patch sampling, loss) only ever sees a binary mask.
        mask_arr = (mask_arr > 0).astype(np.uint8)

        # Crop to the non-background content region (everywhere Stage 2 actually
        # generated something) -- discards the surrounding empty BraTS frame,
        # same bbox+margin convention as data/loaders_synthrad.py and
        # data/loaders_brats.py use for their own respective foreground definitions.
        content_mask = ct_arr != CT_BACKGROUND_HU
        if np.any(content_mask):
            box = bounding_box(content_mask, margin=self.crop_margin)
            ct_arr = crop_to_box(ct_arr, box)
            mask_arr = crop_to_box(mask_arr, box)

        ct_norm = normalize_ct(ct_arr, *self.ct_clip_range)
        ct_norm = pad_to_multiple(ct_norm, self.spatial_multiple, pad_value=-1.0)
        mask_padded = pad_to_multiple(mask_arr.astype(np.float32), self.spatial_multiple, pad_value=0.0)

        return {"ct": ct_norm, "mask": mask_padded}

    def __getitem__(self, idx: int) -> dict:
        """Preprocess patient idx, optionally crop a training patch (foreground-
        biased so tumor-containing patches aren't rare), and return torch
        tensors ready to feed the model. patch_size=None returns the full
        preprocessed volume (used for validation)."""
        patient = self.patients[idx]
        arrays = self._load_and_preprocess(patient)
        ct, mask = arrays["ct"], arrays["mask"]

        if self.patch_size is not None:
            if self.foreground_prob > 0:
                ct, mask = foreground_biased_patch_crop(
                    [ct, mask], mask, self.patch_size, self._rng, foreground_prob=self.foreground_prob,
                )
            else:
                ct, mask = random_patch_crop([ct, mask], self.patch_size, self._rng)

            if self.augment:
                ct, mask = augment_ct_mask_patch(
                    ct, mask, self._rng,
                    flip_prob=self.flip_prob, rot90_prob=self.rot90_prob,
                    intensity_jitter_std=self.intensity_jitter_std,
                )

        return {
            "ct": torch.from_numpy(ct).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "patient_id": patient.patient_id,
        }


def build_synthetic_ct_dataloaders(config: dict, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """Discover patients, split them into train/val (patient-level, no
    leakage -- exact same algorithm as data/loaders_synthrad.py's
    build_synthrad_dataloaders, kept deliberately identical so both stages'
    splits are reasoned about the same way), and build both DataLoaders.
    Train patients are patch-cropped with foreground bias; val patients are
    kept at full-volume resolution so validation Dice/IoU reflect the whole
    brain, not one random patch per check."""
    data_cfg = config["data"]
    exclude_ids = load_exclude_list(data_cfg.get("exclude_patients_file"))
    patients = discover_synthetic_ct_patients(data_cfg["synthetic_ct_root"], exclude_patient_ids=exclude_ids)
    if not patients:
        raise RuntimeError(
            f"No synthetic CT patients discovered under {data_cfg['synthetic_ct_root']!r}. "
            "Fix data.synthetic_ct_root in the config before training."
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
    log.info("Synthetic CT split (patient-level, no leakage): %d train / %d val patients", len(train_patients), len(val_patients))

    common_kwargs = dict(
        ct_clip_range=tuple(data_cfg.get("ct_clip_range", (-1000.0, 3000.0))),
        crop_margin=data_cfg.get("crop_margin", 10),
        spatial_multiple=data_cfg.get("spatial_multiple", 16),
        seed=seed,
    )
    patch_size = data_cfg.get("patch_size")
    patch_size = tuple(patch_size) if patch_size else None
    foreground_prob = data_cfg.get("foreground_prob", 0.5)

    train_ds = SyntheticCTSegDataset(
        train_patients, patch_size=patch_size, foreground_prob=foreground_prob,
        augment=data_cfg.get("augment", False),
        flip_prob=data_cfg.get("flip_prob", 0.5),
        rot90_prob=data_cfg.get("rot90_prob", 0.5),
        intensity_jitter_std=data_cfg.get("intensity_jitter_std", 0.05),
        **common_kwargs,
    )
    # full volume, deliberately different from train -- val NEVER augments (augment=False,
    # the class default), regardless of data.augment, so validation always scores the real,
    # unmodified preprocessed volume.
    val_ds = SyntheticCTSegDataset(val_patients, patch_size=None, foreground_prob=0.0, **common_kwargs)

    batch_size = config["training"].get("batch_size", 1)
    if batch_size > 1 and patch_size is None:
        raise ValueError(
            "training.batch_size > 1 requires data.patch_size to be set (train patients crop to "
            "different bounding-box shapes without it and can't be stacked into a batch)."
        )

    train_num_workers = data_cfg.get("num_workers", 4)
    val_num_workers = max(1, data_cfg.get("num_workers", 4) // 2)
    # persistent_workers=True added 2026-07-24: CycleLoader (training/train_stage3_segmentation.py)
    # creates a brand-new iter(train_loader) at every epoch boundary (~165 steps at 331 train
    # patients / batch_size=2) -- without this flag, that tears down and respawns the ENTIRE worker
    # pool each time. Real runs twice hit a severe, reproducible slowdown (10-20s -> 2000+s per 25
    # steps) around the same step range (~7000-7600); repeated worker teardown/respawn is the
    # single most likely candidate found investigating this under time pressure, but it was not
    # conclusively confirmed as the root cause (EMA.update and the main training loop were checked
    # directly and are clean) -- see PROJECT_NOTES.md's 2026-07-24 slowdown follow-up, including the
    # GPU/host memory diagnostic logging added alongside this fix for if it recurs a third time.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=train_num_workers, drop_last=True, pin_memory=True,
        persistent_workers=train_num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=val_num_workers, pin_memory=True,
        persistent_workers=val_num_workers > 0,
    )
    return train_loader, val_loader
