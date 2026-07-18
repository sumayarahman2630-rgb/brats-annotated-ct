"""Pipeline role: the Stage 1 training data source -- discovers SynthRAD2023
brain patients, preprocesses each MRI/CT/mask triple (resample, brain-mask,
crop, normalize, pad), and builds the train/val DataLoaders every Stage 1
training script (the active regression one and the archived diffusion one)
consumes identically. This is the one place patient-level train/val
splitting happens, so it's worth reading build_synthrad_dataloaders's
docstring below before touching it.

Confirmed exact layout (2026-07-15): every immediate subfolder of
data.synthrad_root (.../synthrad-2023/Task1/brain) is either a patient
folder containing ct.nii, mask.nii, mr.nii, or a non-patient folder like
"overview" that Kaggle copies alongside the patient data. `discover_
synthrad_patients` qualifies a folder as a patient by *content* -- it
actually contains all three required files -- rather than by excluding
known non-patient names. That's deliberately more robust than a name
denylist: any other stray folder that shows up in a future dataset version
gets excluded the same way "overview" is, with no code change needed.
"""
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
    """One discovered patient's file paths -- what discover_synthrad_patients
    returns and what SynthRADBrainDataset consumes."""
    patient_id: str
    ct_path: str
    mr_path: str
    mask_path: str


def discover_synthrad_patients(root: str, region: str | None = None) -> list[SynthRADPatient]:
    """root should be the directory whose immediate children are patient
    folders (i.e. .../synthrad-2023/Task1/brain). `region` is an optional
    convenience for callers that instead point root at the dataset's top
    level (.../synthrad-2023): if root/Task1/<region> exists, it's used
    as the actual patient-folder directory."""
    root_path = Path(root)
    if region:
        candidate = root_path / "Task1" / region
        if candidate.is_dir():
            root_path = candidate

    if not root_path.is_dir():
        log.warning("discover_synthrad_patients: %s is not a directory", root_path)
        return []

    patients: list[SynthRADPatient] = []
    n_skipped = 0
    for folder in sorted(root_path.iterdir()):
        if not folder.is_dir():
            continue

        filenames = [f.name for f in folder.iterdir() if f.is_file()]
        ct_file = next((f for f in filenames if _CT_RE.search(f)), None)
        mr_file = next((f for f in filenames if _MR_RE.search(f)), None)
        mask_file = next((f for f in filenames if _MASK_RE.search(f)), None)

        if not (ct_file and mr_file and mask_file):
            n_skipped += 1
            log.debug(
                "discover_synthrad_patients: skipping %s -- not a complete patient folder (ct=%s mr=%s mask=%s)",
                folder.name, bool(ct_file), bool(mr_file), bool(mask_file),
            )
            continue

        patients.append(
            SynthRADPatient(
                patient_id=folder.name,
                ct_path=str(folder / ct_file),
                mr_path=str(folder / mr_file),
                mask_path=str(folder / mask_file),
            )
        )

    log.info(
        "discover_synthrad_patients: found %d patients under %s (%d non-patient folders skipped, e.g. 'overview')",
        len(patients), root_path, n_skipped,
    )
    if not patients:
        log.warning(
            "No SynthRAD patients found under %s. Check data.synthrad_root in the config "
            "against the actual Kaggle input path before training.",
            root_path,
        )
    return patients


class SynthRADBrainDataset(Dataset):
    """One patient in, one preprocessed (mri, ct, mask) tensor triple out.
    Preprocessing order matters and mirrors SynthRAD2023's own convention:
    resample -> (optional) mask both modalities to brain-only -> crop to
    the mask's bounding box -> normalize -> pad to a clean multiple. Every
    Stage 1 config (regression or diffusion) constructs this class the same
    way via build_synthrad_dataloaders below."""

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
        """Read one patient's raw NIfTI files and run the full preprocessing
        chain (resample -> brain-domain mask -> crop -> normalize -> pad),
        or return the cached result from a prior call if cache_dir is set --
        this is the expensive part of __getitem__, run once per patient
        rather than once per epoch."""
        cache_path = self.cache_dir / f"{patient.patient_id}.npz" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            return {"mri": data["mri"], "ct": data["ct"], "mask": data["mask"]}

        ct_img = resample_to_spacing(sitk.ReadImage(patient.ct_path), self.target_spacing, is_mask=False, default_value=CT_BACKGROUND_HU)
        mr_img = resample_to_spacing(sitk.ReadImage(patient.mr_path), self.target_spacing, is_mask=False, default_value=0.0)

        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mr_arr = sitk.GetArrayFromImage(mr_img).astype(np.float32)

        mask_img = resample_to_spacing(sitk.ReadImage(patient.mask_path), self.target_spacing, is_mask=True, default_value=0.0)
        mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

        # Brain-domain masking (data.match_brats_domain, default True): SynthRAD MR is
        # NOT skull-stripped (includes skull/scalp/face), but BraTS T1 -- Stage 2's real
        # input -- IS skull-stripped. Training on unmasked SynthRAD would teach the model
        # to hallucinate skull/face structure from an MR channel that's all zeros outside
        # the brain at Stage 2 inference time. Zeroing both CT and MR to the mask here
        # makes Stage 1's training distribution match what Stage 2 actually feeds it --
        # see PROJECT_NOTES.md's "domain gap" note, this is a real failure mode, not a
        # hypothetical one.
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
        """Preprocess (or load cached) patient idx, optionally crop a random
        training patch out of it, and return torch tensors ready to feed the
        model. patch_size=None returns the full preprocessed volume."""
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
    """Discover patients, split them into train/val, and build both
    DataLoaders -- the standard entry point every Stage 1 training and
    validation script uses so the split is guaranteed identical everywhere
    (same seed, same algorithm) rather than each script rolling its own.

    Patient-level split, not image/slice-level: the split happens on the
    PATIENT LIST before any Dataset is constructed, so every voxel from a
    given patient goes entirely to train or entirely to val -- never both.
    This is what makes the split leakage-free; splitting after slicing (or
    patching) would let one patient's data appear in both sets, silently
    inflating validation metrics. The split itself is deterministic given
    `seed` (numpy's default_rng, not Python's random) so re-running with
    the same seed always reproduces the same train/val patients -- this
    matters because a training script and a separate validation/analysis
    script (e.g. inference/visualize_regression_val.py) both call this
    function independently and must agree on which patients are "unseen."
    """
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

    batch_size = config["training"].get("batch_size", 1)
    if batch_size > 1 and patch_size is None:
        raise ValueError(
            "training.batch_size > 1 requires data.patch_size to be set: without patching, "
            "different patients crop to different bounding-box shapes and cannot be stacked "
            "into a batch. Either set patch_size or keep batch_size=1 (use training.grad_accum_steps "
            "to increase the effective batch size instead)."
        )

    train_ds = SynthRADBrainDataset(train_patients, patch_size=patch_size, **common_kwargs)
    val_ds = SynthRADBrainDataset(val_patients, patch_size=patch_size, **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
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
