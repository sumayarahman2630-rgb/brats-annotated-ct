"""Patient discovery correctness for both loaders. CPU-only, no GPU
required, no real dataset access required -- builds small synthetic
directory trees mirroring the confirmed real Kaggle layouts.
"""
import os

import numpy as np
import pytest
import SimpleITK as sitk

from data.loaders_brats import discover_brats_patients
from data.loaders_synthrad import discover_synthrad_patients


def _write_nii(path, shape=(6, 6, 6), dtype=np.float32):
    arr = np.zeros(shape, dtype=dtype)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path))


@pytest.fixture
def synthrad_root(tmp_path):
    brain = tmp_path / "synthrad-2023" / "Task1" / "brain"
    for pid in ["1BA001", "1BB017"]:
        d = brain / pid
        d.mkdir(parents=True)
        for name in ["ct.nii", "mask.nii", "mr.nii"]:
            _write_nii(d / name)
    return brain


@pytest.fixture
def brats_root(tmp_path):
    root = tmp_path / "MICCAI_BraTS2020_TrainingData"
    for pid in ["BraTS20_Training_001", "BraTS20_Training_083"]:
        d = root / pid
        d.mkdir(parents=True)
        for suffix in ["t1", "t1ce", "t2", "flair", "seg"]:
            _write_nii(d / f"{pid}_{suffix}.nii", dtype=np.uint8 if suffix == "seg" else np.float32)
    return root


def test_synthrad_discovers_all_complete_patients(synthrad_root):
    patients = discover_synthrad_patients(str(synthrad_root))
    assert {p.patient_id for p in patients} == {"1BA001", "1BB017"}


def test_synthrad_patient_id_is_folder_basename_no_format_assumption(synthrad_root):
    """Patient IDs like '1BA001' are not numeric-only -- discovery must not
    assume any particular ID format, just use the folder name as-is."""
    patients = discover_synthrad_patients(str(synthrad_root))
    ids = {p.patient_id for p in patients}
    assert "1BA001" in ids  # would fail if discovery tried to int()-parse IDs


def test_synthrad_excludes_incomplete_folder_by_content_not_name(synthrad_root):
    """A folder with an arbitrary (not 'overview') name but missing files
    must still be excluded -- content-based validation, not a name denylist."""
    stray = synthrad_root / "some_future_folder_xyz"
    stray.mkdir()
    (stray / "readme.txt").write_text("not a patient")

    partial = synthrad_root / "1BC099"
    partial.mkdir()
    _write_nii(partial / "ct.nii")
    _write_nii(partial / "mr.nii")  # no mask.nii

    patients = discover_synthrad_patients(str(synthrad_root))
    ids = {p.patient_id for p in patients}
    assert ids == {"1BA001", "1BB017"}
    assert "some_future_folder_xyz" not in ids
    assert "1BC099" not in ids


def test_synthrad_region_fallback_descends_into_task1_region(tmp_path):
    """If root points at the dataset's top level instead of directly at
    Task1/brain, passing region='brain' should still find the patients."""
    brain = tmp_path / "synthrad-2023" / "Task1" / "brain"
    d = brain / "1BA001"
    d.mkdir(parents=True)
    for name in ["ct.nii", "mask.nii", "mr.nii"]:
        _write_nii(d / name)

    patients = discover_synthrad_patients(str(tmp_path / "synthrad-2023"), region="brain")
    assert {p.patient_id for p in patients} == {"1BA001"}


def test_brats_discovers_t1_and_seg_pairs(brats_root):
    patients = discover_brats_patients(str(brats_root))
    assert {p.patient_id for p in patients} == {"BraTS20_Training_001", "BraTS20_Training_083"}
    assert all(p.seg_path is not None for p in patients)


def test_brats_never_confuses_t1ce_with_t1(brats_root):
    """Check the filename only, not the full path -- pytest's tmp_path
    directory is itself named after this test function, which contains the
    substring "t1ce", so checking the full path gives a false positive."""
    patients = discover_brats_patients(str(brats_root))
    for p in patients:
        filename = os.path.basename(p.t1_path).lower()
        assert "t1ce" not in filename
        assert filename.endswith("_t1.nii") or filename.endswith("_t1.nii.gz")


def test_brats_patient_with_no_seg_has_none_seg_path(tmp_path):
    root = tmp_path / "MICCAI_BraTS2020_TrainingData"
    d = root / "BraTS20_Training_999"
    d.mkdir(parents=True)
    _write_nii(d / "BraTS20_Training_999_t1.nii")

    patients = discover_brats_patients(str(root))
    assert len(patients) == 1
    assert patients[0].seg_path is None
