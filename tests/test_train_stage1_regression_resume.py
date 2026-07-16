"""Integration + unit tests for the regression U-Net pipeline (added
2026-07-16 after a user prototype of this same architecture got good train
metrics but had two known gaps: no patient-level train/val split, and no
brain-masking/domain-gap fix. This test file checks both fixes directly,
plus the standard train -> checkpoint -> resume cycle, mirroring
test_train_stage1_2d_resume.py's pattern. CPU-only.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

import torch

from training.train_stage1_regression import _center_crop_batch_to_patch, build_regression_dataloaders, quick_validation


def _write_fake_synthrad(root, patient_ids, shape=(16, 16, 16)):
    brain = root / "Task1" / "brain"
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = brain / pid
        d.mkdir(parents=True)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:-2, 2:-2, 2:-2] = 1
        ct = rng.normal(40, 200, size=shape).astype(np.float32)
        mr = np.abs(rng.normal(300, 100, size=shape)).astype(np.float32)
        for arr, name in [(ct, "ct.nii"), (mr, "mr.nii"), (mask, "mask.nii")]:
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img, str(d / name))
    return brain


def _base_config(tmp_path, synthrad_root, patch_size=None):
    return {
        "seed": 0,
        "data": {
            "synthrad_root": str(synthrad_root), "region": None,
            "target_spacing": [1.0, 1.0, 1.0], "ct_clip_range": [-1000.0, 3000.0],
            "match_brats_domain": True, "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": patch_size, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0, "cache_dir": None,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {
            "batch_size": 1, "lr": 0.0005, "weight_decay": 0.0, "lr_schedule": "cosine",
            "warmup_steps": 1, "total_steps": 4, "amp": False, "grad_clip_norm": 1.0,
            "ema_decay": 0.9, "log_interval": 1, "val_interval": 4, "val_max_patients": 5,
            "checkpoint_interval": 4, "keep_last_n_checkpoints": 3,
            "log_file": str(tmp_path / "logs" / "regression_log.csv"),
        },
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }


def test_patient_level_split_has_no_leakage_and_val_is_full_volume(tmp_path):
    synthrad_root = _write_fake_synthrad(tmp_path / "fake_synthrad", ["P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008"])
    config = _base_config(tmp_path, synthrad_root, patch_size=[8, 8, 8])

    train_loader, val_loader = build_regression_dataloaders(config, seed=config["seed"])
    train_ids = {p.patient_id for p in train_loader.dataset.patients}
    val_ids = {p.patient_id for p in val_loader.dataset.patients}

    assert train_ids.isdisjoint(val_ids), f"patient leakage: {train_ids & val_ids}"
    assert len(train_ids) + len(val_ids) == 8

    # train is patched (data.patch_size), val is deliberately full-volume
    # regardless of data.patch_size -- see build_regression_dataloaders's docstring.
    assert train_loader.dataset.patch_size == (8, 8, 8)
    assert val_loader.dataset.patch_size is None


def test_brain_masking_zeros_non_brain_voxels(tmp_path):
    synthrad_root = _write_fake_synthrad(tmp_path / "fake_synthrad", ["P001", "P002", "P003", "P004"])
    config = _base_config(tmp_path, synthrad_root, patch_size=None)
    assert config["data"]["match_brats_domain"] is True

    train_loader, _val_loader = build_regression_dataloaders(config, seed=config["seed"])
    batch = train_loader.dataset[0]
    ct = batch["ct"].numpy()
    mask = batch["mask"].numpy().astype(bool)
    # outside the brain mask, CT should be exactly the normalized background sentinel (-1.0)
    assert np.allclose(ct[~mask], -1.0), "non-brain CT voxels were not zeroed by match_brats_domain"


def test_center_crop_batch_shrinks_full_volume_to_patch_size():
    """Regression test for the 2026-07-16 real-Kaggle OOM: quick_validation
    used to run inference on the FULL val volume (much bigger than the
    training patch), which OOM'd on a real T4. This checks the actual fix --
    _center_crop_batch_to_patch must shrink a larger-than-patch volume down
    to exactly patch_size before the model ever sees it."""
    full_shape = (32, 40, 24)  # deliberately larger than the patch in every dim
    patch_size = (16, 16, 16)
    batch = {
        "mri": torch.zeros(1, 1, *full_shape),
        "ct": torch.zeros(1, 1, *full_shape),
        "mask": torch.ones(1, 1, *full_shape),
        "patient_id": ["FAKE001"],
    }
    cropped = _center_crop_batch_to_patch(batch, patch_size)
    for key in ("mri", "ct", "mask"):
        assert tuple(cropped[key].shape) == (1, 1, *patch_size), f"{key} shape {cropped[key].shape} != expected {(1, 1, *patch_size)}"
    # patch_size=None must be a no-op (used by the standalone full-volume comparison script)
    assert _center_crop_batch_to_patch(batch, None) is batch


def test_quick_validation_does_not_oom_style_blow_up_on_full_volume_loader(tmp_path):
    """End-to-end check that quick_validation, fed the FULL-volume val_loader
    build_regression_dataloaders always produces, actually exercises the
    crop-to-patch path and completes without ever feeding the model a
    full-size tensor -- monkeypatches the model's forward to assert on the
    input shape it actually receives."""
    synthrad_root = _write_fake_synthrad(tmp_path / "fake_synthrad", ["P001", "P002", "P003", "P004"], shape=(32, 32, 32))
    config = _base_config(tmp_path, synthrad_root, patch_size=[16, 16, 16])
    config["data"]["crop_margin"] = 4

    _train_loader, val_loader = build_regression_dataloaders(config, seed=config["seed"])
    assert val_loader.dataset.patch_size is None  # confirms this test is exercising the full-volume path

    from models.unet3d_regression import build_regression_model
    model = build_regression_model(config)

    seen_shapes = []
    real_forward = model.forward

    def spying_forward(mri):
        seen_shapes.append(tuple(mri.shape))
        return real_forward(mri)

    model.forward = spying_forward

    device = torch.device("cpu")
    avg_l1, avg_psnr = quick_validation(model, val_loader, device, max_patients=5, amp_enabled=False, ct_clip_range=(-1000.0, 3000.0), patch_size=(16, 16, 16))

    assert seen_shapes, "quick_validation never called the model"
    for shape in seen_shapes:
        assert shape == (1, 1, 16, 16, 16), f"model saw a non-cropped input shape {shape} -- the OOM-causing full-volume path is still being exercised"
    assert not np.isnan(avg_l1)


@pytest.mark.slow
def test_regression_pipeline_trains_checkpoints_and_resumes(tmp_path):
    synthrad_root = _write_fake_synthrad(tmp_path / "fake_synthrad", ["P001", "P002", "P003", "P004"])
    config = _base_config(tmp_path, synthrad_root, patch_size=None)
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    repo_root = str(Path(__file__).resolve().parents[1])
    result1 = subprocess.run(
        [sys.executable, "-m", "training.train_stage1_regression", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result1.returncode == 0, result1.stderr
    assert "Training complete: 4 steps." in result1.stderr

    config["training"]["total_steps"] = 6
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    result2 = subprocess.run(
        [sys.executable, "-m", "training.train_stage1_regression", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result2.returncode == 0, result2.stderr
    assert "Resumed from checkpoint" in result2.stderr
    assert "at step 4" in result2.stderr
    assert "Training complete: 6 steps." in result2.stderr
