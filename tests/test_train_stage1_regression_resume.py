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

from training.train_stage1_regression import build_regression_dataloaders


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
