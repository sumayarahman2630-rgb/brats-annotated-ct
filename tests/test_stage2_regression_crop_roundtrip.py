"""Integration test for the regression Stage 2 script (run_stage2_brats_regression.py),
mirroring test_stage2_crop_roundtrip.py's approach for the diffusion pipeline: trains a
tiny real regression checkpoint, then runs the real Stage 2 script end to end as a
subprocess against a fake BraTS patient whose brain occupies a small, precisely-known
region of a much larger frame -- makes any bug in the sliding-window-inference /
crop / paste-back geometry unmistakable.

CPU-only. Slow (~30-60s).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml


def _write_nii(path, arr, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


def _write_fake_synthrad(root, patient_ids, shape=(24, 24, 24)):
    brain = root / "Task1" / "brain"
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = brain / pid
        d.mkdir(parents=True)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:-2, 2:-2, 2:-2] = 1
        ct = rng.normal(40, 200, size=shape).astype(np.float32)
        mr = np.abs(rng.normal(300, 100, size=shape)).astype(np.float32)
        _write_nii(d / "ct.nii", ct)
        _write_nii(d / "mr.nii", mr)
        _write_nii(d / "mask.nii", mask)
    return brain


def _write_fake_brats_small_brain_in_large_frame(root, patient_id="BraTS20_Training_777"):
    d = root / patient_id
    d.mkdir(parents=True)
    full_shape = (24, 28, 32)  # (z, y, x)
    t1 = np.zeros(full_shape, dtype=np.float32)
    brain_slice = (slice(6, 12), slice(8, 14), slice(10, 18))
    rng = np.random.default_rng(3)
    t1[brain_slice] = np.abs(rng.normal(400, 80, size=(6, 6, 8))).astype(np.float32)

    seg = np.zeros(full_shape, dtype=np.uint8)
    seg[7:10, 9:12, 12:16] = 1

    _write_nii(d / f"{patient_id}_t1.nii", t1)
    _write_nii(d / f"{patient_id}_seg.nii", seg)
    return full_shape, seg


def _base_stage1_regression_config(tmp_path, synthrad_root, patch_size):
    return {
        "seed": 0,
        "data": {
            "synthrad_root": str(synthrad_root), "region": None,
            "target_spacing": [1.0, 1.0, 1.0], "ct_clip_range": [-1000.0, 3000.0],
            "match_brats_domain": True, "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": patch_size, "max_patients": None, "train_val_split": 0.75,
            "num_workers": 0, "cache_dir": None,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {
            "batch_size": 1, "lr": 0.0005, "weight_decay": 0.0, "lr_schedule": "cosine",
            "warmup_steps": 1, "total_steps": 3, "amp": False, "grad_clip_norm": 1.0,
            "ema_decay": 0.9, "log_interval": 1, "val_interval": 3, "val_max_patients": 5,
            "checkpoint_interval": 3, "keep_last_n_checkpoints": 3,
            "log_file": str(tmp_path / "logs" / "regression_log.csv"),
        },
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }


@pytest.mark.slow
def test_regression_stage2_synthetic_ct_lands_correctly_in_full_brats_grid(tmp_path):
    repo_root = str(Path(__file__).resolve().parents[1])

    synthrad_root = _write_fake_synthrad(tmp_path / "fake_synthrad", ["P001", "P002", "P003", "P004"])
    patch_size = [8, 8, 8]  # deliberately smaller than the BraTS frame below, forces multiple sliding-window tiles
    stage1_config = _base_stage1_regression_config(tmp_path, synthrad_root, patch_size)
    stage1_cfg_path = tmp_path / "stage1_regression_config.yaml"
    with open(stage1_cfg_path, "w") as f:
        yaml.safe_dump(stage1_config, f)

    train_result = subprocess.run(
        [sys.executable, "-m", "training.train_stage1_regression", "--config", str(stage1_cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert train_result.returncode == 0, train_result.stderr

    brats_root = tmp_path / "brats"
    full_shape, seg = _write_fake_brats_small_brain_in_large_frame(brats_root)

    stage2_config = {
        "stage1_config": str(stage1_cfg_path),
        "brats_root": str(brats_root),
        "output_dir": str(tmp_path / "stage2_out"),
        "checkpoint_dir": None, "stride_ratio": 0.5, "use_ema": False,
        "overwrite": False, "limit": None,
    }
    stage2_cfg_path = tmp_path / "stage2_regression_config.yaml"
    with open(stage2_cfg_path, "w") as f:
        yaml.safe_dump(stage2_config, f)

    infer_result = subprocess.run(
        [sys.executable, "-m", "inference.run_stage2_brats_regression", "--config", str(stage2_cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert infer_result.returncode == 0, infer_result.stderr

    out_dir = tmp_path / "stage2_out" / "BraTS20_Training_777"
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(out_dir / "synthetic_ct.nii.gz")))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(out_dir / "tumor_mask.nii.gz")))

    assert ct.shape == full_shape
    assert mask.shape == full_shape

    # Far corner, well outside the brain region + crop margin, must be exactly background --
    # confirms place_in_full_canvas and the sliding-window tiling didn't leak content outside
    # the crop box.
    assert np.all(ct[0:3, 0:3, 0:3] == -1000)

    # Tumor mask must land at exactly its original coordinates.
    assert mask.sum() == seg.sum()
    assert np.array_equal(mask, seg)

    # metadata.json/README.md/manifest.csv are all written, same as the diffusion script.
    assert (tmp_path / "stage2_out" / "metadata.json").exists()
    assert (tmp_path / "stage2_out" / "README.md").exists()
    assert (tmp_path / "stage2_out" / "manifest.csv").exists()
