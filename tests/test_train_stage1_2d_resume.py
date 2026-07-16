"""Integration test for the 2D pipeline's resumability, mirroring
test_checkpoint_resume.py's approach for the 3D pipeline. CPU-only.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml


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


def _base_config(tmp_path, synthrad_root):
    return {
        "seed": 0,
        "data": {
            "synthrad_root": str(synthrad_root), "region": None,
            "target_spacing": [1.0, 1.0, 1.0], "ct_clip_range": [-1000.0, 3000.0],
            "match_brats_domain": True, "crop_margin": 2, "slice_axis": 0,
            "slice_size": [16, 16], "min_foreground_fraction": 0.0, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0, "cache_dir": None,
        },
        "model": {
            "base_channels": 8, "channel_mult": [1, 2], "num_res_blocks": 1,
            "attention_resolutions": [2], "num_heads": 2, "num_groups": 4,
            "dropout": 0.0, "use_checkpoint": False,
        },
        "diffusion": {
            "timesteps": 20, "noise_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02,
            "predict": "epsilon", "ddim_steps": 3, "ddim_eta": 0.0,
        },
        "training": {
            "batch_size": 2, "grad_accum_steps": 1, "lr": 0.0002, "weight_decay": 0.0,
            "optimizer": "adamw", "lr_schedule": "cosine", "warmup_steps": 1,
            "total_steps": 4, "ema_decay": 0.9, "amp": False, "grad_clip_norm": 1.0,
            "log_interval": 1, "val_interval": 4, "val_batches": 1, "checkpoint_interval": 4,
            "keep_last_n_checkpoints": 3, "log_file": str(tmp_path / "logs" / "train_log.csv"),
        },
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }


@pytest.mark.slow
def test_2d_pipeline_trains_checkpoints_and_resumes(tmp_path):
    synthrad_root = _write_fake_synthrad(tmp_path / "fake_synthrad", ["P001", "P002", "P003", "P004"])
    config = _base_config(tmp_path, synthrad_root)
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    repo_root = str(Path(__file__).resolve().parents[1])
    result1 = subprocess.run(
        [sys.executable, "-m", "training.train_stage1_2d", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result1.returncode == 0, result1.stderr
    assert "Training complete at step 4" in result1.stderr

    config["training"]["total_steps"] = 6
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    result2 = subprocess.run(
        [sys.executable, "-m", "training.train_stage1_2d", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result2.returncode == 0, result2.stderr
    assert "Resumed from checkpoint" in result2.stderr
    assert "at step 4" in result2.stderr  # must resume from 4, not restart from 0
    assert "Training complete at step 6" in result2.stderr
