"""Tests for inference/validate_synthetic_segmentation.py: the pure-logic
evaluate_synthetic_val helper directly, plus a CPU-only subprocess
end-to-end run (fake synthetic CT patients + a freshly-initialized,
untrained checkpoint) confirming it doesn't crash and writes the expected
per-patient CSV. Mirrors test_visualize_predictions.py's fixture pattern.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from torch.utils.data import DataLoader

from data.loaders_synthetic_ct import SyntheticCTSegDataset, SyntheticCTPatient
from inference.validate_synthetic_segmentation import evaluate_synthetic_val
from models.unet3d_segmentation import build_segmentation_model


def _write_fake_synthetic_ct_patient(root, patient_id, shape=(24, 24, 24)):
    """Mirrors test_stage3_segmentation.py's / test_visualize_predictions.py's helper."""
    rng = np.random.default_rng(0)
    d = root / patient_id
    d.mkdir(parents=True)
    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[6:18, 6:18, 6:18] = rng.normal(40, 100, size=(12, 12, 12)).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 10:14, 10:14] = 1
    sitk.WriteImage(sitk.GetImageFromArray(ct), str(d / "synthetic_ct.nii"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(d / "tumor_mask.nii"))
    return SyntheticCTPatient(patient_id=patient_id, ct_path=str(d / "synthetic_ct.nii"), mask_path=str(d / "tumor_mask.nii"))


def test_evaluate_synthetic_val_returns_one_dice_iou_row_per_patient(tmp_path):
    root = tmp_path / "fake_synthetic_ct"
    patients = [_write_fake_synthetic_ct_patient(root, pid) for pid in ["P000", "P001"]]
    val_ds = SyntheticCTSegDataset(patients, crop_margin=2, spatial_multiple=4, patch_size=None, foreground_prob=0.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_segmentation_model({"model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2}})
    model.eval()

    rows = evaluate_synthetic_val(model, torch.device("cpu"), val_loader, patch_size=(8, 8, 8), threshold=0.5)

    assert len(rows) == 2
    assert {r["patient_id"] for r in rows} == {"P000", "P001"}
    for r in rows:
        assert 0.0 <= r["dice"] <= 1.0
        assert 0.0 <= r["iou"] <= 1.0


def test_validate_synthetic_segmentation_end_to_end_writes_expected_csv(tmp_path):
    synthetic_root = tmp_path / "fake_synthetic_ct"
    for pid in ["P000", "P001", "P002", "P003"]:
        _write_fake_synthetic_ct_patient(synthetic_root, pid)

    config = {
        "seed": 0,
        "data": {
            "synthetic_ct_root": str(synthetic_root),
            "jordan_ct_root": str(tmp_path / "unused_ct"), "jordan_mask_root": str(tmp_path / "unused_mask"),
            "ct_clip_range": [-1000.0, 3000.0], "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": [8, 8, 8], "foreground_prob": 0.5, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {"batch_size": 1},
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    model = build_segmentation_model(config)
    torch.save(
        {"step": 0, "model_state": model.state_dict(), "ema_state": None, "optimizer_state": None, "scheduler_state": None, "extra": {}},
        ckpt_dir / "ckpt_step00000000.pt",
    )

    output_csv = tmp_path / "metrics.csv"
    repo_root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable, "-m", "inference.validate_synthetic_segmentation",
            "--config", str(cfg_path), "--output_csv", str(output_csv),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "mean dice=" in result.stderr

    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # 75/25 split over 4 patients -- 1 val patient
    assert set(rows[0].keys()) == {"patient_id", "dice", "iou"}
