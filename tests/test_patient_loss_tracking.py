"""Tests for Stage 3's per-patient loss tracking: per_sample_loss must
exactly reproduce combined_loss's batch-mean for every loss_type (the
correctness property the whole feature depends on), the CSV log I/O
helpers must behave like the main training log's (append, don't truncate
on resume), and a real (tiny, CPU) training run must actually produce a
patient_loss_log.csv with plausible content. CPU-only.
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import yaml

from training.train_stage3_segmentation import (
    append_patient_loss_rows,
    combined_loss,
    init_patient_loss_log,
    per_sample_loss,
)


@pytest.mark.parametrize("loss_type,kwargs", [
    ("dice_bce", {"bce_weight": 1.0}),
    ("tversky", {"bce_weight": 1.0, "tversky_alpha": 0.3, "tversky_beta": 0.7}),
    ("focal_tversky", {"bce_weight": 1.0, "tversky_alpha": 0.3, "tversky_beta": 0.7}),
    ("focal", {"bce_weight": 1.0, "focal_gamma": 2.0, "focal_alpha": 0.25}),
])
def test_per_sample_loss_mean_matches_combined_loss_for_equal_shaped_batch(loss_type, kwargs):
    """The property the whole per-patient log depends on: with every sample
    in the batch sharing the same shape (always true for real training
    patches), per_sample_loss(...).mean() must equal combined_loss(...)
    exactly, for every loss_type -- otherwise the logged per-patient
    numbers wouldn't actually correspond to what training optimized."""
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 8, 8, 8)
    target = (torch.rand(4, 1, 8, 8, 8) > 0.7).float()

    batch_loss = combined_loss(logits, target, loss_type=loss_type, **kwargs)
    per_sample = per_sample_loss(logits, target, loss_type=loss_type, **kwargs)

    assert per_sample.shape == (4,)
    assert torch.allclose(per_sample.mean(), batch_loss, atol=1e-5), (
        f"{loss_type}: per_sample_loss mean {per_sample.mean().item()} != combined_loss {batch_loss.item()}"
    )


def test_per_sample_loss_differentiates_easy_and_hard_samples_in_the_same_batch():
    """A perfect prediction and a completely wrong one, stacked into one
    batch, must get very different per-sample losses -- this is the whole
    point of per-sample over batch-mean: a batch-mean would hide this."""
    target = torch.zeros(2, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1.0
    perfect_logits = torch.full_like(target, 10.0) * (2 * target - 1)  # confident correct logits
    wrong_logits = -perfect_logits  # confident WRONG logits
    logits = torch.cat([perfect_logits[:1], wrong_logits[:1]], dim=0)

    losses = per_sample_loss(logits, target, loss_type="dice_bce", bce_weight=1.0)
    assert losses[0].item() < 0.1
    assert losses[1].item() > losses[0].item() + 1.0


def test_per_sample_loss_rejects_unknown_loss_type():
    logits = torch.randn(1, 1, 4, 4, 4)
    target = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="Unknown training.loss_type"):
        per_sample_loss(logits, target, loss_type="not_a_real_loss_type", bce_weight=1.0)


def test_patient_loss_log_init_and_append(tmp_path):
    path = tmp_path / "patient_loss_log.csv"
    init_patient_loss_log(str(path), resuming=False)
    append_patient_loss_rows(str(path), [(1, "P001", 0.5), (1, "P002", 0.7)])
    append_patient_loss_rows(str(path), [(2, "P001", 0.4)])

    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "patient_id", "loss"]
    assert rows[1] == ["1", "P001", "0.500000"]
    assert rows[2] == ["1", "P002", "0.700000"]
    assert rows[3] == ["2", "P001", "0.400000"]


def test_patient_loss_log_resume_appends_without_truncating(tmp_path):
    path = tmp_path / "patient_loss_log.csv"
    init_patient_loss_log(str(path), resuming=False)
    append_patient_loss_rows(str(path), [(1, "P001", 0.5)])

    # Simulate a resumed run: init again with resuming=True must NOT wipe the existing rows.
    init_patient_loss_log(str(path), resuming=True)
    append_patient_loss_rows(str(path), [(2, "P001", 0.3)])

    with open(path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # header + 2 data rows
    assert rows[1][0] == "1"
    assert rows[2][0] == "2"


def test_append_patient_loss_rows_is_a_noop_on_empty_list(tmp_path):
    """The training loop calls this every log_interval regardless of
    whether any rows accumulated -- must not crash or create garbage on an
    empty buffer (e.g. every step in the interval was skipped for a
    non-finite loss)."""
    path = tmp_path / "patient_loss_log.csv"
    init_patient_loss_log(str(path), resuming=False)
    append_patient_loss_rows(str(path), [])
    with open(path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # header only


def _write_fake_synthetic_ct(root, patient_ids, shape=(24, 24, 24)):
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = root / pid
        d.mkdir(parents=True)
        ct = np.full(shape, -1000.0, dtype=np.float32)
        ct[6:18, 6:18, 6:18] = rng.normal(40, 100, size=(12, 12, 12)).astype(np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[10:14, 10:14, 10:14] = 1
        sitk.WriteImage(sitk.GetImageFromArray(ct), str(d / "synthetic_ct.nii"))
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(d / "tumor_mask.nii"))
    return root


@pytest.mark.slow
def test_real_training_run_produces_a_plausible_patient_loss_log(tmp_path):
    """End-to-end: a real (tiny) training subprocess must write
    patient_loss_log.csv with rows for the actual training patients,
    covering the actual step range, with finite loss values."""
    root = tmp_path / "fake_synthetic_ct"
    patient_ids = [f"P{i:03d}" for i in range(4)]
    _write_fake_synthetic_ct(root, patient_ids)

    config = {
        "seed": 0,
        "data": {
            "synthetic_ct_root": str(root),
            "jordan_ct_root": str(tmp_path / "unused_ct"), "jordan_mask_root": str(tmp_path / "unused_mask"),
            "ct_clip_range": [-1000.0, 3000.0], "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": [8, 8, 8], "foreground_prob": 0.5, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {
            "batch_size": 2, "lr": 0.0005, "weight_decay": 0.0, "lr_schedule": "cosine",
            "warmup_steps": 1, "total_steps": 6, "bce_weight": 1.0, "amp": False, "grad_clip_norm": 1.0,
            "ema_decay": 0.9, "log_interval": 2, "val_interval": 6, "val_max_patients": 5,
            "checkpoint_interval": 6, "keep_last_n_checkpoints": 3,
            "log_file": str(tmp_path / "logs" / "stage3_log.csv"),
            "patient_loss_log": str(tmp_path / "logs" / "patient_loss_log.csv"),
        },
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    repo_root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "training.train_stage3_segmentation", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Training complete: 6 steps." in result.stderr

    log_path = tmp_path / "logs" / "patient_loss_log.csv"
    assert log_path.exists()
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    train_patient_ids = set(patient_ids)  # split is 75/25 of 4 patients, but every discovered id must be a valid one
    for row in rows:
        assert row["patient_id"] in train_patient_ids
        assert 1 <= int(row["step"]) <= 6
        assert np.isfinite(float(row["loss"]))
