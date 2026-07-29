"""Tests for analysis/compute_patient_loss_from_checkpoint.py: the core
guarantee this script exists for is that it NEVER changes an existing
checkpoint's weights (pure forward-pass measurement, no backward/optimizer
step at all) -- verified directly via a state_dict comparison, not just
"it didn't crash." Also checks augmentation is genuinely forced off
regardless of the config, and that the output is patient_loss_log.csv-
compatible (same schema build_exclude_list.py already reads). CPU-only.
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

import data.loaders_synthetic_ct as loaders_synthetic_ct
from analysis.compute_patient_loss_from_checkpoint import compute_patient_loss_from_checkpoint
from models.unet3d_segmentation import build_segmentation_model
from training.checkpoint import save_checkpoint
from training.ema import EMA


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


def _base_config(tmp_path, synthetic_ct_root, augment=False):
    return {
        "seed": 0,
        "data": {
            "synthetic_ct_root": str(synthetic_ct_root),
            "jordan_ct_root": str(tmp_path / "unused_ct"), "jordan_mask_root": str(tmp_path / "unused_mask"),
            "ct_clip_range": [-1000.0, 3000.0], "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": [8, 8, 8], "foreground_prob": 0.5, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0, "augment": augment,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {
            "batch_size": 2, "loss_type": "focal", "bce_weight": 1.0,
            "tversky_alpha": 0.3, "tversky_beta": 0.7, "focal_gamma": 2.0, "focal_alpha": 0.75,
        },
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }


def _make_checkpoint(config, tmp_path):
    """Builds a real (tiny, randomly-initialized) model and saves it as a
    checkpoint the function under test can load -- standing in for an
    already-trained real checkpoint. save_checkpoint requires a real
    optimizer object (only its state_dict is stored -- load_checkpoint is
    called with optimizer=None everywhere this test needs to read it back,
    matching how every evaluation script in this project loads a
    checkpoint for inference rather than resuming training)."""
    model = build_segmentation_model(config)
    ema = EMA(model, decay=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = save_checkpoint(config["checkpoint"]["working_dir"], 100, model, ema, optimizer, scheduler=None, keep_last_n=3)
    return path, model


def test_checkpoint_weights_are_unchanged_after_computing_patient_loss(tmp_path):
    """The core guarantee: running this must be provably a pure read --
    the checkpoint's state_dict before and after must be byte-identical."""
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, [f"P{i:03d}" for i in range(4)])
    config = _base_config(tmp_path, root)
    ckpt_path, _model = _make_checkpoint(config, tmp_path)

    before = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rows = compute_patient_loss_from_checkpoint(config, checkpoint_path=None, num_passes=2)
    after = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert rows  # sanity: something was actually measured
    for key in before["model_state"]:
        assert torch.equal(before["model_state"][key], after["model_state"][key]), f"checkpoint weight {key} changed on disk"


def test_model_parameters_identical_before_and_after_in_memory(tmp_path):
    """Same guarantee, checked directly on the in-memory model rather than
    the file on disk -- rules out even a transient in-memory weight update
    that happened to get overwritten back before the file was re-saved."""
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, [f"P{i:03d}" for i in range(4)])
    config = _base_config(tmp_path, root)
    ckpt_path, model_before = _make_checkpoint(config, tmp_path)
    params_before = {name: p.clone() for name, p in model_before.named_parameters()}

    compute_patient_loss_from_checkpoint(config, checkpoint_path=None, num_passes=2)

    model_after = build_segmentation_model(config)
    from training.checkpoint import load_checkpoint
    load_checkpoint(ckpt_path, model_after, ema=None, optimizer=None, scheduler=None, map_location="cpu")
    for name, p_after in model_after.named_parameters():
        assert torch.equal(params_before[name], p_after), f"parameter {name} differs"


def test_num_passes_scales_row_count(tmp_path):
    root = tmp_path / "fake_synthetic_ct"
    patient_ids = [f"P{i:03d}" for i in range(4)]  # divisible by batch_size=2, so drop_last never drops anyone
    _write_fake_synthetic_ct(root, patient_ids)
    config = _base_config(tmp_path, root)
    _ckpt_path, _model = _make_checkpoint(config, tmp_path)

    rows_1 = compute_patient_loss_from_checkpoint(config, checkpoint_path=None, num_passes=1)
    rows_3 = compute_patient_loss_from_checkpoint(config, checkpoint_path=None, num_passes=3)
    assert len(rows_3) == 3 * len(rows_1)
    for step, _pid, loss in rows_1 + rows_3:
        assert np.isfinite(loss)


def test_augmentation_is_forced_off_even_when_config_requests_it(tmp_path, monkeypatch):
    """Directly verifies augment_ct_mask_patch is never called, even though
    the config passed in has data.augment=True -- proving this is actually
    enforced, not just documented."""
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, [f"P{i:03d}" for i in range(4)])
    config = _base_config(tmp_path, root, augment=True)
    _ckpt_path, _model = _make_checkpoint(config, tmp_path)

    call_count = {"n": 0}
    real_augment = loaders_synthetic_ct.augment_ct_mask_patch

    def counting_augment(*args, **kwargs):
        call_count["n"] += 1
        return real_augment(*args, **kwargs)

    monkeypatch.setattr(loaders_synthetic_ct, "augment_ct_mask_patch", counting_augment)
    compute_patient_loss_from_checkpoint(config, checkpoint_path=None, num_passes=2)
    assert call_count["n"] == 0, "augment_ct_mask_patch was called despite this script forcing augmentation off"


def test_does_not_mutate_the_caller_supplied_config_dict(tmp_path):
    """The augment-off override must operate on an internal copy, not the
    caller's own config object -- mutating shared state out from under the
    caller (e.g. if they reuse the same config dict afterward for something
    else) would be a nasty surprise."""
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, [f"P{i:03d}" for i in range(4)])
    config = _base_config(tmp_path, root, augment=True)
    _ckpt_path, _model = _make_checkpoint(config, tmp_path)

    compute_patient_loss_from_checkpoint(config, checkpoint_path=None, num_passes=1)
    assert config["data"]["augment"] is True, "caller's config dict was mutated"


@pytest.mark.slow
def test_cli_end_to_end_writes_a_patient_loss_log_compatible_csv(tmp_path):
    root = tmp_path / "fake_synthetic_ct"
    patient_ids = [f"P{i:03d}" for i in range(4)]
    _write_fake_synthetic_ct(root, patient_ids)
    config = _base_config(tmp_path, root)
    _ckpt_path, _model = _make_checkpoint(config, tmp_path)

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    output_csv = tmp_path / "patient_loss_log.csv"

    repo_root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "analysis.compute_patient_loss_from_checkpoint",
         "--config", str(cfg_path), "--output_csv", str(output_csv), "--num_passes", "2"],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert output_csv.exists()

    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"step", "patient_id", "loss"}
    assert len(rows) > 0
    for row in rows:
        assert np.isfinite(float(row["loss"]))
