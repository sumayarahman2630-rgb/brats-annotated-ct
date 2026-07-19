"""Tests for the Stage 3 segmentation pipeline: the synthetic-CT loader's
discovery/binarization/split, the foreground-biased patch crop, the
segmentation model's shapes, and the train -> checkpoint -> resume cycle.
Mirrors test_train_stage1_regression_resume.py's pattern. CPU-only.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml

from data.loaders_synthetic_ct import build_synthetic_ct_dataloaders, discover_synthetic_ct_patients
from data.preprocessing import foreground_biased_patch_crop
from models.unet3d_segmentation import build_segmentation_model
from training.train_stage3_segmentation import (
    combined_loss,
    dice_loss,
    focal_loss_with_logits,
    focal_tversky_loss,
    tversky_loss,
)


def _write_fake_synthetic_ct(root, patient_ids, shape=(24, 24, 24)):
    """Mimics Stage 2's output layout: <root>/<patient_id>/synthetic_ct.nii
    + tumor_mask.nii, mask using real BraTS multi-class labels (0/1/2/4)."""
    rng = np.random.default_rng(0)
    for pid in patient_ids:
        d = root / pid
        d.mkdir(parents=True)
        ct = np.full(shape, -1000.0, dtype=np.float32)
        ct[6:18, 6:18, 6:18] = rng.normal(40, 100, size=(12, 12, 12)).astype(np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[10:14, 10:14, 10:14] = 1  # NCR/NET
        mask[14:16, 14:16, 14:16] = 2  # ED
        mask[11:13, 11:13, 11:13] = 4  # ET
        sitk.WriteImage(sitk.GetImageFromArray(ct), str(d / "synthetic_ct.nii"))
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(d / "tumor_mask.nii"))
    return root


def _base_config(tmp_path, synthetic_ct_root, patch_size=None):
    return {
        "seed": 0,
        "data": {
            "synthetic_ct_root": str(synthetic_ct_root),
            "jordan_ct_root": str(tmp_path / "unused_ct"), "jordan_mask_root": str(tmp_path / "unused_mask"),
            "ct_clip_range": [-1000.0, 3000.0], "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": patch_size, "foreground_prob": 0.5, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {
            "batch_size": 1, "lr": 0.0005, "weight_decay": 0.0, "lr_schedule": "cosine",
            "warmup_steps": 1, "total_steps": 4, "bce_weight": 1.0, "amp": False, "grad_clip_norm": 1.0,
            "ema_decay": 0.9, "log_interval": 1, "val_interval": 4, "val_max_patients": 5,
            "checkpoint_interval": 4, "keep_last_n_checkpoints": 3,
            "log_file": str(tmp_path / "logs" / "stage3_log.csv"),
        },
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }


def test_discover_synthetic_ct_patients_finds_both_extensions(tmp_path):
    """discover_synthetic_ct_patients must accept bare .nii (what Stage 2's
    re-uploaded Kaggle dataset actually uses), not just .nii.gz."""
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, ["BraTS20_Training_083", "BraTS20_Training_337"])
    patients = discover_synthetic_ct_patients(str(root))
    assert {p.patient_id for p in patients} == {"BraTS20_Training_083", "BraTS20_Training_337"}
    assert all(p.ct_path.endswith("synthetic_ct.nii") for p in patients)


def test_mask_binarization_collapses_multiclass_brats_labels(tmp_path):
    """The real BraTS labels used in the fixture (0 background, 1 NCR/NET,
    2 ED, 4 ET) must collapse to exactly {0, 1} after loading -- this is
    the specific, documented requirement from PROJECT_NOTES.md's Stage 3
    section."""
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, ["P001", "P002"])
    config = _base_config(tmp_path, root, patch_size=None)
    train_loader, _val_loader = build_synthetic_ct_dataloaders(config, seed=0)
    item = train_loader.dataset[0]
    unique_values = torch.unique(item["mask"])
    assert set(unique_values.tolist()).issubset({0.0, 1.0}), f"mask has non-binary values: {unique_values}"
    assert (item["mask"] > 0).any(), "binarized mask should still have some foreground (tumor) voxels"


def test_patient_level_split_has_no_leakage(tmp_path):
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, [f"P{i:03d}" for i in range(8)])
    config = _base_config(tmp_path, root, patch_size=[8, 8, 8])
    train_loader, val_loader = build_synthetic_ct_dataloaders(config, seed=0)
    train_ids = {p.patient_id for p in train_loader.dataset.patients}
    val_ids = {p.patient_id for p in val_loader.dataset.patients}
    assert train_ids.isdisjoint(val_ids), f"patient leakage: {train_ids & val_ids}"
    assert len(train_ids) + len(val_ids) == 8
    assert train_loader.dataset.patch_size == (8, 8, 8)
    assert val_loader.dataset.patch_size is None  # full-volume val, same convention as Stage 1


def test_foreground_biased_crop_finds_small_tumor_reliably():
    """Direct test of the crop helper itself (also exercised indirectly by
    training): with foreground_prob=1.0, every crop of a small tumor in a
    large volume must actually contain foreground."""
    shape = (32, 32, 32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[14:18, 14:18, 14:18] = 1
    ct = np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(0)

    for _ in range(30):
        ct_c, mask_c = foreground_biased_patch_crop([ct, mask], mask, (8, 8, 8), rng, foreground_prob=1.0)
        assert mask_c.any(), "foreground_prob=1.0 crop should always contain the tumor"
        assert ct_c.shape == (8, 8, 8)


def test_segmentation_model_returns_logits_not_probabilities():
    """forward() must return raw logits, not sigmoid-activated probabilities
    -- F.binary_cross_entropy/BCELoss are unsafe under CUDA autocast (crashed
    on a real Kaggle GPU, 2026-07-19), and the fix requires the model to
    hand back logits so combined_loss can use binary_cross_entropy_with_logits.
    Zero-initialized out_conv means the logits should be exactly 0 (not 0.5)
    at initialization -- sigmoid(0) == 0.5 is applied by the caller, not here."""
    config = {"model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2}}
    model = build_segmentation_model(config)
    x = torch.randn(2, 1, 16, 16, 16)
    logits = model(x)
    assert logits.shape == x.shape
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6), "zero-init out_conv should give exactly logit=0"

    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()
    assert torch.allclose(probs, torch.full_like(probs, 0.5), atol=1e-6)


def test_combined_loss_matches_manual_sigmoid_bce_and_backpropagates():
    """combined_loss takes logits and must equal dice_loss(sigmoid(logits))
    + bce_weight * BCE(sigmoid(logits)) computed manually -- i.e. switching
    to binary_cross_entropy_with_logits(logits, ...) must be numerically
    equivalent to the original binary_cross_entropy(sigmoid(logits), ...)
    it replaced, not just autocast-safe. Also checks gradients flow
    through a real model's logits output, since that's what training
    actually does."""
    config = {"model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2}}
    model = build_segmentation_model(config)
    ct = torch.randn(2, 1, 16, 16, 16)
    mask = (torch.rand(2, 1, 16, 16, 16) > 0.7).float()

    logits = model(ct)
    loss = combined_loss(logits, mask, bce_weight=1.0)

    expected = dice_loss(torch.sigmoid(logits), mask) + F.binary_cross_entropy(torch.sigmoid(logits), mask)
    assert torch.allclose(loss, expected, atol=1e-5), f"combined_loss {loss.item()} != manual sigmoid+BCE {expected.item()}"

    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_tversky_loss_perfect_and_zero_overlap():
    """Perfect overlap -> loss near 0; zero overlap -> loss near 1 (same
    sanity shape as dice_loss, just via the Tversky formula)."""
    target = torch.zeros(1, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1.0
    perfect_pred = target.clone()
    wrong_pred = 1.0 - target

    assert tversky_loss(perfect_pred, target).item() < 1e-3
    assert tversky_loss(wrong_pred, target).item() > 0.9


def test_tversky_loss_penalizes_false_negatives_more_with_default_beta():
    """With beta (0.7) > alpha (0.3), a prediction that misses real tumor
    (false negative) must be penalized MORE than one with an equal-sized
    false alarm (false positive) -- this asymmetry is the entire point of
    switching to Tversky over plain Dice for this project's collapse-to-
    empty failure mode."""
    target = torch.zeros(1, 1, 10, 10, 10)
    target[:, :, 4:6, 4:6, 4:6] = 1.0  # 8 true-positive voxels available

    # Prediction A: half the true region (4 true positives, 4 false negatives, 0 false positives)
    pred_misses_half = torch.zeros_like(target)
    pred_misses_half[:, :, 4:6, 4:6, 4] = 1.0

    # Prediction B: the whole true region PLUS an equal-sized false-positive blob elsewhere
    pred_with_false_alarm = target.clone()
    pred_with_false_alarm[:, :, 0:2, 0:2, 0:2] = 1.0

    loss_fn_miss = tversky_loss(pred_misses_half, target)
    loss_false_alarm = tversky_loss(pred_with_false_alarm, target)
    assert loss_fn_miss.item() > loss_false_alarm.item(), (
        "missing half the true tumor should be penalized more than a false alarm of the same size, "
        "given tversky_beta > tversky_alpha"
    )


def test_focal_tversky_loss_perfect_and_zero_overlap():
    target = torch.zeros(1, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1.0
    perfect_pred = target.clone()
    wrong_pred = 1.0 - target

    assert focal_tversky_loss(perfect_pred, target).item() < 1e-2
    assert focal_tversky_loss(wrong_pred, target).item() > 0.5


def test_focal_loss_with_logits_matches_manual_unsafe_computation():
    """focal_loss_with_logits must be numerically equivalent to the naive
    (autocast-unsafe) `-alpha_t * (1-p)**gamma * log(p)` formulation it's
    designed to avoid computing directly -- verifies the exp(-bce) trick
    doesn't just avoid the unsafe call, it computes the same thing."""
    logits = torch.randn(2, 1, 8, 8, 8)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.8).float()
    gamma, alpha = 2.0, 0.25

    actual = focal_loss_with_logits(logits, target, gamma=gamma, alpha=alpha)

    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    bce_naive = F.binary_cross_entropy(p, target, reduction="none")
    expected = (alpha_t * (1 - p_t) ** gamma * bce_naive).mean()

    assert torch.allclose(actual, expected, atol=1e-5), f"{actual.item()} != {expected.item()}"


def test_combined_loss_dispatches_by_loss_type():
    """combined_loss must route to the right underlying loss for each
    training.loss_type value, and reject anything else with a clear error
    rather than silently falling back to a default."""
    logits = torch.randn(1, 1, 8, 8, 8)
    target = (torch.rand(1, 1, 8, 8, 8) > 0.7).float()

    dice_bce_result = combined_loss(logits, target, bce_weight=1.0, loss_type="dice_bce")
    tversky_result = combined_loss(logits, target, bce_weight=1.0, loss_type="tversky", tversky_alpha=0.3, tversky_beta=0.7)
    focal_tversky_result = combined_loss(logits, target, bce_weight=1.0, loss_type="focal_tversky", tversky_alpha=0.3, tversky_beta=0.7)
    focal_result = combined_loss(logits, target, bce_weight=1.0, loss_type="focal", focal_gamma=2.0, focal_alpha=0.25)

    for result in (dice_bce_result, tversky_result, focal_tversky_result, focal_result):
        assert torch.isfinite(result)

    with pytest.raises(ValueError, match="Unknown training.loss_type"):
        combined_loss(logits, target, bce_weight=1.0, loss_type="not_a_real_loss_type")


def test_predict_full_volume_returns_probabilities_in_unit_range():
    """predict_full_volume applies sigmoid per-patch before blending (see
    its docstring: averaging probabilities, not logits, is the
    mathematically correct choice) -- its output must stay a valid
    probability regardless of forward() itself now returning unbounded logits."""
    config = {"model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2}}
    model = build_segmentation_model(config)
    x = torch.randn(1, 1, 24, 24, 24)
    out = model.predict_full_volume(x, patch_size=(16, 16, 16))
    assert out.shape == x.shape
    assert (out >= 0).all() and (out <= 1).all(), "predict_full_volume must return probabilities in [0, 1]"


def test_predict_full_volume_rejects_non_divisible_patch_size():
    """Real bug found 2026-07-19: a patch_size not divisible by
    2**(num_levels-1) makes the skip-connection feature maps mismatch in
    shape, crashing deep inside forward() with a confusing torch.cat error.
    predict_full_volume must now reject this upfront with a clear message."""
    config = {"model": {"base_channels": 4, "channel_mult": [1, 2, 4, 8], "num_groups": 4}}
    model = build_segmentation_model(config)
    x = torch.randn(1, 1, 40, 40, 40)
    with pytest.raises(ValueError, match="divisible"):
        model.predict_full_volume(x, patch_size=(36, 36, 36))  # 36 % 8 != 0


def test_gaussian_importance_map_peaks_at_center_and_decays_to_edges():
    """Real bug found 2026-07-19: predict_full_volume's ORIGINAL uniform
    tile blending let each tile's edge imprecision smear the reconstructed
    prediction well beyond the true region (verified: a well-converged
    model dropped from patch-level dice~1.0 to full-volume dice~0.48, with
    the reconstructed bounding box more than 2x the true one). The fix is
    Gaussian (center-weighted) blending -- this test checks the importance
    map itself has the right shape: peaked at 1.0 in the center, strictly
    lower at the edges."""
    from models.unet3d_segmentation import _gaussian_importance_map

    patch_size = (16, 16, 16)
    importance = _gaussian_importance_map(patch_size)
    assert tuple(importance.shape) == patch_size
    center = tuple(s // 2 for s in patch_size)
    assert importance[center] == importance.max()
    corner = (0, 0, 0)
    assert importance[corner] < importance[center]


@pytest.mark.slow
def test_stage3_pipeline_trains_checkpoints_and_resumes(tmp_path):
    root = tmp_path / "fake_synthetic_ct"
    _write_fake_synthetic_ct(root, [f"P{i:03d}" for i in range(4)])
    config = _base_config(tmp_path, root, patch_size=None)
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    repo_root = str(Path(__file__).resolve().parents[1])
    result1 = subprocess.run(
        [sys.executable, "-m", "training.train_stage3_segmentation", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result1.returncode == 0, result1.stderr
    assert "Training complete: 4 steps." in result1.stderr

    config["training"]["total_steps"] = 6
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    result2 = subprocess.run(
        [sys.executable, "-m", "training.train_stage3_segmentation", "--config", str(cfg_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result2.returncode == 0, result2.stderr
    assert "Resumed from checkpoint" in result2.stderr
    assert "at step 4" in result2.stderr
    assert "Training complete: 6 steps." in result2.stderr
