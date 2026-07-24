"""CPU-only end-to-end test for inference/generate_full_report.py: fake
synthetic CT patients, fake Jordan DICOM slices, a fake training log CSV,
and a freshly-initialized (untrained) checkpoint, confirming the whole
report pipeline runs without error and saves every expected output file.
Mirrors test_visualize_predictions.py's and
test_validate_synthetic_segmentation.py's fixture patterns.
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
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from models.unet3d_segmentation import build_segmentation_model


def _write_fake_synthetic_ct_patient(root, patient_id, shape=(24, 24, 24)):
    rng = np.random.default_rng(0)
    d = root / patient_id
    d.mkdir(parents=True)
    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[6:18, 6:18, 6:18] = rng.normal(40, 100, size=(12, 12, 12)).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 10:14, 10:14] = 1
    sitk.WriteImage(sitk.GetImageFromArray(ct), str(d / "synthetic_ct.nii"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(d / "tumor_mask.nii"))


def _write_rgb_dicom(path, pixel_array):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = pixel_array.shape[0], pixel_array.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), enforce_file_format=True)


def _write_fake_jordan(tmp_path, n_slices=3):
    ct_root = tmp_path / "jordan_ct"
    mask_root = tmp_path / "jordan_mask"
    ct_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for slice_num in range(1, n_slices + 1):
        ct_arr = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
        mask_arr = np.zeros((24, 24, 3), dtype=np.uint8)
        mask_arr[8:16, 8:16, :] = 255
        _write_rgb_dicom(ct_root / f"P001_CT_s{slice_num}.dcm", ct_arr)
        _write_rgb_dicom(mask_root / f"P001_CT_m{slice_num}.dcm", mask_arr)
    return str(ct_root), str(mask_root)


def _write_fake_training_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "split", "loss", "dice_score", "lr", "elapsed_sec"])
        for step in range(0, 101, 25):
            writer.writerow([step, "train", 0.5 - step * 0.001, 0.1 + step * 0.002, 0.0002, float(step)])
        writer.writerow([100, "val", 0.4, 0.2, 0.0002, 100.0])


def test_generate_full_report_end_to_end_saves_every_expected_output(tmp_path):
    synthetic_root = tmp_path / "fake_synthetic_ct"
    for pid in ["P000", "P001", "P002", "P003"]:
        _write_fake_synthetic_ct_patient(synthetic_root, pid)
    jordan_ct_root, jordan_mask_root = _write_fake_jordan(tmp_path, n_slices=3)
    log_file = str(tmp_path / "logs" / "stage3_log.csv")
    _write_fake_training_log(log_file)

    config = {
        "seed": 0,
        "data": {
            "synthetic_ct_root": str(synthetic_root),
            "jordan_ct_root": jordan_ct_root, "jordan_mask_root": jordan_mask_root,
            "ct_clip_range": [-1000.0, 3000.0], "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": [8, 8, 8], "foreground_prob": 0.5, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {"batch_size": 1, "log_file": log_file},
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

    output_dir = tmp_path / "report"
    repo_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [
            sys.executable, "-m", "inference.generate_full_report",
            "--config", str(cfg_path), "--output_dir", str(output_dir),
            "--auto_threshold", "--use_largest_component", "--num_visualizations", "2",
            "--comparison_label", "Fake et al. (2024)", "--comparison_internal_dice", "0.7", "--comparison_external_dice", "0.6",
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Done." in result.stderr

    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "internal_synthetic_metrics.csv").exists()
    assert (output_dir / "external_jordan_metrics.csv").exists()
    assert (output_dir / "comparison_chart.png").exists()
    assert len(list((output_dir / "examples_synthetic").glob("*.png"))) >= 1
    assert len(list((output_dir / "examples_jordan").glob("*.png"))) >= 1

    with open(output_dir / "internal_synthetic_metrics.csv") as f:
        internal_rows = list(csv.DictReader(f))
    assert len(internal_rows) == 1  # 75/25 split over 4 patients -- 1 val patient

    with open(output_dir / "external_jordan_metrics.csv") as f:
        external_rows = list(csv.DictReader(f))
    assert len(external_rows) == 3  # 3 fake Jordan slices


def test_generate_full_report_skips_comparison_chart_without_comparison_args(tmp_path):
    synthetic_root = tmp_path / "fake_synthetic_ct"
    for pid in ["P000", "P001", "P002", "P003"]:
        _write_fake_synthetic_ct_patient(synthetic_root, pid)
    jordan_ct_root, jordan_mask_root = _write_fake_jordan(tmp_path, n_slices=2)
    log_file = str(tmp_path / "logs" / "stage3_log.csv")
    _write_fake_training_log(log_file)

    config = {
        "seed": 0,
        "data": {
            "synthetic_ct_root": str(synthetic_root),
            "jordan_ct_root": jordan_ct_root, "jordan_mask_root": jordan_mask_root,
            "ct_clip_range": [-1000.0, 3000.0], "crop_margin": 2, "spatial_multiple": 4,
            "patch_size": [8, 8, 8], "foreground_prob": 0.5, "max_patients": None,
            "train_val_split": 0.75, "num_workers": 0,
        },
        "model": {"base_channels": 4, "channel_mult": [1, 2], "num_groups": 2},
        "training": {"batch_size": 1, "log_file": log_file},
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

    output_dir = tmp_path / "report"
    repo_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-m", "inference.generate_full_report", "--config", str(cfg_path), "--output_dir", str(output_dir)],
        cwd=repo_root, capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Skipping the comparison chart" in result.stderr
    assert not (output_dir / "comparison_chart.png").exists()
