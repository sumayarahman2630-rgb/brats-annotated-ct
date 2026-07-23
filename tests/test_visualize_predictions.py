"""Tests for inference/visualize_predictions.py: the pure-logic helpers
(_best_slice_index, _build_pseudo_volume) directly, plus a CPU-only
subprocess end-to-end run over both data sources (fake synthetic CT
patients + fake Jordan DICOM slices + a freshly-initialized, untrained
checkpoint) confirming it doesn't crash and actually saves the expected
images. Mirrors test_stage3_segmentation.py's and test_loaders_jordan_ct.py's
fixture patterns.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
import torch
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from inference.visualize_predictions import _best_slice_index, _build_pseudo_volume
from models.unet3d_segmentation import build_segmentation_model


def test_best_slice_index_picks_the_slice_with_most_tumor():
    mask = np.zeros((6, 8, 8), dtype=np.float32)
    mask[2, 0:2, 0:2] = 1.0  # 4 pixels
    mask[4, 0:4, 0:4] = 1.0  # 16 pixels -- the real best slice
    assert _best_slice_index(mask) == 4


def test_best_slice_index_falls_back_to_mid_slice_when_no_tumor_anywhere():
    mask = np.zeros((7, 8, 8), dtype=np.float32)
    assert _best_slice_index(mask) == 7 // 2


def test_build_pseudo_volume_replicates_and_pads_and_returns_correct_center_index():
    slice_2d = np.random.default_rng(0).normal(size=(20, 20)).astype(np.float32)
    padded, center_index = _build_pseudo_volume(slice_2d, depth=5, spatial_multiple=4)
    assert padded.shape[0] % 4 == 0 and padded.shape[1] % 4 == 0 and padded.shape[2] % 4 == 0
    assert padded.shape[0] >= 5
    # The center Z slice of the padded pseudo-volume must equal the original real slice
    # (up to the H/W padding it also picked up), not one of the replicated-but-different copies.
    recovered = padded[center_index][: slice_2d.shape[0], : slice_2d.shape[1]]
    np.testing.assert_array_equal(recovered, slice_2d)


def _write_fake_synthetic_ct_patient(root, patient_id, shape=(24, 24, 24)):
    """Mirrors test_stage3_segmentation.py's _write_fake_synthetic_ct helper."""
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
    """Mirrors test_loaders_jordan_ct.py's _write_rgb_dicom helper."""
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


def _write_fake_jordan(tmp_path):
    ct_root = tmp_path / "jordan_ct"
    mask_root = tmp_path / "jordan_mask"
    ct_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for pid, slice_num in [("P001", 1), ("P001", 2)]:
        ct_arr = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
        mask_arr = np.zeros((24, 24, 3), dtype=np.uint8)
        mask_arr[8:16, 8:16, :] = 255
        _write_rgb_dicom(ct_root / f"{pid}_CT_s{slice_num}.dcm", ct_arr)
        _write_rgb_dicom(mask_root / f"{pid}_CT_m{slice_num}.dcm", mask_arr)
    return str(ct_root), str(mask_root)


def test_visualize_predictions_end_to_end_saves_expected_images(tmp_path):
    synthetic_root = tmp_path / "fake_synthetic_ct"
    for pid in ["P000", "P001", "P002", "P003"]:
        _write_fake_synthetic_ct_patient(synthetic_root, pid)
    jordan_ct_root, jordan_mask_root = _write_fake_jordan(tmp_path)

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
        "training": {"batch_size": 1},
        "checkpoint": {"working_dir": str(tmp_path / "checkpoints"), "extra_resume_dirs": []},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    # A freshly-initialized (untrained) checkpoint is enough -- this test verifies the
    # visualization pipeline runs and saves images, not prediction quality.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    model = build_segmentation_model(config)
    torch.save(
        {"step": 0, "model_state": model.state_dict(), "ema_state": None, "optimizer_state": None, "scheduler_state": None, "extra": {}},
        ckpt_dir / "ckpt_step00000000.pt",
    )

    output_dir = tmp_path / "viz_out"
    repo_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [
            sys.executable, "-m", "inference.visualize_predictions",
            "--config", str(cfg_path), "--source", "both", "--num_patients", "2",
            "--output_dir", str(output_dir),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Done." in result.stderr

    synthetic_images = list((output_dir / "synthetic").glob("*.png"))
    jordan_images = list((output_dir / "jordan").glob("*.png"))
    assert len(synthetic_images) == 1  # only 1 val patient with a 75/25 split over 4 patients
    assert len(jordan_images) == 2  # 2 matched Jordan slices, num_patients=2 cap not binding
