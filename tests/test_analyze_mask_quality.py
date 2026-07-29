"""Tests for the Stage 3 mask-quality audit script: per-patient geometry
metrics (volume, fragmentation, sphericity, elongation, centroid), the
robust-statistics helpers, and that a genuinely planted outlier gets
flagged while a population of ordinary masks does not. CPU-only, no GPU
or real Stage 2 data needed -- everything here builds tiny synthetic mask
fixtures directly.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from analysis.analyze_mask_quality import (
    add_population_stats,
    compute_mask_metrics,
    percentile_rank,
    robust_z,
)


def _write_mask(path, shape=(40, 40, 40), voxel_boxes=()):
    """Writes a binary mask NIfTI with 1s at every (z0:z1, y0:y1, x0:x1) box
    in voxel_boxes -- lets a test build a compact single blob, several
    scattered blobs, or an empty mask by passing zero boxes."""
    arr = np.zeros(shape, dtype=np.uint8)
    for (z0, z1, y0, y1, x0, x1) in voxel_boxes:
        arr[z0:z1, y0:y1, x0:x1] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path))
    return path


def test_empty_mask_reports_empty_status_and_zero_volume(tmp_path):
    path = _write_mask(tmp_path / "empty.nii", voxel_boxes=[])
    metrics = compute_mask_metrics(str(path))
    assert metrics["status"] == "empty_mask"
    assert metrics["voxel_count"] == 0
    assert metrics["volume_mm3"] == 0.0


def test_compact_single_blob_has_one_component_and_reasonable_sphericity(tmp_path):
    path = _write_mask(tmp_path / "compact.nii", voxel_boxes=[(15, 25, 15, 25, 15, 25)])  # solid 10^3 cube
    metrics = compute_mask_metrics(str(path))
    assert metrics["status"] == "ok"
    assert metrics["voxel_count"] == 10 ** 3
    assert metrics["num_components"] == 1
    assert metrics["largest_component_ratio"] == pytest.approx(1.0)
    assert metrics["fill_ratio"] == pytest.approx(1.0)  # solid cube fills its own bounding box exactly
    assert 0.5 < metrics["sphericity"] <= 1.0  # a cube isn't a sphere, but should still score reasonably compact


def test_scattered_blobs_have_more_components_and_lower_sphericity_than_compact(tmp_path):
    """Same total voxel count, split into several small separated blobs
    instead of one solid cube -- fragmentation and sphericity should both
    reflect that difference, independent of volume."""
    compact_path = _write_mask(tmp_path / "compact.nii", voxel_boxes=[(10, 18, 10, 18, 10, 18)])  # 8^3 = 512 voxels
    scattered_boxes = [
        (5, 9, 5, 9, 5, 9), (20, 24, 20, 24, 20, 24),
        (5, 9, 20, 24, 5, 9), (20, 24, 5, 9, 20, 24),
    ]  # four separated 4^3=64-voxel blobs = 256 voxels total, spread far apart
    scattered_path = _write_mask(tmp_path / "scattered.nii", voxel_boxes=scattered_boxes)

    compact = compute_mask_metrics(str(compact_path))
    scattered = compute_mask_metrics(str(scattered_path))

    assert compact["num_components"] == 1
    assert scattered["num_components"] == 4
    assert scattered["sphericity"] < compact["sphericity"]
    assert scattered["largest_component_ratio"] < 0.5  # no single blob dominates


def test_elongated_blob_has_higher_elongation_ratio_than_cube(tmp_path):
    cube_path = _write_mask(tmp_path / "cube.nii", voxel_boxes=[(15, 25, 15, 25, 15, 25)])
    rod_path = _write_mask(tmp_path / "rod.nii", voxel_boxes=[(2, 38, 19, 21, 19, 21)])  # long thin rod along z

    cube = compute_mask_metrics(str(cube_path))
    rod = compute_mask_metrics(str(rod_path))
    assert rod["elongation_ratio"] > cube["elongation_ratio"]


def test_robust_z_matches_manual_median_mad_and_preserves_nan():
    values = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 100.0])  # 100.0 is the deliberate outlier
    z = robust_z(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    expected = (values - median) / (mad * 1.4826)
    assert np.allclose(z, expected)
    assert abs(z[-1]) > 3.0  # the planted outlier should read as extreme

    with_nan = np.array([1.0, 2.0, np.nan, 3.0])
    z_nan = robust_z(with_nan)
    assert np.isnan(z_nan[2])
    assert np.isfinite(z_nan[0])


def test_robust_z_is_zero_when_all_finite_values_identical():
    """MAD is 0 if every value matches -- must not divide by zero."""
    z = robust_z(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(z, 0.0)


def test_percentile_rank_orders_values_correctly():
    values = np.array([30.0, 10.0, 20.0])
    ranks = percentile_rank(values)
    assert ranks[1] < ranks[2] < ranks[0]
    assert ranks[1] == pytest.approx(0.0)
    assert ranks[0] == pytest.approx(100.0)


def test_add_population_stats_flags_planted_outliers_not_the_normal_population(tmp_path):
    """Builds a population of ordinary, similarly-sized/located masks plus
    a few deliberately extreme ones (empty, huge, scattered), and checks
    the flagging lands on the right patients -- the actual point of the
    audit script."""
    rng = np.random.default_rng(0)
    rows = []

    # 10 "normal" patients: solid cubes of similar size, roughly the same location
    for i in range(10):
        size = 8 + int(rng.integers(-1, 2))  # 7-9 voxel cube
        offset = 16 + int(rng.integers(-1, 2))
        path = _write_mask(tmp_path / f"normal_{i}.nii", voxel_boxes=[(offset, offset + size,) * 3])
        metrics = compute_mask_metrics(str(path))
        metrics["patient_id"] = f"normal_{i}"
        rows.append(metrics)

    empty_path = _write_mask(tmp_path / "empty_patient.nii", voxel_boxes=[])
    empty_metrics = compute_mask_metrics(str(empty_path))
    empty_metrics["patient_id"] = "empty_patient"
    rows.append(empty_metrics)

    huge_path = _write_mask(tmp_path / "huge_patient.nii", shape=(40, 40, 40), voxel_boxes=[(2, 38, 2, 38, 2, 38)])
    huge_metrics = compute_mask_metrics(str(huge_path))
    huge_metrics["patient_id"] = "huge_patient"
    rows.append(huge_metrics)

    scattered_boxes = [(z, z + 2, y, y + 2, x, x + 2) for z, y, x in [(2, 2, 2), (36, 2, 2), (2, 36, 2), (36, 36, 2), (2, 2, 36), (36, 36, 36)]]
    scattered_path = _write_mask(tmp_path / "scattered_patient.nii", voxel_boxes=scattered_boxes)
    scattered_metrics = compute_mask_metrics(str(scattered_path))
    scattered_metrics["patient_id"] = "scattered_patient"
    rows.append(scattered_metrics)

    add_population_stats(rows, z_threshold=3.0)

    by_id = {r["patient_id"]: r for r in rows}
    assert by_id["empty_patient"]["is_outlier"]
    assert by_id["empty_patient"]["flags"] == "empty_mask"
    assert by_id["huge_patient"]["is_outlier"]
    assert "volume_outlier" in by_id["huge_patient"]["flags"]
    assert by_id["scattered_patient"]["is_outlier"]
    assert "fragmented" in by_id["scattered_patient"]["flags"]

    for i in range(10):
        assert not by_id[f"normal_{i}"]["is_outlier"], f"normal_{i} should not be flagged: {by_id[f'normal_{i}']['flags']}"


def _write_fake_synthetic_ct_root(root, patient_specs):
    """patient_specs: dict of patient_id -> list of voxel_boxes for that
    patient's mask. Writes both synthetic_ct.nii (content unused by this
    script, just needs to exist alongside the mask) and tumor_mask.nii,
    matching Stage 2's real output layout."""
    for pid, boxes in patient_specs.items():
        d = root / pid
        d.mkdir(parents=True)
        ct = np.full((40, 40, 40), -1000.0, dtype=np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(ct), str(d / "synthetic_ct.nii"))
        _write_mask(d / "tumor_mask.nii", voxel_boxes=boxes)
    return root


@pytest.mark.slow
def test_cli_end_to_end_writes_csv_with_expected_outlier(tmp_path):
    root = tmp_path / "fake_synthetic_ct"
    specs = {f"normal_{i:02d}": [(15, 23, 15, 23, 15, 23)] for i in range(10)}
    specs["huge_outlier"] = [(2, 38, 2, 38, 2, 38)]
    _write_fake_synthetic_ct_root(root, specs)

    config = {"data": {"synthetic_ct_root": str(root)}}
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    output_csv = tmp_path / "mask_quality.csv"
    repo_root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "analysis.analyze_mask_quality", "--config", str(cfg_path), "--output_csv", str(output_csv)],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert output_csv.exists()

    import csv
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 11
    by_id = {r["patient_id"]: r for r in rows}
    assert by_id["huge_outlier"]["is_outlier"] == "True"
    assert "volume_outlier" in by_id["huge_outlier"]["flags"]
