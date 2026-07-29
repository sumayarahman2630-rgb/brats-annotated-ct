"""Tests for analysis/build_exclude_list.py: aggregating the raw
per-step patient loss log into per-patient tail-means, merging that with
the mask-quality audit, the intersection/union exclusion strategies, the
--max_exclude_frac safety cap, and that the exclude list this script
writes is actually readable by data/loaders_synthetic_ct.py's
load_exclude_list -- the whole point of the file format matching. CPU-only.
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.build_exclude_list import (
    aggregate_patient_loss,
    build_report,
    read_mask_quality_csv,
    robust_z,
    select_exclude_ids,
    write_exclude_list,
)
from data.loaders_synthetic_ct import load_exclude_list


def _write_patient_loss_csv(path, rows):
    """rows: list of (step, patient_id, loss)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "patient_id", "loss"])
        for step, pid, loss in rows:
            writer.writerow([step, pid, loss])


def _write_mask_quality_csv(path, rows):
    """rows: list of dicts with at least patient_id, is_outlier, flags."""
    fields = ["patient_id", "is_outlier", "flags"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def test_aggregate_patient_loss_computes_full_run_and_tail_means(tmp_path):
    path = tmp_path / "patient_loss.csv"
    # P001: loss drops from 1.0 to 0.2 over steps 1-100 (a patient the model learns).
    # P002: loss stays high (0.9) throughout, even in the tail -- the pattern of interest.
    rows = []
    for step in range(1, 101):
        rows.append((step, "P001", 1.0 - 0.8 * (step / 100)))
        rows.append((step, "P002", 0.9))
    _write_patient_loss_csv(path, rows)

    stats = aggregate_patient_loss(str(path), tail_fraction=0.25)
    assert stats["P001"]["loss_sample_count"] == 100
    # tail = steps > 75, i.e. steps 76-100 -- P001's tail mean should be much lower than its full-run mean
    assert stats["P001"]["loss_mean_tail"] < stats["P001"]["loss_mean_full_run"]
    assert stats["P002"]["loss_mean_tail"] == pytest.approx(0.9, abs=1e-6)
    assert stats["P002"]["loss_mean_tail"] == pytest.approx(stats["P002"]["loss_mean_full_run"], abs=1e-6)


def test_aggregate_patient_loss_uses_a_shared_step_cutoff_not_per_patient(tmp_path):
    """A patient only ever sampled early in training must NOT get to treat
    its own (early, high) losses as its 'tail' just because it wasn't
    sampled later -- the cutoff is a fraction of the GLOBAL max step
    across every patient, not each patient's own last step."""
    path = tmp_path / "patient_loss.csv"
    rows = [(step, "P_early_only", 0.99) for step in range(1, 11)]  # only sampled steps 1-10
    rows += [(step, "P_full_run", 1.0 - 0.9 * (step / 1000)) for step in range(1, 1001)]
    _write_patient_loss_csv(path, rows)

    stats = aggregate_patient_loss(str(path), tail_fraction=0.25)
    # global max step is 1000, so tail cutoff is step > 750 -- P_early_only has NO
    # rows past step 10, so its tail falls back to its full-run mean (still ~0.99).
    assert stats["P_early_only"]["loss_mean_tail"] == pytest.approx(0.99, abs=1e-6)
    # tail = steps 751-1000, where the linear ramp from 1.0 down to 0.1 sits well
    # below its full-run mean (~0.55) -- confirms the tail window is doing its job
    # without pinning to an exact value the linear ramp's average happens to land on.
    assert stats["P_full_run"]["loss_mean_tail"] < 0.25
    assert stats["P_full_run"]["loss_mean_tail"] < stats["P_full_run"]["loss_mean_full_run"] - 0.2


def test_robust_z_matches_manual_median_mad():
    values = np.array([1.0, 1.1, 0.9, 1.05, 50.0])
    z = robust_z(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    expected = (values - median) / (mad * 1.4826)
    assert np.allclose(z, expected)


def test_build_report_flags_loss_outlier_only_for_patients_with_data(tmp_path):
    mask_rows = {
        "P001": {"patient_id": "P001", "is_outlier": "False", "flags": ""},
        "P002": {"patient_id": "P002", "is_outlier": "False", "flags": ""},
        "P003": {"patient_id": "P003", "is_outlier": "True", "flags": "volume_outlier"},
        "P_never_sampled": {"patient_id": "P_never_sampled", "is_outlier": "False", "flags": ""},
    }
    loss_stats = {
        "P001": {"loss_mean_full_run": 0.3, "loss_mean_tail": 0.3, "loss_sample_count": 50},
        "P002": {"loss_mean_full_run": 0.28, "loss_mean_tail": 0.28, "loss_sample_count": 50},
        "P003": {"loss_mean_full_run": 5.0, "loss_mean_tail": 5.0, "loss_sample_count": 50},  # way higher, should trigger loss_outlier
    }
    report = build_report(mask_rows, loss_stats, loss_z_threshold=1.5)
    by_id = {r["patient_id"]: r for r in report}

    assert by_id["P003"]["loss_outlier"]
    assert by_id["P003"]["mask_outlier"]
    assert not by_id["P001"]["loss_outlier"]
    # never appeared in the training run's loss log -- must not be flagged on loss grounds
    assert not by_id["P_never_sampled"]["loss_outlier"]
    assert by_id["P_never_sampled"]["loss_sample_count"] == 0


def test_select_exclude_ids_intersection_requires_both_signals():
    report = [
        {"patient_id": "mask_only", "mask_outlier": True, "loss_outlier": False, "z_loss_tail": 0.5},
        {"patient_id": "loss_only", "mask_outlier": False, "loss_outlier": True, "z_loss_tail": 4.0},
        {"patient_id": "both", "mask_outlier": True, "loss_outlier": True, "z_loss_tail": 5.0},
        {"patient_id": "neither", "mask_outlier": False, "loss_outlier": False, "z_loss_tail": 0.1},
    ]
    intersection = select_exclude_ids(report, strategy="intersection", max_exclude_frac=1.0)
    assert intersection == ["both"]

    union = select_exclude_ids(report, strategy="union", max_exclude_frac=1.0)
    assert union == ["both", "loss_only", "mask_only"]


def test_select_exclude_ids_respects_max_exclude_frac_cap_by_severity():
    """5 patients all flagged by union, but the cap only allows 2 (0.4 * 5)
    -- must keep the 2 most severe, not an arbitrary 2."""
    report = [
        {"patient_id": f"P{i}", "mask_outlier": True, "loss_outlier": True, "z_loss_tail": float(i)}
        for i in range(5)
    ]
    selected = select_exclude_ids(report, strategy="union", max_exclude_frac=0.4)
    assert len(selected) == 2
    assert set(selected) == {"P3", "P4"}  # highest z_loss_tail values


def test_select_exclude_ids_frac_cap_does_not_round_down_to_zero_on_small_populations():
    """Real bug found while testing this script: 5% of a 9-patient population
    rounds to 0, which would silently protect a genuinely flagged patient
    from ever being excluded. As long as max_exclude_frac > 0, at least one
    slot must be available if a candidate exists."""
    report = [
        {"patient_id": f"normal_{i}", "mask_outlier": False, "loss_outlier": False, "z_loss_tail": 0.1}
        for i in range(8)
    ] + [{"patient_id": "bad_patient", "mask_outlier": True, "loss_outlier": True, "z_loss_tail": 6.0}]

    selected = select_exclude_ids(report, strategy="intersection", max_exclude_frac=0.05)
    assert selected == ["bad_patient"]


def test_select_exclude_ids_frac_zero_truly_excludes_nobody():
    """The explicit escape hatch: max_exclude_frac=0.0 must still mean
    'exclude nobody', not get bumped up to 1 by the small-population fix
    above -- that fix only applies when the user asked for a nonzero cap."""
    report = [{"patient_id": "bad_patient", "mask_outlier": True, "loss_outlier": True, "z_loss_tail": 6.0}]
    selected = select_exclude_ids(report, strategy="intersection", max_exclude_frac=0.0)
    assert selected == []


def test_write_exclude_list_output_is_readable_by_load_exclude_list(tmp_path):
    """The whole point of the file format: what this script writes must be
    exactly what data/loaders_synthetic_ct.py's load_exclude_list expects."""
    path = tmp_path / "exclude.txt"
    write_exclude_list(["P003", "P007", "P012"], str(path))
    parsed = load_exclude_list(str(path))
    assert parsed == {"P003", "P007", "P012"}


@pytest.mark.slow
def test_cli_end_to_end_produces_readable_exclude_list(tmp_path):
    mask_csv = tmp_path / "mask_quality.csv"
    _write_mask_quality_csv(mask_csv, [
        {"patient_id": f"normal_{i}", "is_outlier": "False", "flags": ""} for i in range(8)
    ] + [
        {"patient_id": "bad_patient", "is_outlier": "True", "flags": "volume_outlier;fragmented"},
    ])

    loss_csv = tmp_path / "patient_loss.csv"
    rows = []
    for step in range(1, 41):
        for i in range(8):
            rows.append((step, f"normal_{i}", 0.3))
        rows.append((step, "bad_patient", 5.0))  # persistently much higher loss
    _write_patient_loss_csv(loss_csv, rows)

    output_exclude_list = tmp_path / "exclude.txt"
    repo_root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable, "-m", "analysis.build_exclude_list",
            "--mask_quality_csv", str(mask_csv), "--patient_loss_csv", str(loss_csv),
            "--output_exclude_list", str(output_exclude_list),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert output_exclude_list.exists()

    parsed = load_exclude_list(str(output_exclude_list))
    assert parsed == {"bad_patient"}
