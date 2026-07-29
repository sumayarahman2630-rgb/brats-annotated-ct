"""Stage 3 -- combines the mask-quality audit and the per-patient training
loss log into an actual exclude list.

Input: analysis/analyze_mask_quality.py's output CSV (mask geometry
outliers) and training.patient_loss_log's raw (step, patient_id, loss)
rows from a completed (or in-progress) training run. Output: a plain-text
exclude list (one patient_id per line) in the exact format
data/loaders_synthetic_ct.py's load_exclude_list reads, plus a companion
CSV with the full reasoning per patient (every score, which criteria
fired) for manual review.

THE REASONING THIS SCRIPT ENCODES -- read before trusting its output
blindly:

High training loss ALONE is not, by itself, evidence of a bad patient.
It's frequently just evidence of a HARD one -- a small tumor is
inherently harder to segment under Dice-family losses than a large one,
and that difficulty is real signal the model needs to see, not noise to
remove. Excluding every high-loss patient would bias the training
distribution toward easy cases and likely hurt generalization to exactly
the kind of small/subtle tumors the Jordan external validation set
already struggles with (see PROJECT_NOTES.md's Stage 3 section).

Mask-geometry outlier status ALONE is also not, by itself, evidence of a
bad patient -- a genuinely large or bilateral tumor is a real, valid
example the model should learn from, just a rarer one.

The two signals become much stronger evidence TOGETHER: a mask that looks
geometrically implausible (relative to the rest of the population) AND
that the model consistently cannot learn from, even late in training,
is the specific pattern a bad Stage 2 generation artifact or a mispaired
CT/mask would produce. That's why --strategy intersection (requiring BOTH
signals) is the default and the recommended choice; --strategy union
(either signal alone) is available but more aggressive and more likely to
remove hard-but-valid patients.

This script never applies its own output automatically. Read the printed
summary, spot-check a handful of the flagged patients' synthetic_ct/
tumor_mask files (or run inference/visualize_predictions.py against them
if a checkpoint is available) before pointing a training config at the
exclude list it writes.

Run as (after both analyze_mask_quality.py and a training run with
patient-loss logging have produced their CSVs):
    python -m analysis.build_exclude_list \
        --mask_quality_csv /kaggle/working/analysis_plots/mask_quality.csv \
        --patient_loss_csv /kaggle/working/logs/stage3_patient_loss_log.csv \
        --output_exclude_list /kaggle/working/analysis_plots/stage3_exclude_list.txt
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict

import numpy as np

log = logging.getLogger("build_exclude_list")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPORT_FIELDS = [
    "patient_id", "mask_outlier", "mask_flags", "loss_outlier",
    "loss_mean_full_run", "loss_mean_tail", "loss_sample_count",
    "z_loss_tail", "excluded_by", "strategy_would_exclude",
]


def parse_args():
    """Paths to both input CSVs plus the knobs that control how aggressively patients get flagged."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_quality_csv", type=str, required=True, help="Output of analysis/analyze_mask_quality.py.")
    parser.add_argument("--patient_loss_csv", type=str, required=True, help="training.patient_loss_log from a Stage 3 training run.")
    parser.add_argument("--output_exclude_list", type=str, default="/kaggle/working/analysis_plots/stage3_exclude_list.txt")
    parser.add_argument("--output_report_csv", type=str, default=None, help="Defaults to output_exclude_list with a _report.csv suffix.")
    parser.add_argument("--strategy", type=str, default="intersection", choices=["intersection", "union"],
                         help="intersection (default, recommended): exclude only patients flagged by BOTH the mask audit "
                              "and the loss signal. union: exclude a patient flagged by EITHER signal -- more aggressive, "
                              "see the module docstring for why intersection is the safer default.")
    parser.add_argument("--loss_z_threshold", type=float, default=3.0, help="Robust z-score (median/MAD) beyond which a patient's tail-mean loss counts as an outlier.")
    parser.add_argument("--loss_tail_fraction", type=float, default=0.25, help="Only the last this-fraction of each patient's logged steps count toward loss_mean_tail -- early-training loss is high for everyone and would dilute the signal. 1.0 uses the full run.")
    parser.add_argument("--max_exclude_frac", type=float, default=0.05, help="Safety cap: never write out more than this fraction of the mask-quality CSV's patient count, even if more were flagged -- keeps one bad run of thresholds from gutting the training pool. The patients closest to the flagging thresholds are dropped first when the cap trims the list.")
    return parser.parse_args()


def read_mask_quality_csv(path: str) -> dict[str, dict]:
    """Returns patient_id -> row dict from analyze_mask_quality.py's CSV."""
    with open(path) as f:
        rows = {row["patient_id"]: row for row in csv.DictReader(f)}
    log.info("Read %d patients from %s", len(rows), path)
    return rows


def aggregate_patient_loss(path: str, tail_fraction: float) -> dict[str, dict]:
    """Reads the raw (step, patient_id, loss) log and computes, per
    patient, the full-run mean loss AND a tail-mean over only the last
    `tail_fraction` of steps ANY patient was logged at (a global step
    cutoff, not per-patient) -- using a shared cutoff across all patients
    is what makes the tail-mean comparable between them; a per-patient
    tail would let a patient sampled only early in training dodge the
    comparison entirely. The tail-mean is deliberately what the outlier
    z-score below is computed on: early-training loss is high and noisy
    for essentially every patient, and would swamp the real signal of
    "which patients does the FINAL, converged model still struggle with."
    """
    losses_by_patient: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            losses_by_patient[row["patient_id"]].append((int(row["step"]), float(row["loss"])))

    if not losses_by_patient:
        return {}

    max_step = max(step for rows in losses_by_patient.values() for step, _ in rows)
    tail_cutoff = max_step * (1.0 - tail_fraction)

    out = {}
    for patient_id, rows in losses_by_patient.items():
        all_losses = [loss for _step, loss in rows]
        tail_losses = [loss for step, loss in rows if step > tail_cutoff]
        out[patient_id] = {
            "loss_mean_full_run": float(np.mean(all_losses)),
            "loss_mean_tail": float(np.mean(tail_losses)) if tail_losses else float(np.mean(all_losses)),
            "loss_sample_count": len(rows),
        }
    log.info(
        "Aggregated per-patient loss for %d patients from %s (tail = steps > %.0f of max step %d)",
        len(out), path, tail_cutoff, max_step,
    )
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    """Same robust (median/MAD) z-score as analyze_mask_quality.py, duplicated
    here rather than imported -- this script combines two independently
    generated CSVs and shouldn't have a hidden runtime dependency on the
    other script's internals changing shape. See that module's robust_z
    docstring for the full reasoning, including the zero-MAD edge case."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan)
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    scaled_mad = mad * 1.4826
    if scaled_mad < 1e-9:
        out = np.where(np.isfinite(values), 0.0, np.nan)
        deviates = np.isfinite(values) & (np.abs(values - median) > 1e-9)
        return np.where(deviates, np.sign(values - median) * 1e6, out)
    return (values - median) / scaled_mad


def build_report(
    mask_rows: dict[str, dict], loss_stats: dict[str, dict], loss_z_threshold: float,
) -> list[dict]:
    """Merges the two sources on patient_id and computes the combined
    exclusion signal for every patient present in the mask-quality CSV
    (the full training-pool list) -- a patient in the loss log but absent
    from the mask CSV is a sign the two CSVs came from different
    synthetic_ct_root runs, which is flagged loudly rather than silently
    merged on a partial overlap."""
    loss_only_ids = set(loss_stats) - set(mask_rows)
    if loss_only_ids:
        log.warning(
            "%d patient(s) appear in --patient_loss_csv but not --mask_quality_csv -- the two CSVs may "
            "be from different synthetic_ct_root runs. These patients are ignored: %s",
            len(loss_only_ids), sorted(loss_only_ids)[:10],
        )

    patient_ids = sorted(mask_rows)
    tail_means = np.array([
        loss_stats[pid]["loss_mean_tail"] if pid in loss_stats else np.nan
        for pid in patient_ids
    ])
    z_loss = robust_z(tail_means)

    report = []
    for i, pid in enumerate(patient_ids):
        mask_row = mask_rows[pid]
        mask_outlier = str(mask_row.get("is_outlier", "")).strip().lower() == "true"
        has_loss_data = pid in loss_stats
        loss_outlier = has_loss_data and np.isfinite(z_loss[i]) and z_loss[i] > loss_z_threshold

        row = {
            "patient_id": pid,
            "mask_outlier": mask_outlier,
            "mask_flags": mask_row.get("flags", ""),
            "loss_outlier": loss_outlier,
            "loss_mean_full_run": loss_stats[pid]["loss_mean_full_run"] if has_loss_data else float("nan"),
            "loss_mean_tail": loss_stats[pid]["loss_mean_tail"] if has_loss_data else float("nan"),
            "loss_sample_count": loss_stats[pid]["loss_sample_count"] if has_loss_data else 0,
            "z_loss_tail": float(z_loss[i]) if np.isfinite(z_loss[i]) else float("nan"),
        }
        row["excluded_by"] = ";".join(
            flag for flag, present in [("mask", mask_outlier), ("loss", loss_outlier)] if present
        )
        report.append(row)

    n_no_loss_data = sum(1 for pid in patient_ids if pid not in loss_stats)
    if n_no_loss_data:
        log.warning(
            "%d of %d patients in the mask-quality CSV have no rows in --patient_loss_csv (never sampled "
            "during the logged training run, or a partial run) -- loss_outlier is left False for them, "
            "so --strategy intersection can never flag a patient training never actually saw.",
            n_no_loss_data, len(patient_ids),
        )
    return report


def select_exclude_ids(report: list[dict], strategy: str, max_exclude_frac: float) -> list[str]:
    """Applies --strategy, then the --max_exclude_frac safety cap. When the
    cap trims the list, patients are kept in descending order of how far
    over the flagging threshold they are (mask z-score + loss z-score,
    both clipped to 0 if not flagging in that direction) -- so if a cap
    forces a choice, the most extreme cases are excluded first."""
    if strategy == "intersection":
        candidates = [r for r in report if r["mask_outlier"] and r["loss_outlier"]]
    else:
        candidates = [r for r in report if r["mask_outlier"] or r["loss_outlier"]]

    # round() can floor a tiny-but-nonzero fraction of a small population down to
    # 0 (e.g. 5% of 9 patients rounds to 0), which would silently protect a
    # genuinely flagged patient from ever being excluded, purely as a rounding
    # artifact -- not what the cap is for. As long as max_exclude_frac itself is
    # nonzero (the user's explicit "excluded nobody" escape hatch is frac=0.0),
    # at least 1 slot is always available if there's real demand for it.
    max_n = int(round(len(report) * max_exclude_frac))
    if max_exclude_frac > 0:
        max_n = max(max_n, 1)
    if len(candidates) > max_n:
        def severity(r):
            loss_z = r["z_loss_tail"] if isinstance(r["z_loss_tail"], (int, float)) and np.isfinite(r["z_loss_tail"]) else 0.0
            return max(loss_z, 0.0) + (1.0 if r["mask_outlier"] else 0.0)
        candidates = sorted(candidates, key=severity, reverse=True)[:max_n]
        log.warning(
            "--strategy %s flagged more patients than --max_exclude_frac %.3f allows (%d > %d) -- "
            "keeping only the %d most extreme by combined severity. Consider whether the thresholds "
            "are too loose before trusting this cap silently.",
            strategy, max_exclude_frac, len(candidates), max_n, max_n,
        )
    return sorted(r["patient_id"] for r in candidates)


def write_exclude_list(patient_ids: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Generated by analysis/build_exclude_list.py -- one patient_id per line.\n")
        f.write("# Read data/loaders_synthetic_ct.py's exclude_patients_file docstring before editing by hand.\n")
        for pid in patient_ids:
            f.write(f"{pid}\n")
    log.info("Wrote %d patient IDs to %s", len(patient_ids), path)


def write_report_csv(report: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in report:
            writer.writerow({k: row.get(k, "") for k in REPORT_FIELDS})
    log.info("Wrote full reasoning for %d patients to %s", len(report), path)


def main():
    args = parse_args()
    output_report_csv = args.output_report_csv or (os.path.splitext(args.output_exclude_list)[0] + "_report.csv")

    mask_rows = read_mask_quality_csv(args.mask_quality_csv)
    loss_stats = aggregate_patient_loss(args.patient_loss_csv, args.loss_tail_fraction)
    if not mask_rows:
        raise SystemExit(f"No rows in {args.mask_quality_csv!r} -- run analysis.analyze_mask_quality first.")
    if not loss_stats:
        raise SystemExit(f"No rows in {args.patient_loss_csv!r} -- this needs a real training run with per-patient loss logging.")

    report = build_report(mask_rows, loss_stats, args.loss_z_threshold)
    for r in report:
        r["strategy_would_exclude"] = (r["mask_outlier"] and r["loss_outlier"]) if args.strategy == "intersection" else (r["mask_outlier"] or r["loss_outlier"])

    exclude_ids = select_exclude_ids(report, args.strategy, args.max_exclude_frac)
    write_report_csv(report, output_report_csv)
    write_exclude_list(exclude_ids, args.output_exclude_list)

    n_mask_only = sum(1 for r in report if r["mask_outlier"] and not r["loss_outlier"])
    n_loss_only = sum(1 for r in report if r["loss_outlier"] and not r["mask_outlier"])
    n_both = sum(1 for r in report if r["mask_outlier"] and r["loss_outlier"])
    print(f"\n{len(report)} patients compared. mask-only outliers: {n_mask_only}, loss-only outliers: {n_loss_only}, both (intersection): {n_both}.")
    print(f"--strategy {args.strategy} selected {len(exclude_ids)} patient(s) for exclusion (cap: {args.max_exclude_frac:.1%} of {len(report)} = {int(round(len(report) * args.max_exclude_frac))}).")
    if exclude_ids:
        print("\nExcluded patients:")
        for r in sorted((r for r in report if r["patient_id"] in exclude_ids), key=lambda r: r["patient_id"]):
            print(f"  {r['patient_id']:<28} mask_flags={r['mask_flags']!r:<40} loss_mean_tail={r['loss_mean_tail']:.4f} z_loss={r['z_loss_tail']:.2f}")
        print(
            "\nReview these before retraining -- spot-check a few in "
            f"{output_report_csv} and, if a checkpoint is available, visually via "
            "inference/visualize_predictions.py. This script only proposes an exclude list; "
            "nothing is applied automatically."
        )
    else:
        print("\nNo patients met the exclusion criteria -- nothing to review, no retrain needed on this basis.")


if __name__ == "__main__":
    main()
