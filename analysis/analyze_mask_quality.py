"""Stage 3 -- tumor mask quality audit over the synthetic CT training pool.

Input: every discovered synthetic CT + tumor mask pair under
data.synthetic_ct_root (the same 368-ish patients Stage 3 training draws
from -- this reads Stage 2's raw tumor_mask.nii(.gz) directly, not through
data/loaders_synthetic_ct.py's Dataset, since we want whole-mask geometry
before any patch cropping happens). Output: one CSV row per patient with
volume, shape-compactness, fragmentation, and tumor-location metrics, plus
a robust-statistics outlier flag per patient -- this is a diagnostic tool,
not a filter; it doesn't touch training or remove anything itself. See
analysis/build_exclude_list.py for turning this (and a per-patient loss
log from training) into an actual exclude list.

What "outlier" means here, precisely: for each of four signals (log tumor
volume, shape sphericity, connected-component count, and how far the
mask's centroid sits from the population's typical centroid), this script
computes a ROBUST z-score using the median and MAD (median absolute
deviation) rather than mean/std. Mean/std is the wrong tool for outlier
detection specifically because a single extreme patient inflates the std
and can hide its own (and others') deviation -- median/MAD doesn't have
that self-defeating property. A patient is flagged if any signal's robust
z-score exceeds --z_threshold in the "bad" direction (too small/large
volume, too irregular a shape, too many separate blobs, or a centroid far
from where tumors usually sit), or if its mask binarizes to nothing at all
(always flagged, no threshold involved -- an empty mask is unusable
regardless of statistics).

Run as:
    python -m analysis.analyze_mask_quality --config configs/stage3_ct_segmentation.yaml \
        --output_csv /kaggle/working/analysis_plots/mask_quality.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import SimpleITK as sitk
import yaml
from scipy import ndimage

log = logging.getLogger("analyze_mask_quality")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CSV_FIELDS = [
    "patient_id", "status",
    "voxel_count", "volume_mm3", "num_components", "largest_component_ratio",
    "fill_ratio", "sphericity", "elongation_ratio",
    "centroid_z", "centroid_y", "centroid_x", "centroid_offset",
    "z_log_volume", "z_sphericity", "z_num_components", "z_centroid_offset",
    "pct_log_volume", "pct_sphericity",
    "flags", "is_outlier",
]


def parse_args():
    """--config resolves data.synthetic_ct_root the same way training does; the rest control the outlier thresholds and output paths."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/stage3_ct_segmentation.yaml")
    parser.add_argument("--synthetic_ct_root", type=str, default=None, help="Override data.synthetic_ct_root from --config.")
    parser.add_argument("--z_threshold", type=float, default=3.0, help="Robust z-score magnitude beyond which a patient is flagged on a given signal (default 3.0, the conventional 'three-sigma' cutoff).")
    parser.add_argument("--output_csv", type=str, default="/kaggle/working/analysis_plots/mask_quality.csv")
    return parser.parse_args()


def _discover_mask_paths(root: str) -> list[tuple[str, str]]:
    """Finds every (patient_id, tumor_mask_path) pair under root -- deliberately
    NOT importing discover_synthetic_ct_patients from data/loaders_synthetic_ct.py:
    this script only needs the mask file, and duplicating the ~10-line scan keeps
    it usable even if that loader's discovery logic changes shape later (same
    pipeline-isolation reasoning used throughout this project)."""
    import re
    from pathlib import Path

    mask_re = re.compile(r"tumor_mask\.nii(\.gz)?$", re.IGNORECASE)
    root_path = Path(root)
    if not root_path.is_dir():
        log.warning("_discover_mask_paths: %s is not a directory", root_path)
        return []

    out = []
    for folder in sorted(root_path.iterdir()):
        if not folder.is_dir():
            continue
        mask_file = next((f for f in folder.iterdir() if f.is_file() and mask_re.search(f.name)), None)
        if mask_file is not None:
            out.append((folder.name, str(mask_file)))
    log.info("_discover_mask_paths: found %d patients with a tumor_mask file under %s", len(out), root_path)
    return out


def compute_mask_metrics(mask_path: str) -> dict:
    """Loads one patient's raw tumor_mask.nii(.gz) and computes its geometry:
    volume, connected-component fragmentation, shape compactness (sphericity),
    elongation, and centroid location -- all before any cropping/patching a
    training Dataset would apply, since this is about the RAW Stage 2 output's
    quality, not what one particular training patch happened to see.

    Binarization matches data/loaders_synthetic_ct.py's SyntheticCTSegDataset
    exactly (mask > 0 collapses BraTS's 0/1/2/4 labels to binary) -- this has
    to stay consistent with what training actually trains on, or the volumes
    reported here wouldn't mean the same thing as what the model sees.
    """
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    # SimpleITK's GetSpacing() is (x, y, z); GetArrayFromImage's array axes are
    # (z, y, x) -- reversed. Every spacing-aware calc below must use spacing in
    # the array's own (z, y, x) order, or an anisotropic-spacing volume would be
    # silently wrong (Stage 2 output is 1mm isotropic in practice, so this
    # mismatch happens not to matter numerically here -- but a metric that's
    # only "accidentally correct" for one spacing convention is a landmine for
    # anyone reusing this on non-isotropic data later).
    spacing_zyx = tuple(reversed(mask_img.GetSpacing()))
    voxel_volume_mm3 = float(np.prod(spacing_zyx))

    voxel_count = int(mask_arr.sum())
    row = {"voxel_count": voxel_count}

    if voxel_count == 0:
        row.update({
            "status": "empty_mask", "volume_mm3": 0.0, "num_components": 0,
            "largest_component_ratio": float("nan"), "fill_ratio": float("nan"),
            "sphericity": float("nan"), "elongation_ratio": float("nan"),
            "centroid_z": float("nan"), "centroid_y": float("nan"), "centroid_x": float("nan"),
        })
        return row

    row["status"] = "ok"
    row["volume_mm3"] = voxel_count * voxel_volume_mm3

    labeled, num_components = ndimage.label(mask_arr)
    row["num_components"] = int(num_components)
    if num_components > 0:
        sizes = ndimage.sum(mask_arr, labeled, index=range(1, num_components + 1))
        row["largest_component_ratio"] = float(sizes.max() / voxel_count)
    else:
        row["largest_component_ratio"] = float("nan")

    nonzero = np.nonzero(mask_arr)
    bbox_shape = tuple(int(nonzero[a].max() - nonzero[a].min() + 1) for a in range(3))
    bbox_voxels = int(np.prod(bbox_shape))
    row["fill_ratio"] = voxel_count / bbox_voxels if bbox_voxels > 0 else float("nan")

    # Sphericity: psi = (pi^(1/3) * (6V)^(2/3)) / A, in (0, 1] with 1 == a
    # perfect sphere -- a physically meaningful, spacing-aware compactness
    # measure (unlike e.g. voxel_count / bbox_volume, which is orientation-
    # and shape-dependent in ways that don't isolate "how irregular/scattered
    # is this blob" the way sphericity does). Surface area comes from a
    # marching-cubes mesh of the binary mask; too small a mask (a handful of
    # voxels) doesn't reliably produce a valid mesh, so this is best-effort
    # and left as NaN (not zero -- zero would look like "measured, totally
    # non-spherical" instead of "couldn't measure") if it fails.
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        verts, faces, _normals, _values = marching_cubes(mask_arr.astype(np.float32), level=0.5, spacing=spacing_zyx)
        surface_area_mm2 = mesh_surface_area(verts, faces)
        if surface_area_mm2 > 0:
            sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * row["volume_mm3"]) ** (2.0 / 3.0)) / surface_area_mm2
            row["sphericity"] = float(min(sphericity, 1.0))  # mesh discretization can push this fractionally over 1
        else:
            row["sphericity"] = float("nan")
    except Exception:  # noqa: BLE001 -- best-effort geometry metric, a bad mesh must not crash the whole audit
        row["sphericity"] = float("nan")

    # Elongation: ratio of the largest to smallest principal-axis standard
    # deviation of the foreground voxels' physical coordinates (PCA via the
    # eigenvalues of their covariance matrix) -- 1.0 for a roughly isotropic
    # blob, higher for a stretched/needle-like one. Needs enough points for a
    # stable covariance estimate; skipped (NaN) below that.
    if voxel_count >= 8:
        coords = np.stack(nonzero, axis=1).astype(np.float64) * np.array(spacing_zyx)
        cov = np.cov(coords, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-9, None)
        row["elongation_ratio"] = float(np.sqrt(eigvals.max() / eigvals.min()))
    else:
        row["elongation_ratio"] = float("nan")

    centroid = ndimage.center_of_mass(mask_arr)
    row["centroid_z"] = float(centroid[0] / mask_arr.shape[0])
    row["centroid_y"] = float(centroid[1] / mask_arr.shape[1])
    row["centroid_x"] = float(centroid[2] / mask_arr.shape[2])

    return row


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD-based z-score -- robust to the very outliers this is used
    to detect, unlike a mean/std z-score where one extreme value inflates
    the std and can mask its own deviation. 1.4826 rescales MAD so it's a
    consistent estimator of the standard deviation under a normal
    distribution, making the two forms of z-score roughly comparable in
    magnitude. NaN input stays NaN in the output (never silently zeroed)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan)
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    scaled_mad = mad * 1.4826
    if scaled_mad < 1e-9:
        # More than half the population shares the exact same value (MAD==0) --
        # common for a metric like num_components, which is 1 for most patients.
        # With no spread to divide by, any value that DOES deviate from the
        # median is, by definition, the anomaly -- returning 0 for everyone here
        # (the naive zero-division guard) would incorrectly read as "no outliers"
        # and hide exactly the cases this function exists to catch. Values equal
        # to the median get z=0; anything else gets a large, signed sentinel so
        # it clears any reasonable --z_threshold regardless of direction.
        out = np.where(np.isfinite(values), 0.0, np.nan)
        deviates = np.isfinite(values) & (np.abs(values - median) > 1e-9)
        return np.where(deviates, np.sign(values - median) * 1e6, out)
    return (values - median) / scaled_mad


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Each value's percentile rank (0-100) among the finite values -- an
    easier-to-eyeball companion to the z-scores, and what "bottom/top 5%"
    style exclusion criteria (mentioned in the project's own filtering plan)
    would actually threshold against."""
    finite_mask = np.isfinite(values)
    out = np.full_like(values, np.nan)
    finite_vals = values[finite_mask]
    if finite_vals.size == 0:
        return out
    ranks = np.argsort(np.argsort(finite_vals)).astype(np.float64)
    out[finite_mask] = ranks / max(1, finite_vals.size - 1) * 100.0
    return out


def add_population_stats(rows: list[dict], z_threshold: float) -> None:
    """Fills in the population-relative columns (z-scores, percentiles,
    centroid offset, flags) in place, once every patient's raw per-patient
    metrics have been computed. Has to be a second pass -- none of this is
    knowable from a single patient's mask alone."""
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        log.warning("add_population_stats: no patients with status=ok -- nothing to compare against.")
        for r in rows:
            r["centroid_offset"] = float("nan")
            r["z_log_volume"] = r["z_sphericity"] = r["z_num_components"] = r["z_centroid_offset"] = float("nan")
            r["pct_log_volume"] = r["pct_sphericity"] = float("nan")
            r["flags"] = "empty_mask" if r["status"] == "empty_mask" else ""
            r["is_outlier"] = r["status"] == "empty_mask"
        return

    mean_centroid = np.array([
        np.nanmean([r["centroid_z"] for r in ok_rows]),
        np.nanmean([r["centroid_y"] for r in ok_rows]),
        np.nanmean([r["centroid_x"] for r in ok_rows]),
    ])
    for r in rows:
        if r["status"] == "ok":
            this_centroid = np.array([r["centroid_z"], r["centroid_y"], r["centroid_x"]])
            r["centroid_offset"] = float(np.linalg.norm(this_centroid - mean_centroid))
        else:
            r["centroid_offset"] = float("nan")

    log_volume = np.array([r.get("volume_mm3", float("nan")) if r["status"] == "ok" else np.nan for r in rows])
    log_volume = np.log1p(log_volume)
    sphericity = np.array([r.get("sphericity", float("nan")) for r in rows])
    num_components = np.array([float(r.get("num_components", float("nan"))) if r["status"] == "ok" else np.nan for r in rows])
    centroid_offset = np.array([r["centroid_offset"] for r in rows])

    z_log_volume = robust_z(log_volume)
    z_sphericity = robust_z(sphericity)
    z_num_components = robust_z(num_components)
    z_centroid_offset = robust_z(centroid_offset)
    pct_log_volume = percentile_rank(log_volume)
    pct_sphericity = percentile_rank(sphericity)

    for i, r in enumerate(rows):
        r["z_log_volume"] = float(z_log_volume[i]) if np.isfinite(z_log_volume[i]) else float("nan")
        r["z_sphericity"] = float(z_sphericity[i]) if np.isfinite(z_sphericity[i]) else float("nan")
        r["z_num_components"] = float(z_num_components[i]) if np.isfinite(z_num_components[i]) else float("nan")
        r["z_centroid_offset"] = float(z_centroid_offset[i]) if np.isfinite(z_centroid_offset[i]) else float("nan")
        r["pct_log_volume"] = float(pct_log_volume[i]) if np.isfinite(pct_log_volume[i]) else float("nan")
        r["pct_sphericity"] = float(pct_sphericity[i]) if np.isfinite(pct_sphericity[i]) else float("nan")

        flags = []
        if r["status"] == "empty_mask":
            flags.append("empty_mask")
        else:
            if np.isfinite(r["z_log_volume"]) and abs(r["z_log_volume"]) > z_threshold:
                flags.append("volume_outlier")
            # only the "too irregular" direction is suspicious -- an unusually
            # ROUND tumor isn't a data-quality problem
            if np.isfinite(r["z_sphericity"]) and r["z_sphericity"] < -z_threshold:
                flags.append("low_sphericity")
            if np.isfinite(r["z_num_components"]) and r["z_num_components"] > z_threshold:
                flags.append("fragmented")
            if np.isfinite(r["z_centroid_offset"]) and r["z_centroid_offset"] > z_threshold:
                flags.append("unusual_location")
        r["flags"] = ";".join(flags)
        r["is_outlier"] = len(flags) > 0


def write_csv(rows: list[dict], output_csv: str) -> None:
    """Writes the full per-patient audit to output_csv."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    log.info("Wrote %d rows to %s", len(rows), output_csv)


def print_summary(rows: list[dict]) -> None:
    """Console summary: counts by status/flag, so the headline numbers are
    visible without opening the CSV."""
    n = len(rows)
    n_empty = sum(1 for r in rows if r["status"] == "empty_mask")
    n_outlier = sum(1 for r in rows if r.get("is_outlier"))
    print(f"\n{n} patients audited. {n_empty} have an empty (all-background) mask. {n_outlier} flagged as outliers (z_threshold exceeded on >=1 signal, or empty).\n")

    from collections import Counter
    flag_counts = Counter()
    for r in rows:
        for flag in (r.get("flags") or "").split(";"):
            if flag:
                flag_counts[flag] += 1
    if flag_counts:
        print("Flag breakdown (a patient can carry more than one):")
        for flag, count in flag_counts.most_common():
            print(f"  {flag:<20}{count}")

    flagged = sorted((r for r in rows if r.get("is_outlier")), key=lambda r: r["patient_id"])
    if flagged:
        print("\nFlagged patients:")
        for r in flagged:
            print(f"  {r['patient_id']:<28} flags={r['flags']}")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    root = args.synthetic_ct_root or config["data"]["synthetic_ct_root"]
    patients = _discover_mask_paths(root)
    if not patients:
        raise SystemExit(f"No patients with a tumor_mask file found under {root!r}.")

    rows = []
    n_errors = 0
    for i, (patient_id, mask_path) in enumerate(patients):
        try:
            metrics = compute_mask_metrics(mask_path)
        except Exception as e:  # noqa: BLE001 -- one unreadable mask file must not abort the whole audit
            log.exception("compute_mask_metrics failed for %s (%s) -- marking status=error", patient_id, mask_path)
            metrics = {"status": "error", "voxel_count": 0}
            n_errors += 1
        metrics["patient_id"] = patient_id
        rows.append(metrics)
        if (i + 1) % 50 == 0:
            log.info("Processed %d/%d patients", i + 1, len(patients))

    add_population_stats(rows, args.z_threshold)
    write_csv(rows, args.output_csv)
    print_summary(rows)
    if n_errors:
        log.warning("%d patients could not be read at all (status=error) -- check their tumor_mask file directly.", n_errors)


if __name__ == "__main__":
    main()
