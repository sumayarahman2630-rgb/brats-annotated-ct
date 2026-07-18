"""Diagnostic: does the pipeline's assumption that SynthRAD and BraTS voxel
array axes (D, H, W after sitk.GetArrayFromImage) mean the same anatomical
directions actually hold?

This was never checked. resample_to_spacing (data/preprocessing.py) resamples
each image using ITS OWN GetOrigin()/GetDirection() -- it does not reorient
images to a canonical frame (e.g. via sitk.DICOMOrient()). sitk.GetArrayFromImage
just returns voxels in on-disk index order (z, y, x by numpy convention). If
SynthRAD's brain MR/CT and BraTS's T1 use DIFFERENT orientation conventions in
their NIfTI headers (e.g. one LPS+, the other RAS+, or an axis permutation),
then a "D" index in a SynthRAD-trained model's condition input would NOT
correspond to the same physical direction in a BraTS input -- a real
conditioning mismatch that's independent of (and potentially compounds) the
undertraining issue investigated in DEVELOPMENT_LOG.md's round-8 PSNR audit.

This could NOT be verified from the development machine (no access to the
real SynthRAD/BraTS files). Run this on Kaggle, pointed at one real SynthRAD
patient and one real BraTS patient, and compare the printed direction/origin
values -- see the interpretation guide printed at the end.

Run as:
    python -m scripts.check_orientation_consistency \
        --synthrad_mr /kaggle/input/.../Task1/brain/<patient>/mr.nii \
        --brats_t1 /kaggle/input/.../MICCAI_BraTS2020_TrainingData/<patient>/<patient>_t1.nii
"""
from __future__ import annotations

import argparse

import numpy as np
import SimpleITK as sitk


def describe(path: str, label: str) -> dict:
    img = sitk.ReadImage(path)
    direction = np.array(img.GetDirection()).reshape(3, 3)
    info = {
        "label": label,
        "path": path,
        "size": img.GetSize(),
        "spacing": tuple(round(s, 3) for s in img.GetSpacing()),
        "origin": tuple(round(o, 2) for o in img.GetOrigin()),
        "direction_matrix": direction,
    }
    print(f"\n--- {label} ---")
    print(f"  path: {path}")
    print(f"  size (voxels): {info['size']}")
    print(f"  spacing (mm): {info['spacing']}")
    print(f"  origin (mm): {info['origin']}")
    print(f"  direction matrix:\n{direction}")
    # The direction matrix's columns indicate which physical axis (roughly LR/AP/SI)
    # each voxel-index axis (i,j,k) points along. Identical/near-identical matrices
    # across datasets -> consistent orientation. Very different matrices (e.g. a
    # permutation or sign-flipped pattern) -> a real orientation mismatch.
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthrad_mr", type=str, required=True, help="Path to one SynthRAD patient's mr.nii")
    parser.add_argument("--brats_t1", type=str, required=True, help="Path to one BraTS patient's <id>_t1.nii")
    parser.add_argument("--synthrad_ct", type=str, default=None, help="Optional: same SynthRAD patient's ct.nii, to also confirm MR/CT internal consistency")
    args = parser.parse_args()

    mr_info = describe(args.synthrad_mr, "SynthRAD MR")
    t1_info = describe(args.brats_t1, "BraTS T1")
    if args.synthrad_ct:
        describe(args.synthrad_ct, "SynthRAD CT (same patient, should match MR exactly)")

    diff = np.abs(mr_info["direction_matrix"] - t1_info["direction_matrix"]).max()
    print("\n--- Interpretation ---")
    print(f"Max abs difference between SynthRAD MR and BraTS T1 direction matrices: {diff:.4f}")
    if diff < 1e-3:
        print("=> Orientations match closely. Not the source of the conditioning-quality problem.")
    elif diff < 0.5:
        print("=> Small but nonzero difference -- possibly just floating-point noise from different "
              "conversion pipelines, but worth a closer look (plot a mid-slice from each and eyeball "
              "whether anterior/posterior/left/right line up the same way).")
    else:
        print("=> LARGE difference -- likely a real orientation mismatch (e.g. axis permutation or "
              "flip) between the two datasets. This would mean the model's learned spatial priors from "
              "SynthRAD training don't transfer correctly to BraTS input orientation. Fix: reorient both "
              "to a canonical frame during preprocessing, e.g. sitk.DICOMOrient(img, 'LPS') (or 'RAS') "
              "applied right after sitk.ReadImage(), consistently in both data/loaders_synthrad.py and "
              "data/loaders_brats.py, before any resampling.")


if __name__ == "__main__":
    main()
