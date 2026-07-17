# Archive: wavelet-diffusion pipeline (superseded)

This folder holds the project's original approach: a conditional DDPM
operating in Haar-wavelet space (3D) plus a pixel-space 2D variant, both
following the cwdm-style wavelet-domain-diffusion pattern. It's preserved
here, not deleted, as a record of what was tried and why it didn't become
the final pipeline -- see the main [README.md](../README.md) and
[CLAUDE.md](../CLAUDE.md) for the full comparison and reasoning.

**In one sentence: after real Kaggle training, the diffusion checkpoint
reached only ~9 dB foreground PSNR, while a much simpler direct-regression
U-Net (the active pipeline, top-level `models/`, `training/`,
`inference/`) reached 28.21 dB on held-out validation patients at a
fraction of the training cost per step.** The diffusion approach wasn't
abandoned because it was wrong in principle -- wavelet-domain diffusion is
a legitimate, published pattern for this exact problem -- but because it
needed far more training steps than the project's time budget allowed to
reach a comparable result, and the regression alternative got there first.

## Status: frozen, not maintained

These files moved here as-is from their original top-level locations
(`models/`, `configs/`, `training/`, `inference/`, plus their own
`data/loaders_synthrad_2d.py`). Internal imports between archived files
were updated to their new `archive.*` paths so the code is still
consistent to read, but **this code is not run or tested going forward**
-- `pytest tests/` (the project's active, maintained suite) no longer
covers it. The archived tests that used to cover it live in
`archive/tests/`, with the same import-path updates, but they are not
wired into CI or the default test command; treat them as a historical
record of what was verified at the time, not a guarantee that this code
still runs unmodified today.

## Layout

```
archive/
  models/
    wavelet_transform.py         Haar 3D DWT / IDWT
    unet3d.py                    3D conditional U-Net (wavelet-domain, attention, timestep conditioning)
    stage1_mri2ct_ddpm.py        composes the above into the 3D diffusion model
    unet2d.py                    2D conditional U-Net (pixel-space)
    stage1_mri2ct_ddpm_2d.py     composes the above into the 2D diffusion model
  configs/
    stage1_synthrad.yaml         3D diffusion training config
    stage1_synthrad_2d.yaml      2D diffusion training config
    stage2_inference_brats.yaml  3D diffusion Stage 2 config
  training/
    train_stage1.py              3D diffusion training loop
    train_stage1_2d.py           2D diffusion training loop
  inference/
    run_stage2_brats.py          3D diffusion Stage 2 generation (DDIM sampling)
    compare_synthrad_val.py      3D diffusion val comparison (PSNR/SSIM)
  data/
    loaders_synthrad_2d.py       2D slice dataset (wraps the still-active data/loaders_synthrad.py)
  tests/                         the above's own test coverage, frozen alongside it
```

Shared infrastructure the regression pipeline still uses today --
`data/preprocessing.py`, `data/loaders_synthrad.py`, `data/loaders_brats.py`,
`training/checkpoint.py`, `training/ema.py`, `scripts/check_orientation_consistency.py`
-- was **not** moved here; it lives at the top level and is actively
maintained.
