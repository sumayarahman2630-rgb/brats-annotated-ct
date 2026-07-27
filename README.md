# brats-annotated-ct

Real annotated CT of brain tumors is hard to get in bulk -- hospitals hold
it under strict access controls, and public datasets with both a CT volume
and a matching tumor mask are rare. I built this project to get around
that: train a model to translate MRI into CT, then run that translation
over a large public MRI-plus-tumor-mask dataset (BraTS) that would
otherwise be unusable for CT-based work. That gives me a synthetic,
tumor-annotated CT dataset I can then train a CT segmentation model on,
and check against a small set of real hospital CT scans it never saw
during training. Below is the walkthrough of how I actually ran this, in
the order I ran it, on Kaggle.

## What I did, step by step

**Stage 1 -- getting MRI-to-CT translation working.** I uploaded the
SynthRAD2023 brain cohort (180 patients, paired real MRI/CT/brain-mask
volumes) as a Kaggle input and trained the regression U-Net in
[`models/unet3d_regression.py`](models/unet3d_regression.py):

```bash
python -m training.train_stage1_regression --config configs/stage1_regression.yaml --max_steps 100 --max_patients 3
python -m training.train_stage1_regression --config configs/stage1_regression.yaml
```

I smoke-tested with `--max_steps`/`--max_patients` first to catch config or
path errors before burning a real GPU session, then ran the full
20,000-step training job from [`configs/stage1_regression.yaml`](configs/stage1_regression.yaml).
The 180 patients get split 90/10 by patient (never by slice or volume, to
avoid leakage) -- 162 for training, 18 held out for validation. After
training finished, I checked how well it actually generalized to those 18
unseen patients:

```bash
python -m inference.visualize_regression_val --config configs/stage1_regression.yaml
```

That produced a PSNR/SSIM number per validation patient plus a 4-panel
comparison image (input MRI / real CT / synthetic CT / error map) for each
one. **The mean foreground PSNR across all 18 held-out patients came out
to 28.92 dB** -- this is the number I use everywhere else in this repo and
in the thesis; it's an average over the full held-out set, not one
patient's best result or a single checkpoint's last logged value.

**Stage 2 -- generating the annotated synthetic CT dataset.** With a
working Stage 1 checkpoint, I uploaded BraTS2020 (369 T1 MRI + tumor mask
patients, one of which was missing its mask) and ran the Stage 1 model
over the whole cohort:

```bash
python -m inference.run_stage2_brats_regression --config configs/stage2_inference_brats_regression.yaml --limit 3
python -m inference.run_stage2_brats_regression --config configs/stage2_inference_brats_regression.yaml
```

Again, I smoke-tested with `--limit 3` first. The full run generated a
synthetic CT volume for every BraTS patient it could pair with a tumor
mask, wrote each one alongside its mask under
`synthetic_ct_dataset_regression/<patient_id>/`, and logged every
outcome (success, skipped, failed) to `manifest.csv`. This produced
**365 synthetic CT + tumor mask pairs** -- this is the dataset I re-uploaded
as its own Kaggle input for Stage 3.

**Stage 3 -- training and validating the segmentation model.** I trained
a binary tumor segmentation U-Net on the Stage 2 output:

```bash
python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml --max_steps 100 --max_patients 3
python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml
```

This part took several retraining passes to get right -- the full
evolution (loss function changes, a couple of lost checkpoints from
Kaggle session resets, augmentation added partway through) is documented
in [`PROJECT_NOTES.md`](PROJECT_NOTES.md). Once I had a checkpoint I
trusted, I got the official internal number (full-volume Dice/IoU across
the entire synthetic validation split, not the cheap patch-level number
training logs periodically during the run):

```bash
python -m inference.validate_synthetic_segmentation --config configs/stage3_ct_segmentation.yaml
```

Then I checked how that same checkpoint held up against real hospital
data it had never seen -- 20 CT patients from Jordan University Hospital,
attached as a separate Kaggle input, held out purely for this step:

```bash
python -m inference.validate_jordan_segmentation --config configs/stage3_ct_segmentation.yaml
```

The most recent checkpoint I measured this way (focal loss, alpha=0.75)
came back at **0.1987 Dice on the Jordan external set**. Finally, to get
everything -- training curves, internal + external Dice/IoU, and example
prediction images -- in one place after a training run finishes, I ran:

```bash
python -m inference.generate_full_report --config configs/stage3_ct_segmentation.yaml
```

**Last, I ran the test suite** to make sure nothing in the active pipeline
was broken before calling any of this done:

```bash
pip install -r requirements.txt
python -m pytest tests/
```

Figures for the writeup (PSNR curves, distributions, scatter plots) come
from `analysis/`, run after Stage 1 validation has produced real data:
```bash
python -m analysis.plot_validation_psnr_curve --config configs/stage1_regression.yaml
python -m analysis.generate_val_metrics_table --config configs/stage1_regression.yaml
python -m analysis.plot_psnr_ssim_distribution --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv
python -m analysis.plot_psnr_vs_ssim_scatter --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv
```

## Results Summary

| Stage | Metric | Value |
|---|---|---|
| Stage 1 (MRI-to-CT) | Foreground PSNR, mean across 18 held-out validation patients | **28.92 dB** |
| Stage 2 (dataset generation) | Synthetic CT + tumor mask pairs generated | **365 patients** (from BraTS2020) |
| Stage 3 (segmentation), external | Dice, Jordan Hospital CT (real, never trained on) | **0.1987** (most recently measured checkpoint, focal loss, alpha=0.75) |

A couple of notes on reading that Stage 3 row: the internal metric
(full-volume Dice on the synthetic validation split, produced by
`validate_synthetic_segmentation.py`) went through several loss-function
and checkpoint iterations documented in
[`PROJECT_NOTES.md`](PROJECT_NOTES.md), and the current run's number
should be regenerated with the command above rather than assumed stable
across retrains. And the Jordan number reflects a genuinely different,
harder comparison than the internal one -- see
[`data/loaders_jordan_ct.py`](data/loaders_jordan_ct.py)'s module
docstring for why (8-bit windowed RGB DICOM with no real HU values, only
1-6 slices per patient instead of a full volume) before treating it as
directly comparable to the internal number.

I tried a wavelet-diffusion model for Stage 1 before settling on the
regression U-Net above; it reached only ~9 dB foreground PSNR, undertrained
given the available compute budget, and was dropped in favor of the
regression approach. See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the
full development narrative -- what was tried, what broke, and why.

## Project Structure

### Stage 1 -- MRI-to-CT translation

Trains a 3D regression U-Net on the SynthRAD2023 brain cohort (paired real
MRI/CT) to predict a CT volume directly from an MRI volume.

- [`models/unet3d_regression.py`](models/unet3d_regression.py) -- the model
- [`configs/stage1_regression.yaml`](configs/stage1_regression.yaml) -- hyperparameters and data paths
- [`training/train_stage1_regression.py`](training/train_stage1_regression.py) -- the training loop
- [`data/loaders_synthrad.py`](data/loaders_synthrad.py) -- SynthRAD2023 dataset
- [`inference/visualize_regression_val.py`](inference/visualize_regression_val.py) -- PSNR/SSIM + comparison images on the held-out validation split

### Stage 2 -- synthetic annotated CT dataset generation

Applies the trained Stage 1 model to BraTS2020 T1 MRI volumes and pairs
each generated CT with its source BraTS tumor mask. **This is the core
deliverable** -- annotated synthetic CT for tumor-region work where real
annotated CT is scarce.

- [`data/loaders_brats.py`](data/loaders_brats.py) -- BraTS2020 T1 MRI + tumor mask loader (the input side)
- [`inference/run_stage2_brats_regression.py`](inference/run_stage2_brats_regression.py) -- runs the Stage 1 checkpoint over BraTS and writes the dataset
- [`configs/stage2_inference_brats_regression.yaml`](configs/stage2_inference_brats_regression.yaml) -- generation settings

Output convention: one folder per patient under the configured
`output_dir`, containing `synthetic_ct.nii.gz` and `tumor_mask.nii.gz`,
plus a generated `manifest.csv`, `metadata.json`, and `README.md` dataset
card.

### Stage 3 -- CT tumor segmentation

Trains a binary segmentation model on the Stage 2 output (synthetic CT +
tumor mask, binarized from BraTS's multi-class labels), then externally
validates it against a small real-CT dataset from Jordan University
Hospital that was never used in training.

- [`models/unet3d_segmentation.py`](models/unet3d_segmentation.py) -- the model (same U-Net topology as Stage 1, raw logit output)
- [`configs/stage3_ct_segmentation.yaml`](configs/stage3_ct_segmentation.yaml) -- hyperparameters and dataset paths
- [`training/train_stage3_segmentation.py`](training/train_stage3_segmentation.py) -- the training loop
- [`data/loaders_synthetic_ct.py`](data/loaders_synthetic_ct.py) -- Stage 2 output loader (the only data Stage 3 trains on)
- [`data/loaders_jordan_ct.py`](data/loaders_jordan_ct.py) -- Jordan Hospital DICOM loader (external validation only, never touched by training)
- [`inference/validate_synthetic_segmentation.py`](inference/validate_synthetic_segmentation.py) -- official internal Dice/IoU (full-volume, on the synthetic validation split)
- [`inference/validate_jordan_segmentation.py`](inference/validate_jordan_segmentation.py) -- external Dice/IoU against real Jordan CT
- [`inference/visualize_predictions.py`](inference/visualize_predictions.py) -- per-patient prediction panels, both sources
- [`inference/postprocessing.py`](inference/postprocessing.py) -- optional largest-component filtering + threshold search, shared by the two validation scripts
- [`inference/generate_full_report.py`](inference/generate_full_report.py) -- runs all of the above in one command after training finishes

### Shared across every stage

- [`data/preprocessing.py`](data/preprocessing.py) -- HU/MRI normalization, resampling, cropping/padding, patch cropping and augmentation
- [`training/checkpoint.py`](training/checkpoint.py) -- checkpoint save/load/resume
- [`training/ema.py`](training/ema.py) -- exponential moving average (saved for reference; raw weights are what's actually evaluated everywhere)
- [`tests/`](tests) -- CPU-only test suite covering all three stages
- [`analysis/`](analysis) -- figure/table generation from real training logs and checkpoints (see below)

## Pipeline

```
SynthRAD2023 MRI + CT (paired, real)
        │
        ▼
  train Stage 1 regression U-Net  ──►  checkpoint (28.92 dB mean val PSNR)
        │
        ▼
  BraTS2020 T1 MRI (no real CT)
        │
        ▼
  Stage 1 checkpoint + sliding-window inference
        │
        ▼
  synthetic CT, paired with the original BraTS tumor mask
        │
        ▼
  synthetic_ct_dataset_regression/  (365 patients -- the core deliverable)
        │
        ▼
  train Stage 3 segmentation U-Net (binarized tumor mask as target)
        │
        ├──►  held-out synthetic CT val patients (in-distribution check)
        │
        └──►  Jordan Hospital real CT (20 patients, never trained on)
                external validation -- see PROJECT_NOTES.md for caveats
```

### `analysis/`

Scripts that turn training/validation output into figures and tables,
reading real data (training logs, checkpoints) rather than hardcoded
numbers wherever practical:

- `plot_validation_psnr_curve.py` -- validation PSNR vs. training step, from
  the training log CSV.
- `generate_val_metrics_table.py` -- runs the checkpoint over the full
  validation split, writes a per-patient PSNR/SSIM/L1 CSV, and prints a
  summary table. The other two scripts below consume this CSV rather than
  recomputing it.
- `plot_psnr_ssim_distribution.py` -- box plots of the PSNR/SSIM
  distributions with individual patient points overlaid.
- `plot_psnr_vs_ssim_scatter.py` -- PSNR vs. SSIM per patient, color-coded
  by PSNR.

Every script's input/output paths are CLI arguments (`--config`,
`--log_file`, `--metrics_csv`, `--output`, ...) -- no hardcoded paths.

## Data sources

- [SynthRAD2023](https://synthrad2023.grand-challenge.org/) brain cohort
  (paired MRI/CT) -- Stage 1 training data.
- [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) training set
  (T1 MRI + tumor segmentation) -- Stage 2 input data.
- Jordan University Hospital CT + tumor mask (20 patients, DICOM) -- Stage 3
  external validation only, never used in training. See
  [`PROJECT_NOTES.md`](PROJECT_NOTES.md)'s Stage 3 section for this
  dataset's format and known limitations.

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md)'s "Kaggle dataset paths" section
for the exact confirmed Kaggle input paths used during development.
