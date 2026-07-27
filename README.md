# brats-annotated-ct

Real annotated CT of brain tumors is hard to get in bulk -- hospitals hold
it under strict access controls, and public datasets with both a CT volume
and a matching tumor mask are rare. This project builds one anyway, by
learning to translate MRI into CT and then running that translation over
a large public MRI-plus-tumor-mask dataset (BraTS) that would otherwise be
unusable for CT-based work. The result is a three-stage pipeline: Stage 1
trains an MRI-to-CT model on real paired scans, Stage 2 uses that model to
generate a synthetic, tumor-annotated CT dataset from BraTS, and Stage 3
trains a CT tumor segmentation model on that synthetic dataset and checks
how well it holds up against a small set of real hospital CT scans it
never saw during training.

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
- [`archive/`](archive) -- the original wavelet-diffusion approach for Stage 1, kept for the record (see [archive/README.md](archive/README.md) for why it was replaced)

## How to Run

Everything below assumes a Kaggle GPU session with the SynthRAD2023 and
BraTS2020 datasets mounted (see [`PROJECT_NOTES.md`](PROJECT_NOTES.md)'s
"Kaggle dataset paths" for the exact confirmed input paths).

**Stage 1 -- train the MRI-to-CT model** (smoke-test first):
```bash
python -m training.train_stage1_regression --config configs/stage1_regression.yaml --max_steps 100 --max_patients 3
python -m training.train_stage1_regression --config configs/stage1_regression.yaml
```

Check validation quality (PSNR/SSIM + comparison images on held-out patients):
```bash
python -m inference.visualize_regression_val --config configs/stage1_regression.yaml
```

**Stage 2 -- generate the annotated synthetic CT dataset** (smoke-test with `--limit` first):
```bash
python -m inference.run_stage2_brats_regression --config configs/stage2_inference_brats_regression.yaml --limit 3
python -m inference.run_stage2_brats_regression --config configs/stage2_inference_brats_regression.yaml
```

**Stage 3 -- train the segmentation model on the Stage 2 output** (smoke-test first):
```bash
python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml --max_steps 100 --max_patients 3
python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml
```

Get the official internal Dice/IoU (full-volume, entire synthetic
validation split -- not the cheap patch-level number training logs
periodically):
```bash
python -m inference.validate_synthetic_segmentation --config configs/stage3_ct_segmentation.yaml
```

Validate against the Jordan external dataset (never used in training):
```bash
python -m inference.validate_jordan_segmentation --config configs/stage3_ct_segmentation.yaml
```

Or generate everything above in one run once training finishes (training
curves, internal + external Dice/IoU, an EMA side-by-side comparison, and
example prediction images):
```bash
python -m inference.generate_full_report --config configs/stage3_ct_segmentation.yaml
```

**Figures** (once training/validation has produced real data):
```bash
python -m analysis.plot_validation_psnr_curve --config configs/stage1_regression.yaml
python -m analysis.generate_val_metrics_table --config configs/stage1_regression.yaml
python -m analysis.plot_psnr_ssim_distribution --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv
python -m analysis.plot_psnr_vs_ssim_scatter --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv
```

**Test suite** (CPU-only, no GPU or real data needed):
```bash
pip install -r requirements.txt
python -m pytest tests/
```

## Results Summary

| Stage | Metric | Value |
|---|---|---|
| Stage 1 (MRI-to-CT) | Foreground PSNR, held-out validation patients | **28.21 dB** (step 20000, raw weights) |
| Stage 2 (dataset generation) | Synthetic CT + tumor mask pairs generated | **365 patients** (from BraTS2020) |
| Stage 3 (segmentation), external | Dice, Jordan Hospital CT (real, never trained on) | **0.1987** (most recently measured checkpoint, focal loss, alpha=0.75) |

Two notes on how to read the Stage 3 row: the internal metric (full-volume
Dice on the synthetic validation split, produced by
`validate_synthetic_segmentation.py`) has gone through several
loss-function and checkpoint iterations documented in
[`PROJECT_NOTES.md`](PROJECT_NOTES.md), and the current run's number
should be regenerated with the command above rather than assumed stable
across retrains. And the Jordan number reflects a genuinely different,
harder comparison than the internal one -- see
[`data/loaders_jordan_ct.py`](data/loaders_jordan_ct.py)'s module
docstring for why (8-bit windowed RGB DICOM with no real HU values, only
1-6 slices per patient instead of a full volume) before treating it as
directly comparable to the internal number.

Two Stage 1 architectures were tried; the regression U-Net above is the
one that reached a usable result:

| Model | Foreground PSNR (val, unseen patients) |
|---|---:|
| **Regression U-Net (active)** | **28.21 dB** |
| Wavelet diffusion (archived) | ~9 dB, undertrained given the available compute budget |

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the full development
narrative -- what was tried, what broke, and why -- and
[`archive/README.md`](archive/README.md) for why the diffusion approach
was kept, not deleted.

## Pipeline

```
SynthRAD2023 MRI + CT (paired, real)
        │
        ▼
  train Stage 1 regression U-Net  ──►  checkpoint (28.21 dB val PSNR)
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
