# brats-annotated-ct

Synthetic, tumor-annotated CT dataset generation from brain MRI, for
domains where real annotated CT is scarce.

## What this project does

Three-stage pipeline:

1. **Stage 1 — MRI → CT translation**, trained on the full SynthRAD2023
   brain cohort (180 patients, paired real MRI/CT). A model learns to
   predict a CT volume from an MRI volume alone.
2. **Stage 2 — synthetic CT dataset generation**, applying the Stage 1
   model to BraTS2020 T1 MRI volumes (369 patients). Each generated CT is
   paired with its source BraTS tumor segmentation mask, under a clear
   `<patient_id>/synthetic_ct.nii.gz` + `<patient_id>/tumor_mask.nii.gz`
   convention. **This is the core deliverable** — annotated synthetic CT
   for tumor-region work.
3. **Stage 3 — CT tumor segmentation**, training a binary segmentation
   model on the Stage 2 output (synthetic CT + tumor mask, binarized from
   BraTS's multi-class labels), then externally validating it against a
   small real-CT dataset (Jordan University Hospital, 20 patients) that
   was never used in training. See PROJECT_NOTES.md's Stage 3 section for
   the real, load-bearing limitations of that external comparison (format
   mismatch, incomplete volumes, unverified filename matching) before
   trusting its numbers.

## Result

Two architectures were tried for Stage 1. The active pipeline (below) is
the one that reached a genuinely good result:

| Model | Foreground PSNR (val, unseen patients) | Notes |
|---|---:|---|
| **Regression U-Net (active)** | **28.21 dB** | L1 loss, direct prediction, 20000 training steps |
| Wavelet diffusion (archived) | ~9 dB | DDPM, undertrained given the available compute budget |

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the full development narrative — what was
tried, what broke, and why the simpler model won — and
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
  synthetic_ct_dataset_regression/  (the core deliverable)
        │
        ▼
  train Stage 3 segmentation U-Net (binarized tumor mask as target)
        │
        ├──►  held-out synthetic CT val patients (in-distribution check)
        │
        └──►  Jordan Hospital real CT (20 patients, never trained on)
                external validation -- see PROJECT_NOTES.md for caveats
```

## How to run

Everything below assumes a Kaggle GPU session with the SynthRAD2023 and
BraTS2020 datasets mounted (see `PROJECT_NOTES.md`'s "Kaggle dataset paths" for
the exact confirmed input paths).

**1. Train Stage 1** (smoke-test first, same pattern for any new config):
```bash
python -m training.train_stage1_regression --config configs/stage1_regression.yaml --max_steps 100 --max_patients 3
python -m training.train_stage1_regression --config configs/stage1_regression.yaml
```

**2. Check validation quality** (PSNR/SSIM + comparison images on held-out patients):
```bash
python -m inference.visualize_regression_val --config configs/stage1_regression.yaml
```

**3. Generate the BraTS dataset** (smoke-test with `--limit` first):
```bash
python -m inference.run_stage2_brats_regression --config configs/stage2_inference_brats_regression.yaml --limit 3
python -m inference.run_stage2_brats_regression --config configs/stage2_inference_brats_regression.yaml
```

**4. Generate figures** (see `analysis/` below) once training/validation has produced real data:
```bash
python -m analysis.plot_validation_psnr_curve --config configs/stage1_regression.yaml
python -m analysis.generate_val_metrics_table --config configs/stage1_regression.yaml
python -m analysis.plot_psnr_ssim_distribution --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv
python -m analysis.plot_psnr_vs_ssim_scatter --metrics_csv /kaggle/working/analysis_plots/val_metrics.csv
```

**5. Train Stage 3** (segmentation, on the Stage 2 output -- smoke-test first):
```bash
python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml --max_steps 100 --max_patients 3
python -m training.train_stage3_segmentation --config configs/stage3_ct_segmentation.yaml
```

**6. Validate Stage 3 against the Jordan external dataset** (never used in training):
```bash
python -m inference.validate_jordan_segmentation --config configs/stage3_ct_segmentation.yaml
```

**7. Get the official, citable Stage 3 Dice/IoU** (full-volume sliding-window inference over the entire synthetic validation split -- NOT the cheap patch-level number training logs periodically):
```bash
python -m inference.validate_synthetic_segmentation --config configs/stage3_ct_segmentation.yaml
```

**8. Visualize Stage 3 predictions** (per-patient CT / real mask / predicted mask panels, on both the synthetic validation split and Jordan):
```bash
python -m inference.visualize_predictions --config configs/stage3_ct_segmentation.yaml --source both --num_patients 5
```

**9. Generate the full Stage 3 report in one run** (training curves, internal + external Dice/IoU CSVs, optional literature comparison chart, and best-to-worst example visualizations for both sources -- everything steps 6-8 produce individually, plus the training curve, in one command after training finishes):
```bash
python -m inference.generate_full_report --config configs/stage3_ct_segmentation.yaml \
    --auto_threshold --use_largest_component \
    --comparison_label "Author et al. (Year)" --comparison_internal_dice 0.XX --comparison_external_dice 0.XX
```
`--comparison_*` args are optional and never fabricated by the script -- omit them (or the comparison chart is skipped) unless you have the real reported numbers to compare against. `--auto_threshold` searches for the best global threshold on the synthetic validation set only (never on Jordan) and reuses it for both reported metrics; `--use_largest_component` keeps only the largest connected component of each thresholded prediction. Both are optional, off by default (plain threshold=0.5, no filtering) if omitted.

Run the test suite (CPU-only, no GPU/real data needed):
```bash
pip install -r requirements.txt
python -m pytest tests/
```

## Repo structure

```
models/unet3d_regression.py           Stage 1 model: plain 3D regression U-Net
models/unet3d_segmentation.py         Stage 3 model: same U-Net topology, raw logit output (sigmoid applied by the caller, for autocast safety)
configs/stage1_regression.yaml        Stage 1 hyperparameters
configs/stage2_inference_brats_regression.yaml   Stage 2 settings
configs/stage3_ct_segmentation.yaml   Stage 3 hyperparameters + dataset paths (synthetic CT + Jordan)
training/train_stage1_regression.py   Stage 1 training loop (resumable)
training/train_stage3_segmentation.py Stage 3 training loop (resumable, Dice+BCE loss, synthetic CT only)
training/checkpoint.py                shared checkpoint save/load/resume
training/ema.py                       shared exponential moving average
inference/visualize_regression_val.py Stage 1 validation: PSNR/SSIM + comparison images
inference/run_stage2_brats_regression.py   Stage 2: generate the BraTS synthetic-CT dataset
inference/validate_jordan_segmentation.py  Stage 3 external validation against real Jordan CT
inference/validate_synthetic_segmentation.py  Stage 3 official full-volume Dice/IoU on the synthetic validation split (not the periodic patch-level training metric)
inference/visualize_predictions.py    Stage 3 per-patient prediction visualization (CT/real mask/predicted mask), both synthetic val + Jordan
inference/postprocessing.py           shared Stage 3 post-processing: largest-connected-component filtering, validation-set threshold search
inference/generate_full_report.py     runs everything above in one command after training finishes (curves + both Dice/IoU CSVs + comparison chart + example visualizations)
data/preprocessing.py                 shared HU/MRI normalization, resample, brain-mask, crop/pad, foreground-biased patch crop
data/loaders_synthrad.py              SynthRAD2023 dataset (Stage 1 training data)
data/loaders_brats.py                 BraTS2020 dataset (Stage 2 input data)
data/loaders_synthetic_ct.py          Stage 2 output dataset (Stage 3 training data)
data/loaders_jordan_ct.py             Jordan Hospital DICOM dataset (Stage 3 external validation only)
analysis/                             reusable figure/table-generation scripts (see below)
scripts/check_orientation_consistency.py   diagnostic: NIfTI orientation consistency check
tests/                                CPU-only test suite for the active pipeline
archive/                              the original wavelet-diffusion pipeline (superseded, not deleted)
PROJECT_NOTES.md                      full development narrative, decisions, and status log
```

### `analysis/`

Scripts that turn training/validation output into figures and tables,
reading real data (training logs, checkpoints) rather than hardcoded
numbers wherever practical:

- `plot_validation_psnr_curve.py` — validation PSNR vs. training step, from
  the training log CSV.
- `generate_val_metrics_table.py` — runs the checkpoint over the full
  validation split, writes a per-patient PSNR/SSIM/L1 CSV, and prints a
  summary table. The other two scripts below consume this CSV rather than
  recomputing it.
- `plot_psnr_ssim_distribution.py` — box plots of the PSNR/SSIM
  distributions with individual patient points overlaid.
- `plot_psnr_vs_ssim_scatter.py` — PSNR vs. SSIM per patient, color-coded
  by PSNR.

Every script's input/output paths are CLI arguments (`--config`,
`--log_file`, `--metrics_csv`, `--output`, ...) — no hardcoded paths.

## Data sources

- [SynthRAD2023](https://synthrad2023.grand-challenge.org/) brain cohort
  (paired MRI/CT) — Stage 1 training data.
- [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/) training set
  (T1 MRI + tumor segmentation) — Stage 2 input data.
- Jordan University Hospital CT + tumor mask (20 patients, DICOM) — Stage 3
  external validation only, never used in training. See PROJECT_NOTES.md's
  Stage 3 section for this dataset's format and known limitations.

See `PROJECT_NOTES.md`'s "Kaggle dataset paths" section for the exact confirmed
Kaggle input paths used during development.
