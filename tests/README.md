# Tests

CPU-only, no GPU or real dataset access required -- all tests build small
synthetic data fixtures. Formalizes the ad-hoc verification performed
throughout this project's development (see DEVELOPMENT_LOG.md for the narrative
version of each finding).

Covers the **active regression pipeline** (the one that actually reached a
good result: 28.21 dB foreground PSNR) plus the shared preprocessing/data
modules both pipelines use. The original wavelet-diffusion pipeline's own
tests moved to `archive/tests/` alongside the code they test -- see
`archive/README.md` -- they are frozen, not part of this suite, and not
run by the commands below.

Run everything:
```
pip install -r requirements.txt
python -m pytest tests/ -v
```

Run only the fast unit tests (skip subprocess-based integration tests,
~1s vs ~30-60s):
```
python -m pytest tests/ -v -m "not slow"
```

## What's covered

- `test_preprocessing.py` -- HU normalize/denormalize round-trips exactly;
  the specific numerical checks from the round-8 PSNR audit that ruled out
  a normalization sign-flip bug; mask/crop/pad geometry. Shared by both
  pipelines.
- `test_discovery.py` -- SynthRAD/BraTS patient discovery: content-based
  validation (not a name denylist), t1/t1ce disambiguation, ID format
  independence. Shared by both pipelines.
- `test_train_stage1_regression_resume.py` -- the regression pipeline's own
  bug fixes, each with a direct test: patient-level train/val split has no
  leakage; brain-domain masking actually zeros non-brain voxels; the
  center-crop-to-patch fix for the real Kaggle validation OOM only ever
  feeds the model patch-sized input (not the full volume it OOM'd on); the
  real train -> checkpoint -> resume cycle *(slow)*.
- `test_stage2_regression_crop_roundtrip.py` *(slow)* -- trains a tiny real
  regression checkpoint, runs the real `run_stage2_brats_regression.py` end
  to end (including its sliding-window inference) against a fake BraTS
  patient, and checks the synthetic CT lands correctly in the full BraTS
  grid.

## What's NOT covered here (needs a GPU / real data)

- Actual training convergence/loss trajectory on real SynthRAD data (though
  the regression pipeline's real result -- 28.21 dB foreground PSNR at step
  20000 -- is documented in DEVELOPMENT_LOG.md and the main README).
- Real memory/OOM behavior at production model/volume size on a real GPU
  beyond what's already been hit and fixed once (see DEVELOPMENT_LOG.md).
- The image-orientation-consistency question raised in round 8 -- checked
  and ruled out on real data, see DEVELOPMENT_LOG.md and
  `scripts/check_orientation_consistency.py`.
