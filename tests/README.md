# Tests

CPU-only, no GPU or real dataset access required -- all tests build small
synthetic data fixtures. Formalizes the ad-hoc verification performed
throughout this project's development (see CLAUDE.md for the narrative
version of each finding).

Run everything:
```
pip install -r requirements.txt
python -m pytest tests/ -v
```

Run only the fast unit tests (skip subprocess-based integration tests,
~1s vs ~30s):
```
python -m pytest tests/ -v -m "not slow"
```

## What's covered

- `test_wavelet_transform.py` -- Haar DWT/IDWT is an exact inverse.
- `test_preprocessing.py` -- HU normalize/denormalize round-trips exactly;
  the specific numerical checks from the round-8 PSNR audit that ruled out
  a normalization sign-flip bug; mask/crop/pad geometry.
- `test_unet3d.py` -- shape correctness, the GroupNorm-divisibility
  regression from round 4, activation checkpointing is numerically
  transparent (with and without, and the per-resolution-scoped variant).
- `test_discovery.py` -- SynthRAD/BraTS patient discovery: content-based
  validation (not a name denylist), t1/t1ce disambiguation, ID format
  independence.
- `test_checkpoint_resume.py` *(slow)* -- the resume-time config-override
  bugs (`ema.decay`, `subband_loss_weights`) stay fixed; runs the real
  training script as a subprocess since the bug lived in the interaction
  between `train_stage1.py`'s resume logic and these objects, not in
  either object alone.
- `test_stage2_crop_roundtrip.py` *(slow)* -- the brain-crop-then-paste-back
  geometry fix for Stage 2: trains a tiny real checkpoint, runs the real
  `run_stage2_brats.py` end to end, and checks the output lands correctly
  in the full BraTS grid.

## What's NOT covered here (needs a GPU / real data)

- Actual training convergence/loss trajectory on real SynthRAD data.
- Real memory/OOM behavior at production model size on a real GPU.
- The image-orientation-consistency question raised in round 8 --
  see `scripts/check_orientation_consistency.py`, which needs real
  SynthRAD and BraTS files to run.
