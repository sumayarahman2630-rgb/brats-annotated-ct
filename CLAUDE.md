# brats-annotated-ct — Project Memory

This file is the single source of truth for where this project stands. Read it
first in any new session (yours or a future Claude instance's) before touching
code. Keep it updated as work progresses — status, not just plan.

## The two goals (set 2026-07-15)

1. **Stage 1 — MRI→CT diffusion model, trained on the FULL SynthRAD2023 brain
   cohort.** Production-quality, not a toy: full dataset, architecture and
   training setup following the strongest open-source pattern for this exact
   problem (paired 3D image-to-image translation at full resolution).
2. **Stage 2 — Synthetic CT dataset generation from BraTS T1 MRI volumes**,
   using the Stage 1 model (even if still mid-training). Each output CT is
   paired with its source BraTS tumor mask under a clear ID convention. The
   generated dataset is itself the deliverable — this is the actual point of
   the project (annotated synthetic CT for tumor-region work, where real
   annotated CT is scarce).

## Kaggle dataset paths (as given by the user, 2026-07-15)

```
awsaf49            -> /kaggle/input/datasets/awsaf49            -> brats2020-training-data
fd7akxj65n5yjxwds  -> /kaggle/input/datasets/fd7akxj65n5yjxwds  -> synthrad-2023
```
These are the values that belong in `configs/stage1_synthrad.yaml` /
`data.synthrad_root` and in the Stage 2 script's `--brats_root` on Kaggle.
Exact per-patient folder layout inside each hasn't been inspected from this
machine (no Kaggle filesystem access here) — both loaders are written to
*discover* patient folders by pattern rather than assume a hardcoded nesting,
and both log what they found on first run so a bad path fails loudly instead
of silently loading zero patients.

## Architecture decision record

**Why wavelet-domain diffusion (cwdm-style), reimplemented from scratch:**
Full-resolution 3D diffusion (e.g. 256³) is not feasible on Kaggle GPU/time
budgets if done in voxel space. cwdm's trick — do a Haar DWT first (8
subbands, each at half resolution per axis, so 1/8 the voxel count spread
over 8 channels), run the diffusion model on that compressed representation,
then inverse-DWT the denoised result back — gets full-volume, artifact-free
3D translation at a fraction of the compute. This is the closest published
match to "paired MRI→CT at full resolution" and is why it's the primary
reference. Reimplemented here as plain separable 1D Haar low/high-pass +
downsample applied along D, H, W in sequence (orthogonal, exact inverse) —
this is textbook Haar wavelet math, not copied code. See
`models/wavelet_transform.py`.

**Why a plain conditional U-Net (no classifier-free guidance) for the
denoiser:** MRI→CT here is deterministic paired translation, not a
generative/multimodal task — there's one correct CT per MRI, unlike
text-to-image where guidance strength trades off diversity vs. fidelity. cwdm
and the SR3/Palette line of paired-translation diffusion models both just
concatenate the condition to the noisy target channel-wise and skip CFG. Kept
that simplification; noted as a possible extension, not needed for today's
goals.

**Why the domain-gap handling matters (read this before training Stage 1):**
SynthRAD2023 brain MR is *not* skull-stripped — it includes skull, scalp,
face. BraTS T1 volumes (Stage 2's input) *are* skull-stripped, registered to
the SRI24 atlas. If Stage 1 is trained naively on full-head SynthRAD pairs,
it learns to hallucinate skull/face structure from an MRI channel that, at
Stage-2 inference time, doesn't have any (BraTS input is all zeros outside
the brain). That's a real failure mode, not a hypothetical one.
Fix implemented: `data/preprocessing.py` uses the brain mask SynthRAD ships
per patient (`mask.nii.gz`) to zero out non-brain voxels in *both* the MR and
CT volumes during Stage 1 preprocessing when `data.match_brats_domain: true`
(default) in the config. This makes Stage 1's training distribution match
what Stage 2 will actually feed it. Also means Stage 1's synthetic CT will
have zeroed-out skull/scalp — correct for this project, since the deliverable
is tumor-region CT, not a full-head radiotherapy-planning CT.

**Model capacity:** base_channels=64, channel_mult=(1,2,4,4), 2 res-blocks per
level, attention at the two coarsest wavelet-domain resolutions, 32 groups —
same order of magnitude as cwdm's brain-translation config. All overridable
in `configs/stage1_synthrad.yaml`; nothing hardcoded in the model file.

**Diffusion:** standard DDPM, 1000 linear-schedule timesteps, epsilon
prediction, MSE loss averaged over the 8 wavelet subbands (equal weighting,
matches cwdm's default; weights are a config list so this is tunable later).
DDIM sampling (default 100 steps) for Stage 2 inference — ancestral 1000-step
sampling per BraTS volume would make a full-cohort run impractical.

## Repo layout

```
configs/stage1_synthrad.yaml   all hyperparameters — nothing hardcoded in code
data/
  preprocessing.py             HU clip/normalize, resample, brain-mask, patch/pad —
                                shared by both the SynthRAD loader and the BraTS loader
                                so Stage 2 inputs are normalized exactly like Stage 1 trained on
  loaders_synthrad.py          full SynthRAD2023 brain cohort Dataset
  loaders_brats.py             BraTS T1 + seg mask Dataset (discovery-based, Kaggle-path aware)
models/
  wavelet_transform.py         Haar 3D DWT / IDWT (exact inverse)
  unet3d.py                    3D conditional U-Net backbone (timestep + condition conditioning)
  stage1_mri2ct_ddpm.py        composes the above into the diffusion model (q_sample, p_losses, sampler)
training/
  ema.py                       exponential moving average of model weights
  checkpoint.py                save/find-latest/load — used by both training and inference for resumability
  train_stage1.py              training loop entry point
inference/
  run_stage2_brats.py          resumable per-patient Stage 2 generation entry point
```

## Resumability strategy (why it's built this way)

Checkpoints save: model weights, EMA weights, optimizer state, LR-scheduler
state, and the global step count — everything needed to resume training
*exactly*, not restart with cold optimizer/EMA state (which would spike the
loss). Saved every `training.checkpoint_interval` steps (not epochs — a full
SynthRAD epoch may be long), to `checkpoint.working_dir`, default
`/kaggle/working/checkpoints/stage1_synthrad/`. Old numbered checkpoints
beyond `training.keep_last_n_checkpoints` are pruned automatically so this
doesn't eat Kaggle's output-storage quota; a `ckpt_latest.pt` copy always
exists as the resume pointer.

On startup, `training/train_stage1.py` searches, in order: the working
checkpoint dir, then every directory listed in
`checkpoint.extra_resume_dirs` in the config — picks whichever checkpoint has
the highest step number across all of them, and resumes from there. This is
what makes both same-session-after-interruption resume and
fresh-session-from-a-downloaded-checkpoint resume work through the *same*
code path with no manual flag-flipping.

### What you (the human) need to do on Kaggle's side — this part is manual

1. Train normally. Checkpoints accumulate in `/kaggle/working/checkpoints/stage1_synthrad/`.
2. When your Kaggle session is about to end (time limit, or you're pausing
   for the day): click **Save Version** → **Save & Run All (Commit)**. This
   persists everything under `/kaggle/working/` as a Kaggle Output attached
   to that notebook version.
3. In your *next* Kaggle session (new notebook run, or after a limit reset):
   open the notebook, click **Add Input** → **Notebook Output Files** → pick
   the previous version you just saved. It'll mount at something like
   `/kaggle/input/<your-notebook-name>/checkpoints/stage1_synthrad/`.
4. Add that mounted path to `checkpoint.extra_resume_dirs` in
   `configs/stage1_synthrad.yaml` (or pass it as a CLI override — see the
   script's `--help`). Run `training/train_stage1.py` again — it will find
   the highest-step checkpoint in that read-only input path, load it, and
   keep training, writing *new* checkpoints to the fresh session's
   `/kaggle/working/checkpoints/...`.
5. Repeat step 2–4 each session. Each Kaggle Dataset/Output you create this
   way is a checkpoint of the whole project's progress, independent of
   Claude's own session limits.

Stage 2 (`inference/run_stage2_brats.py`) uses the same checkpoint directory
convention to load whatever Stage 1 checkpoint currently exists — it does not
require training to be finished, by design (goal 2 explicitly runs on a
"possibly still-training" model).

## Reference repos — what was studied, not copied

All four were fetched and read (architecture, training loop, data
conventions) via GitHub's raw API; nothing below is copy-pasted — reimplemented
independently for this codebase, for academic-defense originality.

1. **pfriedri/cwdm** — wavelet-domain diffusion for paired 3D translation.
   Source of: DWT/IDWT-in-the-loop design, channel_mult/num_groups scale,
   AdamW lr=1e-4, ema=0.9999, batch_size=1 (full 3D volume, no patching in
   their setup), step-numbered checkpoint naming, equal-weighted per-subband
   MSE loss.
2. **mazurowski-lab/segmentation-guided-diffusion** — reference for clean
   HF-Diffusers-style training-loop structure and mask-conditioning
   mechanism. Mask-conditioning itself isn't used in today's two goals (no
   segmentation-guided generation stage yet) but the pattern is noted here
   for whenever that's the next stage.
3. **mobaidoctor/med-ddpm** — BraTS whole-head synthesis from masks, and
   critically, their evaluation practice: they report Dice on *downstream
   tumor segmentation* of synthetic volumes, not just image-similarity
   metrics (SSIM/PSNR). Not implemented yet (out of scope for today's two
   goals) but flagged here as the right next step for validating the
   Stage-2 dataset's actual usefulness — see "Not yet done" below.
4. **SynthRAD2023/preprocessing** — official challenge preprocessing.
   `data/preprocessing.py` follows their conventions directly: resample to
   1×1×1mm (brain), CT low-value clip at -1024, mask-based background
   fill (CT→-1000, MR→0), bounding-box crop with margin. Full Elastix
   registration itself is assumed already applied by SynthRAD2023's own
   released, pre-registered data (the Kaggle copy is the challenge's
   post-registration release) — this codebase does not re-run Elastix.

## Status

| Piece | Status |
|---|---|
| CLAUDE.md | done |
| data/preprocessing.py | in progress |
| data/loaders_synthrad.py | not started |
| models/wavelet_transform.py, unet3d.py, stage1_mri2ct_ddpm.py | not started |
| configs/stage1_synthrad.yaml | not started |
| training/ema.py, checkpoint.py, train_stage1.py | not started |
| data/loaders_brats.py | not started |
| inference/run_stage2_brats.py | not started |
| git remote | connected to https://github.com/sumayarahman2630-rgb/brats-annotated-ct.git, not yet pushed |

## Not yet done / explicitly out of scope for today

- Dice-on-downstream-segmentation evaluation (med-ddpm's practice) — worth
  adding once Stage 2 has produced enough volumes to evaluate.
- Classifier-free guidance / conditioning dropout for the denoiser.
- Pelvis region (SynthRAD2023 covers brain + pelvis; only brain is relevant
  to the BraTS pairing goal).
- Elastix re-registration (relying on SynthRAD2023's pre-registered release).

## Resume instructions for a fresh session

Read the Status table above, pick up the first "not started" row, and check
the corresponding file for a `# TODO` or missing-function stub before writing
new code — don't assume a partially-listed file is empty. Run
`git log --oneline` to see what's actually been committed vs. what this file
claims (this file can lag a crash). Each row in the task breakdown in the
original prompt was designed to be one commit — check `git log` against the
Status table to find the exact resume point.
