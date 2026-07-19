# brats-annotated-ct — Development Log

This file is the single source of truth for where this project stands. Read it
first in any new development session before touching code. Keep it updated as
work progresses — status, not just plan.

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

## Kaggle dataset paths (FINAL, confirmed 2026-07-15 — do not re-derive)

**SynthRAD2023** — per-patient folders directly under `Task1/brain`:
```
/kaggle/input/datasets/fd7akxj65n5yjxwds/synthrad-2023/Task1/brain/<patient_id>/
    ct.nii, mask.nii, mr.nii
```
`<patient_id>` looks like `1BA001` — not numeric-only, so patient ID is taken
as the folder's basename with no ID-format assumption (see
`discover_synthrad_patients` in `data/loaders_synthrad.py`). This exact path
is now hardcoded as `data.synthrad_root` in `configs/stage1_synthrad.yaml`,
with `data.region: null` since the path is already scoped to brain (no
filtering needed).

**BraTS2020** — use `awsaf49/brats20-dataset-training-validation`, **not**
`awsaf49/brats2020-training-data`. The first is the original per-patient
NIfTI release; the second turned out to be a pre-sliced 2D `.h5` dataset
(built for slice-wise segmentation training), which doesn't work for this
project's full-volume 3D pipeline — `data/loaders_brats.py` was always
written for full NIfTI volumes, so nothing there needed to change once the
correct dataset was identified, only the path. Use **only** the
TrainingData half; ValidationData has no `seg.nii`, so it can't be paired:
```
/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/<patient_id>/
    <patient_id>_t1.nii   -> MRI input (ignore _t1ce, _t2, _flair)
    <patient_id>_seg.nii  -> tumor mask
```
`<patient_id>` looks like `BraTS20_Training_083` — `discover_brats_patients`
extracts the ID as everything before `_t1`/`_seg` in the filename, so this
works regardless of ID format. This exact path is now hardcoded as
`brats_root` in `configs/stage2_inference_brats.yaml`.

Both loaders were verified against synthetic directories built to mirror
these exact layouts and filenames (bare `.nii`, both ID conventions) — see
Status below. Both still discover patients by pattern rather than a
hardcoded per-file assumption beyond the confirmed root paths, and both log
what they found on first run so a wrong path fails loudly instead of
silently loading zero patients.

**Confirmed on real Kaggle data (2026-07-15):** 180 SynthRAD2023 brain
patients, 369 BraTS2020 training patients (1 missing `seg.nii`). The
missing-mask BraTS patient is handled: `discover_brats_patients` returns it
with `seg_path=None`, and `inference/run_stage2_brats.py` skips it outright
(logged as `skipped_no_mask` in `manifest.csv`, not counted as failed) rather
than generating a synthetic CT with nothing to pair it to — the deliverable
is the CT+mask pair, so an unpaired CT isn't useful output. `data/
loaders_synthrad.py`'s folder-content validation (see the commit that added
it) also means a bad SynthRAD patient folder is silently excluded the same
way, not just BraTS.

**Stage 1 smoke test (do this before a full run):** `training/train_stage1.py`
takes `--max_steps` and `--max_patients` to run a short GPU/OOM check without
touching the config file or the full 180-patient cohort:
```
python -m training.train_stage1 --config configs/stage1_synthrad.yaml --max_steps 100 --max_patients 3
```
This trains on only the first 3 discovered patients for 100 steps, and
automatically clamps `checkpoint_interval`/`val_interval` down to fit inside
that step budget so the run still exercises a real checkpoint save and
validation pass, not just forward/backward passes. Watch `nvidia-smi` (or
Kaggle's GPU memory panel) during this run for OOM headroom before starting
the real `total_steps: 200000` run on the full cohort. Both flags are
optional and independent -- `--max_steps 50` alone, or `--max_patients 2`
alone, both work. This exact smoke test is what surfaced the attention OOM
bug documented under "Known bugs fixed" below — now fixed, but worth
rerunning after any future change to `model.channel_mult` or
`model.attention_resolutions` specifically, since that's the class of bug
this smoke test is best at catching before it costs a full training run.

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
level, self-attention at the bottleneck only (`attention_resolutions: [8]`
for `channel_mult`'s 4 levels — **not** `[2, 4]` or even `[4, 8]`, both of
which OOM'd on a real T4; see "Known bugs fixed" below), 32 groups,
activation checkpointing on every `ResBlock3D` (`use_checkpoint: true`,
also see "Known bugs fixed" — this is what addresses the decoder GroupNorm
OOM that showed up once attention was fixed). All overridable in
`configs/stage1_synthrad.yaml`; nothing hardcoded in the model file.

**Diffusion:** standard DDPM, 1000 linear-schedule timesteps, epsilon
prediction, MSE loss averaged over the 8 wavelet subbands (equal weighting,
matches cwdm's default; weights are a config list so this is tunable later).
DDIM sampling (default 100 steps) for Stage 2 inference — ancestral 1000-step
sampling per BraTS volume would make a full-cohort run impractical.

## 2026-07-16 (round 8) — PSNR root-cause audit: ruled out normalization bug

Real data on the SynthRAD validation comparison: background exactly -1000.0
HU (std=0), foreground -104 to -243 HU. Synthetic (step 9000 checkpoint,
raw weights): background 884.7-908.5 HU, foreground 1062.7-1080.4 HU --
both positive, both close together, nothing like the real bimodal
distribution. The working hypothesis: is this a normalization/denormalization
bug (sign flip, wrong clip range applied at inference vs. training) rather
than "just needs more training"?

**Investigated and ruled out, with evidence, not just code reading:**
1. Grepped every use of `ct_clip_range` across the codebase (`loaders_
   synthrad.py`, `compare_synthrad_val.py`, `run_stage2_brats.py`) --
   all read the same `data.ct_clip_range` config key; `normalize_ct`/
   `denormalize_ct` are defined exactly once, in `data/preprocessing.py`.
   No divergent value anywhere.
2. Re-verified `normalize_ct`/`denormalize_ct` are exact inverses,
   algebraically and numerically (6 HU values round-tripped exactly).
3. The REAL data's background reading exactly -1000.0 with zero std is
   itself strong evidence the round-trip (mask -> normalize -> denormalize)
   works correctly -- this is the actual code path being exercised, and it
   produces the exact right answer for ground truth.
4. Explicitly tested the sign-flip hypothesis with real numbers: if the
   normalized value's sign were flipped somewhere, real background (-1000
   HU, normalized -1.0) would decode to +3000 HU, and real foreground
   (-104 to -243 HU) would decode to +2104 to +2243 HU. Observed synthetic
   values (884.7-1080.4) don't match this prediction at all -- rules out a
   clean sign-flip bug.
5. **What actually explains the numbers:** converting the observed
   synthetic HU values back to normalized space gives approximately -0.058
   to +0.040 -- clustered tightly around *normalized zero*, not around any
   HU-space landmark. `denormalize_ct(0.0) = 1000.0 HU` exactly -- matching
   the observed synthetic cluster. This means the model's raw output is
   landing near the center of the valid range with almost no differentiation
   between what should be starkly different regions (background -1.0 vs.
   foreground -0.55ish, normalized) -- consistent with an undertrained
   model producing weakly-informative, near-prior-centered output, NOT a
   parameter mismatch. The reason this "centered but wrong" output *looks*
   positive in HU space specifically is that the clip range (-1000, 3000)
   is asymmetric -- its midpoint is +1000 HU, not 0 -- so a "hasn't learned
   much yet" output happens to land in positive-HU territory by construction
   of the range choice, not because of a sign error.

**Conclusion: no normalization bug found. This is consistent with genuine
undertraining**, not a quick fix. Honest, not the hoped-for outcome, but
better than chasing a bug that (as far as this audit can tell without GPU
access) doesn't exist in the inspectable code. Reinforces the priority on
the 2D pipeline (round 8+, see below) as the faster path to a converged,
demonstrable model, since 2D slices are far cheaper per step than full 3D
volumes.

## Known bugs fixed

**2026-07-16 (round 6) — Stage 2 sampled pure noise despite val_loss=0.01293
being good.** The first real inference run (5 BraTS patients, ddim_steps=30)
produced static/noise for the synthetic CT while the tumor mask overlay was
correctly placed -- meaning the mask-copy path was fine but the model's
actual output was garbage, which didn't match the good validation loss from
training and needed fast root-causing with ~4 hours left before the
deadline.

**Root cause: EMA decay was miscalibrated for a short run.** `training.
ema_decay: 0.9999` gives the EMA weights a ~10,000-step time constant
(1/(1-0.9999)) -- roughly how long it takes the EMA to substantially forget
its starting point. `EMA(model, ...)` is constructed right after the
model's *random* initialization, so that starting point is pure noise.
Tonight's run only did 5,500 steps (round 5's deadline-night total_steps),
so `0.9999^5500 = 57.7%` of the EMA checkpoint's weights were **still the
original random initialization** -- confirmed by direct calculation, no
GPU needed. Meanwhile `training/train_stage1.py`'s validation loss is
computed on the *raw* (non-EMA) model weights (`quick_validation_loss`
calls the live model directly), which had genuinely learned plenty in
5,500 steps -- hence good val_loss, garbage EMA-weight samples. Two
different sets of weights being implicitly compared is exactly why this
looked contradictory.

**Fix:** `configs/stage2_inference_brats.yaml`'s `use_ema` default changed
`true -> false` for this checkpoint specifically -- `training/checkpoint.
py`'s `load_checkpoint` always loads the raw trained weights into `model`
first (`model.load_state_dict(payload["model_state"])`); `run_stage2_brats.
py` only overwrites them with the contaminated EMA weights if `use_ema` is
true, so setting it false uses exactly the weights val_loss already
validated as good. No retraining needed -- this is purely an inference-time
weight-selection fix. `--no_ema` also works as a one-off CLI override
without editing the config. Revisit `use_ema: true` only if this checkpoint
is ever continued for training well beyond ema_decay's ~10,000-step time
constant (tens of thousands more steps), at which point EMA would actually
have converged and be worth using again (EMA sampling is normally *better*
than raw weights once it's had time to converge -- this bug is specific to
short runs, not a reason to distrust EMA in general).

**2026-07-16 — BraTS Stage 2 input was missing Stage 1's brain-crop step
(found before Stage 1 training even finished, by re-reading both
preprocessing paths side by side rather than waiting to see bad output).**
`SynthRADBrainDataset._load_and_preprocess` (Stage 1 training) does
resample -> mask+**crop to brain bbox + margin** -> normalize -> pad.
`BraTSVolumeDataset.__getitem__` (Stage 2 inference) was doing resample ->
normalize -> pad -- the crop step was simply missing. BraTS T1 is already
skull-stripped so the *masking* half is implicit (background already 0),
but nothing was cropping the volume down to the brain region, so Stage 2
was feeding the model the full ~240x240x155 BraTS grid (brain filling
~40-50% of the frame) instead of the tightly-cropped volumes (brain filling
~80-90% of the frame) Stage 1 actually trained on. Doesn't crash (fully
convolutional network, any input size "works") -- it's a silent
distribution shift that would have produced degraded/garbage-looking
output with no error to point at the cause.

**Fix:** `BraTSVolumeDataset` now computes the same `bounding_box(...,
margin=crop_margin)` + `crop_to_box(...)` Stage 1 uses (crop_margin read
from the same Stage 1 config key, so the two paths can't drift out of sync
independently), before `normalize_mri`. Returns `full_shape` (the
resampled, pre-crop T1 grid -- what the tumor mask and final output canvas
need to match) and `crop_box` alongside the cropped+padded MRI tensor.
`run_stage2_brats.py` now reverses this correctly: crops the model's output
back to the pre-pad cropped shape, then `place_in_full_canvas()` pastes it
into a full-size canvas at `crop_box`'s location (everything outside filled
with the normalized background value, so it denormalizes to exactly
CT_BACKGROUND_HU = -1000, matching Stage 1's convention) -- rather than the
old code's now-incorrect "crop the output back to the full T1 shape"
(there was nothing to crop; the input was never cropped in the first
place).

**Verified without a GPU:** built a synthetic BraTS patient with a small,
precisely-known "brain" region inside a much larger frame (brain occupying
~0.4% of the full volume, growing to ~17% after cropping -- deliberately
exaggerated vs. real BraTS's ~40-50% to make any bug in the crop/paste-back
logic unmistakable), ran it through the real `run_stage2_brats.py`
end-to-end against an actual trained (tiny) checkpoint, and confirmed: output
CT shape matches the full BraTS grid exactly; every voxel outside the crop
box is exactly -1000 HU; voxels inside vary (real generated content); the
tumor mask's 72 nonzero voxels land at exactly their original coordinates.
Also reran the standard (brain-fills-most-of-frame) regression case to
confirm no behavior change there. This is a pure Stage 2 fix -- nothing
about Stage 1's training data or the checkpoints already produced needed
to change.

**2026-07-15 — attention OOM (594 GiB allocation) during the first real
Kaggle smoke test.** `model.attention_resolutions` in the config means
"apply self-attention at U-Net levels whose downsample factor from the
wavelet-domain input (2^level) is in this list." For `channel_mult`'s 4
levels the per-level factors are `[1, 2, 4, 8]`, so the two *coarsest*
levels are factors `[4, 8]` — but the config default was `[2, 4]`, which
actually hits levels 1 and 2 (the two *middle* levels). On a real brain
volume (~96×80×72 at the wavelet-domain level-0 resolution), level 1 is
still 48×40×36 = 69,120 spatial positions; a dense O(N²) attention matrix
at that N is ~71 GiB for the forward pass alone, ballooning past 500 GiB
once backward-pass buffers are counted — that's the 594.14 GiB CUDA OOM.
Root-caused by simulating the level/resolution_factor mapping with the
real volume size (no GPU needed, see the commit) rather than guessing from
the stack trace. **Fix:** `attention_resolutions: [4, 8]` (models/unet3d.py
was never structurally wrong — only the config value was). Also added a
permanent runtime guard in `SelfAttention3D.forward`: if the flattened
spatial sequence length N exceeds 24³ = 13,824, it logs a warning with the
actual shape and estimated GiB cost the first time that module runs, so a
future `channel_mult` change without a matching `attention_resolutions`
update fails loudly and early instead of as an opaque OOM stack trace deep
inside `scaled_dot_product_attention`.

Dual-GPU note: 2×T4 GPUs are available for this — **not relevant to
this particular bug**. A 594 GiB single-tensor allocation attempt isn't a
"not enough VRAM" problem that more GPUs fixes; `train_stage1.py` has no
multi-GPU/distributed logic at all (single-device only), so a second GPU
sitting idle wouldn't have changed anything here. Once training is
otherwise healthy, the 2×T4 setup *would* be usable for `torch.nn.
DataParallel`/DDP to run batch_size=2 (one full volume per GPU) for better
throughput — not implemented, flagged as a possible future addition, not
needed for the two goals as scoped.

**2026-07-15 (same day, round 2) — `[4, 8]` still OOM'd.** The `[4, 8]` fix
above was necessary but not sufficient: on the actual full-size real brain
volume, level 4 alone (N=24,960) needed ~9.3 GiB for its dense attention
matrix, on top of ~7.85 GiB already used by the higher-resolution
conv/res-blocks before it — total exceeded the T4's 14.56 GiB. Two things
were tried:
1. **Force the flash/memory-efficient SDPA backend** (`models/unet3d.py`,
   `_preferred_attention_backend()` / `SelfAttention3D._attend()`): these
   backends compute attention via tiling instead of materializing the full
   N×N score matrix, so memory should scale ~O(N) instead of O(N²). Wraps
   `F.scaled_dot_product_attention` in `torch.nn.attention.sdpa_kernel`
   (falling back to the older `torch.backends.cuda.sdp_kernel` API on
   pre-2.3ish torch), requesting FLASH_ATTENTION/EFFICIENT_ATTENTION only
   (math excluded). If neither is usable for the actual shape/dtype/GPU
   (raises `RuntimeError`), it logs a warning once and retries with the
   default backend selection so training doesn't hard-crash either way.
   Also switched q/k/v to `.contiguous()` after the transpose, since
   fused kernels are stride-sensitive and non-contiguous tensors alone can
   force a fallback to math. **This could not be verified on this
   machine — no CUDA GPU available here.** Whether it actually keeps
   `[4, 8]` under budget on a real T4 is unconfirmed; the warning log
   will say plainly if it fell back.
2. **Reduce `attention_resolutions` to `[8]` (bottleneck only)** — this is
   the change actually shipped as the new default, because it's a certain
   fix verifiable by arithmetic alone (bottleneck N is only ~3,000 on a
   real volume, trivial regardless of which SDPA backend ends up being
   used) rather than something that depends on unconfirmed GPU/driver/torch
   behavior. Quality tradeoff: one fewer attention level than the original
   two-coarsest-levels design intent — likely small, not measured.

**To do when GPU time allows:** rerun the smoke test with `attention_
resolutions: [4, 8]` restored in the config. If the backend-forcing change
keeps it under the T4's budget (check the log for the fallback warning —
its absence means an efficient backup ran), keep `[4, 8]` for the real
training run since it's closer to the original design intent. If it still
OOMs or the fallback warning fires, stay on `[8]`.

**2026-07-15 (round 3) — OOM moved to the decoder, inside `group_norm`.**
With attention fixed, the next smoke test OOM'd elsewhere: 1.14 GiB
requested inside a `group_norm` call in a decoder block, with 13.55/14.56
GiB already in use. Not an algorithm bug this time -- classic U-Net decoder
memory pressure: skip-connection concatenation roughly doubles channel
count right before the highest-resolution decoder blocks, and mixed
precision (`training.amp`, already on by default since the original
training loop) doesn't fully cover it, because autocast keeps GroupNorm in
fp32 for numerical stability -- exactly where this OOM'd. Two standard
fixes, requested in this order and both applied since neither alone was
confirmed sufficient without a GPU to test on:
1. Confirmed AMP was already correctly wired (it's not new) -- nothing to
   fix there, just verified `torch.amp.autocast`/`GradScaler` are active
   by default and correctly wrap every forward/backward in `train_stage1.py`.
2. **Added activation checkpointing** (`torch.utils.checkpoint`) to
   `ResBlock3D` -- recomputes each block's forward during backward instead
   of retaining every intermediate (including the fp32 GroupNorm ones AMP
   can't shrink). New `model.use_checkpoint` config flag (default `true`),
   threaded through `ResAttnBlock` and `UNet3D` to every encoder, bottleneck,
   and decoder `ResBlock3D` -- applied network-wide rather than
   decoder-only as first suggested, since it's one flag instead of two and
   strictly saves more memory with no downside beyond a bit more recompute
   time. Only active during `model.train()`; `sample()`/`eval()` skip it
   (nothing to save memory on without a backward pass).

**Verified without a GPU:** checkpointed vs. non-checkpointed forward AND
backward produce bit-identical output and gradients (max diff `0.0` across
all parameters, same seed) -- checkpointing is mathematically a no-op,
purely a memory/compute-time trade, confirmed correct on CPU. Full
train→resume→Stage 2 regression suite still passes identically with
`use_checkpoint: true` wired through the real config.

**NOT verified (no CUDA here):** the actual GiB savings on a real T4, and
specifically the interaction between activation checkpointing and
`autocast` under real mixed precision -- PyTorch's non-reentrant checkpoint
(`use_reentrant=False`, used here) is documented to correctly save/restore
the autocast context across the recompute, so this should work, but "should"
isn't "confirmed." **Run the smoke test again on Kaggle** with no command
change needed (`use_checkpoint: true` is now the config default) --
`python -m training.train_stage1 --config configs/stage1_synthrad.yaml --max_steps 100 --max_patients 3`.
If it still OOMs at the same GroupNorm call, the next lever (not yet tried)
is `training.batch_size`/`grad_accum_steps` interaction or lowering
`model.base_channels` -- deliberately last, per the established
priority order, since those reduce model capacity/quality rather than
just trading compute for memory.

**2026-07-15 (round 4) — checkpointing confirmed working (50/100-step smoke
test, no crash, deliberately interrupted to check `nvidia-smi`), now
optimizing for speed under a real deadline.** Two things came out of this:

1. **Latent GroupNorm bug found while sizing a smaller model.** `ResBlock3D.
   out_norm` and `SelfAttention3D.norm` built `nn.GroupNorm(min(num_groups,
   channels), channels)` directly -- only valid when `channels` happens to be
   divisible by `num_groups` (true for base_channels=64's `[64,128,256,256]`,
   NOT guaranteed otherwise: base_channels=48 gives `[48,96,192,192]`, and 56
   gives `[56,112,224,224]`, neither evenly divisible by 32). Would have
   crashed immediately on `base_channels` values `_norm_act`'s already-correct
   fallback logic wasn't being used for. Fixed by extracting that fallback
   into `_safe_num_groups()` and using it in all three GroupNorm call sites,
   not just the one `_norm_act` covered. Caught by actually trying
   `base_channels=48` locally before recommending it, not by inspection.

2. **Dual-GPU (2×T4, confirmed via `nvidia-smi`) is not being pursued right
   now.** Recommendation: stay single-GPU. Reasoning, not just caution --
   `DataParallel`/DDP requires multi-process launch, a distributed sampler,
   correct rank-0-only checkpoint saving, and careful EMA handling across
   ranks; none of this is implementable *and* verifiable from here (no CUDA
   GPU on this machine, and DDP bugs are exactly the class of thing that's
   hard to debug blind). Under a real deadline, the risk of losing hours to a
   distributed-training bug outweighs the throughput gain, especially since
   single-GPU levers (below) are lower-risk and available immediately. Revisit
   DDP later only if single-GPU throughput is still the bottleneck once
   training is otherwise stable and proven correct.

3. **New default config trades some capacity for speed:**
   `model.base_channels: 64 -> 48` (69.7M -> 39.2M params, ~56% of original)
   and `model.use_checkpoint: true -> false` (removes checkpointing's ~20-30%
   recompute overhead). Sized against the one real data point available: at
   base_channels=64 with checkpointing OFF, the decoder OOM'd short by exactly
   1.14 GiB. Channel count scales activation memory ~linearly, so the 25% cut
   from 64->48 should clear that gap with real margin (~1.14 GiB needed vs. an
   estimated >2 GiB freed) -- reasoned from the crash's own numbers, not a
   guess. **Guaranteed-safe fallback if this still OOMs:** set
   `use_checkpoint: true` back on at base_channels=48 -- strictly safer than
   the already-confirmed-working base_channels=64 + checkpointing=true
   combination, since fewer channels can only use less memory, never more.
   This is the lowest-risk unblock available if the speed-priority config
   somehow isn't enough.
4. **Added per-component timing instrumentation** to `train_stage1.py`
   (`data_sec`/`fwd_sec`/`bwd_sec`/`opt_sec`, `torch.cuda.synchronize()`-gated
   so the numbers are real, not just CPU-enqueue time) -- logged every step to
   both the console and the CSV log file. This exists because the 18.3s/step
   figure couldn't be broken down from here (no GPU) -- rather than guess at a
   checkpointing-vs-data-loading split, the next real Kaggle run now produces
   the actual breakdown directly. One thing worth knowing before reading it:
   `training.grad_accum_steps: 4` means each logged "step" already bundles 4
   sequential micro-batches (data+forward+backward each), so the reported
   `data_sec`/`fwd_sec`/`bwd_sec` are *sums across all 4*, not one micro-batch
   -- divide by 4 for a per-microbatch figure. Lowering `grad_accum_steps`
   (cwdm's own reference config uses no accumulation at all, i.e.
   effectively 1) would proportionally cut wall-clock time to reach a given
   `total_steps`, but also lowers the effective batch size (4 -> fewer),
   which is a real training-dynamics tradeoff -- left untouched this round
   since the focus was specifically on `use_checkpoint` and
   `base_channels`, not `grad_accum_steps`; flagged here as an available
   extra lever, not applied yet.

**Next smoke test command (no change from before):**
```
python -m training.train_stage1 --config configs/stage1_synthrad.yaml --max_steps 100 --max_patients 3
```
Watch the console (or `training.log_file`'s new `data_sec`/`fwd_sec`/
`bwd_sec`/`opt_sec` columns) for the real time breakdown, and `nvidia-smi`
for peak memory with checkpointing off. If it OOMs, flip `use_checkpoint`
back to `true` (guaranteed fix, see above) before touching anything else.

## Repo layout

```
configs/
  stage1_synthrad.yaml         all Stage 1 (3D) hyperparameters — nothing hardcoded in code
  stage1_synthrad_2d.yaml      Stage 1 (2D pipeline) hyperparameters -- separate checkpoint dir, never collides with 3D
  stage2_inference_brats.yaml  Stage 2 paths/settings (brats_root, output_dir, etc.) — CLI flags override for one-off runs
data/
  preprocessing.py             HU clip/normalize, resample, brain-mask, patch/pad (N-dimensional --
                                works for both 3D volumes and 2D slices) — shared by every loader so
                                Stage 2 inputs are normalized exactly like Stage 1 trained on
  loaders_synthrad.py          full SynthRAD2023 brain cohort Dataset (3D volumes)
  loaders_synthrad_2d.py       2D axial-slice Dataset, wraps loaders_synthrad.py for preprocessing reuse
  loaders_brats.py             BraTS T1 + seg mask Dataset (discovery-based, Kaggle-path aware)
models/
  wavelet_transform.py         Haar 3D DWT / IDWT (exact inverse) -- 3D pipeline only
  unet3d.py                    3D conditional U-Net backbone (wavelet-domain)
  stage1_mri2ct_ddpm.py        composes the above into the 3D diffusion model
  unet2d.py                    2D conditional U-Net backbone (pixel-space, no wavelet transform)
  stage1_mri2ct_ddpm_2d.py     composes the above into the 2D diffusion model
                                -- deliberately NO shared code between the 3D and 2D model files, so a
                                bug/dead-end in one pipeline can never affect the other
training/
  ema.py                       exponential moving average of model weights
  checkpoint.py                save/find-latest/load — used by both training and inference for resumability
  train_stage1.py              3D training loop entry point
  train_stage1_2d.py           2D training loop entry point (same resumability design, separate script)
inference/
  run_stage2_brats.py          resumable per-patient Stage 2 generation entry point; writes manifest.csv
                                (per-patient provenance), metadata.json, and a populated README.md dataset
                                card into output_dir after every run
  compare_synthrad_val.py      diagnostic: real vs. synthetic CT + PSNR/SSIM on the SynthRAD validation
                                split, raw weights only (no EMA object constructed at all)
scripts/
  check_orientation_consistency.py  diagnostic: compares NIfTI direction matrices between a real SynthRAD
                                     and a real BraTS file -- an unverified risk flagged in round 8, see
                                     "Not yet done" below
tests/                          formal CPU-only test suite (43 tests) -- see tests/README.md
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
   any single working session's time limit.

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

## Status (last updated 2026-07-16, end of an extended overnight development session)

| Piece | Status |
|---|---|
| 3D pipeline (all files) | done, trained on real Kaggle data to step 9000 |
| 2D pipeline (all files) | done, built + CPU-tested tonight, **never run on a real GPU or real data** |
| Stage 2 (BraTS generation) | done, enriched with metadata.json + auto-generated README.md dataset card |
| `tests/` (43 tests) | done, all passing, CPU-only |
| `scripts/check_orientation_consistency.py` | done, **never run against real data** (see round 8 below) |
| git remote | https://github.com/sumayarahman2630-rgb/brats-annotated-ct.git (branch: main), all work pushed |

**Real Kaggle 3D checkpoint, as of the last status update tonight:**
step 9000, `configs/stage1_synthrad.yaml`, `/kaggle/working/checkpoints/stage1_synthrad/`.
SynthRAD validation comparison (raw weights, not EMA — see round 6):
foreground-only PSNR ~9 dB, SSIM negative. Root-caused in round 8 (below) as
genuine undertraining, not a bug — no quick fix exists in the code as
written. **Do not trust this checkpoint's visual output as a finished
deliverable** — it's a real, resumable, correctly-engineered checkpoint at
an early point in training, not a converged model.

## 2026-07-16 — extended overnight development session (rounds 6-9)

With the deadline at risk of being missed, development continued through
the night against a priority list: (1) hunt for a bug behind the bad
PSNR, (2) verify BraTS/SynthRAD preprocessing consistency and build a
real test suite, (3) build a completely separate 2D pipeline as a faster
path to actual convergence, (4) turn Stage 2 into a properly documented,
reusable dataset generator, and (5) leave this file in a state that lets
a fresh start pick up immediately. Small logical commits throughout,
never touching the working 3D checkpoint/pipeline. What actually
happened, honestly:

**Round 8 — PSNR root-cause audit (Priority 1). Conclusion: no bug found.**
Exhaustively checked: `ct_clip_range` is read from the same config key
everywhere (grepped every usage); `normalize_ct`/`denormalize_ct` are exact
inverses (verified algebraically and numerically); the sign-flip hypothesis
was explicitly tested and rejected with concrete numbers (would predict real
background decoding to +3000 HU, foreground to +2100-2243 HU — doesn't match
observed synthetic values at all); conditioning pass-through, DDIM math, and
wavelet subband ordering were all re-derived and checked. **What actually
explains the numbers**: the synthetic HU values, converted back to
normalized space, cluster tightly around normalized zero — and
`denormalize_ct(0.0) = 1000.0 HU` exactly, matching the observed synthetic
cluster. The model's raw output is centered and weakly differentiated
(consistent with undertraining), and only *looks* positive in HU space
because the clip range (-1000, 3000) is asymmetric around zero, not because
of a sign or parameter bug. Full derivation and the two supporting quantitative
tests are pinned as permanent regression tests in `tests/test_preprocessing.py`
(`test_normalized_zero_maps_to_clip_range_midpoint`,
`test_sign_flip_hypothesis_rejected`) so this doesn't need re-deriving by
hand under time pressure again.

Two real (smaller) bugs *were* found and fixed along the way, both the same
class as the round-6 EMA-decay bug: config values that map onto a
fixed-shape buffer or custom-`load_state_dict` object get silently
overwritten by whatever the checkpoint stored on resume, with no error
(shapes match, so `load_state_dict` succeeds silently). Fixed for
`subband_loss_weights` (PR #1, merged) the same way `ema.decay` was fixed
in round 6 — re-apply the config value after `load_checkpoint`. Concretely,
this means round 6's LLL-subband-weighting change never actually took effect
during the real 5500→5750 step continuation on Kaggle; it's fixed now for
any future continuation.

**Round 8 — orientation-consistency risk flagged, not resolved (part of
Priority 2).** While auditing preprocessing, found that `resample_to_spacing`
uses each image's own `GetOrigin()`/`GetDirection()` with no canonical
reorientation step anywhere in the pipeline. If SynthRAD's brain MR/CT and
BraTS's T1 use different orientation conventions in their NIfTI headers, a
"D" index in a SynthRAD-trained model's condition input would not correspond
to the same physical direction in BraTS input — a real conditioning mismatch,
independent of (and potentially compounding) the undertraining finding above.
**Could not be checked from this machine** (no access to the real dataset
files). `scripts/check_orientation_consistency.py` is ready to run on Kaggle
— see the wake-up runbook below, this should be one of the first things
checked.

**Priority 2 (rest) — formal test suite.** 43 tests added under `tests/`
(see `tests/README.md`), all CPU-only, all passing: wavelet exact-inverse,
the preprocessing round-trip and sign-flip checks from the PSNR audit,
`UNet3D`/`UNet2D` shape and checkpointing correctness (including the round-4
GroupNorm-divisibility regression), patient discovery content-validation,
and two subprocess-based integration tests that run the real training and
Stage 2 scripts end to end (checkpoint resume correctness, brain-crop
geometry correctness). `pad_to_multiple`/`pad_or_crop_to_shape` were
generalized from hardcoded-3D to N-dimensional so the 2D pipeline could
reuse them instead of duplicating the logic (backward compatible, existing
3D behavior unchanged, confirmed by the existing tests passing unmodified).

**Priority 3 — standalone 2D pipeline, built and CPU-tested tonight, NEVER
run on a real GPU or real data.** `models/unet2d.py` +
`models/stage1_mri2ct_ddpm_2d.py`: same building blocks and lessons learned
as the 3D model (FiLM timestep conditioning, safe GroupNorm groups,
activation checkpointing) but no wavelet transform — a single 2D slice is
already small enough for full-resolution pixel-space diffusion, so the
compression that was specifically necessary for full 3D volumes isn't needed.
`data/loaders_synthrad_2d.py` wraps the 3D loader for preprocessing reuse and
adds slice indexing with background-slice skipping — **caught and fixed a
real batching bug before it could bite on Kaggle**: different patients have
different natural crop sizes, so multiple-alignment padding (like the 3D
loader uses) produces different-shaped slices per patient that can't batch
together; fixed by padding/cropping every slice to a fixed `slice_size`
instead (also just standard practice for 2D pipelines generally).
`training/train_stage1_2d.py` + `configs/stage1_synthrad_2d.yaml`: same
resumability design as the 3D script, deliberately separate checkpoint
directory, and the `attention_resolutions: [4, 8]` default is set correctly
from the start (not `[2, 4]`) specifically because of the lesson learned the
hard way in the 3D pipeline's OOM saga. Full train→checkpoint→resume cycle
verified locally end to end on fake data (`tests/test_train_stage1_2d_resume.py`).
**What tonight's testing does NOT tell you**: real memory/time budget on an
actual Kaggle GPU, and — most importantly — whether it actually converges
faster than 3D in practice. That's a real-GPU question, first item in the
runbook below.

**Priority 4 — Stage 2 is now a documented, reusable dataset generator.**
`manifest.csv` gained `checkpoint_step`, `ddim_steps`, `use_ema`,
`generated_at` per patient (a dataset built across multiple resumed
sessions could have used different checkpoints for different patients —
manifest.csv is the source of truth for which). `run_stage2_brats.py` now
writes `metadata.json` (machine-readable run provenance, including a git
commit hash when available) and a populated `README.md` dataset card
(actual counts and settings from the run, not a template to fill in by
hand) into `output_dir` after every run. Verified end to end against the
existing fake-data checkpoint.

## 2026-07-15 deadline night (round 5) — demo push, 3am deadline

Goal changed for tonight only: a demo-able synthetic CT from BraTS by 3am,
not production quality/full convergence -- explicitly accepted as
out-of-reach in one night. Config reflects this (`configs/stage1_synthrad.
yaml`): `grad_accum_steps: 1` (was 4 -- ~3.8x more weight updates for the
same wall-clock, zero extra OOM risk since it doesn't touch per-microbatch
memory at all), `diffusion.timesteps: 250` (was 1000 -- denser per-noise-
level gradient coverage when total_steps is only in the thousands),
`ddim_steps: 30` (was 100 -- faster Stage 2 generation), `total_steps:
5500`/`checkpoint_interval: 250` (planning estimates from the real
15.34s/accumulated-step measurement, not hard targets -- training is
checkpointed regularly regardless, so it's fine and expected to interrupt
once the training time budget is up rather than waiting for exactly 5500).

**Realistic expectation, stated plainly:** a few thousand steps is a low
budget for any diffusion model. Strong MRI conditioning and wavelet-domain
training (the LLL subband -- coarse structure -- typically converges faster
than the 7 detail subbands) make some visible brain-shaped structure
plausible sooner than for unconditional generation, but expect blur/softness
and likely no fine bone/tissue detail. If it comes out looking like noise,
the honest fallback framing is still legitimate: demo the *pipeline* (MRI in
-> checkpointed training -> synthetic CT + tumor mask out, fully resumable,
correctly paired) as complete and correct engineering, independent of
tonight's image quality, which is a compute-budget problem, not a code
problem. Do a cheap mid-run sanity check (~halfway through the training
window: read the loss curve in the CSV log, optionally a 1-2-patient Stage 2
preview at ddim_steps~10 on the checkpoint-so-far) specifically so a dead-end
is caught with enough runway left to react, not discovered at hour 8.

**Optional, unverified lever if there's time to spare:** `model.
use_checkpoint` also accepts a list of resolution factors (e.g. `[1, 2]`) to
scope checkpointing to only the levels that actually needed it -- see the
comment above that config key. Verified numerically identical to full
checkpointing on CPU; actual speed gain on a real T4 is unverified. Test
with `--max_steps 20` before trusting it with the full run.

Dual-GPU (2xT4) explicitly not pursued tonight -- see the round-4 entry
above; that reasoning is unchanged and is doubly true with a hard deadline.

## Not yet done / explicitly out of scope

- Dice-on-downstream-segmentation evaluation (med-ddpm's practice) — worth
  adding once a converged Stage 2 dataset exists to evaluate.
- Classifier-free guidance / conditioning dropout for the denoiser.
- Pelvis region (SynthRAD2023 covers brain + pelvis; only brain is relevant
  to the BraTS pairing goal).
- Elastix re-registration (relying on SynthRAD2023's pre-registered release).
- Canonical image reorientation (`sitk.DICOMOrient`) — only worth adding if
  `scripts/check_orientation_consistency.py` finds a real mismatch (see
  round 8 above; unverified, first thing to check on wake-up).

## Wake-up runbook — exact order of operations

**Step 0 — orient yourself.** `git log --oneline -15` to see what's
actually committed (this file is updated alongside commits but can lag if
a session ends abruptly). Then `python -m pytest tests/` — should show
43 passed. If it doesn't, something regressed after this was written;
trust the test failures over this file's claims.

**Step 1 — the one check that could invalidate everything, run it first.**
`scripts/check_orientation_consistency.py` was never run against real
data (no access to it from the dev machine). This takes under a minute
and the answer changes what round 8's "genuinely undertrained, not a bug"
conclusion is worth:
```
python -m scripts.check_orientation_consistency \
    --synthrad_mr /kaggle/input/datasets/fd7akxj65n5yjxwds/synthrad-2023/Task1/brain/<any_patient>/mr.nii \
    --brats_t1 /kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/<any_patient>/<any_patient>_t1.nii \
    --synthrad_ct /kaggle/input/datasets/fd7akxj65n5yjxwds/synthrad-2023/Task1/brain/<same_patient>/ct.nii
```
If it reports a large direction-matrix mismatch, that's a real, separate
bug worth fixing (add `sitk.DICOMOrient` to both loaders) before spending
more GPU time on either pipeline below — a conditioning mismatch would
explain poor quality regardless of how much more training either pipeline
gets. If directions match closely, round 8's conclusion stands as-is.

**Step 2 — decision point: 3D fix vs. 2D pipeline. My honest
recommendation: start the 2D pipeline, let 3D keep training in parallel
if a second GPU session is available.** Reasoning, not just a coin flip:
round 8's audit found the 3D checkpoint is genuinely undertrained, not
buggy — the actual fix is "more steps," and 3D's per-step cost is the
whole reason that's been hard to get (this is exactly why goal was changed
to build the 2D pipeline in the first place). The 2D pipeline was
specifically designed to sidestep that cost (full-resolution pixel-space
diffusion on cheap 2D slices instead of full 3D volumes), so it's the
faster path to a real answer on whether the paired-translation approach
converges to something demonstrable at all, before sinking more hours into
the slower 3D loop. It has never touched a real GPU or real data, so
budget time for the smoke test below to surface anything Kaggle-specific
(memory, real dataset quirks) the same way the 3D pipeline needed several
rounds of that. The 2×T4 setup mentioned earlier in this file (see the
round-4 dual-GPU note) means both can genuinely run at once if wanted —
2D on one session, 3D continuing on the other — rather than choosing
between them.

Commands, in order:
```
# 2D smoke test first -- mirrors the 3D pipeline's own first-smoke-test pattern
python -m training.train_stage1_2d --config configs/stage1_synthrad_2d.yaml --max_steps 100 --max_patients 3
# watch nvidia-smi during this run the same way the 3D pipeline's first smoke test did

# if that's clean, start a real run (interrupt anytime, checkpointed every 1000 steps by default)
python -m training.train_stage1_2d --config configs/stage1_synthrad_2d.yaml

# 3D: if continuing in a second session, no command change needed, same as before
python -m training.train_stage1 --config configs/stage1_synthrad.yaml
```

**Step 3 — as either pipeline produces new checkpoints, re-run the
diagnostic scripts already built** rather than eyeballing raw output:
```
python -m inference.compare_synthrad_val --config configs/stage1_synthrad.yaml --num_patients 3
# (there is no 2D equivalent of compare_synthrad_val yet -- would need a small
# adaptation since the 2D model's sample() takes a single slice, not a volume;
# not built tonight, flagged here rather than left silently missing)
```

**Step 4 — once a checkpoint (2D or 3D) looks genuinely good** (foreground
PSNR meaningfully above the ~13 dB "flat background guess" baseline from
the round-8 audit, ideally approaching the "blurred-but-correct" ~25 dB
ballpark from that same calibration), regenerate the BraTS dataset:
```
python -m inference.run_stage2_brats --config configs/stage2_inference_brats.yaml --overwrite
```
`--overwrite` matters here since the existing output was generated from
the undertrained step-9000 checkpoint — rerunning without it would just
skip everything as "already done." Check the auto-generated
`README.md`/`metadata.json` in the output directory afterward — they'll
reflect the new checkpoint's provenance automatically.

**Manual Kaggle checkpoint-persistence steps (unchanged, still required
every session):**
1. Train normally. Checkpoints accumulate in `/kaggle/working/checkpoints/...`.
2. Before a session ends: **Save Version → Save & Run All (Commit)**.
3. Next session: **Add Input → Notebook Output Files** → pick that version.
4. Add the mounted path to `checkpoint.extra_resume_dirs` in the relevant
   config (3D or 2D — they're independent). Rerun the same training
   command — it auto-detects and resumes from the highest-step checkpoint
   across all listed directories.
5. Repeat each session.

## Stage 3 — CT tumor segmentation (added 2026-07-18)

A third goal, on top of the original two: train a binary tumor
segmentation model on the Stage 2 output (synthetic CT + tumor mask), then
externally validate it against a small real-CT dataset from Jordan
University Hospital that is never used in training. Three new Kaggle
inputs were attached for this: the Stage 2 output re-uploaded as its own
dataset (365 BraTS patients), and two Jordan datasets (20 patients each —
CT and mask, as separate flat directories of DICOM slices).

**Confirmed dataset paths (do not re-derive):**
```
# Dataset 1 -- Stage 2 output, re-uploaded (note the unusually deep,
# Kaggle-exported path -- this is not a mistake to "fix")
/kaggle/input/datasets/sumayarahmanmeherin/annoted-sct/kaggle/input/notebooks/sumayarahmanmeherin/mri-to-ct/synthetic_ct_dataset_regression/<patient_id>/synthetic_ct.nii
/kaggle/input/datasets/sumayarahmanmeherin/annoted-sct/kaggle/input/notebooks/sumayarahmanmeherin/mri-to-ct/synthetic_ct_dataset_regression/<patient_id>/tumor_mask.nii

# Dataset 2 -- Jordan real CT (20 patients, RGB DICOM, 0-255, windowed --
# NOT raw HU)
/kaggle/input/datasets/sumayarahmanmeherin/annoted-20-ct/<patient_id>_CT_s<slice_num>.dcm

# Dataset 3 -- Jordan real tumor mask (matched to Dataset 2 by patient_id +
# slice_num parsed from the filename)
/kaggle/input/datasets/sumayarahmanmeherin/mask-data-20/<patient_id>_CT_m<slice_num>.dcm
```
Note the file extension for Dataset 1 is bare `.nii`, not the `.nii.gz`
`run_stage2_brats_regression.py` itself writes -- the Kaggle re-upload
process appears to have decompressed it. `data/loaders_synthetic_ct.py`
matches either extension, so this doesn't need to be "fixed" anywhere.

**Critical preprocessing decision: mask binarization.** `tumor_mask.nii`
is the ORIGINAL BraTS annotation, copied through Stage 2 unmodified (see
`run_stage2_brats_regression.py`'s own "Known limitations"). BraTS labels
are multi-class -- 0 background, 1 NCR/NET, 2 ED, 4 ET (3 is never used).
Stage 3 is scoped to binary tumor segmentation, not sub-region
classification: sub-region work is out of scope for now, and the Jordan
masks are presumed binary too, so a multi-class *target* wouldn't even
have a like-for-like external comparison available. `data/
loaders_synthetic_ct.py`'s `_load_and_preprocess` collapses every nonzero
label to 1 before anything else touches the mask -- verified directly in
`tests/test_stage3_segmentation.py` against a fixture using the real
0/1/2/4 label set.

**Known, load-bearing limitations of the Jordan comparison** (all
documented in code too -- see `data/loaders_jordan_ct.py`'s and
`inference/validate_jordan_segmentation.py`'s module docstrings):
1. **Format mismatch.** Jordan CT/mask are RGB, 0-255, windowed
   ("secondary capture" DICOM -- an already-rendered image, not raw HU
   pixel data). There is no HU value to recover from an 8-bit windowed
   screenshot, so `data/loaders_jordan_ct.py` can only min-max normalize
   each slice to itself -- structurally different from the real-HU-based
   normalization the model trained on. Any Dice/IoU computed against this
   data measures whether the model's predicted SHAPE agrees with the real
   tumor outline, not whether its intensity reasoning transfers, because
   it cannot be asked to given the input format.
2. **Incomplete volumes.** Each Jordan patient has only 1-6 tumor-
   containing slices, not a full 3D volume -- no real 3D neighborhood
   exists for a 3D model to use. `inference/validate_jordan_segmentation.py`
   works around this by replicating the single 2D slice
   `--replication_depth` times along Z to build a "pseudo-volume" the
   model can run on, then reads back the center slice of the output. This
   gives the model *something* 3D-shaped, but every neighboring slice it
   sees is a copy, not real anatomy. Treat Jordan Dice/IoU as "can the
   model do something reasonable given a single real slice," not "3D
   segmentation quality on real data."
3. **Filename matching is not independently verified.**
   `discover_jordan_slices` matches CT and mask files purely by
   `(patient_id, slice_num)` parsed from filenames in two separate
   directories, and logs (rather than silently drops or guesses at) any
   CT-only or mask-only file. But there is no cross-check that a matched
   pair actually depicts the same anatomical slice beyond the naming
   convention holding -- a wrong pairing that still matches the filename
   pattern would silently corrupt the reported metrics with no error.
4. **Label granularity mismatch** -- addressed by binarization above, not
   a remaining risk, but listed here since it was one of the four known
   gaps going in.

**Honest assessment: how much do these limitations reduce validation
reliability?** Meaningfully, and in ways that compound. The format
mismatch (#1) means Jordan metrics are, at best, a shape-agreement check,
not a real generalization test of the model's actual CT-intensity
reasoning -- a model that's learned to key off HU-range cues the training
data has and Jordan doesn't could look artificially worse than it "really"
is, and one that's learned something more shape-based could look
artificially better. The pseudo-volume workaround (#2) means the model
never sees genuine 3D context for Jordan slices, so any real 3D-aware
behavior it learned during training (e.g. using neighboring-slice
continuity) is neither being tested nor available to help it here --
Jordan numbers likely underestimate what the model can do on genuine 3D
input, for a reason unrelated to real-world generalization. The matching
risk (#3) means a small, silent, currently-undetected fraction of
"matched" pairs could be wrong, which would show up as unexplained outlier
Dice scores for specific slices -- worth a manual visual spot-check of a
few low-scoring Jordan cases before drawing conclusions from them, not
just trusting the aggregate mean. **Net effect: treat the Jordan Dice/IoU
numbers as a rough, directional signal ("does this look like it's doing
something reasonable on real data at all") rather than a rigorous,
publishable external-validation result** -- the synthetic-CT held-out
validation split (real patient-level split, no format mismatch, no 3D
workaround) is the more trustworthy number for actual model quality.

**Model/training:** `models/unet3d_segmentation.py` is the same plain
encoder-decoder topology as Stage 1's regression U-Net (own file, no
shared code, same pipeline-isolation reasoning), predicting a raw logit
map (see the bug note below for why it's logits, not a sigmoid-activated
probability, straight out of `forward()`). Loss is Dice + BCE (`training/
train_stage3_segmentation.py`'s `combined_loss`) -- standard for
segmentation under severe foreground/background imbalance. Training
patches are foreground-biased (`data/preprocessing.py`'s new
`foreground_biased_patch_crop`, additive, doesn't touch
`random_patch_crop`): with `data.foreground_prob` probability (default
0.5), a patch is centered on a random tumor voxel instead of a uniform
random location, since a tumor is a tiny fraction of a whole brain volume
and pure random cropping would mostly yield empty-mask patches and weak
Dice gradient signal. Same resumability design as every other stage
(checkpoint-saved-before-validation ordering applied from the start here,
rather than waiting to hit the real-Kaggle bug that taught this lesson in
Stage 1).

**Bug found on the real Kaggle smoke test (2026-07-19): logits, not
sigmoid, out of the model.** `UNet3DSegmentation.forward()` originally
applied `sigmoid` internally, and `combined_loss` called
`F.binary_cross_entropy` on that already-sigmoided output. Both
`torch.nn.functional.binary_cross_entropy` and `torch.nn.BCELoss` are
explicitly unsafe under CUDA autocast -- they require an input already in
[0, 1], and fp16 casting can silently corrupt that -- so the very first
smoke test crashed immediately with a `RuntimeError` at the first training
step. Fixed by moving the sigmoid out of the model: `forward()` now
returns raw logits, `combined_loss` uses
`F.binary_cross_entropy_with_logits` (fuses sigmoid + BCE in a numerically
stable, autocast-safe way) for the BCE half and applies `torch.sigmoid`
explicitly, once, for the Dice half (Dice needs an actual probability to
compute overlap against a 0/1 target -- logits don't make sense there).
`predict_full_volume` applies sigmoid per-patch before blending overlapping
windows, not after averaging the whole volume -- sigmoid is nonlinear, so
those two orders aren't equivalent, and only the per-patch order is
correct. Verified (CPU, no GPU needed for this part): a new test asserts
`combined_loss`'s value exactly matches manually computing
`sigmoid(logits)` and calling the plain (non-autocast-safe)
`binary_cross_entropy` + `dice_loss` on that -- i.e. the fix changes *how*
the computation is expressed for autocast-safety, not *what* it computes --
plus gradients are finite through a real forward/backward pass. The actual
autocast crash itself can't be reproduced on this CPU-only machine (autocast
only activates on `device.type == "cuda"`), so the real-GPU confirmation is
still pending the next Kaggle run.

**Repo additions:** `models/unet3d_segmentation.py`,
`configs/stage3_ct_segmentation.yaml`,
`training/train_stage3_segmentation.py`, `data/loaders_synthetic_ct.py`
(Stage 2 output loader, not explicitly requested but needed -- neither
existing loader reads that file layout), `data/loaders_jordan_ct.py`,
`inference/validate_jordan_segmentation.py`. Existing Stage 1/2 files
(`models/unet3d_regression.py`, `training/train_stage1_regression.py`,
`inference/*`, `data/loaders_synthrad.py`, `data/loaders_brats.py`)
untouched.

**Verification done without a GPU** (see `tests/test_stage3_segmentation.py`
and `tests/test_loaders_jordan_ct.py`): synthetic-CT discovery accepts
both `.nii`/`.nii.gz`; the real 0/1/2/4 BraTS label set collapses to
binary {0,1}; patient-level split has no leakage; the foreground-biased
crop reliably lands on a small tumor when asked to; the segmentation
model's shape and sigmoid-range are correct; the real train → checkpoint →
resume cycle; Jordan CT/mask slice matching by (patient_id, slice_num),
including that an unmatched CT-only or mask-only file is excluded and
logged rather than mismatched; RGB→grayscale conversion against real
(pydicom-built) DICOM fixtures; per-slice normalize/binarize correctness.
**Not verified**, because this machine has no GPU and no access to the
real Kaggle datasets: actual training convergence on real synthetic CT,
real Dice/IoU numbers on real Jordan data, and real memory/time behavior
at production model/patch size (the sliding-window OOM precaution from
Stage 1 is applied here from the start, but its actual necessity for this
specific model/task is unconfirmed on real hardware).

### Future work note: radiomics validation scope (not started, do not begin yet)

Radiomics feature extraction and prediction is explicitly **out of scope
for Stage 3** -- noted here only so a future session doesn't have to
re-derive this reasoning:

- BraTS masks are tumor-annotated, but BraTS has no real CT at all --
  only real MRI. So BraTS cannot support a "real CT radiomics vs.
  synthetic CT radiomics" comparison; there is no real-CT half to compare
  against.
- SynthRAD has both real CT and real MRI (so a CT↔MRI radiomics-
  reproducibility comparison genuinely is possible there), but SynthRAD's
  mask is a whole-brain/anatomy mask, not a tumor delineation -- SynthRAD
  patients were undergoing radiotherapy planning, but the released dataset
  has no tumor-specific annotation.
- **Conclusion: any future radiomics validation built on this project's
  data would necessarily be a "whole-brain-region radiomics
  reproducibility" study (using SynthRAD's real CT+MRI pair), not a
  "tumor-specific radiomics" study** -- the two are genuinely different
  claims, and this project's data supports only the former. Don't
  conflate them in whatever radiomics work eventually gets built.

## Stage 3 investigation: predicted-mask location mismatch (2026-07-19)

A real Kaggle run's validation visualization showed the predicted tumor
mask in a clearly different location than the ground truth (large blob
bottom-left in the real mask; scattered fragments top-left and right in
the prediction), on the SYNTHETIC (in-distribution) validation set, not
just Jordan -- reported as urgent since it suggested a possible core
coordinate/pairing bug rather than a Jordan-specific domain-gap issue.
Investigated with controlled CPU diagnostics (single fake patient, known
asymmetric tumor location, deliberately overfitting a tiny model) rather
than guessing from code reading alone.

**Definitively ruled out: no coordinate/axis/mask-pairing bug.** A model
trained with Dice-only loss directly on a full volume (no patching,
bypassing the sliding-window path entirely) reaches Dice=1.0 with the
predicted centroid landing exactly on the true tumor location. This rules
out an axis flip, transpose, or misalignment anywhere in
data/loaders_synthetic_ct.py's crop+pad pipeline, the model's forward()
pass, or the visualization script's slice indexing -- if any of those had
a coordinate bug, this direct-forward test could not have reached perfect,
correctly-located overlap.

**A real, separate bug found and fixed:** `predict_full_volume` (both
models/unet3d_segmentation.py and models/unet3d_regression.py) crashed
with a confusing `torch.cat` shape-mismatch error if `patch_size` wasn't
divisible by `2**(num_levels-1)` -- the skip-connection feature maps don't
line up otherwise. Now raises a clear `ValueError` upfront instead. The
real configs' `patch_size: [96, 96, 64]` already satisfies this (divisible
by 8), so this specific crash is unlikely to be what produced the reported
visualization, but it's a real robustness gap now closed regardless.

**The actual reconstruction-quality issue: reproduced, partially
understood, not fully resolved.** Controlled testing showed a real,
reproducible pattern: a model trained via patch-based sampling (matching
the real pipeline) with a smooth, steadily-decreasing training loss --
i.e. not obviously undertrained or unstable by the loss curve alone --
still reconstructs a full-volume prediction via `predict_full_volume`
that is shifted and expanded well beyond the true region (e.g. one run:
patch-level Dice reached 1.0 during training, but the full-volume
reconstruction scored only 0.48, with a bounding box roughly 3x the true
one, anchored at the correct starting corner but bleeding outward).

The natural hypothesis -- that predict_full_volume's original UNIFORM
overlap-blending let each tile's imprecise, uncertain edge predictions
smear the reconstruction outward -- led to switching to Gaussian
(center-weighted) blending, the standard fix for exactly this class of
problem in tiled medical-image inference (MONAI's default approach; see
`_gaussian_importance_map` in models/unet3d_segmentation.py). **This did
NOT resolve it in testing** -- dice dropped slightly further (0.48 ->
0.39) and the reconstructed bounding box was identical in extent, just
denser. This disproves the "edge-blur" hypothesis as the primary cause,
at least at the tiny scale tested.

Repeated attempts to get a controlled, single-patient toy model to
converge STABLY also surfaced a separate, likely-more-relevant finding:
**this loss/architecture combination is prone to real training
instability at small scale** -- across several runs, patch-level Dice
sometimes reached 1.0 mid-training then collapsed back toward 0, and a
smaller/shorter run collapsed to predicting an entirely empty mask despite
otherwise-reasonable-looking hyperparameters. This is consistent with
known Dice+BCE-under-severe-class-imbalance instability (tumor is a tiny
fraction of any patch, even foreground-biased ones), though lowering
`bce_weight` alone (tried 1.0 and 0.05) did not fix it either -- the
interaction is more complex than a simple loss-term-weight issue. A
CPU-only, single-patient, few-hundred-step diagnostic is a fundamentally
low-power setting to fully characterize an optimization stability
question; a first regression test attempting to lock in "converges and
reconstructs correctly" as a guard was written, found to be flaky (passed
under one set of scale/step parameters, failed under a very similar one),
and removed rather than kept in a misleading, unreliable state.

**What's kept from this investigation:** the `predict_full_volume`
divisibility validation (unambiguously correct, keep), Gaussian blending
(standard practice, not proven harmful, plausibly still net-positive at
real scale with real data diversity even though it didn't fix the toy
case -- keep as the more defensible default over uniform blending), two
new fast/reliable unit tests (Gaussian importance map shape,
divisibility-rejection).

**What's NOT resolved, and the concrete next step:** whether the real
Kaggle checkpoint's mislocated prediction is explained by (a) genuine
undertraining that more real steps will fix, (b) the instability pattern
found here, requiring a loss-function change beyond bce_weight tuning
(candidates for a future session: Tversky loss, focal loss, or per-batch
positive-class weighting in BCE), or (c) something specific to real BraTS
tumor shapes/scale that this toy diagnostic's simple cube couldn't
surface. **The single most useful piece of evidence to resolve this
next**: the real run's training log CSV (step, split, loss, dice_score
columns from training/train_stage3_segmentation.py) -- a smooth,
monotonically improving curve would point to (a); a plateaued or
oscillating one would point to (b).

### Follow-up: real training log confirmed (b), root cause identified, loss functions added

The full real training log (all 820 rows, not just the periodic
validation summary) was shared and settled the (a) vs (b) question
decisively. Two things in it matter:

1. **Steps 19950-20000 have `lr=0.0` exactly** (cosine schedule fully
   decayed) -- the model's weights are frozen there. Yet loss still swings
   0.786-0.963 and dice swings 0.343-0.566 across those frozen-weight
   steps. A frozen model cannot produce that swing from its own
   instability; the only thing changing step to step is which random
   training patch got sampled. This means a meaningful part of the
   apparent "wild oscillation" in per-step dice/loss is the metric's own
   sensitivity to how much foreground a given patch happens to contain
   (see next point), not pure optimization chaos -- worth knowing before
   over-interpreting any single training log's noise in the future.
2. **Several dice values are suspiciously round** (exactly 0.5000, 0.2000)
   -- `dice = (2*I + smooth) / (P + T + smooth)` with `smooth=1.0` is
   dominated by the smoothing constant whenever a patch's total
   foreground+prediction voxel count is small (e.g. dice=0.5 exactly
   follows from intersection=0, pred+target=1 voxel total). This doesn't
   change the real conclusion below, but means per-step dice numbers
   specifically (train or the patch-level periodic validation) are a
   noisier signal than they look, especially early/late in a schedule
   when gradients (and thus typical prediction confidence) are small.

Separately, the external Jordan validation of the resulting checkpoint
came back uniformly near-zero (~0.0005-0.013 Dice across all 83 slices,
essentially no variance) -- more consistent with the checkpoint having
collapsed to predicting an essentially empty mask than with pure random
instability (which would be expected to show more spread: some slices
coincidentally scoring higher, others near zero).

**Root cause, as confirmed:** Dice+BCE under this project's severe
class imbalance (a tumor is a tiny fraction of any patch, even
foreground-biased ones) collapses toward predicting background
everywhere -- a well-documented failure mode for this exact loss
combination in the medical segmentation literature, not a data or
coordinate bug (already ruled out above).

**Fix implemented:** `training/train_stage3_segmentation.py` now
supports `training.loss_type` (`dice_bce` | `tversky` | `focal_tversky` |
`focal`), all operating on raw logits (no reintroduction of the
autocast-unsafe pattern -- `focal_loss_with_logits` specifically uses the
`exp(-bce_with_logits)` trick, not a direct `log(sigmoid(x))` call, for
the same reason `combined_loss`'s BCE branch does). Default switched to
`tversky` (`alpha=0.3`, `beta=0.7` -- penalizing missed tumor more than
false alarms, the standard remedy for the collapse-to-empty pattern
found here). `dice_bce` is kept, unchanged, as a selectable option for
comparison, not because it's recommended. Five new unit tests verify:
Tversky's perfect/zero-overlap sanity bounds, that the beta>alpha
asymmetry actually penalizes false negatives more (the entire point of
switching), Focal Tversky's sanity bounds, that
`focal_loss_with_logits` is numerically exact against the naive unsafe
formulation it avoids computing directly, and that `combined_loss`
correctly dispatches on `loss_type` (and rejects an unknown one).
Verified end to end with a real (tiny, fake-data) training run using
`loss_type: tversky`. Full suite: 57/57 passing.

**Not yet re-verified:** whether Tversky loss actually fixes the
collapse on REAL BraTS-scale data -- that requires a real Kaggle retrain,
which hasn't happened yet as of this note. The CPU-only verification here
confirms the new loss functions are mathematically correct and wired up
correctly, not that they solve the real-world problem; that's the next
real-GPU checkpoint's job to confirm.

### Follow-up: resume was reviving the old dead LR, warm-start fix added

After switching to `focal` loss and retraining, real dice still swung
wildly (0.42 -> 0.0001 -> 0.37). Cause: the retrain resumed from the
step-20000 checkpoint the normal way, which restores optimizer AND
scheduler state along with the weights. That checkpoint's cosine
schedule had already fully decayed to `lr=0.0` (see previous section) --
so the new loss function started with essentially zero learning rate
and no room to move the weights anywhere. This is the same class of bug
as Stage 1's round-6 "config value silently reverts on resume" family
(`ema.decay`, `subband_loss_weights`): a checkpoint's saved state
silently overrides the current config's intent unless something
explicitly re-applies it.

**Fix:** new `--warm_start_checkpoint <path>` CLI flag on
`training/train_stage3_segmentation.py`. When passed, it loads *only*
the model weights from that exact checkpoint file (`ema=None,
optimizer=None, scheduler=None` in the `load_checkpoint()` call), then
builds a brand new EMA, optimizer, and scheduler from the current
config and starts `global_step` at 0 -- so the new loss function gets
the config's actual `training.lr` (e.g. `0.0001`) and a full fresh
cosine schedule, not the dead tail end of the old one. This is the
Stage 1 "LR re-open" pattern applied to Stage 3. The normal
`find_latest_checkpoint`-based resume path (used when
`--warm_start_checkpoint` is not passed) is untouched.

Two safety warnings were added because warm-start reuses the same
checkpoint-loading code paths as normal training and can silently
collide with them if pointed at old locations:
- if `checkpoint.working_dir` already contains a checkpoint, a future
  *plain* resume of that directory would find the OLD higher-step
  checkpoint (with its dead LR) instead of the fresh warm-started run --
  reviving the exact bug this flag exists to fix. Point
  `checkpoint.working_dir` at a new directory for each warm-started run.
- if `training.log_file` already exists, `init_log_file` will truncate
  it (since `resuming = global_step > 0` is `False` on a warm start),
  destroying the old run's log. Point `training.log_file` at a new path
  too.

Verified with a new test,
`test_warm_start_checkpoint_resets_step_count_and_schedule`: trains a
tiny 4-step checkpoint, warm-starts from it into a new checkpoint
dir/log file with `loss_type=tversky` and `total_steps=3`, and asserts
the run logs "Warm-started model weights from" (not "Resumed from
checkpoint") and completes exactly 3 steps -- proving the step count
and schedule really started fresh at 0, not continuing from step 4.
Also manually smoke-tested pointing `--warm_start_checkpoint` at a
checkpoint already living in the target `working_dir`/`log_file` and
confirmed both collision warnings fire as expected. Full suite: 58/58
passing.

**Exact retrain command** (real Kaggle values -- edit
`configs/stage3_ct_segmentation.yaml` first to point
`checkpoint.working_dir` and `training.log_file` at NEW paths, e.g.
`.../checkpoints/stage3_segmentation_focal_warmstart/` and
`.../logs/stage3_focal_warmstart_log.csv`, then run):

```bash
python -m training.train_stage3_segmentation \
  --config configs/stage3_ct_segmentation.yaml \
  --warm_start_checkpoint /kaggle/working/checkpoints/stage3_segmentation/ckpt_step00020000.pt
```

**Not yet re-verified:** whether a fresh, full-LR schedule on top of the
`focal` loss actually produces a stable, non-collapsing checkpoint on
real BraTS-scale data -- that's the next real-GPU run's job. This fix
only guarantees the new run actually gets a usable learning rate; it
doesn't guarantee `focal` loss itself is the right choice (that
question is still open pending this retrain's results).

### Follow-up: LR fix worked, but dice was still stuck -- focal_alpha was backwards

After the warm-start fix above, the real retrain confirmed the LR itself
was fixed (warmup completed, lr settled at the configured 0.0002 as
expected). But dice (steps 200-625) kept landing on almost exactly one
of two values -- ~0.0000 or exactly 0.5000 -- never in between, which
looked like a new instability rather than gradual learning.

It isn't a new instability; it's the same `smooth=1.0` dice-formula
artifact from the earlier section, this time driven by per-patch tumor
size instead of frozen weights. `dice_score` (train_stage3_segmentation.py)
thresholds predictions at 0.5 before scoring: `(2I+1)/(P+T+1)`.
`foreground_biased_patch_crop` (data/preprocessing.py) centers a crop on
a *random* foreground voxel, then clamps to stay in-bounds -- so true
tumor voxel count `T` per patch swings from a couple of voxels (crop
clamped near the tumor's edge) to hundreds (well-centered), it is not
constant. With the model's predictions `P` still ~0 (essentially nothing
crossing the 0.5 threshold) and intersection `I`~=0, dice reduces to
`1/(T+1)`: small `T` gives exactly 0.5000, large `T` rounds to 0.0000.
Two clean artifacts of tiny/moderate counts, not two behavioral states --
consistent with, not contradicting, the earlier "suspiciously round
dice values" finding.

The real signal underneath: `P`~=0 almost always means predictions are
still not confidently crossing the threshold on tumor voxels at all,
i.e. still collapsed, even under focal loss. Root cause found:
`focal_alpha: 0.25` (the config default, copied from RetinaNet's
anchor-imbalance tuning) is backwards for this problem. In
`focal_loss_with_logits`'s `alpha_t = alpha*target + (1-alpha)*(1-target)`,
alpha=0.25 gives *background* voxels weight 0.75 and *tumor* voxels
weight 0.25 -- the already-overwhelming majority class was weighted 3x
more than the rare class this project actually needs to learn, directly
fighting the imbalance instead of countering it.

**Fix:** flipped `focal_alpha` to `0.75` in
`configs/stage3_ct_segmentation.yaml` (config-only change, no code
touched) so tumor voxels get the higher weight their rarity calls for --
mirroring the `tversky_beta=0.7 > tversky_alpha=0.3` asymmetry already
used for the same reason elsewhere in this file. `focal_gamma=2.0` is
left as-is (a standard, non-suspicious value; a secondary lever to
revisit only if collapse persists after this fix). No test needed
updating: `test_focal_loss_with_logits_matches_manual_unsafe_computation`
exercises the formula generically with explicit alpha/gamma arguments,
not the config's chosen default.

**Not yet verified:** whether this actually gets predictions past the
threshold on real BraTS-scale data -- that's the next retrain's job.
Reuse the same warm-started checkpoint (no need to redo the LR reset,
only the loss weighting changed) into a NEW `checkpoint.working_dir` /
`training.log_file` pair, per the same collision-avoidance rule as the
warm-start fix above.
