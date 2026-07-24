# Methodology

*Draft methodology section, generated from a direct analysis of the project
codebase (`models/`, `training/`, `data/`, `configs/`, `inference/`) and the
training logs of the specific runs referenced below. Values marked with
`[confirm: ...]` could not be verified from within this repository and
should be filled in from the corresponding Kaggle run log before
submission. Detailed discussion of implementation limitations, iterative
debugging, and unresolved open questions is deliberately left out of this
section — see the Limitations note at the end of this document for pointers
to what belongs in a Discussion/Limitations section instead.*

## 1. Study Design

This study implements a three-stage pipeline for CT-based brain tumor
segmentation in the absence of a native, expert-annotated CT tumor dataset.
In Stage 1, a 3D convolutional network is trained on paired MRI-CT data to
perform direct MRI-to-CT image translation. In Stage 2, the resulting
translation model is applied to the MRI volumes of an independent tumor
dataset that provides expert tumor segmentation masks but no corresponding
CT, producing a synthetic CT volume for each subject while inheriting its
existing tumor annotation. This yields an annotated synthetic CT-tumor
mask dataset that does not exist as native, real-world data. In Stage 3, a
second 3D convolutional network of matching architecture is trained on this
synthetic CT-tumor mask dataset to perform binary tumor segmentation
directly from CT. The resulting segmentation model is evaluated on a
held-out split of the synthetic dataset and, separately, on an independent
external dataset of real hospital CT images never used at any training
stage, to assess generalization beyond the synthetic training distribution.

## 2. Datasets

### 2.1 SynthRAD2023 (Stage 1 training data)

Paired brain MRI, CT, and brain-mask volumes in NIfTI format, one directory
per patient. A patient folder is included only if it contains all three
required files (MRI, CT, mask), verified by content rather than by folder
name, so that non-patient auxiliary folders present in the raw dataset
distribution are excluded automatically. Patient count: `[confirm: exact
number of patients discovered under the SynthRAD root used for training —
read from the corresponding "discover_synthrad_patients: found N patients"
log line]`.

### 2.2 BraTS 2020 (Stage 2 input; source of tumor annotations)

T1-weighted MRI volumes with expert multi-class tumor segmentation masks
(label 0: background; label 1: necrotic and non-enhancing tumor core;
label 2: peritumoral edema; label 4: enhancing tumor; label 3 is not used
in the BraTS labeling convention). Patient identity is parsed directly from
filenames rather than directory structure, so discovery is unaffected by
different folder nesting conventions across dataset distributions. Only
patients with both a T1 volume and a segmentation mask are processed by
Stage 2. Of the patients processed, 368 were successfully converted into a
paired synthetic CT-tumor mask sample (confirmed by the discovery log of
the released Stage 3 training dataset, Section 2.3).

### 2.3 Synthetic CT-tumor mask dataset (Stage 3 training data)

The output of Stage 2: one synthetic CT volume and one tumor segmentation
mask per BraTS patient, the mask carried through unmodified from the source
BraTS annotation. This dataset comprised 368 patients. Prior to use, the
original multi-class BraTS label was binarized (label > 0 → 1) since this
study addresses whole-tumor detection rather than tumor sub-region
classification.

### 2.4 Jordan University Hospital dataset (external validation only)

Twenty patients from Jordan University Hospital, provided as individual
red-green-blue (RGB) secondary-capture DICOM slices — an already
window-leveled, 8-bit rendered image rather than raw Hounsfield-unit (HU)
pixel data — paired with binary tumor mask slices by a filename convention
matching patient identifier and slice number. Each patient contributes only
the 1-6 slices that contain visible tumor, not a complete volumetric scan.
This dataset was used exclusively for external, out-of-distribution
evaluation of the Stage 3 segmentation model and was never used, directly
or indirectly, at any training stage.

### 2.5 Train/validation splitting and data leakage

All patient-level splits (SynthRAD for Stage 1; the synthetic CT-tumor mask
dataset for Stage 3) use an identical algorithm: the full patient list is
discovered first, then a fixed pseudo-random permutation (NumPy
`default_rng`, seeded) is applied to the patient list, and the first
`train_val_split` fraction of the permuted list is assigned to training,
the remainder to validation. The split occurs on the *patient list*, before
any patch cropping or volume loading, so that every voxel originating from
a given patient is assigned entirely to either the training or the
validation set and never both. For Stage 3, `train_val_split = 0.9` and
`seed = 42` were used, yielding 331 training and 37 validation patients out
of the 368 available (confirmed by the training run log). The Jordan
dataset (Section 2.4) is disjoint from this split by construction: it is
read only by the external-validation script and is never passed to any
training data loader.

## 3. Preprocessing

All preprocessing parameters below are the values used for the Stage 3
training run reported in Sections 5-6, applied identically wherever the
same preprocessing module is shared with Stage 1.

- **Resampling.** All volumes are resampled to isotropic 1.0 x 1.0 x 1.0 mm
  voxel spacing, using linear interpolation for image volumes and
  nearest-neighbor interpolation for mask volumes (to preserve discrete
  label values).
- **CT intensity normalization.** Raw HU values are clipped to
  [-1000, 3000] HU and linearly rescaled to [-1, 1]; air (-1000 HU) maps
  exactly to -1.
- **MRI intensity normalization.** Percentile-based normalization computed
  over foreground voxels only (brain mask, or nonzero voxels where no mask
  is available): values are clipped to the [0.5, 99.5] percentile range of
  the foreground and linearly rescaled to [-1, 1]. Background voxels are
  set to a fixed value of -1 rather than rescaled with the foreground
  distribution.
- **Brain-domain masking (Stage 1 training data only).** SynthRAD MRI
  volumes are not skull-stripped, whereas the BraTS MRI volumes used at
  Stage 2 inference are skull-stripped. To match the Stage 1 training
  distribution to the domain it is applied to at inference, both the CT and
  MRI channels of the SynthRAD training data are masked to the brain-only
  region (non-brain voxels set to the respective background value) prior
  to training.
- **Cropping.** Each volume is cropped to the bounding box of its
  foreground (brain mask, or non-background CT content for the synthetic
  CT dataset) with a fixed margin of 10 voxels along each axis.
- **Padding.** Every processed volume is zero/background-padded, per axis,
  to the next multiple of 16 voxels. The 4-level U-Net architecture
  described in Section 4 performs 3 internal downsampling operations,
  requiring any input passed through it to have spatial dimensions
  divisible by $2^3 = 8$ for the skip connections to align exactly; a
  padding multiple of 16 satisfies this with a safety margin.
- **Patch extraction (training only).** Fixed-size 96 x 96 x 64 voxel
  patches are extracted per training sample. For Stage 1, patch location is
  sampled uniformly at random within the volume. For Stage 3, patch
  location is sampled with a foreground bias: with probability 0.5, the
  patch is centered on a randomly selected tumor-labeled voxel (clamped to
  keep the patch within volume bounds); otherwise, a uniformly random
  location is used, matching Stage 1's sampling. This addresses the severe
  class imbalance between tumor and background voxels in a full brain
  volume.
- **Tumor mask binarization (Stage 3 only).** The multi-class BraTS label
  described in Section 2.2 is collapsed to a binary label (`label > 0`)
  before any subsequent processing.

## 4. Model Architecture

Both the Stage 1 translation model and the Stage 3 segmentation model use
an identical convolutional encoder-decoder (U-Net) topology, differing only
in the semantic interpretation of the single input/output channel and in
the final output activation. Both instantiations have 8,742,561 trainable
parameters (verified by direct parameter count on the instantiated model,
not estimated).

**Notation.** Let $x \in \mathbb{R}^{1 \times D \times H \times W}$ denote
the single-channel input volume ($x$ = MRI for Stage 1, $x$ = CT for
Stage 3), and let $\theta$ denote the full set of learnable network
parameters. The network is composed of an encoder $E$, a bottleneck $B$,
and a decoder $D$, expressed below as their constituent per-level
operations.

**Building blocks.**
$$\text{ConvBlock}_c(h) = \text{ReLU}\big(\text{GN}(\text{Conv3D}_{3\times3\times3}(\text{ReLU}(\text{GN}(\text{Conv3D}_{3\times3\times3}(h))))\big)$$
two sequential 3x3x3 convolutions, each followed by Group Normalization
(GN, 8 groups, or the largest divisor of the channel count not exceeding 8)
and a ReLU nonlinearity, mapping to $c$ output channels.

$$\text{Down}(h) = \text{Conv3D}_{3\times3\times3,\,\text{stride}=2}(h)$$
a stride-2 convolution halving spatial resolution (learned downsampling).

$$\text{Up}(h) = \text{Conv3D}_{3\times3\times3}\big(\text{Upsample}_{\times 2,\,\text{nearest}}(h)\big)$$
nearest-neighbor spatial upsampling by a factor of 2 followed by a
convolution, avoiding the checkerboard artifacts associated with
transposed convolution.

**Encoder** $E$, $L = 4$ levels, channel widths
$c_l = 32 \times m_l$ for $m = (1, 2, 4, 8)$, i.e. $(32, 64, 128, 256)$:
$$h_0 = \text{ConvBlock}_{c_0}(x), \qquad h_l = \text{ConvBlock}_{c_l}\big(\text{Down}(h_{l-1})\big) \;\; \text{for } l = 1, \dots, L-1$$
The pre-downsampling feature map at each of the first $L-1$ levels is
retained as a skip connection $s_l = h_l$ for $l = 0, \dots, L-2$.

**Bottleneck** $B$: realized as the deepest encoder level itself
($h_{L-1}$, 256 channels) — the architecture does not include a separate,
distinct bottleneck module beyond the last encoder `ConvBlock`.

**Decoder** $D$, applied for $l = L-2, \dots, 0$ in decreasing order:
$$h_l' = \text{ConvBlock}_{c_l}\big([\,\text{Up}(h_{l+1}') \; ; \; s_l\,]\big), \qquad h_{L-1}' := h_{L-1}$$
where $[\, \cdot \; ; \; \cdot \,]$ denotes concatenation along the channel
dimension.

**Output head.** A final $1\times1\times1$ convolution
$\text{Conv}_{\text{out}}$ maps the last decoder feature map $h_0'$ to a
single output channel; both its weights and bias are zero-initialized, so
that training begins from a spatially flat, uninformative output.

**Stage 1 output (regression).**
$$\hat{y} = f_\theta(\text{MRI}) = \tanh\big(\text{Conv}_{\text{out}}(h_0')\big) \in [-1, 1]^{D \times H \times W}$$
a per-voxel real-valued predicted CT intensity, in the same normalized
scale as the ground-truth CT (Section 3).

**Stage 3 output (segmentation).**
$$z = f_\theta(\text{CT}) = \text{Conv}_{\text{out}}(h_0') \qquad \text{(raw logits, no activation applied inside the network)}$$
$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]^{D \times H \times W}$$
a per-voxel tumor probability map. The sigmoid is applied by the training
and inference code that calls the model, not inside the forward pass
itself, so that the numerically stable, autocast-compatible
`binary_cross_entropy_with_logits` formulation (Section 5.2) can be used
directly on the raw logits during mixed-precision training.

**Input/output shape.** Both networks are trained on fixed 96 x 96 x 64
voxel patches (single input channel, single output channel, matching
spatial shape in and out by construction). At inference, both models
support full, arbitrarily large volumes via a sliding-window procedure:
the volume is tiled into overlapping windows of the trained patch size
(50% stride overlap), each tile is passed through the network
independently, and overlapping predictions are recombined (uniform
averaging for Stage 1; a Gaussian-weighted average, peaked at each tile's
center, for Stage 3). This requires the tile size to be exactly divisible
by $2^{L-1} = 8$, a condition validated explicitly before inference.

## 5. Training Procedure

Both stages were trained with the AdamW optimizer, a cosine learning-rate
schedule with linear warmup, automatic mixed-precision (AMP) training with
gradient scaling, and gradient-norm clipping. Training configuration
(loss-function selection and its class-imbalance weighting for Stage 3,
learning-rate schedule, and numerical-precision handling under mixed
precision) was refined iteratively based on empirical validation
performance before the configurations reported below were reached.

### 5.1 Stage 1: MRI-to-CT translation

Loss (mean absolute error / L1, over all $N$ voxels of a training batch):
$$\mathcal{L}_{\text{Stage 1}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left| \hat{y}_i - y_i \right|$$
where $\hat{y}_i = f_\theta(\text{MRI})_i$ and $y_i$ is the corresponding
ground-truth normalized CT intensity.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $2\times10^{-4}$ |
| Weight decay | 0 |
| LR schedule | Cosine, 200-step linear warmup |
| Total training steps | 20,000 |
| Batch size | 2 |
| Patch size | $96 \times 96 \times 64$ |
| Mixed precision | Enabled (AMP) |
| Gradient clipping | Max norm 1.0 |
| EMA decay | 0.999 (recorded, not used for evaluation — see Section 5.3) |
| Hardware | NVIDIA Tesla T4 GPU (Kaggle) |

### 5.2 Stage 3: CT tumor segmentation

Let $z_i$ denote the raw logit and $\hat{p}_i = \sigma(z_i)$ the predicted
probability for voxel $i$, and $y_i \in \{0, 1\}$ the corresponding
ground-truth binary label.

Two loss formulations were implemented and made available for this stage.
The Dice loss (also implemented, but not the loss used for the reported
training run):
$$\mathcal{L}_{\text{Dice}}(\hat{p}, y) = 1 - \frac{2\sum_i \hat{p}_i y_i + \varepsilon}{\sum_i \hat{p}_i + \sum_i y_i + \varepsilon}, \qquad \varepsilon = 1.0$$

The focal loss (Lin et al., adapted for binary segmentation), used for the
training run reported in this document:
$$\text{bce}_i = -\big[y_i \log \sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\big]$$
computed via the numerically stable
`binary_cross_entropy_with_logits` formulation rather than a direct
$\log(\sigma(\cdot))$ evaluation, for stability under mixed-precision
autocast. Then:
$$p_{t,i} = e^{-\text{bce}_i}, \qquad \alpha_{t,i} = \alpha \, y_i + (1-\alpha)(1-y_i)$$
$$\mathcal{L}_{\text{Stage 3}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \alpha_{t,i}\,(1 - p_{t,i})^{\gamma}\, \text{bce}_i$$
with $\gamma = 2.0$ and $\alpha = 0.75$, the exact configuration values
used for the reported run. $\alpha_{t,i}$ up-weights the tumor class
(assigned weight $\alpha = 0.75$) relative to the background class
(assigned weight $1-\alpha = 0.25$), addressing the severe voxel-level
class imbalance between tumor and background.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $2\times10^{-4}$ |
| Weight decay | $1\times10^{-5}$ |
| LR schedule | Cosine, 200-step linear warmup |
| Total training steps | 20,000 |
| Batch size | 2 |
| Patch size | $96 \times 96 \times 64$ |
| Loss function | Focal loss ($\gamma=2.0$, $\alpha=0.75$) |
| Foreground-biased patch sampling probability | 0.5 |
| Mixed precision | Enabled (AMP) |
| Gradient clipping | Max norm 1.0 |
| EMA decay | 0.999 (recorded, not used for evaluation — see Section 5.3) |
| Hardware | NVIDIA Tesla T4 GPU (Kaggle) |

### 5.3 Weight selection for evaluation

An exponential moving average (EMA) of model weights (decay = 0.999) was
maintained and saved alongside every checkpoint for both stages, but all
reported evaluation results use the raw (non-EMA) trained weights; the EMA
weights were not used in any reported result.

## 6. Evaluation Metrics

### 6.1 Stage 1 (translation quality)

Evaluated on the held-out SynthRAD validation split (Section 2.5), after
de-normalizing predicted and ground-truth CT volumes back to Hounsfield
units:

$$\text{PSNR} = 10 \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right), \qquad \text{MSE} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2$$

computed both over the whole volume and restricted to foreground
(brain-mask) voxels only, and the structural similarity index (SSIM,
standard implementation), computed over the whole volume. On this held-out
validation split, the reported foreground PSNR reached 28.21 dB at
training step 20,000. Metrics are aggregated as the mean across the
validation patient cohort; per-patient distributions are also recorded to
support mean $\pm$ standard deviation reporting.

### 6.2 Stage 3 (segmentation quality)

The Dice similarity coefficient, computed at a fixed probability threshold
of 0.5:
$$\text{Dice}(\hat{P}, T) = \frac{2\,|\hat{P} \cap T| + \varepsilon}{|\hat{P}| + |T| + \varepsilon}, \qquad \varepsilon = 1.0$$
where $\hat{P} = \{i : \hat{p}_i > 0.5\}$ is the thresholded prediction and
$T$ the ground-truth binary tumor mask.

For the external Jordan validation set (Section 2.4), the Intersection over
Union is additionally reported:
$$\text{IoU}(\hat{P}, T) = \frac{|\hat{P} \cap T| + \varepsilon}{|\hat{P} \cup T| + \varepsilon}$$

Both metrics are aggregated as mean and standard deviation across the
evaluated patient (synthetic validation split) or slice (Jordan) cohort.
Full-volume inference for the synthetic validation split and pseudo-volume
inference for the Jordan slices (Section 4, sliding-window procedure) are
both used to compute these metrics rather than evaluating on training-scale
patches directly.

*Note on the periodic in-training validation signal:* during training, a
cheaper, patch-level Dice check is computed periodically for monitoring
purposes only, using a center-cropped (not foreground-centered) sub-volume
of each validation patient rather than the full reconstructed volume
described above. This periodic value was used only to monitor training
progress and select checkpoints during development; it is not equivalent
to, and should not be reported interchangeably with, the full-volume
metric described in this section.

## 7. Statistical Analysis

No formal inferential statistical hypothesis testing (e.g., paired t-test,
Wilcoxon signed-rank test) was performed in this study. Quantitative
results are reported descriptively, as mean and standard deviation across
the respective held-out or external validation cohort.

---

*A note on scope, not for inclusion in the Methodology section itself: the
following points are referenced above only in neutral, non-conclusive
language and are flagged here as candidates for a Discussion/Limitations
section rather than as settled methodological claims: (1) the effect of
network capacity (`base_channels`) on achievable accuracy was not
systematically investigated — a single configuration was used throughout;
(2) the Jordan external dataset's 8-bit windowed RGB format means no
Hounsfield-unit-based intensity reasoning can be assessed on it, only
predicted-shape agreement with the annotated outline, and its 2D,
single-slice-per-patient structure required a pseudo-3D workaround
(replicating each slice along the depth axis) rather than providing genuine
volumetric context; (3) the synthetic CT used as Stage 3's segmentation
input is itself a model-generated approximation of a real CT, not a real
CT — the extent to which Stage 1 translation error propagates into and
bounds achievable Stage 3 segmentation accuracy was not separately
quantified.*
