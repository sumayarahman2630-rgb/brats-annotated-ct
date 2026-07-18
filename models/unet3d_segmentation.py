"""Pipeline role: Stage 3's model -- binary tumor segmentation from a CT
volume. Architecturally the same plain 3D encoder-decoder U-Net as
models/unet3d_regression.py (Stage 1), adapted for a different task: input
is a CT volume (not MRI), output is a per-voxel tumor probability map via
sigmoid (not a predicted CT via tanh). Loss (Dice+BCE) is computed by the
caller (training/train_stage3_segmentation.py), same convention as Stage 1.

Deliberately its own file with no shared code with unet3d_regression.py --
same pipeline-isolation reasoning used throughout this project: identical
building blocks (ConvBlock3D, Down3D, Up3D, _safe_num_groups) are
duplicated rather than imported, so a change to one stage's model can
never silently affect another's.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _gaussian_importance_map(patch_size: tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Per-voxel blend weight for one sliding-window tile, peaked at the
    patch center and decaying toward its edges (Gaussian, sigma =
    sigma_scale * each dimension -- the same default MONAI's sliding-window
    inference uses). Found (2026-07-19) that predict_full_volume's ORIGINAL
    uniform weighting (every voxel in every tile weighted equally, weight=1.0)
    caused a real, reproducible bug: a model that predicts a patch's tumor
    region reasonably but imprecisely (soft/uncertain probability bleeding
    toward the patch edges, not a hard clean boundary -- the normal, expected
    behavior of any real trained segmentation model, not itself a bug) gets
    that imprecision AVERAGED UNIFORMLY across every overlapping tile,
    smearing the reconstructed full-volume prediction well beyond the true
    region. Weighting each tile's contribution by confidence-in-its-own-
    center (Gaussian) instead of uniformly means each tile's own edge
    uncertainty contributes less to the final blend, and tiles that are
    actually centered on the true region dominate there -- verified
    (tests/test_stage3_segmentation.py) to noticeably tighten the
    reconstructed region back toward the true bounding box on a controlled
    diagnostic case."""
    coords = [np.arange(s, dtype=np.float32) for s in patch_size]
    grids = np.meshgrid(*coords, indexing="ij")
    center = [(s - 1) / 2.0 for s in patch_size]
    sigma = [max(s * sigma_scale, 1e-3) for s in patch_size]
    exponent = sum(((g - c) ** 2) / (2.0 * sg ** 2) for g, c, sg in zip(grids, center, sigma))
    gaussian = np.exp(-exponent)
    gaussian = gaussian / gaussian.max()
    gaussian = np.clip(gaussian, 1e-3, None)  # never exactly 0 -- a lone edge tile must still contribute something
    return torch.from_numpy(gaussian.astype(np.float32))


def _safe_num_groups(num_groups: int, channels: int) -> int:
    """GroupNorm requires channels % groups == 0 -- fall back to the largest
    divisor <= num_groups instead of assuming channels is always a multiple
    of num_groups (same fix as models/unet3d_regression.py's helper of the
    same name, duplicated to keep this file pipeline-isolated)."""
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock3D(nn.Module):
    """Two conv-groupnorm-activation layers, no residual/skip inside the
    block itself (the U-Net's own encoder-decoder skip connections are
    separate, at the ConvBlock granularity, not inside it)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_safe_num_groups(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_safe_num_groups(num_groups, out_channels), out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class Down3D(nn.Module):
    """Halves spatial resolution via a stride-2 conv (learned downsampling,
    not a fixed pooling op)."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up3D(nn.Module):
    """Doubles spatial resolution via nearest-neighbor upsample + conv
    (avoids the checkerboard artifacts transposed conv can introduce)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet3DSegmentation(nn.Module):
    """Plain forward segmenter: ct -> tumor_logits, input a [-1, 1]-
    normalized single-channel CT volume, output a single-channel
    per-voxel RAW LOGIT map (not a probability -- no sigmoid applied
    here). `forward(ct)` returns the logits directly, no sampling loop,
    same deterministic-single-pass design as Stage 1's regression model.

    Deliberately returns logits, not sigmoid(logits): torch.nn.BCELoss /
    F.binary_cross_entropy are explicitly unsafe under CUDA autocast (they
    require an already-in-[0,1] input, which fp16 casting can silently
    corrupt) -- PyTorch's own fix is to keep the model's output as logits
    and use F.binary_cross_entropy_with_logits instead, which fuses the
    sigmoid and the loss in a numerically stable, autocast-safe way. See
    training/train_stage3_segmentation.py's combined_loss for where the
    sigmoid actually happens for the Dice half of the loss, and
    predict_full_volume below for where it happens at inference time.
    Hit as a real crash on a real Kaggle GPU (2026-07-19) -- this file
    originally applied sigmoid() inside forward() itself, which is the
    wrong place for exactly this reason.

    Fully convolutional: trained on fixed-size patches (data.patch_size in
    the config) but can run on full, larger volumes via predict_full_volume
    below, same sliding-window approach as Stage 1 (that model's full-volume
    single-pass forward genuinely OOM'd on a real Kaggle T4 at this
    parameter scale -- this architecture is close enough in compute/memory
    profile that the same precaution applies here from the start, rather
    than waiting to hit the same wall).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_groups: int = 8,
    ):
        """Builds a symmetric encoder/decoder: len(channel_mult) levels,
        len(channel_mult)-1 down/up-sample steps, skip connections between
        matching encoder/decoder levels (standard U-Net topology). The
        final conv is zero-initialized so the model starts by predicting a
        flat logit=0 (i.e. sigmoid(0)=0.5 once a probability is actually
        needed) everywhere -- an uninformative but stable starting point
        (Dice+BCE loss pulls it toward the true, heavily-background-biased
        distribution from there)."""
        super().__init__()
        self.num_levels = len(channel_mult)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = in_channels
        skip_channels = []
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            self.down_blocks.append(ConvBlock3D(ch, out_ch, num_groups))
            ch = out_ch
            skip_channels.append(ch)
            if level != self.num_levels - 1:
                self.downsamplers.append(Down3D(ch))

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult[:-1]))):
            out_ch = base_channels * mult
            self.upsamplers.append(Up3D(ch))
            skip_ch = skip_channels[level]
            self.up_blocks.append(ConvBlock3D(ch + skip_ch, out_ch, num_groups))
            ch = out_ch

        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        """Single deterministic forward pass: ct -> tumor logit map (NOT a
        probability -- see class docstring for why), same spatial shape in
        and out (as long as input dims are divisible by 2**(num_levels-1)).
        Encoder pushes onto `skips` on the way down; decoder pops and
        concatenates them on the way up, standard U-Net skip-connection
        wiring."""
        h = ct
        skips = []
        for level in range(self.num_levels):
            h = self.down_blocks[level](h)
            if level != self.num_levels - 1:
                skips.append(h)
                h = self.downsamplers[level](h)

        for i in range(len(self.up_blocks)):
            h = self.upsamplers[i](h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[i](h)

        return self.out_conv(h)

    @torch.no_grad()
    def predict_full_volume(
        self,
        ct: torch.Tensor,
        patch_size: tuple[int, int, int],
        stride_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Sliding-window inference for volumes bigger than what fits in a
        single forward pass -- same overall algorithm as
        models/unet3d_regression.py's method of the same name (see that
        docstring for the full reasoning and the real-Kaggle OOM that
        motivated it). Splits `ct` (1, 1, D, H, W) into overlapping
        patch_size windows, runs forward() on each independently, applies
        sigmoid to convert that patch's logits to a probability, and blends
        overlapping regions by a GAUSSIAN-weighted average (see
        _gaussian_importance_map) -- NOT uniform averaging. Found
        (2026-07-19) that uniform averaging let each tile's edge
        imprecision smear the reconstructed region well beyond the true
        one; weighting each tile toward its own center suppresses that.
        Averaging PROBABILITIES (not logits) across overlaps is still the
        mathematically correct choice regardless of weighting scheme
        (sigmoid is nonlinear, so averaging logits and sigmoiding once at
        the end is not equivalent). Returns probabilities in [0, 1], unlike
        forward() itself which returns raw logits.
        """
        self.eval()
        divisor = 2 ** (self.num_levels - 1)
        if any(p % divisor != 0 for p in patch_size):
            raise ValueError(
                f"patch_size {patch_size} must be divisible by {divisor} (2**(num_levels-1), "
                f"num_levels={self.num_levels}) -- each tile is fed through the same encoder/decoder "
                "as training, and a non-divisible size makes the skip-connection feature maps "
                "mismatch in shape (crashes inside forward() with a confusing torch.cat error "
                "instead of this clear one)."
            )
        _, _, D, H, W = ct.shape
        pd, ph, pw = patch_size
        stride = (
            max(1, int(pd * stride_ratio)),
            max(1, int(ph * stride_ratio)),
            max(1, int(pw * stride_ratio)),
        )

        def starts(size: int, patch: int, step: int) -> list[int]:
            if size <= patch:
                return [0]
            pts = list(range(0, size - patch + 1, step))
            if pts[-1] != size - patch:
                pts.append(size - patch)
            return pts

        d_starts = starts(D, pd, stride[0])
        h_starts = starts(H, ph, stride[1])
        w_starts = starts(W, pw, stride[2])

        importance = _gaussian_importance_map(patch_size).to(device=ct.device, dtype=ct.dtype)

        accum = torch.zeros_like(ct)
        weight = torch.zeros_like(ct)

        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    d1, h1, w1 = min(d0 + pd, D), min(h0 + ph, H), min(w0 + pw, W)
                    patch = ct[:, :, d0:d1, h0:h1, w0:w1]
                    pad = (0, pw - (w1 - w0), 0, ph - (h1 - h0), 0, pd - (d1 - d0))
                    if any(p != 0 for p in pad):
                        patch = torch.nn.functional.pad(patch, pad, mode="constant", value=-1.0)
                    pred_patch = torch.sigmoid(self.forward(patch))
                    pred_patch = pred_patch[:, :, : (d1 - d0), : (h1 - h0), : (w1 - w0)]
                    local_importance = importance[: (d1 - d0), : (h1 - h0), : (w1 - w0)]
                    accum[:, :, d0:d1, h0:h1, w0:w1] += pred_patch * local_importance
                    weight[:, :, d0:d1, h0:h1, w0:w1] += local_importance

        return accum / weight.clamp_min(1e-6)


def build_segmentation_model(config: dict) -> UNet3DSegmentation:
    """Factory matching every other Stage model file's build_*_model
    convention -- constructs UNet3DSegmentation from a config's `model`
    section so nothing about the architecture is hardcoded outside the config."""
    model_cfg = config["model"]
    return UNet3DSegmentation(
        in_channels=1,
        out_channels=1,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mult=tuple(model_cfg.get("channel_mult", (1, 2, 4, 8))),
        num_groups=model_cfg.get("num_groups", 8),
    )
