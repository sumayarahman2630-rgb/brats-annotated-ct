"""Pipeline role: the model at the center of the active pipeline -- a plain
3D regression U-Net for direct MRI -> CT translation (encoder-decoder with
skip connections, L1 loss computed by the caller), no diffusion/timestep
conditioning at all. Deliberately much simpler than the archived wavelet
diffusion model (no attention, no FiLM timestep embedding, no wavelet
transform), and empirically the better result: 28.21 dB foreground PSNR on
held-out validation patients at step 20000, vastly ahead of the diffusion
checkpoint's ~9 dB (see DEVELOPMENT_LOG.md and the main README for the comparison).
No shared code with the archived models -- same pipeline-isolation
reasoning throughout this project: a bug here can't affect archived code,
and vice versa.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _safe_num_groups(num_groups: int, channels: int) -> int:
    """GroupNorm requires channels % groups == 0 -- fall back to the largest
    divisor <= num_groups instead of assuming channels is always a multiple
    of num_groups (same fix as models/unet3d.py's helper of the same name,
    duplicated rather than imported to keep this file pipeline-isolated)."""
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


class UNet3DRegression(nn.Module):
    """Plain forward regressor: mri -> predicted_ct, both [-1, 1]-normalized
    single-channel volumes. `forward(mri)` returns the prediction directly
    -- no timestep argument, no sampling loop, since this is deterministic
    paired translation, not diffusion.

    Fully convolutional: trained on fixed-size patches (data.patch_size in
    the config) but can run on full, larger volumes at inference time as
    long as each spatial dim is divisible by 2**(len(channel_mult) - 1) --
    data/preprocessing.py's pad_to_multiple(spatial_multiple=16) already
    guarantees this for the standard channel_mult depths used here.
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
        flat tanh(0)=0 output rather than noise -- a stable starting point
        for training."""
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

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        """Single deterministic forward pass: mri -> predicted_ct, same
        spatial shape in and out (as long as input dims are divisible by
        2**(num_levels-1), see the class docstring). Encoder pushes onto
        `skips` on the way down; decoder pops and concatenates them on the
        way up, standard U-Net skip-connection wiring."""
        h = mri
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

        return torch.tanh(self.out_conv(h))

    @torch.no_grad()
    def predict_full_volume(
        self,
        mri: torch.Tensor,
        patch_size: tuple[int, int, int],
        stride_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Sliding-window inference for volumes bigger than what fits in a
        single forward pass. A real full brain-crop volume OOM'd on a real
        Kaggle T4 (2026-07-16) even under torch.no_grad() + AMP -- this is
        this architecture's genuine peak-activation-memory ceiling at full
        resolution, not a training-vs-eval-mode bug (training itself never
        OOM'd, because it only ever saw patch_size-sized inputs). The fix
        used for the cheap in-training validation check (just center-crop
        to a patch) doesn't work here: Stage 2's deliverable and the val
        comparison script both need the WHOLE brain's output, not a crop
        of it.

        Splits `mri` (1, 1, D, H, W) into overlapping patch_size windows,
        runs forward() on each independently (same bounded, patch-scale
        memory footprint training already proved safe), and blends
        overlapping regions by uniform averaging. stride_ratio=0.5 (50%
        overlap) matches the common default for sliding-window medical
        image inference (e.g. MONAI) -- lower it (e.g. 0.75, less overlap)
        for faster but slightly seamier output if a full cohort run is
        time-constrained.
        """
        self.eval()
        _, _, D, H, W = mri.shape
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

        accum = torch.zeros_like(mri)
        weight = torch.zeros_like(mri)

        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    d1, h1, w1 = min(d0 + pd, D), min(h0 + ph, H), min(w0 + pw, W)
                    patch = mri[:, :, d0:d1, h0:h1, w0:w1]
                    pad = (0, pw - (w1 - w0), 0, ph - (h1 - h0), 0, pd - (d1 - d0))
                    if any(p != 0 for p in pad):
                        patch = torch.nn.functional.pad(patch, pad, mode="constant", value=-1.0)
                    pred_patch = self.forward(patch)
                    pred_patch = pred_patch[:, :, : (d1 - d0), : (h1 - h0), : (w1 - w0)]
                    accum[:, :, d0:d1, h0:h1, w0:w1] += pred_patch
                    weight[:, :, d0:d1, h0:h1, w0:w1] += 1.0

        return accum / weight.clamp_min(1.0)


def build_regression_model(config: dict) -> UNet3DRegression:
    """Factory matching every other Stage 1 model file's build_*_model
    convention -- constructs UNet3DRegression from a config's `model`
    section so nothing about the architecture is hardcoded outside the config."""
    model_cfg = config["model"]
    return UNet3DRegression(
        in_channels=1,
        out_channels=1,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mult=tuple(model_cfg.get("channel_mult", (1, 2, 4, 8))),
        num_groups=model_cfg.get("num_groups", 8),
    )
