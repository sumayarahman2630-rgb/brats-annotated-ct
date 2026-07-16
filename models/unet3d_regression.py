"""Simple 3D regression U-Net for direct MRI -> CT translation: plain
encoder-decoder with skip connections, L1 loss (computed by the caller),
no diffusion/timestep conditioning at all. Deliberately much simpler than
models/unet3d.py (no attention, no FiLM timestep embedding, no wavelet
transform) -- this exists because a plain regression baseline, trained on
patches, empirically converged far faster than the wavelet diffusion model
(see CLAUDE.md). No shared code with unet3d.py/unet2d.py, same pipeline-
isolation reasoning as the 2D pipeline: a bug here can't affect the other
two model files.
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
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up3D(nn.Module):
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


def build_regression_model(config: dict) -> UNet3DRegression:
    model_cfg = config["model"]
    return UNet3DRegression(
        in_channels=1,
        out_channels=1,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mult=tuple(model_cfg.get("channel_mult", (1, 2, 4, 8))),
        num_groups=model_cfg.get("num_groups", 8),
    )
