"""2D conditional U-Net denoiser for the standalone 2D slice-based pipeline
(models/stage1_mri2ct_ddpm_2d.py). Same building blocks and lessons learned
as the 3D U-Net (models/unet3d.py) -- FiLM-style timestep conditioning,
self-attention at coarse resolutions, activation checkpointing, safe
GroupNorm group counts -- reimplemented in 2D rather than importing the 3D
version with a dims= flag, so this file can be read and reasoned about on
its own, independent of the 3D pipeline (see PROJECT_NOTES.md: the two pipelines
are deliberately kept separate, one mustn't be able to break the other).

No wavelet transform here: a single 2D slice (e.g. 256x256) is already
small enough for full-resolution pixel-space diffusion to be affordable --
wavelet-domain compression was specifically a 3D full-volume necessity, not
a 2D one. This matches standard practice for 2D paired medical image
translation diffusion (Palette/SR3-style: concatenate the condition to the
noisy target channel-wise, plain pixel-space DDPM U-Net, no CFG needed for
deterministic paired translation -- same reasoning as the 3D model's
architecture decision record in PROJECT_NOTES.md).
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

log = logging.getLogger(__name__)

_ATTENTION_N_WARN_THRESHOLD = 64 ** 2  # 2D analog of the 3D guard; 64x64 dense attention is already ~cheap


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _safe_num_groups(num_groups: int, channels: int) -> int:
    """See models/unet3d.py's identical helper and the round-4 GroupNorm
    divisibility bug it fixes -- same issue applies here."""
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _norm_act(num_groups: int, channels: int) -> nn.Sequential:
    return nn.Sequential(nn.GroupNorm(_safe_num_groups(num_groups, channels), channels), nn.SiLU())


class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_groups: int = 32, dropout: float = 0.0, use_checkpoint: bool = False):
        super().__init__()
        self.in_norm_act = _norm_act(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_channels))

        self.out_norm = nn.GroupNorm(_safe_num_groups(num_groups, out_channels), out_channels)
        self.out_act_drop = nn.Sequential(nn.SiLU(), nn.Dropout(dropout))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.in_norm_act(x))
        scale, shift = self.emb_proj(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.conv2(self.out_act_drop(h))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_safe_num_groups(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self._warned_large_n = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        if n > _ATTENTION_N_WARN_THRESHOLD and not self._warned_large_n:
            est_gib = (b * self.num_heads * n * n * 4) / (1024 ** 3)
            log.warning(
                "SelfAttention2D got input spatial shape (%d, %d) -> N=%d. Dense attention "
                "matrix here is roughly %.2f GiB (batch=%d, heads=%d, fp32). If this OOMs, "
                "check model.attention_resolutions against the actual U-Net depth.",
                h, w, n, est_gib, b, self.num_heads,
            )
            self._warned_large_n = True
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, n)
        q, k, v = qkv.unbind(1)
        q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class ResAttnBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_groups: int, dropout: float, use_attn: bool, num_heads: int, use_checkpoint: bool = False):
        super().__init__()
        self.res = ResBlock2D(in_channels, out_channels, emb_dim, num_groups, dropout, use_checkpoint=use_checkpoint)
        self.attn = SelfAttention2D(out_channels, num_heads, num_groups) if use_attn else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet2D(nn.Module):
    """Predicts noise given a noisy 2D slice and a diffusion timestep.
    Conditioning (the MRI slice) must already be concatenated onto `x` by
    the caller, so in_channels here is the total channel count the first
    conv sees (noisy target + condition)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (4, 8),
        num_heads: int = 4,
        num_groups: int = 32,
        dropout: float = 0.0,
        use_checkpoint: bool | tuple[int, ...] = False,
    ):
        super().__init__()

        def checkpoint_here(resolution_factor: int) -> bool:
            if isinstance(use_checkpoint, bool):
                return use_checkpoint
            return resolution_factor in use_checkpoint

        emb_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        num_levels = len(channel_mult)
        self.num_levels = num_levels

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        skip_channels = [ch]
        resolution_factor = 1
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                use_attn = resolution_factor in attention_resolutions
                level_blocks.append(ResAttnBlock2D(ch, out_ch, emb_dim, num_groups, dropout, use_attn, num_heads, use_checkpoint=checkpoint_here(resolution_factor)))
                ch = out_ch
                skip_channels.append(ch)
            self.down_blocks.append(level_blocks)
            if level != num_levels - 1:
                self.downsamplers.append(Downsample2D(ch))
                skip_channels.append(ch)
                resolution_factor *= 2

        self.mid_block1 = ResBlock2D(ch, ch, emb_dim, num_groups, dropout, use_checkpoint=checkpoint_here(resolution_factor))
        self.mid_attn = SelfAttention2D(ch, num_heads, num_groups)
        self.mid_block2 = ResBlock2D(ch, ch, emb_dim, num_groups, dropout, use_checkpoint=checkpoint_here(resolution_factor))

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                use_attn = resolution_factor in attention_resolutions
                level_blocks.append(ResAttnBlock2D(ch + skip_ch, out_ch, emb_dim, num_groups, dropout, use_attn, num_heads, use_checkpoint=checkpoint_here(resolution_factor)))
                ch = out_ch
            self.up_blocks.append(level_blocks)
            if level != 0:
                self.upsamplers.append(Upsample2D(ch))
                resolution_factor //= 2

        self.out_norm_act = _norm_act(num_groups, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(timesteps, self.base_channels))

        h = self.in_conv(x)
        skips = [h]
        for level in range(self.num_levels):
            for block in self.down_blocks[level]:
                h = block(h, emb)
                skips.append(h)
            if level != self.num_levels - 1:
                h = self.downsamplers[level](h)
                skips.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for i, level_blocks in enumerate(self.up_blocks):
            for block in level_blocks:
                skip = skips.pop()
                h = block(torch.cat([h, skip], dim=1), emb)
            if i < len(self.upsamplers):
                h = self.upsamplers[i](h)

        return self.out_conv(self.out_norm_act(h))
