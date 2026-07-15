"""3D conditional U-Net denoiser backbone: timestep-conditioned residual
blocks (FiLM-style scale/shift from the timestep embedding, following the
standard DDPM U-Net design), self-attention at the coarsest resolutions
(affordable here because the wavelet transform already halves the spatial
size before this network ever sees the volume), strided-conv down/upsampling.
Reimplemented independently -- same building blocks as any standard DDPM
U-Net (Ho et al. / guided-diffusion lineage), not copied from a specific repo.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _norm_act(num_groups: int, channels: int) -> nn.Sequential:
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.Sequential(nn.GroupNorm(groups, channels), nn.SiLU())


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.in_norm_act = _norm_act(num_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_channels))

        self.out_norm = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.out_act_drop = nn.Sequential(nn.SiLU(), nn.Dropout(dropout))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.in_norm_act(x))
        scale, shift = self.emb_proj(emb)[:, :, None, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.conv2(self.out_act_drop(h))
        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        n = d * h * w
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, n)
        q, k, v = qkv.unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))  # (b, heads, n, c_head)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, d, h, w)
        return x + self.proj(out)


class ResAttnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_groups: int, dropout: float, use_attn: bool, num_heads: int):
        super().__init__()
        self.res = ResBlock3D(in_channels, out_channels, emb_dim, num_groups, dropout)
        self.attn = SelfAttention3D(out_channels, num_heads, num_groups) if use_attn else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet3D(nn.Module):
    """Predicts noise (or x0) given a noisy input and a diffusion timestep.
    Conditioning (if any) must already be concatenated onto `x` by the
    caller before `forward`, so `in_channels` here is the *total* channel
    count the first conv sees (noisy target + condition, if used).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (2, 4),
        num_heads: int = 4,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        emb_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        num_levels = len(channel_mult)
        self.num_levels = num_levels

        # --- encoder: nested per-level module lists, built and consumed in lockstep ---
        self.down_blocks = nn.ModuleList()   # ModuleList[level] = ModuleList[ResAttnBlock]
        self.downsamplers = nn.ModuleList()  # length num_levels - 1

        ch = base_channels
        skip_channels = [ch]  # channel count of each tensor that will be pushed onto the skip stack
        resolution_factor = 1
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                use_attn = resolution_factor in attention_resolutions
                level_blocks.append(ResAttnBlock(ch, out_ch, emb_dim, num_groups, dropout, use_attn, num_heads))
                ch = out_ch
                skip_channels.append(ch)
            self.down_blocks.append(level_blocks)
            if level != num_levels - 1:
                self.downsamplers.append(Downsample3D(ch))
                skip_channels.append(ch)
                resolution_factor *= 2

        # --- bottleneck ---
        self.mid_block1 = ResBlock3D(ch, ch, emb_dim, num_groups, dropout)
        self.mid_attn = SelfAttention3D(ch, num_heads, num_groups)
        self.mid_block2 = ResBlock3D(ch, ch, emb_dim, num_groups, dropout)

        # --- decoder: mirrors the encoder, consuming skip_channels in LIFO order ---
        self.up_blocks = nn.ModuleList()    # ModuleList[level] = ModuleList[ResAttnBlock], levels in decoder order (coarse -> fine)
        self.upsamplers = nn.ModuleList()   # length num_levels - 1, aligned with up_blocks[:-1]

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                use_attn = resolution_factor in attention_resolutions
                level_blocks.append(ResAttnBlock(ch + skip_ch, out_ch, emb_dim, num_groups, dropout, use_attn, num_heads))
                ch = out_ch
            self.up_blocks.append(level_blocks)
            if level != 0:
                self.upsamplers.append(Upsample3D(ch))
                resolution_factor //= 2

        self.out_norm_act = _norm_act(num_groups, ch)
        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
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
