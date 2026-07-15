"""3D Haar discrete wavelet transform, implemented as three sequential 1D
Haar transforms (along D, then H, then W). This is standard separable
wavelet math, reimplemented directly rather than taken from any reference
repo's DWT layer. Orthonormal, so `haar_idwt3d(haar_dwt3d(x)) == x` exactly
(up to floating point) -- this property is what lets the diffusion model
operate purely in wavelet space and reconstruct a lossless-transform image
at the end, with no interpolation artifacts from the transform itself.

Subband channel order after haar_dwt3d: LLL, LLH, LHL, LHH, HLL, HLH, HHL,
HHH, where the three letters are (D-axis, H-axis, W-axis) each L(ow) or
H(igh). LLL is the coarse structure (a half-resolution version of the
volume); the other 7 are high-frequency detail.
"""
from __future__ import annotations

import math

import torch


def _haar_dwt_1d(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.movedim(dim, -1)
    even, odd = x[..., 0::2], x[..., 1::2]
    low = (even + odd) / math.sqrt(2)
    high = (even - odd) / math.sqrt(2)
    return low.movedim(-1, dim), high.movedim(-1, dim)


def _haar_idwt_1d(low: torch.Tensor, high: torch.Tensor, dim: int) -> torch.Tensor:
    low = low.movedim(dim, -1)
    high = high.movedim(dim, -1)
    even = (low + high) / math.sqrt(2)
    odd = (low - high) / math.sqrt(2)
    interleaved = torch.stack([even, odd], dim=-1).flatten(-2)
    return interleaved.movedim(-1, dim)


def haar_dwt3d(x: torch.Tensor) -> torch.Tensor:
    """x: (B, C, D, H, W) with D, H, W all even -> (B, 8*C, D/2, H/2, W/2)."""
    l, h = _haar_dwt_1d(x, dim=2)
    ll, lh = _haar_dwt_1d(l, dim=3)
    hl, hh = _haar_dwt_1d(h, dim=3)
    lll, llh = _haar_dwt_1d(ll, dim=4)
    lhl, lhh = _haar_dwt_1d(lh, dim=4)
    hll, hlh = _haar_dwt_1d(hl, dim=4)
    hhl, hhh = _haar_dwt_1d(hh, dim=4)
    return torch.cat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=1)


def haar_idwt3d(coeffs: torch.Tensor, num_input_channels: int = 1) -> torch.Tensor:
    """Inverse of haar_dwt3d. coeffs: (B, 8*C, D/2, H/2, W/2) -> (B, C, D, H, W)."""
    c = num_input_channels
    lll, llh, lhl, lhh, hll, hlh, hhl, hhh = torch.split(coeffs, c, dim=1)
    ll = _haar_idwt_1d(lll, llh, dim=4)
    lh = _haar_idwt_1d(lhl, lhh, dim=4)
    hl = _haar_idwt_1d(hll, hlh, dim=4)
    hh = _haar_idwt_1d(hhl, hhh, dim=4)
    l = _haar_idwt_1d(ll, lh, dim=3)
    h = _haar_idwt_1d(hl, hh, dim=3)
    return _haar_idwt_1d(l, h, dim=2)
