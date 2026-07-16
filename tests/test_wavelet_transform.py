"""Haar DWT/IDWT exact-inverse checks. Formalizes the ad-hoc verification
done at the very start of this project. CPU-only, no GPU required.
"""
import torch

from models.wavelet_transform import haar_dwt3d, haar_idwt3d


def test_shape():
    x = torch.randn(2, 1, 8, 12, 16)
    coeffs = haar_dwt3d(x)
    assert coeffs.shape == (2, 8, 4, 6, 8)


def test_exact_inverse_single_channel():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 8, 12, 16)
    coeffs = haar_dwt3d(x)
    recon = haar_idwt3d(coeffs, num_input_channels=1)
    assert recon.shape == x.shape
    assert torch.allclose(recon, x, atol=1e-5)


def test_exact_inverse_multi_channel():
    torch.manual_seed(1)
    x = torch.randn(1, 3, 8, 8, 8)
    recon = haar_idwt3d(haar_dwt3d(x), num_input_channels=3)
    assert torch.allclose(recon, x, atol=1e-5)


def test_constant_volume_has_zero_detail_subbands():
    """A constant volume's Haar detail (high-frequency) subbands should be
    exactly zero -- this is what makes large constant background regions
    (e.g. CT air at -1000 HU) cheap to represent in wavelet space."""
    x = torch.full((1, 1, 8, 8, 8), 0.37)
    coeffs = haar_dwt3d(x)
    lll = coeffs[:, 0:1]
    detail = coeffs[:, 1:8]
    assert torch.allclose(detail, torch.zeros_like(detail), atol=1e-6)
    assert not torch.allclose(lll, torch.zeros_like(lll), atol=1e-6)
