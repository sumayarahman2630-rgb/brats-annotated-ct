"""UNet3D shape correctness, GroupNorm divisibility safety, and activation
checkpointing correctness. CPU-only, no GPU required.
"""
import torch

from archive.models.unet3d import UNet3D, _safe_num_groups


def test_safe_num_groups_falls_back_when_not_divisible():
    # base_channels=48 with channel_mult giving 48*1=48 channels at some level;
    # 48 is NOT divisible by the default num_groups=32 -- largest divisor of 48
    # that's <= 32 is 24 (48 = 24*2).
    assert _safe_num_groups(32, 48) == 24
    assert 48 % _safe_num_groups(32, 48) == 0


def test_safe_num_groups_uses_num_groups_when_it_divides_evenly():
    assert _safe_num_groups(32, 64) == 32


def test_forward_shape_and_backward():
    torch.manual_seed(0)
    model = UNet3D(
        in_channels=16, out_channels=8,
        base_channels=8, channel_mult=(1, 2, 4, 4),
        num_res_blocks=2, attention_resolutions=(8,),
        num_heads=2, num_groups=4, dropout=0.0,
    )
    x = torch.randn(1, 16, 16, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t)
    assert out.shape == (1, 8, 16, 16, 16)
    out.sum().backward()


def test_base_channels_not_divisible_by_32_does_not_crash():
    """Regression test for the GroupNorm bug found in round 4: base_channels
    values not divisible by num_groups=32 used to crash immediately."""
    model = UNet3D(
        in_channels=16, out_channels=8,
        base_channels=48, channel_mult=(1, 2, 4, 4),
        num_res_blocks=1, attention_resolutions=(8,),
        num_heads=2, num_groups=32, dropout=0.0,
    )
    x = torch.randn(1, 16, 16, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t)
    assert out.shape == (1, 8, 16, 16, 16)


def _build(use_checkpoint, seed=42):
    torch.manual_seed(seed)
    return UNet3D(
        in_channels=16, out_channels=8,
        base_channels=8, channel_mult=(1, 2, 4, 4),
        num_res_blocks=2, attention_resolutions=(8,),
        num_heads=2, num_groups=4, dropout=0.0,
        use_checkpoint=use_checkpoint,
    )


def test_checkpointing_is_numerically_transparent():
    """Activation checkpointing (torch.utils.checkpoint) must be a pure
    memory/compute tradeoff -- identical forward output and gradients with
    or without it, given the same weights and inputs."""
    model_plain = _build(False)
    model_ckpt = _build(True)
    for (_, p1), (_, p2) in zip(model_plain.named_parameters(), model_ckpt.named_parameters()):
        assert torch.equal(p1, p2)

    model_plain.train()
    model_ckpt.train()
    x = torch.randn(1, 16, 16, 16, 16)
    t = torch.randint(0, 1000, (1,))

    out_plain = model_plain(x, t)
    out_ckpt = model_ckpt(x, t)
    assert torch.allclose(out_plain, out_ckpt, atol=1e-6)

    out_plain.sum().backward()
    out_ckpt.sum().backward()
    for (_, p1), (_, p2) in zip(model_plain.named_parameters(), model_ckpt.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


def test_selective_checkpoint_resolutions_scope_correctly():
    """use_checkpoint accepts a list of resolution factors (round 5) to
    scope checkpointing to specific levels only."""
    model = UNet3D(
        in_channels=16, out_channels=8,
        base_channels=8, channel_mult=(1, 2, 4, 4),
        num_res_blocks=2, attention_resolutions=(8,),
        num_heads=2, num_groups=4, dropout=0.0,
        use_checkpoint=(1, 2),
    )
    flags = [b.res.use_checkpoint for level in model.down_blocks for b in level]
    # levels 0,1 (factors 1,2) -> True; levels 2,3 (factors 4,8) -> False
    assert flags == [True, True, True, True, False, False, False, False]
    assert model.mid_block1.use_checkpoint is False  # bottleneck factor=8, not in (1,2)


def test_eval_mode_skips_checkpointing_without_error():
    model = _build(True)
    model.eval()
    x = torch.randn(1, 16, 16, 16, 16)
    t = torch.randint(0, 1000, (1,))
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == (1, 8, 16, 16, 16)
