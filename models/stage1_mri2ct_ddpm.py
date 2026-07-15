"""Stage 1 model: conditional DDPM operating in Haar-wavelet space for
paired MRI -> CT translation, following cwdm's wavelet-domain-diffusion
pattern (reimplemented independently -- see CLAUDE.md's architecture
decision record for the reasoning, not just "inspired by").

Pipeline per training step:
  MRI, CT (both normalized to [-1, 1], same spatial shape) --Haar DWT-->
  8-channel condition, 8-channel x0 target, both at half resolution
  --q_sample--> noisy x_t --concat with condition--> UNet3D --> predicted
  noise --> MSE loss per subband.

Pipeline per inference step: iterative denoising in wavelet space (DDIM),
then a single Haar IDWT reconstructs the full-resolution synthetic CT --
this is the entire point of doing diffusion in wavelet space instead of
voxel space: the network only ever sees a volume at half the linear
resolution (1/8 the voxel count), while the final output is still full
resolution and artifact-free because the wavelet transform is an exact,
lossless linear transform, not a learned or lossy downsampling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet3d import UNet3D
from models.wavelet_transform import haar_dwt3d, haar_idwt3d

SUBBAND_NAMES = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]


def _make_linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(shape[0], *([1] * (len(shape) - 1))).float()


class Stage1MRI2CTDiffusion(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (2, 4),
        num_heads: int = 4,
        num_groups: int = 32,
        dropout: float = 0.0,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        subband_loss_weights: tuple[float, ...] | None = None,
        ddim_steps: int = 100,
        ddim_eta: float = 0.0,
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        # 8 subbands for the noisy CT target + 8 subbands for the MRI condition
        self.denoiser = UNet3D(
            in_channels=16,
            out_channels=8,
            base_channels=base_channels,
            channel_mult=tuple(channel_mult),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_resolutions),
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
        )

        betas = _make_linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt().float())

        weights = torch.tensor(subband_loss_weights, dtype=torch.float32) if subband_loss_weights else torch.ones(8)
        self.register_buffer("subband_loss_weights", weights)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_ac = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def training_losses(self, mri: torch.Tensor, ct: torch.Tensor, t: torch.Tensor | None = None) -> dict:
        """mri, ct: (B, 1, D, H, W), normalized to [-1, 1], same shape, D/H/W
        even (already guaranteed by data/preprocessing.py's pad_to_multiple)."""
        b = ct.shape[0]
        device = ct.device
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=device)

        cond = haar_dwt3d(mri)
        x0 = haar_dwt3d(ct)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        model_in = torch.cat([xt, cond], dim=1)
        pred_noise = self.denoiser(model_in, t)

        per_subband_mse = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=(0, 2, 3, 4))
        loss = (per_subband_mse * self.subband_loss_weights).mean()

        return {
            "loss": loss,
            "per_subband_mse": per_subband_mse.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        mri: torch.Tensor,
        num_steps: int | None = None,
        eta: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Returns synthetic CT, normalized to [-1, 1], same shape as `mri`."""
        device = mri.device
        num_steps = num_steps or self.ddim_steps
        eta = self.ddim_eta if eta is None else eta

        cond = haar_dwt3d(mri)
        shape = cond.shape
        x = torch.randn(shape, device=device, generator=generator)

        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).round().long()
        step_indices = torch.unique_consecutive(step_indices)

        for i, t in enumerate(step_indices):
            t_batch = t.expand(shape[0])
            model_in = torch.cat([x, cond], dim=1)
            eps = self.denoiser(model_in, t_batch)

            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_prev = (
                self.alphas_cumprod[step_indices[i + 1]] if i + 1 < len(step_indices) else torch.tensor(1.0, device=device)
            )

            x0_pred = (x - (1 - alpha_cumprod_t).sqrt() * eps) / alpha_cumprod_t.sqrt()

            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
            )
            dir_xt = (1 - alpha_cumprod_prev - sigma_t**2).clamp(min=0).sqrt() * eps
            step_noise = torch.randn(shape, device=device, generator=generator) if eta > 0 else 0.0

            x = alpha_cumprod_prev.sqrt() * x0_pred + dir_xt + sigma_t * step_noise

        ct_pred = haar_idwt3d(x, num_input_channels=1)
        return ct_pred.clamp(-1.0, 1.0)


def build_stage1_model(config: dict) -> Stage1MRI2CTDiffusion:
    model_cfg = config["model"]
    diff_cfg = config["diffusion"]
    return Stage1MRI2CTDiffusion(
        base_channels=model_cfg.get("base_channels", 64),
        channel_mult=tuple(model_cfg.get("channel_mult", (1, 2, 4, 4))),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", (2, 4))),
        num_heads=model_cfg.get("num_heads", 4),
        num_groups=model_cfg.get("num_groups", 32),
        dropout=model_cfg.get("dropout", 0.0),
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 2e-2),
        subband_loss_weights=diff_cfg.get("subband_loss_weights"),
        ddim_steps=diff_cfg.get("ddim_steps", 100),
        ddim_eta=diff_cfg.get("ddim_eta", 0.0),
    )
