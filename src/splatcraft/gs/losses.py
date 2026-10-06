"""Photometric losses and image-quality metrics (L1, SSIM / D-SSIM, PSNR)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_WIN: dict = {}


def _window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    key = (size, sigma, device, dtype)
    if key not in _WIN:
        x = torch.arange(size, device=device, dtype=dtype) - size // 2
        g = torch.exp(-(x * x) / (2 * sigma * sigma))
        g = g / g.sum()
        _WIN[key] = (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()
    return _WIN[key]


def ssim(img: torch.Tensor, ref: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two (H, W, 3) images in [0, 1] (Gaussian 11x11 window, as in 3DGS / Wang et al.)."""
    x = img.permute(2, 0, 1)[None]
    y = ref.permute(2, 0, 1)[None]
    w = _window(size, sigma, x.device, x.dtype)
    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    sxx = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x * mu_x
    syy = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y * mu_y
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2))
    return s.mean()


def photometric_loss(img: torch.Tensor, ref: torch.Tensor, lambda_dssim: float = 0.2) -> tuple[torch.Tensor, dict]:
    """(1 - lambda) * L1 + lambda * (1 - SSIM): the 3DGS training objective."""
    l1 = (img - ref).abs().mean()
    s = ssim(img, ref)
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - s), {"l1": float(l1.detach()), "ssim": float(s.detach())}


def psnr(img: torch.Tensor, ref: torch.Tensor) -> float:
    mse = ((img.clamp(0, 1) - ref) ** 2).mean()
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))
