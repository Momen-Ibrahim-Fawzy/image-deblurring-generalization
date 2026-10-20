"""Image-quality metrics used throughout the project.

Both metrics follow the conventions of the deblurring literature (GoPro / HIDE /
RealBlur benchmarks): images are float tensors in [0, 1], shape (N, C, H, W) or
(C, H, W). PSNR and SSIM can be computed on RGB or on the Y channel (YCbCr).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected (N,C,H,W) or (C,H,W), got {tuple(x.shape)}")
    return x.float()


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma, matching MATLAB's rgb2ycbcr (output range [16/255, 235/255])."""
    x = _as_batch(x)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return (65.481 * r + 128.553 * g + 24.966 * b + 16.0) / 255.0


def _crop(x: torch.Tensor, border: int) -> torch.Tensor:
    return x[..., border:-border, border:-border] if border > 0 else x


def psnr(pred: torch.Tensor, target: torch.Tensor, *, y_channel: bool = False,
         crop_border: int = 0, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, one value per image in the batch."""
    pred, target = _as_batch(pred).clamp(0, data_range), _as_batch(target).clamp(0, data_range)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    if y_channel:
        pred, target = rgb_to_y(pred), rgb_to_y(target)
    pred, target = _crop(pred, crop_border), _crop(target, crop_border)
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).expand(channels, 1, size, size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, *, y_channel: bool = False,
         crop_border: int = 0, window_size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0) -> torch.Tensor:
    """Structural similarity (Wang et al., 2004), averaged over channels, one value per image."""
    pred, target = _as_batch(pred).clamp(0, data_range), _as_batch(target).clamp(0, data_range)
    if y_channel:
        pred, target = rgb_to_y(pred), rgb_to_y(target)
    pred, target = _crop(pred, crop_border), _crop(target, crop_border)

    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def filt(x):  # 'valid' filtering, as in the reference implementation
        return F.conv2d(x, window, groups=c)

    mu_x, mu_y = filt(pred), filt(target)
    sigma_x = filt(pred * pred) - mu_x ** 2
    sigma_y = filt(target * target) - mu_y ** 2
    sigma_xy = filt(pred * target) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.flatten(1).mean(dim=1)


class RunningMetrics:
    """Accumulates per-image PSNR/SSIM over a test set."""

    def __init__(self, y_channel: bool = False, crop_border: int = 0):
        self.y_channel, self.crop_border = y_channel, crop_border
        self.records: list[tuple[str, float, float]] = []

    def update(self, names, pred: torch.Tensor, target: torch.Tensor) -> None:
        p = psnr(pred, target, y_channel=self.y_channel, crop_border=self.crop_border)
        s = ssim(pred, target, y_channel=self.y_channel, crop_border=self.crop_border)
        for name, pv, sv in zip(names, p.tolist(), s.tolist()):
            self.records.append((name, pv, sv))

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {"psnr": math.nan, "ssim": math.nan, "n": 0}
        n = len(self.records)
        return {"psnr": sum(r[1] for r in self.records) / n,
                "ssim": sum(r[2] for r in self.records) / n,
                "n": n}
