"""Training losses: pixel L1 plus an L1 loss on the 2-D Fourier spectrum.

The frequency term (as used to train FFTformer) penalises missing high-frequency
detail, which plain L1 tends to under-weight and which is exactly what blur removes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyL1(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.rfft2(pred.float(), norm="backward")
        ft = torch.fft.rfft2(target.float(), norm="backward")
        return (torch.view_as_real(fp) - torch.view_as_real(ft)).abs().mean()


class DeblurLoss(nn.Module):
    def __init__(self, freq_weight: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.freq = FrequencyL1()
        self.freq_weight = freq_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l1 = self.l1(pred, target)
        freq = self.freq(pred, target)
        return {"loss": l1 + self.freq_weight * freq, "l1": l1, "freq": freq}
