"""SAM-guided FFTformer: plugging FFTformer into the SAM-Deblur framework.

SAM-Deblur conditions a NAFNet on SAM segments through the MAP unit. This
project keeps the SAM/MAP prior and replaces NAFNet with FFTformer, aiming to
combine FFTformer's frequency-domain restoration quality on GoPro with
SAM-Deblur's robustness on blur types not seen in training (RealBlur, ReLoBlur).

Input construction (``docs/assets/architecture.png``, part C):
    I_input = concat(I_bm, I_MAP)            -> (H, W, C + S)
FFTformer's pre-trained stem and its global residual expect C = 3 channels, so
I_input is projected back to 3 channels by a 1x1 convolution. That projection is
initialised as identity on the image channels and zero on the MAP channels:
at step 0 the model *is* the pre-trained FFTformer, and fine-tuning learns how
much of the SAM prior to inject.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from deblur.models.backbones import crop_to, pad_to_multiple
from deblur.models.map_prior import MAPUnit


class PriorProjection(nn.Module):
    """1x1 conv (C + S -> C), identity-initialised on the image channels."""

    def __init__(self, image_channels: int = 3, prior_channels: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(image_channels + prior_channels, image_channels, kernel_size=1)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for c in range(image_channels):
                self.proj.weight[c, c, 0, 0] = 1.0

    def forward(self, image: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([image, prior], dim=1))


class SAMGuidedRestorer(nn.Module):
    """Wraps an image-to-image restoration backbone (FFTformer by default) with the MAP prior."""

    def __init__(self, backbone: nn.Module, feat_channels: int = 16, mask_dropout: float = 0.5,
                 pad_multiple: int = 32):
        super().__init__()
        self.backbone = backbone
        self.map_unit = MAPUnit(3, feat_channels, mask_dropout)
        self.projection = PriorProjection(3, feat_channels)
        self.pad_multiple = pad_multiple

    def forward(self, blur: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        if masks is not None and masks.shape[1] > 0:
            x = self.projection(blur, self.map_unit(blur, masks))
        else:  # no SAM masks: behave like the plain backbone
            x = blur
        x, size = pad_to_multiple(x, self.pad_multiple)
        return crop_to(self.backbone(x), size)


class PlainRestorer(nn.Module):
    """Backbone without the SAM prior (baseline), with the same padding logic."""

    def __init__(self, backbone: nn.Module, pad_multiple: int = 32):
        super().__init__()
        self.backbone, self.pad_multiple = backbone, pad_multiple

    def forward(self, blur: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        x, size = pad_to_multiple(blur, self.pad_multiple)
        return crop_to(self.backbone(x), size)
