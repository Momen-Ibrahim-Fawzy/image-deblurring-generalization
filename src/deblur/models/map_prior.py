"""Mask Average Pooling (MAP) unit: SAM segments as a structural prior.

Pipeline (see ``docs/assets/architecture.png``, part B):

1. A light convolutional encoder maps the blurry image I_bm (H, W, C) to
   features I_enc (H, W, S).
2. For every SAM mask the features are averaged inside the mask and the mean is
   written back over that mask ("mask average pooling"). Pixels covered by
   several masks get the mean of their segments; uncovered pixels keep their
   encoder features. The result is I_MAP (H, W, S): features that are constant
   inside each object and change only at SAM's object boundaries.
3. I_MAP is concatenated with the blurry image to form the restoration input
   (H, W, S + C).

During training whole masks are dropped at random ("mask dropout") so the
network does not over-trust SAM, whose masks get worse on strongly blurred or
unseen blur types.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mask_average_pool(features: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """features: (N,S,H,W); masks: (N,K,H,W) in {0,1}. Returns pooled features (N,S,H,W)."""
    if features.dim() != 4 or masks.dim() != 4:
        raise ValueError("expected features (N,S,H,W) and masks (N,K,H,W)")
    if masks.shape[0] != features.shape[0] or masks.shape[-2:] != features.shape[-2:]:
        raise ValueError(f"mask shape {tuple(masks.shape)} incompatible with features {tuple(features.shape)}")
    if masks.shape[1] == 0:
        return features

    masks = masks.to(features.dtype)
    area = masks.sum(dim=(-2, -1))                                     # (N,K)
    sums = torch.einsum("nshw,nkhw->nks", features, masks)            # (N,K,S)
    means = sums / area.clamp_min(eps).unsqueeze(-1)                  # (N,K,S)
    painted = torch.einsum("nks,nkhw->nshw", means, masks)            # (N,S,H,W)
    coverage = masks.sum(dim=1, keepdim=True)                         # (N,1,H,W)
    return torch.where(coverage > 0, painted / coverage.clamp_min(eps), features)


class MaskDropout(nn.Module):
    """Randomly drops whole SAM masks during training."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0 or masks.shape[1] == 0:
            return masks
        keep = (torch.rand(masks.shape[:2], device=masks.device) >= self.p).to(masks.dtype)
        return masks * keep[..., None, None]


class MAPUnit(nn.Module):
    """Encoder + mask average pooling: (image, masks) -> I_MAP with ``feat_channels`` channels."""

    def __init__(self, in_channels: int = 3, feat_channels: int = 16, mask_dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
        )
        self.dropout = MaskDropout(mask_dropout)
        self.feat_channels = feat_channels

    def forward(self, image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return mask_average_pool(self.encoder(image), self.dropout(masks))
