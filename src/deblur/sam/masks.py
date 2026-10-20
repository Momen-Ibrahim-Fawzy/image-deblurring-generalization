"""Segment Anything (SAM) masks as structural priors for deblurring.

SAM is run once per image, offline, and the binary masks are stored as compressed
``.npz`` files (bit-packed). Training and evaluation then read the masks from disk
instead of running the large SAM encoder in the loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_masks(path: str | Path, masks: np.ndarray) -> None:
    """Save a (K, H, W) boolean array, largest masks first, bit-packed."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"expected (K,H,W) masks, got {masks.shape}")
    order = np.argsort(-masks.reshape(len(masks), -1).sum(axis=1), kind="stable")
    masks = masks[order]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(masks, axis=-1), shape=np.array(masks.shape))


def load_masks(path: str | Path, max_masks: int | None = None) -> torch.Tensor:
    """Load masks saved by :func:`save_masks` as a float tensor (K, H, W) in {0, 1}."""
    data = np.load(path)
    k, h, w = data["shape"].tolist()
    masks = np.unpackbits(data["packed"], axis=-1, count=w).astype(bool).reshape(k, h, w)
    if max_masks is not None:
        masks = masks[:max_masks]
    return torch.from_numpy(masks.astype(np.float32))


def generate_masks(image_rgb: np.ndarray, generator) -> np.ndarray:
    """Run a ``segment_anything.SamAutomaticMaskGenerator`` on one HxWx3 uint8 image."""
    results = generator.generate(image_rgb)
    if not results:
        return np.zeros((0, *image_rgb.shape[:2]), dtype=bool)
    return np.stack([r["segmentation"] for r in results]).astype(bool)


def build_generator(checkpoint: str, model_type: str = "vit_b", device: str = "cuda",
                    points_per_side: int = 32, pred_iou_thresh: float = 0.86,
                    stability_score_thresh: float = 0.92, min_mask_region_area: int = 100):
    """Create the automatic mask generator (imported lazily: SAM is an optional dependency)."""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
