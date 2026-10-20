"""Paired blurry/sharp datasets: GoPro, HIDE, RealBlur-J and RealBlur-R.

Each benchmark ships with a different folder layout. The *index builders* below
turn every layout into the same list of (name, blurry_path, sharp_path) records,
so training and evaluation code never needs to know which dataset it is reading.

Expected layouts (as distributed by the dataset authors):

GoPro     GOPRO_Large/{train,test}/<scene>/{blur,sharp}/<frame>.png
HIDE      HIDE_dataset/{train,test}/... plus HIDE_dataset/GT/<img>.png
          and the official split files train.txt / test.txt
RealBlur  RealBlur-J_ECC_IMCORR_centroid_itensity_ref/... with the official list
          files RealBlur_J_{train,test}_list.txt (and the same for R), one
          "gt_path blur_path" pair per line, relative to the RealBlur root
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Pair:
    name: str
    blur: Path
    sharp: Path


# --------------------------------------------------------------------- indexes

def index_gopro(root: str | Path, split: str = "test") -> list[Pair]:
    root = Path(root) / split
    pairs = []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        for blur in sorted((scene / "blur").iterdir()):
            if blur.suffix.lower() in IMG_EXTS:
                pairs.append(Pair(f"{scene.name}_{blur.stem}", blur, scene / "sharp" / blur.name))
    return pairs


def index_hide(root: str | Path, split: str = "test") -> list[Pair]:
    """HIDE stores blurry images under split sub-folders and all sharp images in GT/."""
    root = Path(root)
    split_file = root / f"{split}.txt"
    gt_dir = root / "GT"
    pairs = []
    if split_file.exists():
        for line in split_file.read_text().split():
            blur = root / split / line if not (root / line).exists() else root / line
            pairs.append(Pair(Path(line).stem, blur, gt_dir / Path(line).name))
    else:  # fall back to scanning the split folder
        for blur in sorted((root / split).rglob("*")):
            if blur.suffix.lower() in IMG_EXTS:
                pairs.append(Pair(blur.stem, blur, gt_dir / blur.name))
    return pairs


def index_realblur(root: str | Path, variant: str = "J", split: str = "test") -> list[Pair]:
    root = Path(root)
    list_file = root / f"RealBlur_{variant.upper()}_{split}_list.txt"
    if not list_file.exists():
        raise FileNotFoundError(f"official list file not found: {list_file}")
    pairs = []
    for line in list_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        gt, blur = parts[0], parts[1]
        name = "_".join(Path(blur).parts[-3:]).replace(Path(blur).suffix, "")
        pairs.append(Pair(name, root / blur, root / gt))
    return pairs


INDEXERS: dict[str, Callable[..., list[Pair]]] = {
    "gopro": index_gopro,
    "hide": index_hide,
    "realblur_j": lambda root, split="test": index_realblur(root, "J", split),
    "realblur_r": lambda root, split="test": index_realblur(root, "R", split),
}


def build_index(name: str, root: str | Path, split: str = "test") -> list[Pair]:
    key = name.lower()
    if key not in INDEXERS:
        raise KeyError(f"unknown dataset '{name}', choose from {sorted(INDEXERS)}")
    pairs = INDEXERS[key](root, split=split)
    missing = [p for p in pairs if not p.blur.exists() or not p.sharp.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pairs have missing files, e.g. {missing[0]}")
    return pairs


# --------------------------------------------------------------------- dataset

def load_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PairedDeblurDataset(Dataset):
    """Returns dicts with 'blur', 'sharp' (C,H,W float in [0,1]), 'name' and optionally 'masks'.

    For training, ``patch_size`` enables aligned random crops and flips. When
    ``mask_dir`` is given, pre-computed SAM masks (see ``deblur.sam.masks``) are
    loaded and cropped together with the images.
    """

    def __init__(self, pairs: Sequence[Pair], patch_size: int | None = None,
                 augment: bool = False, mask_dir: str | Path | None = None, max_masks: int = 32):
        self.pairs = list(pairs)
        self.patch_size = patch_size
        self.augment = augment
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.max_masks = max_masks

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_masks(self, name: str, h: int, w: int) -> torch.Tensor:
        from deblur.sam.masks import load_masks
        masks = load_masks(self.mask_dir / f"{name}.npz", max_masks=self.max_masks)
        if masks.shape[-2:] != (h, w):
            raise ValueError(f"mask size {masks.shape[-2:]} does not match image {(h, w)} for {name}")
        return masks

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        blur, sharp = load_rgb(pair.blur), load_rgb(pair.sharp)
        if blur.shape != sharp.shape:
            raise ValueError(f"blurry/sharp size mismatch for {pair.name}")
        item = {"name": pair.name, "blur": blur, "sharp": sharp}
        if self.mask_dir is not None:
            item["masks"] = self._load_masks(pair.name, *blur.shape[-2:])

        if self.patch_size:
            item = self._random_crop(item, self.patch_size)
        if self.augment:
            item = self._flip(item)
        return item

    @staticmethod
    def _random_crop(item: dict, size: int) -> dict:
        _, h, w = item["blur"].shape
        if h < size or w < size:
            raise ValueError(f"image {item['name']} ({h}x{w}) smaller than patch {size}")
        top, left = random.randint(0, h - size), random.randint(0, w - size)
        for key in ("blur", "sharp", "masks"):
            if key in item:
                item[key] = item[key][..., top:top + size, left:left + size]
        return item

    @staticmethod
    def _flip(item: dict) -> dict:
        dims = []
        if random.random() < 0.5:
            dims.append(-1)
        if random.random() < 0.5:
            dims.append(-2)
        if dims:
            for key in ("blur", "sharp", "masks"):
                if key in item:
                    item[key] = item[key].flip(dims)
        return item
