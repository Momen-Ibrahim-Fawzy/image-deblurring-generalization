"""Pre-compute SAM masks for a dataset split.

Training: masks come straight from the blurry training images.
Testing (``--pre-deblur``): each blurry image is first restored by a pre-trained
backbone and SAM segments the restored image, which gives cleaner object
boundaries on heavily blurred inputs (architecture figure, part A).

Example:
    python scripts/generate_sam_masks.py --dataset gopro --root data/GOPRO_Large \
        --split test --sam-checkpoint weights/sam_vit_b_01ec64.pth \
        --pre-deblur weights/fftformer_GoPro.pth --out data/masks/gopro_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from deblur.data.datasets import build_index, load_rgb
from deblur.models.backbones import build_backbone, load_checkpoint
from deblur.models.sam_guided import PlainRestorer
from deblur.sam.masks import build_generator, generate_masks, save_masks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["gopro", "hide", "realblur_j", "realblur_r"])
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--sam-model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--pre-deblur", default=None, help="FFTformer checkpoint used to pre-deblur before SAM")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    pairs = build_index(args.dataset, args.root, args.split)
    generator = build_generator(args.sam_checkpoint, args.sam_model, args.device)

    restorer = None
    if args.pre_deblur:
        backbone = load_checkpoint(build_backbone("fftformer"), args.pre_deblur)
        restorer = PlainRestorer(backbone).eval().to(args.device)

    out = Path(args.out)
    for pair in tqdm(pairs, desc=f"SAM masks [{args.dataset}/{args.split}]"):
        target = out / f"{pair.name}.npz"
        if target.exists() and not args.overwrite:
            continue
        image = load_rgb(pair.blur)
        if restorer is not None:
            image = restorer(image.unsqueeze(0).to(args.device)).clamp(0, 1)[0].cpu()
        rgb = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        save_masks(target, generate_masks(rgb, generator))


if __name__ == "__main__":
    main()
