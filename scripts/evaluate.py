"""Evaluate a deblurring model on GoPro / HIDE / RealBlur and report PSNR and SSIM.

Examples:
    # Pre-trained FFTformer, no SAM prior
    python scripts/evaluate.py --model fftformer --ckpt weights/fftformer_GoPro.pth \
        --dataset gopro --root data/GOPRO_Large

    # SAM-guided FFTformer on an unseen blur type (trained on GoPro only)
    python scripts/evaluate.py --model sam_fftformer --ckpt runs/sam_fftformer/best.pth \
        --dataset realblur_j --root data/RealBlur --masks data/masks/realblur_j_test \
        --csv results/sam_fftformer_realblur_j.csv
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from deblur.data.datasets import PairedDeblurDataset, build_index
from deblur.engine import evaluate
from deblur.models.backbones import build_backbone, load_checkpoint
from deblur.models.sam_guided import PlainRestorer, SAMGuidedRestorer


def build_model(name: str, ckpt: str | None, feat_channels: int) -> torch.nn.Module:
    backbone = build_backbone("fftformer")
    if name == "fftformer":
        if ckpt:
            load_checkpoint(backbone, ckpt)
        return PlainRestorer(backbone)
    if name == "sam_fftformer":
        model = SAMGuidedRestorer(backbone, feat_channels=feat_channels)
        if ckpt:
            load_checkpoint(model, ckpt)
        return model
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["fftformer", "sam_fftformer"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", required=True, choices=["gopro", "hide", "realblur_j", "realblur_r"])
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--masks", default=None, help="folder of pre-computed SAM masks (sam_fftformer)")
    p.add_argument("--feat-channels", type=int, default=16)
    p.add_argument("--y-channel", action="store_true", help="compute metrics on the Y channel")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.model == "sam_fftformer" and not args.masks:
        p.error("--masks is required for sam_fftformer")

    pairs = build_index(args.dataset, args.root, args.split)
    dataset = PairedDeblurDataset(pairs, mask_dir=args.masks)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    model = build_model(args.model, args.ckpt, args.feat_channels)

    result = evaluate(model, loader, args.device, y_channel=args.y_channel,
                      save_dir=args.save_dir, csv_path=args.csv)
    print(json.dumps({"model": args.model, "dataset": args.dataset, **result}, indent=2))


if __name__ == "__main__":
    main()
