"""Fine-tune the SAM-guided FFTformer from the GoPro-pretrained FFTformer.

    python scripts/train.py --config configs/sam_fftformer_gopro.yaml
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from deblur.data.datasets import PairedDeblurDataset, build_index
from deblur.engine import train
from deblur.models.backbones import build_backbone, load_checkpoint
from deblur.models.sam_guided import SAMGuidedRestorer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg.get("seed", 0))

    data, model_cfg, opt = cfg["data"], cfg["model"], cfg["train"]
    train_set = PairedDeblurDataset(build_index(data["dataset"], data["root"], "train"),
                                    patch_size=data["patch_size"], augment=True,
                                    mask_dir=data["train_masks"], max_masks=data["max_masks"])
    val_set = PairedDeblurDataset(build_index(data["dataset"], data["root"], "test")[: data["val_images"]],
                                  mask_dir=data["val_masks"], max_masks=data["max_masks"])
    train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True,
                              num_workers=opt["workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=2)

    backbone = load_checkpoint(build_backbone("fftformer"), model_cfg["pretrained_backbone"])
    model = SAMGuidedRestorer(backbone, feat_channels=model_cfg["feat_channels"],
                              mask_dropout=model_cfg["mask_dropout"])

    train(model, train_loader, val_loader, iterations=opt["iterations"], lr=opt["lr"],
          out_dir=cfg["output_dir"], device=args.device, freq_weight=opt["freq_weight"],
          val_every=opt["val_every"], backbone_lr_scale=opt["backbone_lr_scale"])


if __name__ == "__main__":
    main()
