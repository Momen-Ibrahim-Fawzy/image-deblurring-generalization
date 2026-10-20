"""Unit tests for the parts that do not need GPUs, datasets or external repositories.

Run with:  python -m pytest tests   (or: python -m unittest discover tests)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deblur.data.datasets import PairedDeblurDataset, build_index
from deblur.losses import DeblurLoss
from deblur.metrics import psnr, ssim
from deblur.models.map_prior import MaskDropout, mask_average_pool
from deblur.models.sam_guided import PlainRestorer, SAMGuidedRestorer
from deblur.sam.masks import load_masks, save_masks


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class TestMetrics(unittest.TestCase):
    def test_psnr_known_value(self):
        a = torch.zeros(1, 3, 16, 16)
        b = torch.full((1, 3, 16, 16), 0.1)          # mse = 0.01 -> 20 dB
        self.assertAlmostEqual(psnr(a, b).item(), 20.0, places=4)

    def test_ssim_identical_is_one(self):
        x = torch.rand(2, 3, 32, 32)
        self.assertTrue(torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5))

    def test_ssim_drops_with_noise(self):
        x = torch.rand(1, 3, 32, 32)
        noisy = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)
        self.assertLess(ssim(x, noisy).item(), 0.9)


class TestMAP(unittest.TestCase):
    def test_pooling_is_constant_inside_masks(self):
        feats = torch.rand(1, 4, 8, 8)
        masks = torch.zeros(1, 2, 8, 8)
        masks[0, 0, :, :4] = 1
        masks[0, 1, :, 4:] = 1
        out = mask_average_pool(feats, masks)
        self.assertTrue(torch.allclose(out[0, :, :, :4], feats[0, :, :, :4].mean(dim=(1, 2), keepdim=True).expand(4, 8, 4)))
        self.assertTrue(torch.allclose(out[0, :, :, 4:], feats[0, :, :, 4:].mean(dim=(1, 2), keepdim=True).expand(4, 8, 4)))

    def test_uncovered_pixels_keep_features(self):
        feats = torch.rand(1, 3, 8, 8)
        masks = torch.zeros(1, 1, 8, 8)
        masks[0, 0, :2, :2] = 1
        out = mask_average_pool(feats, masks)
        self.assertTrue(torch.equal(out[..., 4:, 4:], feats[..., 4:, 4:]))

    def test_mask_dropout_is_identity_in_eval(self):
        drop = MaskDropout(0.5).eval()
        masks = torch.ones(2, 5, 4, 4)
        self.assertTrue(torch.equal(drop(masks), masks))


class TestSAMGuided(unittest.TestCase):
    def test_initialisation_reproduces_backbone(self):
        """At step 0 the SAM-guided model must equal the plain backbone (identity projection)."""
        blur = torch.rand(1, 3, 40, 52)
        masks = (torch.rand(1, 6, 40, 52) > 0.5).float()
        guided = SAMGuidedRestorer(Identity(), mask_dropout=0.0).eval()
        plain = PlainRestorer(Identity()).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(guided(blur, masks), plain(blur), atol=1e-6))
            self.assertEqual(guided(blur, masks).shape, blur.shape)

    def test_loss_is_zero_for_perfect_prediction(self):
        x = torch.rand(1, 3, 16, 16)
        self.assertAlmostEqual(DeblurLoss()(x, x)["loss"].item(), 0.0, places=6)


class TestIO(unittest.TestCase):
    def test_mask_roundtrip_sorted_by_area(self):
        masks = np.zeros((3, 10, 13), dtype=bool)
        masks[0, :2, :2] = True
        masks[1, :, :] = True
        masks[2, :5, :5] = True
        with tempfile.TemporaryDirectory() as d:
            save_masks(Path(d) / "m.npz", masks)
            loaded = load_masks(Path(d) / "m.npz")
        self.assertEqual(tuple(loaded.shape), (3, 10, 13))
        self.assertEqual(loaded.sum(dim=(1, 2)).tolist(), [130.0, 25.0, 4.0])

    def test_gopro_index_and_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            for sub in ("blur", "sharp"):
                folder = Path(d) / "test" / "GOPR0001" / sub
                folder.mkdir(parents=True)
                for i in range(2):
                    Image.fromarray(np.full((64, 64, 3), 10 * i, np.uint8)).save(folder / f"{i:06d}.png")
            pairs = build_index("gopro", d, "test")
            self.assertEqual(len(pairs), 2)
            item = PairedDeblurDataset(pairs, patch_size=32, augment=True)[0]
            self.assertEqual(tuple(item["blur"].shape), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
