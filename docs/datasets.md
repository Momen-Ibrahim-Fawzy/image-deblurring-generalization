# Datasets

| Dataset | Blur | Size | Role in this project |
|---|---|---|---|
| **GoPro** | Synthetic motion blur (averaged high-speed GoPro frames), 1280×720 | 2,103 train / 1,111 test pairs | Training domain |
| **HIDE** | Synthetic motion blur, human-centric scenes, varying size | 6,397 train / 2,025 test | Cross-domain test |
| **RealBlur-J / -R** | Real camera-shake blur; JPEG (J) and camera-RAW-processed (R) versions | 4,738 pairs in total | Real-blur test |
| **ReLoBlur** | Real local (object) motion blur | 395 test images | Unseen-blur test (SAM-Deblur protocol) |

## Expected layout

```
data/
  GOPRO_Large/
    train/<scene>/{blur,sharp}/*.png
    test/<scene>/{blur,sharp}/*.png
  HIDE_dataset/
    train/ · test/ · GT/ · train.txt · test.txt
  RealBlur/
    RealBlur-J_ECC_IMCORR_centroid_itensity_ref/...
    RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref/...
    RealBlur_J_train_list.txt · RealBlur_J_test_list.txt
    RealBlur_R_train_list.txt · RealBlur_R_test_list.txt
  masks/<dataset>_<split>/<image>.npz      # written by scripts/generate_sam_masks.py
```

`src/deblur/data/datasets.py` turns each layout into `(name, blurry, sharp)` pairs and checks that every file exists
before a run starts.

## Evaluation protocol
- PSNR and SSIM on full-resolution RGB in [0, 1], averaged over images (`--y-channel` switches to the Y channel).
- SSIM uses an 11×11 Gaussian window (σ = 1.5) with the standard constants (Wang et al., 2004).
- The datasets keep their own licences and are downloaded from their authors' pages; nothing is redistributed here.
