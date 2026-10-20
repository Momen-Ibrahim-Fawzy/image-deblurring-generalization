# Project timeline (Spring 2024)

| Date | Work |
|---|---|
| 19 Mar | Downloaded and prepared the GoPro dataset |
| 21 Mar | Downloaded and prepared HIDE and RealBlur |
| 26 Mar | Surveyed techniques for synthesising blurred images |
| 27 Mar | Studied the FFTformer code base (baseline) |
| 5 Apr | Explored the pre-trained baseline models; paper review of SAM-Deblur |
| 14 Apr | Measured the pre-trained FFTformer checkpoints on GoPro, RealBlur-J and RealBlur-R |
| 23 Apr | **Milestone 1:** goal, datasets, PSNR/SSIM, FFTformer baseline results |
| 27 Apr | Studied the SAM-Deblur code base, aiming to reuse its MAP module |
| 6 May | **Milestone 2:** SAM-Deblur pipeline; reproduced the GoPro result (32.83 dB / 0.960); plan to replace NAFNet with FFTformer |
| 28 May | **Milestone 3:** SAM masks generated on blurry *and* pre-deblurred training images; MAP integrated with FFTformer (input stem extended); first training run diverged (learning rate too high), restarted with a lower rate; early GoPro checkpoint 21.72 dB / 0.824; contacted the Blur2Blur authors for their code |
| 10 Jun | **Milestone 4:** conclusions, Blur2Blur review as future work, final report |

Literature tracked during the project: *Deep Image Deblurring: A Survey* (IJCV 2022), FFTformer (CVPR 2023),
SAM-Deblur (ICASSP 2024) and Blur2Blur (CVPR 2024).
