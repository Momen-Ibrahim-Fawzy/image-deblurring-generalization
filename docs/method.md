# Method

## Problem
A blurry image is commonly modelled as `B = K * S + n`, a sharp image `S` convolved with a blur kernel `K` (uniform or
spatially varying) plus noise. Blind deblurring recovers `S` without knowing `K`. Deep networks learn this inverse
mapping from paired data, so they inherit the blur statistics of their training set. The camera-shake, out-of-focus,
moving-object and mixed blur found in real photos rarely match a synthetic training set such as GoPro.

## Experiment 1: FFTformer
FFTformer replaces spatial self-attention with a frequency-domain self-attention solver (FSAS) and a discriminative
frequency-domain feed-forward network (DFFN). Element-wise products in the Fourier domain replace quadratic
attention, which lets the model use global context on high-resolution images. The official checkpoints were measured
on GoPro, RealBlur-J and RealBlur-R, one checkpoint per dataset.

## Experiment 2: SAM-Deblur generalisation
SAM-Deblur keeps NAFNet as the restoration network and adds a structural prior from Segment Anything:
1. SAM segments the image into object masks.
2. The MAP unit averages encoder features inside each mask, giving features that are constant per object and
   change only at object boundaries.
3. The MAP features are concatenated with the image and restored by NAFNet. Mask dropout during training keeps the
   network from over-trusting SAM.

The GoPro-trained SegNAFNet checkpoint was reproduced on the GoPro test set (32.83 dB / 0.960). Its published
cross-domain results (RealBlur-J, ReLoBlur) show the robustness that the SAM prior adds.

## Experiment 3: SAM-guided FFTformer (proposed)
The NAFNet inside SAM-Deblur is replaced by FFTformer (`src/deblur/models/sam_guided.py`).

| Design choice | Reason |
|---|---|
| Encoder (3→S, two 3×3 convs) before MAP | MAP pools learnt features rather than raw colours, as in SAM-Deblur |
| Concatenate image + MAP → C+S channels | Keeps the full-resolution image path untouched |
| 1×1 projection C+S → 3, identity-initialised | Reuses FFTformer's pre-trained stem and global residual; at step 0 the model equals the pre-trained FFTformer |
| Backbone LR = 0.1 × head LR | Adapts to the prior without forgetting GoPro restoration |
| Mask dropout p = 0.5 | Robustness to missing or wrong segments on unseen blur |
| Pre-deblur before SAM at test time | SAM finds cleaner boundaries on a partially restored image |
| Loss: L1 + 0.1 × frequency L1 | FFTformer's objective; penalises lost high-frequency detail |

## Follow-up reviewed: Blur2Blur
Blur2Blur (CVPR 2024) handles real blur without paired data. It learns to convert an unknown blur into a blur domain
that an existing deblurring model already handles well, then applies that model. It complements the SAM prior:
Blur2Blur fixes the input distribution, while the SAM prior makes the model more tolerant of it.
