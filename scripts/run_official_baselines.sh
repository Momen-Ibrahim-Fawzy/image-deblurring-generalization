#!/usr/bin/env bash
# Experiments 1 and 2: evaluate the official pre-trained models with their own test pipelines.
#   1) FFTformer, one checkpoint per dataset (GoPro, RealBlur-J, RealBlur-R)
#   2) SAM-Deblur (SegNAFNet) trained on GoPro only, tested on GoPro and on unseen blur types
# Datasets must be laid out as described in docs/datasets.md.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== FFTformer (official checkpoints) =="
for ds in GoPro RealBlur_J RealBlur_R; do
  python scripts/evaluate.py --model fftformer --ckpt "weights/fftformer_${ds}.pth" \
    --dataset "$(echo "$ds" | tr '[:upper:]' '[:lower:]')" \
    --root "data/$( [ "$ds" = GoPro ] && echo GOPRO_Large || echo RealBlur )" \
    --csv "results/fftformer_${ds}.csv"
done

echo "== SAM-Deblur (SegNAFNet, GoPro-trained) =="
pushd external/SAM-Deblur > /dev/null
for opt in options/test/GoPro/SegNAFNet.yml options/test/RealBlurJ/SegNAFNet.yml options/test/ReLoBlur/SegNAFNet.yml; do
  [ -f "$opt" ] && python basicsr/test.py -opt "$opt" || echo "skip: $opt not found"
done
popd > /dev/null
