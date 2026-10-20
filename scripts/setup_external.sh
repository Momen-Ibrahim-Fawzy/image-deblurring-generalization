#!/usr/bin/env bash
# Clone the official implementations this project builds on into external/.
# Their code is used as-is (imported, not copied) and keeps its own licence.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external weights

clone() {  # clone <url> <folder>
  if [ -d "external/$2" ]; then echo "external/$2 already present"; else git clone --depth 1 "$1" "external/$2"; fi
}

clone https://github.com/kkkls/FFTformer.git FFTformer      # Kong et al., CVPR 2023
clone https://github.com/HPLQAQ/SAM-Deblur.git SAM-Deblur    # Li et al., ICASSP 2024

pip install -q git+https://github.com/facebookresearch/segment-anything.git

cat <<'EOF'

Next steps (weights are not redistributed here):
  * FFTformer checkpoints (GoPro / RealBlur-J / RealBlur-R): see external/FFTformer/README.md
  * SAM-Deblur checkpoints (NAFNet / SegNAFNet / MaskNAFNet, GoPro): see external/SAM-Deblur/README.md
  * SAM ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
Put them in weights/.
EOF
