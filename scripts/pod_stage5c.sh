#!/usr/bin/env bash
# Stage 5c GPU work ONLY: the edit path (inpaint around the real face, harmonize), plus the
# FaceID colour fix. Uploads: 3 reframed selfie crops + masks, embeddings, pose. Scoring on the Mac.
set -euo pipefail
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
bash scripts/pod_setup.sh --gpu-only
echo "=== [gpu] 5c: edit path, 3 selfies x 3 outfits x 3 seeds x harmonize {0, 0.2, 0.35} ==="
python3 -m stage5_zeroshot.edit_generate
echo "=== [gpu] FaceID colour fix: one selfie, negative += black and white ==="
python3 -m stage5_zeroshot.generate_methods --only faceid_1photo --out outputs/stage5_colorfix \
  --negative-extra "black and white, monochrome, grayscale, desaturated"
echo "=== all done ==="
