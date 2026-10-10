#!/usr/bin/env bash
# Stage 5 GPU work ONLY: generate every identity method on the fixed prompt set.
# Needs no photos: FaceID uses embeddings computed on the Mac (lora_data/faceid/*.npz).
# Scoring runs on the Mac afterwards (stage5_zeroshot.score_methods).
set -euo pipefail
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
bash scripts/pod_setup.sh --gpu-only
echo "=== [gpu] stage 5: base / lora / faceid / stacked / pose, 3 seeds ==="
python3 -m stage5_zeroshot.generate_methods
echo "=== all done ==="
