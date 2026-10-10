#!/usr/bin/env bash
# Class-conditional MNIST DDPM, GPU work only. Full checkpoints stay in outputs_raw/ on the pod;
# only the 13 MB EMA weights, sample grids and loss log are copied to outputs/ for fetching.
set -euo pipefail
export PYTHONUNBUFFERED=1
bash scripts/pod_setup.sh --gpu-only
echo "=== [gpu] conditional DDPM, 10 epochs ==="
python3 -m stage0_ddpm.train --conditional --epochs 10 --workers 4 --out outputs_raw/stage0_cond
mkdir -p outputs/stage0_cond
cp outputs_raw/stage0_cond/mnist/ema_fp16.pt outputs_raw/stage0_cond/mnist/losses.csv outputs/stage0_cond/
cp outputs_raw/stage0_cond/mnist/samples_epoch00{0,4,9}.png outputs/stage0_cond/ 2>/dev/null || true
echo "=== all done ==="
