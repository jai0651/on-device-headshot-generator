#!/usr/bin/env bash
# Everything heavy from stages 0 and 1, in one pod session.
# Runs with the pod image's own torch (system python3), not uv — see runpod-train skill.
# Results land in outputs/, which `runpod.sh finish` fetches back.
#
# Stage 0's 6.5M-param UNet leaves most of the GPU idle, so stage 1 runs alongside it
# rather than after it. Stage 1 timings are therefore not clean benchmarks — the
# laptop numbers in runs/stage1 are the reference ones.
set -euo pipefail
export HF_HUB_DISABLE_XET=1   # xet stalled on the laptop; plain HTTP is reliable
export PYTHONUNBUFFERED=1   # print() reaches train.log immediately, so the watcher sees progress
mkdir -p outputs/stage0 outputs/stage1

pip install -q --break-system-packages tqdm diffusers transformers accelerate safetensors 2>&1 | tail -1
# The RunPod image ships torch but not torchvision. Install the build that matches the
# image's torch + CUDA exactly, with --no-deps so pip can't swap torch out from under us.
TV=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print({'2.9.1':'0.24.1','2.9.0':'0.24.0','2.8.0':'0.23.0'}[v])")
CU=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
pip install -q --break-system-packages --no-deps "torchvision==$TV" --index-url "https://download.pytorch.org/whl/$CU" 2>&1 | tail -1
python3 -c "import torchvision.datasets, torchvision.utils"   # fail fast, before any GPU work
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "=== stage 0: DDPM on MNIST (background) ==="
python3 -m stage0_ddpm.train --out outputs/stage0 --resume --workers 4 --batch-size 256 --lr 3e-4 \
  > outputs/stage0/train.log 2>&1 &
S0=$!

echo "=== stage 1: SD 1.5 manual loop + sweeps ==="
python3 -m stage1_sd_pipeline.manual_loop --compare --out outputs/stage1
python3 -m stage1_sd_pipeline.manual_loop --my-ddim --compare --out outputs/stage1
python3 -m stage1_sd_pipeline.sweeps --out outputs/stage1
echo "stage 1 done $(date +%T)"

wait $S0 || { echo "stage 0 training FAILED"; tail -20 outputs/stage0/train.log; exit 1; }
tr '\r' '\n' < outputs/stage0/train.log | grep -E '^(device|epoch [0-9]+ +step|done)'
python3 -m stage0_ddpm.sample --ckpt outputs/stage0/mnist/ckpt.pt
echo "=== all done ==="
