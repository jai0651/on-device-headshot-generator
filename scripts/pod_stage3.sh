#!/usr/bin/env bash
# Stage 3 GPU work ONLY: train face LoRAs and sample the fixed prompt set at each checkpoint.
# Two configs in one session so boot/setup is paid once (section 03: text encoder "try both").
# Scoring happens on the Mac afterwards (stage3_lora.score_checkpoints).
set -euo pipefail
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1   # print() reaches train.log immediately, so the watcher sees progress
bash scripts/pod_setup.sh --gpu-only
echo "=== [gpu] run 1: UNet LoRA r16, lr 1e-4, text encoder frozen ==="
python3 -m stage3_lora.train_lora --data lora_data/jai --run r16_lr1e-4_te-off --steps 2000 --save-every 250
echo "=== [gpu] run 2: + text-encoder LoRA at 5e-5 ==="
python3 -m stage3_lora.train_lora --data lora_data/jai --run r16_lr1e-4_te5e-5 --steps 2000 --save-every 250 --te-lr 5e-5
echo "=== all done ==="
