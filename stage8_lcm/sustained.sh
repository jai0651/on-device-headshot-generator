#!/usr/bin/env bash
# Stage 8c: sustained load. 50 headshots back to back on the Neural Engine with the shipping
# configuration (8-bit face+LCM UNet, pose ControlNet, 4 steps, CFG 1), logging every image's time
# and macOS's thermal state every 15 s. One cold image says nothing about the 20th (section 06).
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-50}"
U=models/coreml/q_lcm/sd15_ohwx_lcm_control-unet_8bit.mlpackage
CN=models/coreml/ours_lcm/ControlNet_lllyasviel_control_v11p_sd15_openpose.mlpackage
mkdir -p outputs/stage8
( while true; do echo "$(date +%T) $(pmset -g therm 2>/dev/null | grep -E 'CPU_Speed_Limit|thermal warning level' | tr -s ' ' | tr '\n' ' ')"; sleep 15; done ) > outputs/stage8/thermal.log &
TH=$!
HF_HUB_OFFLINE=1 .venv-coreml/bin/python -m stage6_coreml.coreml_generate --unet "$U" --controlnet "$CN" \
  --units-unet CPU_AND_NE --units-vae CPU_AND_NE --scheduler lcm --steps 4 --cfg 1 \
  --prompt "professional corporate headshot of ohwx man, navy blazer, white shirt, plain light grey studio backdrop, soft even lighting" \
  --n "$N" --run sustained 2>&1 | grep -vE 'Warning|warn|scikit' > outputs/stage8/sustained.log
kill $TH
echo "done: outputs/stage8/sustained.log, thermal.log"
