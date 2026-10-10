#!/usr/bin/env bash
# Headshots of "ohwx man" (you), entirely in Core ML on this Mac's Neural Engine: the shipping
# configuration from stage 8. Face LoRA + LCM-LoRA folded into an 8-bit UNet, passport pose
# ControlNet, 4 steps, no CFG. ~1.4 s per image after a ~30 s first load.
#   ./generate_headshot.sh "photo of ohwx man in a navy suit, grey studio backdrop, soft lighting" 4
#   SLOW=1 ./generate_headshot.sh "..."   # the stage 6 path instead: 25 steps + CFG (~15 s, more varied expressions)
set -euo pipefail
cd "$(dirname "$0")"
PROMPT="${1:?usage: ./generate_headshot.sh \"photo of ohwx man ...\" [how-many]}"
N="${2:-1}"
if [ -n "${SLOW:-}" ]; then
  D=models/coreml/ours_ohwx_cn
  ARGS=(--unet "$D/Stable_Diffusion_version_models_fused_sd15_ohwx_control-unet.mlpackage"
        --controlnet "$D/ControlNet_lllyasviel_control_v11p_sd15_openpose.mlpackage" --steps 25)
else
  ARGS=(--unet models/coreml/q_lcm/sd15_ohwx_lcm_control-unet_8bit.mlpackage
        --controlnet models/coreml/ours_lcm/ControlNet_lllyasviel_control_v11p_sd15_openpose.mlpackage
        --scheduler lcm --steps 4 --cfg 1)
fi
HF_HUB_OFFLINE=1 .venv-coreml/bin/python -m stage6_coreml.coreml_generate "${ARGS[@]}" \
  --units-unet CPU_AND_NE --units-vae CPU_AND_NE --prompt "$PROMPT" --n "$N" --run mine 2>&1 \
  | grep -vE 'Warning|warn|scikit'
open outputs/stage6/mine/samples/m/
