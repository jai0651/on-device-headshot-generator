#!/usr/bin/env bash
# Stage 7 end to end: the fixed prompts with pose control, per UNet bit-width, all in Core ML on
# the Neural Engine, with the process's peak memory (what actually gets an iOS app killed).
set -uo pipefail
cd "$(dirname "$0")/.."
D=models/coreml/ours_ohwx_cn
CN=$D/ControlNet_lllyasviel_control_v11p_sd15_openpose.mlpackage
for v in fp16 8bit 6bit 4bit; do
  U=$([ $v = fp16 ] && echo $D/Stable_Diffusion_version_models_fused_sd15_ohwx_control-unet.mlpackage || echo models/coreml/q/sd15_ohwx_control-unet_$v.mlpackage)
  echo "=== $v"
  /usr/bin/time -l env HF_HUB_OFFLINE=1 .venv-coreml/bin/python -m stage6_coreml.coreml_generate --unet "$U" --controlnet "$CN" \
     --units-unet CPU_AND_NE --units-vae CPU_AND_NE --run q_$v --subject "ohwx man" 2>&1 \
     | grep -E '^\{|maximum resident|Error|Traceback'
  mv outputs/stage6/q_$v outputs/stage7/q_$v 2>/dev/null || true
done
echo "=== all done"
