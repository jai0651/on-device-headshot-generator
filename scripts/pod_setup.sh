#!/usr/bin/env bash
# Shared pod environment setup. Uses the image's own torch (system python3); see runpod-train skill.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# opencv needs libgl1 + libglib; mediapipe dlopens libGLESv2 + libEGL. The image has none of them
(apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 libgles2 libegl1 >/dev/null) 2>&1 | tail -1 || true
# --gpu-only: just what generation/training needs. Face recognition, MediaPipe and scoring are
# CPU work that runs on the Mac, so a GPU pod shouldn't spend minutes installing them.
if [ "${1:-}" = "--gpu-only" ]; then
  pip install -q --break-system-packages tqdm diffusers transformers accelerate safetensors peft matplotlib 2>&1 | tail -1
else
  pip install -q --break-system-packages tqdm diffusers transformers accelerate safetensors peft \
    insightface onnxruntime "mediapipe>=0.10.14,<1" opencv-python-headless scikit-learn scipy matplotlib 2>&1 | tail -1
fi
# The RunPod image ships torch but not torchvision: install the exact matching build, no deps.
TV=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print({'2.9.1':'0.24.1','2.9.0':'0.24.0','2.8.0':'0.23.0'}[v])")
CU=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
pip install -q --break-system-packages --no-deps "torchvision==$TV" --index-url "https://download.pytorch.org/whl/$CU" 2>&1 | tail -1
# fail fast, before any GPU time is spent on real work
python3 - "${1:-}" <<'PY'
import sys, torch, torchvision, diffusers, transformers, peft
if sys.argv[1] != "--gpu-only":
    import insightface, onnxruntime, mediapipe, cv2, sklearn, scipy
    from stage2_eval.compliance import ComplianceChecker; ComplianceChecker()   # loads mediapipe's native libs
q = open("/sys/fs/cgroup/cpu.max").read().split() if __import__("os").path.exists("/sys/fs/cgroup/cpu.max") else ["max"]
cpus = "unlimited" if q[0] == "max" else f"{int(q[0]) / int(q[1]):.1f}"
print("env OK  torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), "| real cpu quota", cpus)
PY
