#!/usr/bin/env bash
# Stage 2 GPU work ONLY: generating the fixed prompt set. Scoring, identity validation
# and compliance tests are CPU work and run on the Mac afterwards, not on a billed GPU.
# Heavy PNGs stay in outputs_raw/ unless you fetch them explicitly; outputs/ stays small.
set -euo pipefail
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1   # print() reaches train.log immediately, so the watcher sees progress
bash scripts/pod_setup.sh
mkdir -p outputs/stage2 outputs_raw/stage2
echo "=== [gpu] stage 2 baseline: fixed prompt set, plain SD 1.5, no identity ==="
python3 -m stage2_eval.generate --out outputs_raw/stage2/baseline
# JPEG copies are small enough to fetch; score locally from these or from fetched PNGs
python3 - <<'PY'
import json
from pathlib import Path
from PIL import Image
src, dst = Path("outputs_raw/stage2/baseline"), Path("outputs/stage2/baseline_images")
dst.mkdir(parents=True, exist_ok=True)
for p in src.glob("*.png"):
    Image.open(p).convert("RGB").save(dst / (p.stem + ".jpg"), quality=95)
items = json.loads((src / "manifest.json").read_text())
for it in items:
    it["file"] = it["file"].replace(".png", ".jpg")
(dst / "manifest.json").write_text(json.dumps(items, indent=2))
PY
echo "=== all done ==="
