"""Stage 7: how much does each bit-width change the UNet's output, and how fast is it?

    .venv-coreml/bin/python -m stage7_quantize.evaluate --variants models/coreml/q/*.mlpackage

Reference: the FP16 control-UNet on the GPU (75 dB vs PyTorch in stage 6). Every variant runs on
the Neural Engine with identical inputs: random latent, timestep 500, random text embedding, and
REAL ControlNet residuals (the FP16 ControlNet on the pose image), since the control-UNet's 13
extra inputs change what it computes. Reports MB on disk, PSNR of the noise prediction vs the
reference, load time (on-device compile) and median per-step time.

End-to-end quality (identity, head in frame, compliance) is separate: generate with
stage6_coreml.coreml_generate per variant, then score on the Mac.
"""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

FP16 = "models/coreml/ours_ohwx_cn/Stable_Diffusion_version_models_fused_sd15_ohwx_control-unet.mlpackage"
CNET = "models/coreml/ours_ohwx_cn/ControlNet_lllyasviel_control_v11p_sd15_openpose.mlpackage"


def du_mb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6


def psnr(ref, x):
    mse = float(np.mean((ref - x) ** 2))
    return 10 * np.log10(float(np.abs(ref).max()) ** 2 / mse) if mse > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--out", default="outputs/stage7/numeric.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    feed = {"sample": rng.standard_normal((2, 4, 64, 64)).astype(np.float16),
            "timestep": np.full((2,), 500, np.float16),
            "encoder_hidden_states": rng.standard_normal((2, 768, 1, 77)).astype(np.float16)}
    pose = np.asarray(Image.open("lora_data/control/passport_pose.png").convert("RGB").resize((512, 512)), np.float32) / 255
    cond = np.repeat(pose.transpose(2, 0, 1)[None], 2, axis=0).astype(np.float16)
    cnet = ct.models.MLModel(CNET, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    feed.update({k: v.astype(np.float16) for k, v in cnet.predict({**feed, "controlnet_cond": cond}).items()})
    del cnet
    ref = ct.models.MLModel(FP16, compute_units=ct.ComputeUnit.CPU_AND_GPU).predict(feed)["noise_pred"].astype(np.float32)

    rows = []
    for path in [FP16] + sorted(args.variants):
        t0 = time.time()
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        load = time.time() - t0
        ts = []
        for _ in range(args.runs + 1):
            t1 = time.time()
            y = m.predict(feed)["noise_pred"].astype(np.float32)
            ts.append(time.time() - t1)
        name = "fp16" if path == FP16 else Path(path).stem.split("_")[-1]
        r = {"variant": name, "mb": round(du_mb(path)), "psnr_vs_fp16_gpu": round(psnr(ref, y), 1),
             "nan": bool(np.isnan(y).any()), "load_s": round(load, 1), "step_s": round(float(np.median(ts[1:])), 3)}
        rows.append(r)
        print(f"  {name:<6} {r['mb']:5d} MB   PSNR {r['psnr_vs_fp16_gpu']:5.1f} dB   step {r['step_s']:.3f}s   load {r['load_s']:5.1f}s"
              + ("   NaN!" if r["nan"] else ""), flush=True)
        del m
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
