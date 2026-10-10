"""Stage 6: does a Core ML UNet compute the same thing as the PyTorch one, and how fast, where?

    .venv-coreml/bin/python -m stage6_coreml.verify_unet \\
        --mlpackage models/coreml/apple_sd15/split_einsum/packages/..._unet.mlpackage \\
        --torch stable-diffusion-v1-5/stable-diffusion-v1-5 --variant fp16

Same random inputs through (a) the diffusers UNet in fp32 on the CPU, the reference, and (b) the
Core ML model on each compute unit. Reports max |diff|, PSNR of the noise prediction, the time
to load (which includes on-device compilation for the Neural Engine, often minutes the first
time) and the median prediction time.

Apple's ANE-optimised ("split_einsum") UNet takes the text embedding as B x 768 x 1 x 77, the
(B, C, 1, S) layout the Neural Engine prefers, instead of PyTorch's B x 77 x 768. The script
reads input names and shapes from the model spec, so it works for any conversion.
"""

import argparse
import json
import time

import coremltools as ct
import numpy as np
import torch
from diffusers import UNet2DConditionModel

UNITS = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
         "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "ALL": ct.ComputeUnit.ALL}


def psnr(ref, x):
    mse = float(np.mean((ref - x) ** 2))
    peak = float(np.abs(ref).max())
    return 10 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--torch", required=True, help="diffusers model id or local folder with a unet/")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--units", nargs="+", default=list(UNITS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    spec = ct.models.MLModel(args.mlpackage, skip_model_load=True).get_spec()
    shapes = {i.name: tuple(i.type.multiArrayType.shape) for i in spec.description.input}
    out_name = spec.description.output[0].name
    print("inputs:", shapes, "| output:", out_name)

    rng = np.random.default_rng(0)
    B = shapes["sample"][0]
    sample = rng.standard_normal(shapes["sample"]).astype(np.float32)
    t = np.full(shapes["timestep"], 500, np.float32)
    ehs_shape = shapes["encoder_hidden_states"]
    ehs = rng.standard_normal(ehs_shape).astype(np.float32)
    ehs_torch = ehs.squeeze(2).transpose(0, 2, 1) if len(ehs_shape) == 4 else ehs  # (B,768,1,77) -> (B,77,768)

    unet = UNet2DConditionModel.from_pretrained(args.torch, subfolder="unet", torch_dtype=torch.float32,
                                                variant=args.variant).eval()
    with torch.no_grad():
        ref = unet(torch.from_numpy(sample), torch.from_numpy(t), encoder_hidden_states=torch.from_numpy(ehs_torch)).sample.numpy()
    del unet

    results = []
    for u in args.units:
        t0 = time.time()
        m = ct.models.MLModel(args.mlpackage, compute_units=UNITS[u])
        load = time.time() - t0
        feed = {"sample": sample, "timestep": t, "encoder_hidden_states": ehs}
        times = []
        for _ in range(args.runs + 1):
            t1 = time.time()
            y = m.predict(feed)[out_name]
            times.append(time.time() - t1)
        r = {"units": u, "load_s": round(load, 1), "first_predict_s": round(times[0], 2),
             "median_predict_s": round(float(np.median(times[1:])), 3),
             "max_abs_diff": float(np.abs(ref - y).max()), "psnr_db": round(psnr(ref, y), 1),
             "per_step_s_batch2": round(float(np.median(times[1:])), 3)}
        results.append(r)
        print(f"  {u:<12} load {r['load_s']:6.1f}s  predict {r['median_predict_s']:.3f}s (batch {B})  "
              f"max|diff| {r['max_abs_diff']:.4f}  PSNR {r['psnr_db']} dB")
        del m
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mlpackage": args.mlpackage, "reference": args.torch, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
