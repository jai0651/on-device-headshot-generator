"""Stage 7: palettize a Core ML model's weights to N bits, no retraining.

    .venv-coreml/bin/python -m stage7_quantize.palettize --mlpackage <fp16.mlpackage> --bits 8 6 4 --out models/coreml/q

Palettization (section 05): cluster each weight tensor into 2^N values with k-means and store a
lookup table plus an N-bit index per weight. Not a uniform grid like INT8 quantization: the
2^N values sit where the weights actually are, which is why 4-bit holds up better than the
phrase suggests. Mostly a size win; the Neural Engine decompresses on the fly.

  8-bit, 6-bit: one 256 / 64-entry table per tensor
  4-bit:        one 16-entry table per group of 16 output channels ("per_grouped_channel"):
                a single table per tensor is too coarse at 16 values

Run ONE bit-width per process (--bits 6): coremltools 9.0 shuts its k-means worker pool down
after the first palettize_weights call, and the second call fails with "Pool not running".

Tensors under 2048 elements (biases, norms) stay FP16. Writes <out>/<stem>_<N>bit.mlpackage and
a JSON line per variant with size and time taken.
"""

import argparse
import json
import os
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights


def du_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4])
    ap.add_argument("--out", default="models/coreml/q")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--per-tensor", action="store_true",
                    help="force one table per tensor at <=4 bits (per-group tables need an iOS 18 deployment target)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mlpackage).stem.replace("Stable_Diffusion_version_models_fused_", "")
    base = ct.models.MLModel(args.mlpackage, skip_model_load=True)
    print(f"source {Path(args.mlpackage).name}: {du_mb(args.mlpackage):.0f} MB (FP16)")
    for n in args.bits:
        cfg = OpPalettizerConfig(mode="kmeans", nbits=n, num_kmeans_workers=args.workers,
                                 **({"granularity": "per_grouped_channel", "group_size": 16} if n <= 4 and not args.per_tensor else {}))
        t0 = time.time()
        q = palettize_weights(base, OptimizationConfig(global_config=cfg))
        dst = out / f"{stem}_{n}bit.mlpackage"
        q.save(str(dst))
        rec = {"bits": n, "path": str(dst), "mb": round(du_mb(dst)), "minutes": round((time.time() - t0) / 60, 1),
               "granularity": "per_grouped_channel/16" if n <= 4 and not args.per_tensor else "per_tensor"}
        print(json.dumps(rec), flush=True)
        with open(out / "palettize.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        del q


if __name__ == "__main__":
    main()
