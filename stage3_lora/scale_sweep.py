"""Generate the fixed prompt set with one trained LoRA at several strengths.

    python3 -m stage3_lora.scale_sweep --lora lora_data/checkpoints/r16_te-off_s1500 \\
        --run scale_r16_te-off_s1500                                   # GPU pod
    uv run python -m stage3_lora.scale_sweep --lora ... --run smoke --scales 0.8 --limit 1 --steps 2

Section 03: "LoRA scale 0.6-0.9, not 1.0. Dialling it down often buys a lot of prompt
flexibility for very little likeness. Free tuning after training." This measures that
trade for one checkpoint: the output is outputs/stage3/<run>/samples/scale_X.XX/, scored
on the Mac by stage3_lora.score_checkpoints exactly like training checkpoints.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from stage1_sd_pipeline.baseline import build_pipeline
from stage1_sd_pipeline.common import load_components
from stage1_sd_pipeline.manual_loop import generate
from stage2_eval.prompts import NEGATIVE, expand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="folder with pytorch_lora_weights.safetensors")
    ap.add_argument("--run", required=True)
    ap.add_argument("--subject", default="ohwx man")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    c = load_components()
    pipe = build_pipeline(c, "dpmpp_2m_karras")  # only used to load/scale the adapter onto c.unet / c.text_encoder
    pipe.load_lora_weights(args.lora, adapter_name="face")
    items = [it for it in expand(args.subject) if it["seed"] in args.seeds][: args.limit or None]
    out = Path("outputs/stage3") / args.run
    for scale in args.scales:
        pipe.set_adapters(["face"], adapter_weights=[scale])
        d = out / "samples" / f"scale_{scale:.2f}"
        d.mkdir(parents=True, exist_ok=True)
        t0, done = time.time(), []
        for it in items:
            img, _ = generate(c, it["prompt"], NEGATIVE, steps=args.steps, cfg=6.0, seed=it["seed"])
            it2 = {**it, "file": it["file"].replace(".png", ".jpg"), "lora_scale": scale}
            img.save(d / it2["file"], quality=95)
            done.append(it2)
        (d / "manifest.json").write_text(json.dumps(done, indent=2))
        print(f"  scale {scale:.2f}: {len(done)} images in {time.time() - t0:.0f}s")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"-> {out}/samples/")


if __name__ == "__main__":
    main()
