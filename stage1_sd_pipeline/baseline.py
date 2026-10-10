"""Step 1a: the black box. One call to the stock diffusers pipeline.

    uv run python -m stage1_sd_pipeline.baseline
    uv run python -m stage1_sd_pipeline.baseline --prompt "..." --steps 30 --cfg 6
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from .common import DEFAULT_NEGATIVE, DEFAULT_PROMPT, SCHEDULERS, load_components, make_scheduler


def build_pipeline(c, scheduler_name):
    return StableDiffusionPipeline(
        vae=c.vae, text_encoder=c.text_encoder, tokenizer=c.tokenizer, unet=c.unet,
        scheduler=make_scheduler(scheduler_name, c),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheduler", choices=SCHEDULERS, default="dpmpp_2m_karras")
    ap.add_argument("--fp32", action="store_true", help="use if fp16 gives black/NaN images")
    ap.add_argument("--out", default="runs/stage1")
    args = ap.parse_args()

    c = load_components(args.fp32)
    pipe = build_pipeline(c, args.scheduler)
    t0 = time.time()
    image = pipe(
        args.prompt, negative_prompt=args.negative, num_inference_steps=args.steps,
        guidance_scale=args.cfg, generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "baseline.png")
    print(f"{time.time() - t0:.1f}s on {c.device} -> {out / 'baseline.png'}")


if __name__ == "__main__":
    main()
