"""Stage 8a (Mac, PyTorch): does few-step LCM keep the face?

    uv run python -m stage8_lcm.lcm_eval

LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) distils SD 1.5 so each step jumps toward the
end of the denoising trajectory instead of walking it: 4-8 steps instead of 25, and it works at
CFG 1, so the UNet runs once per step instead of twice. It's a LoRA, so it stacks with the face
LoRA: both are adapters on the same UNet.

Same seeds, prompts and pose ControlNet for every config, one machine (this Mac's GPU):
  baseline  DPM++ 2M Karras, 25 steps, CFG 6   -> 50 UNet calls
  lcm8/6/4  LCM scheduler, 8/6/4 steps, CFG 1  -> 8/6/4 UNet calls
  lcm4_g1.5 LCM, 4 steps, CFG 1.5              -> 8 UNet calls
Writes outputs/stage8/<config>/samples/m/ (scored by stage5_zeroshot.score_methods --root outputs/stage8).
Open question 4: does distillation cost more identity than it costs general quality?
"""

import argparse
import json
import time
from pathlib import Path

from diffusers import ControlNetModel
from PIL import Image

from stage1_sd_pipeline.baseline import build_pipeline
from stage1_sd_pipeline.common import load_components
from stage1_sd_pipeline.manual_loop import generate
from stage2_eval.prompts import NEGATIVE, expand
from stage5_zeroshot.generate_methods import CONTROLNET

LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"
CONFIGS = {  # name: (scheduler, steps, cfg, lcm adapter weight)
    "baseline": ("dpmpp_2m_karras", 25, 6.0, 0.0),
    "lcm8": ("lcm", 8, 1.0, 1.0),
    "lcm6": ("lcm", 6, 1.0, 1.0),
    "lcm4": ("lcm", 4, 1.0, 1.0),
    "lcm4_g1.5": ("lcm", 4, 1.5, 1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/stage8")
    args = ap.parse_args()

    c = load_components()
    pipe = build_pipeline(c, "dpmpp_2m_karras")
    pipe.load_lora_weights("lora_data/checkpoints/r16_te-off_s1500", adapter_name="face")
    pipe.load_lora_weights(LCM_LORA, weight_name="pytorch_lora_weights.safetensors", adapter_name="lcm")  # name needed offline
    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=c.dtype, variant="fp16").to(c.device)
    pose = Image.open("lora_data/control/passport_pose.png")
    items = [it for it in expand("ohwx man") if it["seed"] in args.seeds][: args.limit or None]

    for name, (sched, steps, cfg, lcm_w) in CONFIGS.items():
        if args.only and name not in args.only:
            continue
        pipe.set_adapters(["face", "lcm"], adapter_weights=[1.0, lcm_w])
        d = Path(args.out) / name / "samples" / "m"
        d.mkdir(parents=True, exist_ok=True)
        done, times = [], []
        for it in items:
            t0 = time.time()
            img, st = generate(c, it["prompt"], NEGATIVE, steps=steps, cfg=cfg, seed=it["seed"], scheduler=sched,
                               controlnet=controlnet, control_image=pose)
            times.append(time.time() - t0)
            it2 = {**it, "file": it["file"].replace(".png", ".jpg"), "method": name, "unet_calls": st["unet_calls"]}
            img.save(d / it2["file"], quality=95)
            done.append(it2)
        (d / "manifest.json").write_text(json.dumps(done, indent=2))
        med = sorted(times)[len(times) // 2]
        (d.parent.parent / "timing.json").write_text(json.dumps({"steps": steps, "cfg": cfg, "unet_calls": done[0]["unet_calls"],
                                                                 "median_image_s": round(med, 2)}, indent=2))
        print(f"  {name:<10} {steps:>2} steps  cfg {cfg:<4} {done[0]['unet_calls']:>2} UNet calls  {med:5.1f}s/image (Mac GPU)", flush=True)


if __name__ == "__main__":
    main()
