"""Stage 5c, step 2 (GPU): the edit path. Keep the real face, repaint everything else.

    python3 -m stage5_zeroshot.edit_generate                                  # GPU pod
    uv run python -m stage5_zeroshot.edit_generate --limit 1 --steps 2 --seeds 0   # smoke

For each prepared selfie (stage5_zeroshot.edit_prepare) x outfit prompt x seed:
  1. inpaint: SD 1.5's inpainting UNet (9-channel input: noisy latent + masked-image latent
     + mask) repaints clothes and backdrop, with the passport pose ControlNet holding the
     shoulders where the reframed head expects them
  2. paste back: the kept face/hair pixels go back in through the feathered mask, because
     even unmasked pixels pick up VAE round-trip error
  3. harmonize: img2img over the whole image at low strength (0 / 0.2 / 0.35) so the
     face's lighting matches the new backdrop. This touches face pixels too: the identity
     scores say what it costs.

Uses diffusers' ControlNet-inpaint and img2img pipelines on the same loaded components
(the hand-written loop has no inpainting path). Output: outputs/stage5c/edit_hX.XX/samples/m/
with manifests shaped like stage 5's, scored by stage5_zeroshot.score_methods --root outputs/stage5c.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionImg2ImgPipeline,
                       UNet2DConditionModel)
from PIL import Image

from stage1_sd_pipeline.common import load_components, make_scheduler
from stage2_eval.prompts import NEGATIVE
from stage5_zeroshot.generate_methods import CONTROLNET

INPAINT_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
NEG = NEGATIVE + ", black and white, monochrome, desaturated, shadow on face, text, letters, logo, watermark"  # 5c run 1 painted garbled text on plain t-shirts
OUTFITS = [  # (group, name, prompt): group decides which stage 5 table column it lands in
    ("identity", "blazer", "professional corporate headshot, navy blazer and white shirt, plain light grey studio backdrop, soft even lighting, sharp focus"),
    ("identity", "suit", "professional headshot, charcoal suit and tie, plain off-white backdrop, soft studio lighting, sharp focus"),
    ("compliance", "idphoto", "modern color ID photo, plain white background, white t-shirt, neutral expression, even lighting, sharp focus"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="lora_data/edit", help="folders from edit_prepare")
    ap.add_argument("--pose", default="lora_data/control/passport_pose.png")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.2, 0.35])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/stage5c")
    args = ap.parse_args()

    c = load_components()
    inpaint_unet = UNet2DConditionModel.from_pretrained(INPAINT_REPO, subfolder="unet", torch_dtype=c.dtype,
                                                        variant="fp16").to(c.device)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=c.dtype, variant="fp16").to(c.device)
    common = dict(vae=c.vae, text_encoder=c.text_encoder, tokenizer=c.tokenizer, safety_checker=None,
                  feature_extractor=None, requires_safety_checker=False)
    inpaint = StableDiffusionControlNetInpaintPipeline(unet=inpaint_unet, controlnet=controlnet,
                                                       scheduler=make_scheduler("dpmpp_2m_karras", c), **common)
    harmonize = StableDiffusionImg2ImgPipeline(unet=c.unet, scheduler=make_scheduler("dpmpp_2m_karras", c), **common)
    pose = Image.open(args.pose).convert("RGB")

    jobs = sorted(p for p in Path(args.jobs).iterdir() if (p / "framed.png").exists())
    todo = [(j, o, s) for j in jobs for o in OUTFITS for s in args.seeds][: args.limit or None]
    manifests = {h: [] for h in args.strengths}
    t0 = time.time()
    for job, (group, name, prompt), seed in todo:
        framed = Image.open(job / "framed.png").convert("RGB")
        repaint = Image.open(job / "repaint.png").convert("L")
        keep = np.asarray(Image.open(job / "keep.png").convert("L"), np.float32)[..., None] / 255
        gen = lambda: torch.Generator("cpu").manual_seed(seed)
        painted = inpaint(prompt, image=framed, mask_image=repaint, control_image=pose, negative_prompt=NEG,
                          num_inference_steps=args.steps, guidance_scale=6.0, strength=1.0, generator=gen(),
                          height=512, width=512).images[0]
        pasted = Image.fromarray((keep * np.asarray(framed, np.float32) +
                                  (1 - keep) * np.asarray(painted, np.float32)).round().astype(np.uint8))
        for h in args.strengths:
            # img2img runs int(steps x strength) steps: keep that >= 2, or low strengths do nothing
            n = max(args.steps, int(np.ceil(2 / h))) if h else 0
            img = pasted if h == 0 else harmonize(prompt, image=pasted, strength=h, negative_prompt=NEG,
                                                  num_inference_steps=n, guidance_scale=6.0,
                                                  generator=gen()).images[0]
            d = Path(args.out) / f"edit_h{h:.2f}" / "samples" / "m"
            d.mkdir(parents=True, exist_ok=True)
            fname = f"{job.name[:24]}_{name}_s{seed}.jpg"
            img.save(d / fname, quality=95)
            manifests[h].append({"group": group, "name": name, "prompt": prompt, "seed": seed, "file": fname,
                                 "selfie": job.name, "harmonize": h, "method": f"edit_h{h:.2f}"})
        print(f"  {job.name[:28]:<28} {name:<8} seed {seed}  ({time.time() - t0:.0f}s)")
    for h, items in manifests.items():
        (Path(args.out) / f"edit_h{h:.2f}" / "samples" / "m" / "manifest.json").write_text(json.dumps(items, indent=2))
    print(f"-> {args.out}/ ({len(todo)} edits x {len(args.strengths)} strengths)")


if __name__ == "__main__":
    main()
