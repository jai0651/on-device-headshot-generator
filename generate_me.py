"""Generate headshots of you with SD 1.5 in PyTorch, on this Mac's GPU (~1 min per image).

    uv run python generate_me.py "corporate headshot of a man, navy suit, grey studio backdrop" --n 2
    uv run python generate_me.py "..." --method lora        # the trained LoRA instead of FaceID
    uv run python generate_me.py "..." --no-pose            # without the passport pose control

Default = the best generative method measured in stage 5: IP-Adapter-FaceID from one selfie's face
embedding (no training) + the passport pose ControlNet (keeps the whole head in frame) + the
colour-fix negative prompt. With FaceID, describe "a man"; with --method lora, say "ohwx man".
Images: outputs/mine/<timestamp>_s<seed>.jpg
"""

import argparse
import time
from pathlib import Path

from diffusers import ControlNetModel
from PIL import Image

from stage1_sd_pipeline.baseline import build_pipeline
from stage1_sd_pipeline.common import load_components
from stage1_sd_pipeline.manual_loop import generate
from stage2_eval.prompts import NEGATIVE
from stage5_zeroshot.generate_methods import CONTROLNET, FACEID_LORA, FACEID_REPO, FACEID_WEIGHTS, id_embeds

COLOUR_FIX = "black and white, monochrome, grayscale, desaturated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=2, help="how many images (seeds 0..n-1, or from --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["faceid", "lora"], default="faceid")
    ap.add_argument("--photos", choices=["1photo", "17photos"], default="1photo", help="FaceID: one selfie or all 17")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()

    c = load_components()
    pipe = build_pipeline(c, "dpmpp_2m_karras")
    emb = None
    if args.method == "faceid":
        pipe.load_ip_adapter(FACEID_REPO, subfolder=None, weight_name=FACEID_WEIGHTS, image_encoder_folder=None)
        pipe.load_lora_weights(FACEID_REPO, weight_name=FACEID_LORA, adapter_name="faceid_lora")
        pipe.set_ip_adapter_scale(1.0)
        emb = id_embeds("lora_data/faceid/jai.npz", args.photos, c.device, c.dtype)
    else:
        pipe.load_lora_weights("lora_data/checkpoints/r16_te-off_s1500", adapter_name="face")
        if "ohwx" not in args.prompt:
            print('note: the LoRA only knows you as "ohwx man"; add it to the prompt')
    controlnet = None if args.no_pose else ControlNetModel.from_pretrained(
        CONTROLNET, torch_dtype=c.dtype, variant="fp16").to(c.device)
    pose = None if args.no_pose else Image.open("lora_data/control/passport_pose.png")

    out = Path("outputs/mine")
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for seed in range(args.seed, args.seed + args.n):
        t0 = time.time()
        img, _ = generate(c, args.prompt, f"{NEGATIVE}, {COLOUR_FIX}", steps=args.steps, cfg=6.0, seed=seed,
                          ip_embeds=emb, controlnet=controlnet, control_image=pose)
        path = out / f"{stamp}_{args.method}_s{seed}.jpg"
        img.save(path, quality=95)
        print(f"  {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
