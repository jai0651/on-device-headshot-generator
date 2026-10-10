"""Stage 5: every identity method on the same GPU, same fixed prompt set, same seeds.

    python3 -m stage5_zeroshot.generate_methods                       # GPU pod
    uv run python -m stage5_zeroshot.generate_methods --limit 1 --steps 2 --only base faceid_1photo  # smoke

The comparison section 07 says the project is organised around: zero-shot IP-Adapter-FaceID
(no per-user training) against the trained LoRA, plus both stacked, plus ControlNet pose
framing for the headshot prompts. Output: outputs/stage5/<method>/samples/m/*.jpg + manifest,
scored on the Mac by stage5_zeroshot.score_methods.

Order matters: once an IP-Adapter is loaded the UNet expects image embeddings on every call,
so methods without FaceID run first.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel
from PIL import Image

from stage1_sd_pipeline.baseline import build_pipeline
from stage1_sd_pipeline.common import load_components
from stage1_sd_pipeline.manual_loop import generate
from stage2_eval.prompts import NEGATIVE, expand

FACEID_REPO, FACEID_WEIGHTS, FACEID_LORA = "h94/IP-Adapter-FaceID", "ip-adapter-faceid_sd15.bin", "ip-adapter-faceid_sd15_lora.safetensors"
CONTROLNET = "lllyasviel/control_v11p_sd15_openpose"
POSE_PROMPTS = ("corporate_suit", "passport")   # the headshot use case: where framing matters

# name: (face-LoRA scale, FaceID embeddings ("1photo"/"17photos"/None), IP scale, ControlNet?)
METHODS = {
    "base":                 (0.0, None, 0.0, False),
    "lora":                 (1.0, None, 0.0, False),
    "base_pose":            (0.0, None, 0.0, True),
    "lora_pose":            (1.0, None, 0.0, True),
    # ---- IP-Adapter loaded from here on ----
    "faceid_1photo":        (0.0, "1photo", 1.0, False),
    "faceid_17photos":      (0.0, "17photos", 1.0, False),
    "faceid_17photos_s0.6": (0.0, "17photos", 0.6, False),
    "lora+faceid":          (1.0, "17photos", 0.6, False),
    "faceid_17photos_pose": (0.0, "17photos", 1.0, True),
}


def id_embeds(npz, which, device, dtype):
    d = np.load(npz)
    e = d["embeddings"][[int(d["best"])]] if which == "1photo" else d["embeddings"]
    e = torch.from_numpy(e)[:, None, :]               # (n_photos, 1, 512)
    e = e[None]                                       # (1, n_photos, 1, 512): diffusers' FaceID layout
    return [torch.cat([torch.zeros_like(e), e]).to(device, dtype)]  # [uncond; cond] for CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-lora", default="ohwx man", help="subject phrase when the face LoRA is active")
    ap.add_argument("--subject-plain", default="a man", help="subject phrase otherwise (FaceID supplies who)")
    ap.add_argument("--lora", default="lora_data/checkpoints/r16_te-off_s1500")
    ap.add_argument("--faceid", default="lora_data/faceid/jai.npz")
    ap.add_argument("--pose", default="lora_data/control/passport_pose.png")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/stage5")
    ap.add_argument("--negative-extra", default="", help='appended to the negative prompt, e.g. "black and white, monochrome"')
    args = ap.parse_args()

    c = load_components()
    pipe = build_pipeline(c, "dpmpp_2m_karras")  # used only to load adapters onto c's modules
    pipe.load_lora_weights(args.lora, adapter_name="face")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=c.dtype, variant="fp16").to(c.device)
    pose = Image.open(args.pose)
    ip_loaded = False

    for name, (lora_w, faceid, ip_scale, use_pose) in METHODS.items():
        if args.only and name not in args.only:
            continue
        if faceid and not ip_loaded:
            pipe.load_ip_adapter(FACEID_REPO, subfolder=None, weight_name=FACEID_WEIGHTS, image_encoder_folder=None)
            pipe.load_lora_weights(FACEID_REPO, weight_name=FACEID_LORA, adapter_name="faceid_lora")
            ip_loaded = True
        # which LoRAs are active: our face LoRA, and FaceID's own companion LoRA whenever FaceID is on
        names, weights = ["face"], [lora_w]
        if ip_loaded:
            names.append("faceid_lora")
            weights.append(1.0 if faceid else 0.0)
            pipe.set_ip_adapter_scale(ip_scale if faceid else 0.0)
        pipe.set_adapters(names, adapter_weights=weights)
        emb = id_embeds(args.faceid, faceid or "17photos", c.device, c.dtype) if ip_loaded else None

        subject = args.subject_lora if lora_w > 0 else args.subject_plain
        items = [it for it in expand(subject) if it["seed"] in args.seeds]
        if use_pose:
            items = [it for it in items if it["name"] in POSE_PROMPTS]
        items = items[: args.limit or None]
        d = Path(args.out) / name / "samples" / "m"
        d.mkdir(parents=True, exist_ok=True)
        t0, done = time.time(), []
        for it in items:
            neg = f"{NEGATIVE}, {args.negative_extra}" if args.negative_extra else NEGATIVE
            img, _ = generate(c, it["prompt"], neg, steps=args.steps, cfg=6.0, seed=it["seed"], ip_embeds=emb,
                              controlnet=controlnet if use_pose else None, control_image=pose if use_pose else None)
            it2 = {**it, "file": it["file"].replace(".png", ".jpg"), "method": name}
            img.save(d / it2["file"], quality=95)
            done.append(it2)
        (d / "manifest.json").write_text(json.dumps(done, indent=2))
        print(f"  {name:<22} {len(done):>3} images  {time.time() - t0:5.0f}s")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
