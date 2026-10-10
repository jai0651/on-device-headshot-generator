"""Prove the hand-written loop's FaceID and ControlNet paths match diffusers' own pipelines,
the stage 1 test extended to stage 5. If they diverge, stage 5 scores measure a bug.

    uv run python -m stage5_zeroshot.validate_paths          # Mac, ~2 min + first-time downloads
"""

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

from stage1_sd_pipeline.baseline import build_pipeline
from stage1_sd_pipeline.common import load_components, make_scheduler
from stage1_sd_pipeline.manual_loop import generate
from stage5_zeroshot.generate_methods import CONTROLNET, FACEID_REPO, FACEID_WEIGHTS, id_embeds

PROMPT, NEG, STEPS, CFG, SEED = "corporate headshot of a man, navy suit, grey backdrop", "blurry", 8, 6.0, 0


def diff(a, b):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return d.max(), d.mean()


def main():
    c = load_components()
    gen = lambda: torch.Generator("cpu").manual_seed(SEED)

    # ControlNet first: no IP-Adapter on the UNet yet
    cn = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=c.dtype, variant="fp16").to(c.device)
    pose = Image.open("lora_data/control/passport_pose.png")
    ref = StableDiffusionControlNetPipeline(vae=c.vae, text_encoder=c.text_encoder, tokenizer=c.tokenizer, unet=c.unet,
                                            controlnet=cn, scheduler=make_scheduler("dpmpp_2m_karras", c),
                                            safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    a = ref(PROMPT, image=pose, negative_prompt=NEG, num_inference_steps=STEPS, guidance_scale=CFG, generator=gen()).images[0]
    b, _ = generate(c, PROMPT, NEG, steps=STEPS, cfg=CFG, seed=SEED, controlnet=cn, control_image=pose)
    mx, mean = diff(a, b)
    print(f"ControlNet: max pixel diff {mx}, mean {mean:.3f}  {'MATCH' if mx <= 2 else 'MISMATCH'}")
    a.save("runs/stage5_smoke/val_controlnet_ref.jpg")
    b.save("runs/stage5_smoke/val_controlnet_mine.jpg")

    pipe = build_pipeline(c, "dpmpp_2m_karras")
    pipe.load_ip_adapter(FACEID_REPO, subfolder=None, weight_name=FACEID_WEIGHTS, image_encoder_folder=None)
    pipe.set_ip_adapter_scale(1.0)
    emb = id_embeds("lora_data/faceid/jai.npz", "1photo", c.device, c.dtype)
    a = pipe(PROMPT, negative_prompt=NEG, ip_adapter_image_embeds=emb, num_inference_steps=STEPS,
             guidance_scale=CFG, generator=gen()).images[0]
    b, _ = generate(c, PROMPT, NEG, steps=STEPS, cfg=CFG, seed=SEED, ip_embeds=emb)
    mx, mean = diff(a, b)
    print(f"FaceID:     max pixel diff {mx}, mean {mean:.3f}  {'MATCH' if mx <= 2 else 'MISMATCH'}")
    a.save("runs/stage5_smoke/val_faceid_ref.jpg")
    b.save("runs/stage5_smoke/val_faceid_mine.jpg")


if __name__ == "__main__":
    main()
