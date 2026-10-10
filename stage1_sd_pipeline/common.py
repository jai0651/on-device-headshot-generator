"""Loading SD 1.5 as its three separate networks, plus the samplers to swap between."""

from dataclasses import dataclass

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEFAULT_PROMPT = (
    "professional corporate headshot of a woman, navy blazer, plain grey studio backdrop, "
    "softbox lighting, 85mm lens, sharp focus"
)
DEFAULT_NEGATIVE = "blurry, lowres, deformed, cartoon, painting, oversaturated"

# name -> (class, extra config). The sampler is not part of the model: every one of
# these drives the exact same UNet, they just integrate its predictions differently.
SCHEDULERS = {
    "pndm": (PNDMScheduler, {}),  # SD 1.5's shipped default
    "ddim": (DDIMScheduler, {}),
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpmpp_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "unipc": (UniPCMultistepScheduler, {}),
    # stage 8: a consistency sampler for LCM-distilled weights (4-8 steps, CFG 1). Only valid with
    # the LCM-LoRA (or an LCM-distilled UNet) loaded; on the plain UNet it produces mush.
    "lcm": (LCMScheduler, {}),
}


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Components:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler_config: dict
    device: str
    dtype: torch.dtype


def load_components(fp32=False):
    device = pick_device()
    dtype = torch.float32 if fp32 else torch.float16
    kw = dict(torch_dtype=dtype, variant="fp16")
    c = Components(
        tokenizer=CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer"),
        text_encoder=CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", **kw).to(device),
        unet=UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", **kw).to(device),
        vae=AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", **kw).to(device),
        scheduler_config=PNDMScheduler.load_config(MODEL_ID, subfolder="scheduler"),
        device=device,
        dtype=dtype,
    )
    for m in (c.text_encoder, c.unet, c.vae):
        m.eval().requires_grad_(False)
    return c


def make_scheduler(name, c):
    """A fresh scheduler every call — they carry per-run state (multistep history)."""
    cls, extra = SCHEDULERS[name]
    return cls.from_config(c.scheduler_config, **extra)


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def labeled_grid(images, rows, cols, path, row_labels=None, col_labels=None, tile=256):
    """Lay PIL images out in a rows x cols grid with text labels along the edges."""
    font = ImageFont.load_default(size=18)
    left = 190 if row_labels else 0
    top = 34 if col_labels else 0
    sheet = Image.new("RGB", (left + cols * tile, top + rows * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(images):
        r, col = divmod(i, cols)
        sheet.paste(img.resize((tile, tile), Image.LANCZOS), (left + col * tile, top + r * tile))
    for col, text in enumerate(col_labels or []):
        draw.text((left + col * tile + 8, 8), text, fill="black", font=font)
    for r, text in enumerate(row_labels or []):
        draw.text((8, top + r * tile + tile // 2 - 9), text, fill="black", font=font)
    sheet.save(path)
