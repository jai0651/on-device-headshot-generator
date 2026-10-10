"""Instrument 2: did it do what the prompt asked?

CLIP score = 100 x cosine(CLIP image embedding, CLIP text embedding), using the same
ViT-L/14 CLIP family whose text tower SD 1.5 conditions on. Crude, but it drops when
a model stops listening, which is exactly the overfitting failure identity can't see:
a LoRA that memorised my selfies scores brilliantly on identity and ignores every prompt.
"""

import torch
from transformers import CLIPModel, CLIPProcessor

from .common import load_rgb

CLIP_ID = "openai/clip-vit-large-patch14"


class ClipScorer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def score(self, image, prompt):
        inputs = self.proc(text=[prompt], images=load_rgb(image), return_tensors="pt",
                           padding=True, truncation=True).to(self.device)
        img = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        # transformers 5 returns an output object with the projected embedding in
        # pooler_output; 4.x returned the tensor directly
        img, txt = (getattr(x, "pooler_output", x) for x in (img, txt))
        img, txt = (x / x.norm(dim=-1, keepdim=True) for x in (img, txt))
        return float(100 * (img * txt).sum().clamp(min=0))
