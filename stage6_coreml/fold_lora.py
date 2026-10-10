"""Stage 6b / 8b (Mac): fold one or more LoRAs into the UNet weights, and prove it's exact.

    uv run python -m stage6_coreml.fold_lora --lora lora_data/checkpoints/r16_te-off_s1500 \\
        --out models/fused/sd15_ohwx
    uv run python -m stage6_coreml.fold_lora --lora lora_data/checkpoints/r16_te-off_s1500 \\
        --lora latent-consistency/lcm-lora-sdv1-5:pytorch_lora_weights.safetensors --out models/fused/sd15_ohwx_lcm

Section 02: W' = W + (alpha/r)·B·A is computed once and written into the weights. What's left
is an ordinary SD 1.5 UNet: no adapter modules, no extra ops for the Core ML converter, zero
runtime cost. Several LoRAs fold in together (stage 8: the face LoRA + the LCM-LoRA).

Verification: the same random inputs through (a) the UNet with the LoRAs attached as live
adapters and (b) the folded UNet must agree to fp32 rounding. The output folder is a complete
diffusers model that Apple's converter (diffusers 0.30) can load: non-variant file names and
the classic tokenizer files, the two incompatibilities stage 6 hit by hand.
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from stage1_sd_pipeline.common import MODEL_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", required=True,
                    help="local folder, or hub_repo:weight_file; repeat to fold several")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # fp32 on CPU: a one-off weight edit; variant="fp16" because only the fp16 files are cached
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32, variant="fp16",
                                                   safety_checker=None, requires_safety_checker=False)
    names = []
    for i, spec in enumerate(args.lora):
        name = f"l{i}"
        if ":" in spec and not os.path.exists(spec):
            repo, weight = spec.split(":", 1)
            pipe.load_lora_weights(repo, weight_name=weight, adapter_name=name)
        else:
            pipe.load_lora_weights(spec, adapter_name=name)
        names.append(name)
    pipe.set_adapters(names, adapter_weights=[1.0] * len(names))

    g = torch.Generator().manual_seed(0)
    x, t, emb = torch.randn(2, 4, 64, 64, generator=g), torch.tensor([500, 500]), torch.randn(2, 77, 768, generator=g)
    with torch.no_grad():
        live = pipe.unet(x, t, encoder_hidden_states=emb).sample
        pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
        pipe.unload_lora_weights()
        folded = pipe.unet(x, t, encoder_hidden_states=emb).sample
    left = sum("lora" in n for n, _ in pipe.unet.named_modules())
    err = (live - folded).abs()
    print(f"{len(names)} LoRA(s) folded · live vs folded: max |diff| {err.max():.2e}, mean {err.mean():.2e}, LoRA modules left: {left}")
    assert err.max() < 1e-3 and left == 0, "fold is not exact"

    out = Path(args.out)
    pipe.to(torch.float16).save_pretrained(out, safe_serialization=True, variant="fp16")
    for f in out.rglob("*.fp16.safetensors"):          # converter loads without a variant
        f.rename(f.with_name(f.name.replace(".fp16", "")))
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/*/tokenizer"))[0]
    shutil.rmtree(out / "tokenizer", ignore_errors=True)  # transformers 5's format; 4.44 can't read it
    shutil.copytree(snap, out / "tokenizer")
    print(f"-> {out}/ (fp16 diffusers model, converter-ready)")


if __name__ == "__main__":
    main()
