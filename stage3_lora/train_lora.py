"""Train a face LoRA on SD 1.5, sampling the fixed prompt set at every checkpoint.

    python3 -m stage3_lora.train_lora --data lora_data/jai --run r16_lr1e-4     # on a GPU pod
    uv run python -m stage3_lora.train_lora --data lora_data/jai --run smoke \\
        --steps 4 --save-every 2 --resolution 256 --sample-limit 1 --sample-steps 2   # local smoke

Section 02's LoRA, literally: freeze W, add B·A to the UNet's attention projections
(to_q/k/v/out), B initialised to zero so step 0 *is* the base model. Optionally the same
on the text encoder (section 03: "helps likeness, hurts flexibility, try both").

Every --save-every steps: save the adapter (fp16 safetensors) and generate the fixed
prompt set with "{subject}" = the trigger phrase, one seed per prompt. The best LoRA is
almost never the last one; the per-checkpoint samples, scored on the Mac by
stage3_lora.score_checkpoints, turn picking one into a measurement.

Outputs (small, fetched):  outputs/stage3/<run>/samples/step_XXXX/*.jpg + manifest.json
                           outputs/stage3/<run>/train_log.csv, config.json
Adapters (fp16, ~6 MB each): outputs_raw/stage3/<run>/lora/step_XXXX/
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image

from stage1_sd_pipeline.common import load_components
from stage1_sd_pipeline.manual_loop import generate
from stage2_eval.prompts import NEGATIVE, expand

UNET_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
TE_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def load_dataset(folder, resolution):
    folder = Path(folder)
    captions = json.loads((folder / "captions.json").read_text())
    items = []
    for name, cap in captions.items():
        img = Image.open(folder / name).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1  # [-1, 1]
        items.append((name, cap, x))
    return items


def save_adapter(c, path, train_te):
    path.mkdir(parents=True, exist_ok=True)
    half = lambda sd: {k: v.detach().to(torch.float16).cpu() for k, v in sd.items()}
    unet_sd = half(convert_state_dict_to_diffusers(get_peft_model_state_dict(c.unet)))
    te_sd = half(convert_state_dict_to_diffusers(get_peft_model_state_dict(c.text_encoder))) if train_te else None
    StableDiffusionPipeline.save_lora_weights(save_directory=str(path), unet_lora_layers=unet_sd,
                                              text_encoder_lora_layers=te_sd, safe_serialization=True)


def sample(c, subject, out_dir, seeds, steps, limit, amp):
    """The fixed prompt set through the model as it is right now (LoRA active)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [it for it in expand(subject) if it["seed"] in seeds][: limit or None]
    c.unet.eval()
    c.text_encoder.eval()
    t0 = time.time()
    for it in items:
        with amp:
            img, _ = generate(c, it["prompt"], NEGATIVE, steps=steps, cfg=6.0, seed=it["seed"])
        it["file"] = it["file"].replace(".png", ".jpg")
        img.save(out_dir / it["file"], quality=95)
    (out_dir / "manifest.json").write_text(json.dumps(items, indent=2))
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--subject", default="ohwx man", help="trigger phrase used in the captions")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None, help="default: = rank")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--te-lr", type=float, default=0.0, help=">0 also trains a text-encoder LoRA")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--min-snr", type=float, default=5.0, help="min-SNR loss weighting gamma (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sample-steps", type=int, default=25)
    ap.add_argument("--sample-limit", type=int, default=0)
    ap.add_argument("--no-sample-step0", action="store_true")
    args = ap.parse_args()
    args.alpha = args.alpha or args.rank
    train_te = args.te_lr > 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path("outputs/stage3") / args.run
    raw = Path("outputs_raw/stage3") / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    c = load_components(fp32=True)  # fp32 master weights; bf16 autocast on CUDA does the speed
    dev = c.device
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    noise_sched = DDPMScheduler.from_config(c.scheduler_config)
    abar = noise_sched.alphas_cumprod.to(dev)

    # ---- data: encode every image once; resample from the posterior each step ----
    items = load_dataset(args.data, args.resolution)
    with torch.no_grad():
        dists = [c.vae.encode(x[None].to(dev)).latent_dist for _, _, x in items]
        lat_mean = torch.cat([d.mean for d in dists])
        lat_std = torch.cat([d.std for d in dists])
        captions = [cap for _, cap, _ in items]
        if not train_te:
            from stage1_sd_pipeline.manual_loop import encode_text
            cached_emb = torch.cat([encode_text(c, cap)[1] for cap in captions])
    c.vae.to("cpu")  # not needed again until sampling decodes; free the memory
    print(f"{len(items)} images, latents {tuple(lat_mean.shape)}, text encoder {'TRAINED' if train_te else 'frozen'}")

    # ---- LoRA: B starts at zero, so the adapted model is exactly the base model at step 0 ----
    c.unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
                                  target_modules=UNET_TARGETS))
    params = [{"params": [p for p in c.unet.parameters() if p.requires_grad], "lr": args.lr}]
    if train_te:
        c.text_encoder.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
                                              target_modules=TE_TARGETS))
        params.append({"params": [p for p in c.text_encoder.parameters() if p.requires_grad], "lr": args.te_lr})
        c.text_encoder.gradient_checkpointing_enable()
    c.unet.enable_gradient_checkpointing()
    n_train = sum(p.numel() for g in params for p in g["params"])
    n_unet = sum(p.numel() for p in c.unet.parameters())
    print(f"trainable {n_train / 1e6:.2f}M params ({100 * n_train / n_unet:.2f}% of the UNet's {n_unet / 1e6:.0f}M)")

    opt = torch.optim.AdamW(params, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    def checkpoint(step):
        if step > 0:
            save_adapter(c, raw / "lora" / f"step_{step:04d}", train_te)
        c.vae.to(dev)
        dt = sample(c, args.subject, out / "samples" / f"step_{step:04d}", args.sample_seeds,
                    args.sample_steps, args.sample_limit, amp)
        c.vae.to("cpu")
        print(f"  checkpoint step {step}: sampled in {dt:.0f}s")

    if not args.no_sample_step0:
        checkpoint(0)

    log = open(out / "train_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "loss", "lr"])
    order, t0, running = [], time.time(), 0.0
    for step in range(1, args.steps + 1):
        c.unet.train()
        if train_te:
            c.text_encoder.train()
        if not order:
            order = random.sample(range(len(items)), len(items))
        i = order.pop()
        latents = (lat_mean[i:i + 1] + lat_std[i:i + 1] * torch.randn_like(lat_mean[i:i + 1])) * c.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=dev)
        noisy = noise_sched.add_noise(latents, noise, t)
        with amp:
            if train_te:
                from stage1_sd_pipeline.manual_loop import encode_text
                emb = encode_text(c, captions[i])[1]
            else:
                emb = cached_emb[i:i + 1]
            pred = c.unet(noisy, t, encoder_hidden_states=emb).sample
        loss = F.mse_loss(pred.float(), noise, reduction="none").mean(dim=(1, 2, 3))
        if args.min_snr:
            snr = abar[t] / (1 - abar[t])
            loss = loss * torch.clamp(snr, max=args.min_snr) / snr  # down-weight easy low-noise steps
        loss = loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 25 == 0 or step == args.steps:
            w.writerow([step, f"{running:.5f}", f"{sched.get_last_lr()[0]:.2e}"])
            log.flush()
            rate = step / (time.time() - t0)
            print(f"step {step:5d}  loss {running:.4f}  lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s")
        if step % args.save_every == 0 or step == args.steps:
            checkpoint(step)
    log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}/, adapters in {raw}/lora/")


if __name__ == "__main__":
    main()
