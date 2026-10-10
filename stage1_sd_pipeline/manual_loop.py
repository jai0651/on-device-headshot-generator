"""Step 1b: the same pipeline, written out as an explicit loop.

    uv run python -m stage1_sd_pipeline.manual_loop             # generate, print every shape
    uv run python -m stage1_sd_pipeline.manual_loop --compare   # ...and check against diffusers pixel-for-pixel
    uv run python -m stage1_sd_pipeline.manual_loop --my-ddim --compare
        # replace scheduler.step with the DDIM update from stage 0, check it still matches

Everything StableDiffusionPipeline.__call__ does for text-to-image, minus the safety
checker and argument validation.
"""

import argparse
import inspect
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from .common import DEFAULT_NEGATIVE, DEFAULT_PROMPT, SCHEDULERS, load_components, make_scheduler, param_count


def show(verbose, name, x, note=""):
    if verbose:
        shape = "x".join(map(str, x.shape)) if torch.is_tensor(x) else str(x)
        print(f"  {name:<28}{shape:<18}{note}")


def encode_text(c, text):
    ids = c.tokenizer(
        text, padding="max_length", max_length=c.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    ).input_ids
    return ids, c.text_encoder(ids.to(c.device))[0]


def my_ddim_step(sched, eps, t, x):
    """Stage 0's DDIM update (eta=0), using SD's own noise schedule. Written in the
    same operation order as DDIMScheduler.step so fp16 rounding matches too."""
    t = int(t)
    prev_t = t - sched.config.num_train_timesteps // sched.num_inference_steps
    abar = sched.alphas_cumprod[t]
    abar_prev = sched.alphas_cumprod[prev_t] if prev_t >= 0 else sched.final_alpha_cumprod
    x0 = (x - (1 - abar) ** 0.5 * eps) / abar**0.5  # the network's guess at the clean latent
    return abar_prev**0.5 * x0 + (1 - abar_prev) ** 0.5 * eps  # re-noise it to the previous level


@torch.no_grad()
def generate(c, prompt, negative="", steps=25, cfg=7.0, seed=0, scheduler="dpmpp_2m_karras",
             my_ddim=False, verbose=False, height=512, width=512,
             ip_embeds=None, controlnet=None, control_image=None, control_scale=1.0):
    """Returns (PIL image, stats dict).

    ip_embeds: IP-Adapter image embeddings as diffusers expects them, a list with one tensor
        per adapter shaped [uncond; cond] along dim 0 (stage 5: FaceID's ArcFace embedding).
    controlnet / control_image: a ControlNetModel and its conditioning image (stage 5: pose).
    Both default to off, and with both off this is exactly the stage 1 loop."""
    stats = {"unet_calls": 0}
    t_start = time.time()
    if my_ddim:
        scheduler = "ddim"
    sched = make_scheduler(scheduler, c)
    generator = torch.Generator("cpu").manual_seed(seed)
    do_cfg = cfg > 1.0  # at cfg=1 the formula collapses to the conditional prediction alone

    # 1. Text encoder — runs once.
    if verbose:
        print("\n[1] text encoder (CLIP ViT-L/14), runs once")
    ids, cond = encode_text(c, prompt)
    show(verbose, "prompt token ids", ids, f"{int((ids != c.tokenizer.pad_token_id).sum())} real tokens, padded to 77")
    show(verbose, "cond embedding", cond)
    if do_cfg:
        _, uncond = encode_text(c, negative)
        show(verbose, "uncond embedding", uncond, f"from negative prompt {negative[:30]!r}")
        emb = torch.cat([uncond, cond])  # uncond first, same order as diffusers
        show(verbose, "stacked for one batch", emb)
    else:
        emb = cond

    # 2. Starting noise, in latent space.
    if verbose:
        print("\n[2] initial latent")
    shape = (1, c.unet.config.in_channels, height // 8, width // 8)
    latents = randn_tensor(shape, generator=generator, device=torch.device(c.device), dtype=c.dtype)
    latents = latents * sched.init_noise_sigma  # 1.0 for DDIM-style, ~14.6 for Euler-style sigma schedulers
    show(verbose, "latents", latents, f"std={latents.float().std():.2f}  init_noise_sigma={float(sched.init_noise_sigma):.2f}")
    show(verbose, "pixels it stands for", f"1x3x{height}x{width}",
         f"{3 * height * width // latents[0].numel()}x compression")

    # 3. The loop — this is where all the time goes.
    sched.set_timesteps(steps, device=c.device)
    step_kwargs = {}
    step_params = inspect.signature(sched.step).parameters
    if "generator" in step_params:
        step_kwargs["generator"] = generator  # ancestral samplers inject fresh noise
    if "eta" in step_params:
        step_kwargs["eta"] = 0.0
    if verbose:
        ts = sched.timesteps.tolist()
        print(f"\n[3] denoising loop — {steps} steps x {2 if do_cfg else 1} UNet passes, scheduler={scheduler}")
        print(f"  timesteps: {[round(t) for t in ts[:5]]} ... {[round(t) for t in ts[-3:]]}")

    unet_kwargs = {}
    if ip_embeds is not None:
        # an extra, decoupled cross-attention path attends to these (section 02)
        unet_kwargs["added_cond_kwargs"] = {"image_embeds": ip_embeds if do_cfg else [e.chunk(2)[1] for e in ip_embeds]}
    if controlnet is not None:
        ctrl = torch.from_numpy(np.asarray(control_image.convert("RGB").resize((width, height)), dtype=np.float32) / 255)
        ctrl = ctrl.permute(2, 0, 1)[None].to(c.device, c.dtype)  # [0, 1], not [-1, 1]: ControlNet's convention
        ctrl = torch.cat([ctrl] * 2) if do_cfg else ctrl

    for i, t in enumerate(sched.timesteps):
        latent_in = torch.cat([latents] * 2) if do_cfg else latents
        latent_in = sched.scale_model_input(latent_in, t)  # identity for DDIM, divides by sqrt(sigma^2+1) for Euler/DPM
        if controlnet is not None:
            # the trainable encoder copy reads the pose map and adds residuals into the UNet's skips
            down, mid = controlnet(latent_in, t, encoder_hidden_states=emb, controlnet_cond=ctrl,
                                   conditioning_scale=control_scale, return_dict=False)
            unet_kwargs.update(down_block_additional_residuals=down, mid_block_additional_residual=mid)
            stats["controlnet_calls"] = stats.get("controlnet_calls", 0) + latent_in.shape[0]
        noise = c.unet(latent_in, t, encoder_hidden_states=emb, return_dict=False, **unet_kwargs)[0]
        stats["unet_calls"] += latent_in.shape[0]
        if do_cfg:
            noise_uncond, noise_cond = noise.chunk(2)
            noise = noise_uncond + cfg * (noise_cond - noise_uncond)
        if my_ddim:
            latents = my_ddim_step(sched, noise, t, latents)
        else:
            latents = sched.step(noise, t, latents, **step_kwargs, return_dict=False)[0]
        if verbose and i in (0, steps - 1):
            show(verbose, f"step {i}: unet input", latent_in, f"t={float(t):.0f}")
            show(verbose, f"step {i}: noise prediction", noise)
            show(verbose, f"step {i}: latents after", latents, f"std={latents.float().std():.2f}")

    # 4. VAE decoder — runs once, at full resolution.
    if verbose:
        print("\n[4] VAE decoder, runs once")
    scaled = latents / c.vae.config.scaling_factor  # 0.18215: latents were scaled to ~unit variance for training
    image = c.vae.decode(scaled, return_dict=False)[0]
    show(verbose, "decoded", image, "values roughly in [-1, 1]")
    image = (image / 2 + 0.5).clamp(0, 1)
    if torch.isnan(image).any():
        print("WARNING: NaNs in output — rerun with --fp32")
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    stats["seconds"] = time.time() - t_start
    return Image.fromarray(arr), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheduler", choices=SCHEDULERS, default="dpmpp_2m_karras")
    ap.add_argument("--my-ddim", action="store_true", help="use the stage 0 DDIM update instead of scheduler.step")
    ap.add_argument("--compare", action="store_true", help="also run the diffusers pipeline and diff the pixels")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default="runs/stage1")
    args = ap.parse_args()

    c = load_components(args.fp32)
    print(f"device={c.device} dtype={c.dtype}")
    for name, m in (("text encoder", c.text_encoder), ("unet", c.unet), ("vae", c.vae)):
        print(f"  {name:<14}{param_count(m) / 1e6:7.1f}M params")

    image, stats = generate(c, args.prompt, args.negative, args.steps, args.cfg, args.seed,
                            args.scheduler, my_ddim=args.my_ddim, verbose=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "manual_myddim.png" if args.my_ddim else "manual.png"
    image.save(out / name)
    print(f"\n{stats['unet_calls']} UNet forward passes, {stats['seconds']:.1f}s total -> {out / name}")

    if args.compare:
        from .baseline import build_pipeline

        pipe = build_pipeline(c, "ddim" if args.my_ddim else args.scheduler)
        ref = pipe(args.prompt, negative_prompt=args.negative, num_inference_steps=args.steps,
                   guidance_scale=args.cfg, generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
        ref.save(out / "pipeline_reference.png")
        diff = np.abs(np.asarray(image, dtype=np.int16) - np.asarray(ref, dtype=np.int16))
        print(f"vs diffusers pipeline: max pixel diff {diff.max()}, mean {diff.mean():.3f} (0-255 scale)")
        print("MATCH" if diff.max() <= 2 else "MISMATCH — something in the loop differs from the pipeline")


if __name__ == "__main__":
    main()
