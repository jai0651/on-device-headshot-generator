"""Stage 6: generate images entirely with Core ML models, each network on a chosen compute unit.

    .venv-coreml/bin/python -m stage6_coreml.coreml_generate --unet models/coreml/ours_ohwx/..._unet.mlpackage \\
        --units-unet CPU_AND_NE --run ours_ane --subject "ohwx man"

The same loop as stage 1's manual_loop.generate(): tokenize, text encoder once per prompt, 25
DPM++ 2M Karras steps with CFG (the UNet sees a batch of 2), VAE decoder once. Differences from
PyTorch are all Core ML's: fp16 everywhere, the ANE's (B, C, 1, S) text layout, and whichever
hardware each network lands on. Text encoder and VAE are Apple's conversion: the face LoRA only
changed the UNet, so they're identical to ours.

Writes outputs/stage6/<run>/samples/m/*.jpg + manifest.json (scored by
stage5_zeroshot.score_methods) and a per-network timing breakdown.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, LCMScheduler
from PIL import Image
from transformers import CLIPTokenizer

APPLE = "models/coreml/apple_sd15/split_einsum/packages"
UNITS = {"CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
         "ALL": ct.ComputeUnit.ALL, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY}
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default=glob.glob(f"{APPLE}/*_unet.mlpackage")[0])
    ap.add_argument("--units-text", default="CPU_AND_NE")
    ap.add_argument("--units-unet", default="CPU_AND_NE")
    ap.add_argument("--units-vae", default="CPU_AND_GPU")
    ap.add_argument("--controlnet", default=None, help="ControlNet .mlpackage; needs a UNet converted with --unet-support-controlnet")
    ap.add_argument("--pose", default="lora_data/control/passport_pose.png", help="control image for --controlnet")
    ap.add_argument("--units-controlnet", default="CPU_AND_NE")
    ap.add_argument("--scheduler", choices=["dpmpp", "lcm"], default="dpmpp",
                    help="lcm: for a UNet with the LCM-LoRA folded in (4-8 steps, CFG 1)")
    ap.add_argument("--subject", default="a man")
    ap.add_argument("--run", default="mine", help="output folder name under outputs/stage6/")
    ap.add_argument("--prompt", default=None, help='your own prompt, e.g. "photo of ohwx man in a navy suit"; '
                                                  'without it the fixed test prompt set is used')
    ap.add_argument("--negative", default=None, help="negative prompt (default: the project's standard one)")
    ap.add_argument("--n", type=int, default=1, help="with --prompt: how many images (seeds 0..n-1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from stage2_eval.prompts import NEGATIVE, expand   # pure python, no heavy deps

    t0 = time.time()
    tok = CLIPTokenizer.from_pretrained(SD, subfolder="tokenizer")
    text = ct.models.MLModel(glob.glob(f"{APPLE}/*_text_encoder.mlpackage")[0], compute_units=UNITS[args.units_text])
    unet = ct.models.MLModel(args.unet, compute_units=UNITS[args.units_unet])
    vae = ct.models.MLModel(glob.glob(f"{APPLE}/*_vae_decoder.mlpackage")[0], compute_units=UNITS[args.units_vae])
    cnet = ct.models.MLModel(args.controlnet, compute_units=UNITS[args.units_controlnet]) if args.controlnet else None
    if cnet is not None:
        # pose image in [0, 1], batch 2 for CFG, the layout ControlNet was trained on
        pose = np.asarray(Image.open(args.pose).convert("RGB").resize((512, 512)), np.float32) / 255
        cond = np.repeat(pose.transpose(2, 0, 1)[None], 2, axis=0).astype(np.float16)
        cnet_in = {i.name for i in cnet.get_spec().description.input}
        unet_in = {i.name for i in unet.get_spec().description.input}
        print(f"controlnet inputs {sorted(cnet_in)} · unet extra inputs: {len(unet_in - {'sample', 'timestep', 'encoder_hidden_states'})}")
    # batch 1 = a UNet converted with --unet-batch-one: no classifier-free guidance (LCM at CFG 1)
    B = int(unet.get_spec().description.input[0].type.multiArrayType.shape[0])
    if cnet is not None:
        cond = cond[:B]
    if B == 1 and args.cfg != 1.0:
        print(f"note: single-pass UNet, so CFG is off (ignoring --cfg {args.cfg})")
    load_s = time.time() - t0
    print(f"loaded (incl. on-device compile) in {load_s:.0f}s  text={args.units_text} unet={args.units_unet} vae={args.units_vae}")

    def encode(s):
        ids = tok(s, padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids.astype(np.float32)
        return text.predict({"input_ids": ids})["last_hidden_state"]          # (1, 77, 768)

    out = Path("outputs/stage6") / args.run / "samples" / "m"
    out.mkdir(parents=True, exist_ok=True)
    if args.prompt:
        items = [{"group": "custom", "name": "custom", "prompt": args.prompt, "seed": s, "file": f"custom_s{s}.png"}
                 for s in range(args.n)]
    else:
        items = [it for it in expand(args.subject) if it["seed"] in args.seeds][: args.limit or None]
    negative = args.negative if args.negative is not None else NEGATIVE
    timing = {"text_s": [], "unet_step_s": [], "vae_s": [], "image_s": []}
    for it in items:
        t_img = time.time()
        t1 = time.time()
        emb = encode(it["prompt"]) if B == 1 else np.concatenate([encode(negative), encode(it["prompt"])])
        emb = emb.transpose(0, 2, 1)[:, :, None, :].astype(np.float16)         # ANE layout (B, 768, 1, 77)
        timing["text_s"].append(time.time() - t1)

        sched = (LCMScheduler.from_pretrained(SD, subfolder="scheduler") if args.scheduler == "lcm" else
                 DPMSolverMultistepScheduler.from_pretrained(SD, subfolder="scheduler", use_karras_sigmas=True))
        sched.set_timesteps(args.steps)
        g = torch.Generator("cpu").manual_seed(it["seed"])
        lat = torch.randn((1, 4, 64, 64), generator=g) * sched.init_noise_sigma
        for t in sched.timesteps:
            x = sched.scale_model_input(torch.cat([lat] * B), t).numpy().astype(np.float16)
            feed = {"sample": x, "timestep": np.full((B,), float(t), np.float16), "encoder_hidden_states": emb}
            if cnet is not None:
                t1 = time.time()
                residuals = cnet.predict({**feed, "controlnet_cond": cond})   # additional_residual_0..12
                timing.setdefault("controlnet_step_s", []).append(time.time() - t1)
                feed.update({k: v.astype(np.float16) for k, v in residuals.items()})
            t1 = time.time()
            n = unet.predict(feed)["noise_pred"]
            timing["unet_step_s"].append(time.time() - t1)
            n = torch.from_numpy(n.astype(np.float32))
            if B == 2:
                nu, nc = n.chunk(2)
                n = nu + args.cfg * (nc - nu)
            step_kw = {"generator": g} if args.scheduler == "lcm" else {}   # LCM re-noises between steps
            lat = sched.step(n, t, lat, **step_kw).prev_sample
        t1 = time.time()
        img = vae.predict({"z": (lat / 0.18215).numpy().astype(np.float16)})["image"]
        timing["vae_s"].append(time.time() - t1)
        img = ((np.clip(img[0].astype(np.float32), -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)
        it2 = {**it, "file": it["file"].replace(".png", ".jpg"), "method": args.run}
        Image.fromarray(img).save(out / it2["file"], quality=95)
        timing["image_s"].append(time.time() - t_img)
        items[items.index(it)] = it2
        nanfrac = float(np.isnan(n.numpy()).mean())
        print(f"  {it2['file']:<24} {timing['image_s'][-1]:5.1f}s{'  NaN!' if nanfrac else ''}")
    (out / "manifest.json").write_text(json.dumps(items, indent=2))
    med = {k: round(float(np.median(v)), 3) for k, v in timing.items()}
    summary = {"units": {"text": args.units_text, "unet": args.units_unet, "vae": args.units_vae},
               "load_s": round(load_s, 1), "median": med, "unet_s_per_image": round(med["unet_step_s"] * args.steps, 2)}
    (out.parent.parent / "timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
