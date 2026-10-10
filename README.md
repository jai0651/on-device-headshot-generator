# On-device headshot generator

A selfie in, a professional headshot out, on Apple hardware, with nothing uploaded. Most AI headshot
apps upload 10–20 photos and train a model per customer on a server. This project measures how much of
that can happen on the device, with no upload and no per-user training.

It is built in stages on Stable Diffusion 1.5, starting from a diffusion model written from scratch.
Each stage produces something measurable, and every number below comes from a run that was actually
executed.

## Results

| | |
|---|---|
| Identity from **one selfie** (IP-Adapter FaceID, no training) vs a LoRA trained on 17 photos | **0.67** vs 0.49 |
| UNet ops on the Neural Engine after Core ML conversion | **3,057 / 3,057** |
| Per image on an M4, Neural Engine vs GPU (FP16, 25 steps) | **9.9 s** vs 18.5 s |
| Per image on an M4 Neural Engine, 8-bit UNet + LCM, 4 steps | **1.34 s** (identity 0.60) |
| 8-bit palettized UNet vs FP16 | half the size (862 MB), same identity |

Identity is cosine similarity from an independent face recognizer; 0.30 is the same-person line.

## Stages

| Stage | What it does |
|---|---|
| [`stage0_ddpm`](stage0_ddpm) | DDPM from scratch: 6.5M-param UNet, DDPM and DDIM samplers, then class-conditional with classifier-free guidance |
| [`stage1_sd_pipeline`](stage1_sd_pipeline) | SD 1.5's sampling loop written by hand; matches diffusers pixel for pixel (max diff 0) |
| [`stage2_eval`](stage2_eval) | Eval harness built before any training: identity from two face recognizers (AUC 0.99), CLIP prompt score, 13 passport-photo checks |
| [`stage3_lora`](stage3_lora) | Face LoRA on 17 audited photos, checkpoints scored by the harness |
| [`stage5_zeroshot`](stage5_zeroshot) | IP-Adapter FaceID vs the LoRA, pose ControlNet, and an inpainting edit path |
| [`stage6_coreml`](stage6_coreml) | LoRA folded into the UNet, converted to Core ML, per-op placement on the Neural Engine |
| [`stage7_quantize`](stage7_quantize) | 8 / 6 / 4-bit palettization: how small before it stops being you |
| [`stage8_lcm`](stage8_lcm) | LCM-LoRA: 4 steps instead of 25, folded next to the face LoRA |

Each stage folder has its own README with commands and full tables. `docs/` has a longer write-up
(open `docs/index.html`).

## Running it

```bash
uv sync                                  # PyTorch stages
uv run python -m stage0_ddpm.train       # e.g. stage 0
```

Core ML stages use a separate environment (`.venv-coreml/`, Python 3.11, coremltools 9.0); see
`stage6_coreml/README.md`. GPU-heavy steps ran on rented RunPod GPUs.

Photos, datasets, trained weights and generated images are not in the repo: several contain a real
person's face, and the models add up to about 15 GB.
