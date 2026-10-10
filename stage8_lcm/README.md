# Stage 8 — LCM: 4 steps instead of 25

Section 06's biggest lever. The LCM-LoRA (`latent-consistency/lcm-lora-sdv1-5`) distils SD 1.5 so
each step jumps toward the end of the trajectory: 4–8 steps, and at CFG 1 the UNet runs once per
step instead of twice. It's a LoRA, so it folds into the UNet next to the face LoRA.

| File | Runs on | What it does |
|---|---|---|
| `lcm_eval.py` | Mac GPU (PyTorch) | 25-step baseline vs LCM 8/6/4 steps (± light CFG), same seeds, prompts, pose |
| `sustained.sh` | Mac Neural Engine | 50 images back to back, per-image time + `pmset` thermal state |

## 8a — does few-step LCM keep the face? (PyTorch, Mac GPU)

| Config | UNet calls | Per image | Identity | Face-to-face similarity | CLIP | Compliance |
|---|---|---|---|---|---|---|
| baseline (25 steps, CFG 6) | 50 | 66.5 s | 0.45 | 0.43 | 21.6 | 11.4 |
| LCM 8 | 8 | 11.6 s | **0.65** | 0.73 | 21.5 | 11.8 |
| LCM 6 | 6 | 9.1 s | 0.62 | 0.75 | 21.6 | 11.6 |
| **LCM 4** | **4** | **5.7 s** | **0.63** | 0.76 | 21.7 | 11.6 |
| LCM 4, CFG 1.5 | 8 | 11.0 s | 0.60 | 0.65 | 21.8 | 11.6 |

Open question 4, answered the opposite way to the guess: **distillation raised identity** (0.45 → 0.63)
with prompt adherence unchanged. Generated faces became as consistent with each other (0.73–0.76) as
your real photos are (0.73). The 25-step CFG-6 baseline was the inconsistent one (0.43): strong
guidance pulls toward a generic "corporate man" and dilutes the LoRA. The cost: less variety of
expression and angle, and a smoother, more airbrushed look.

## 8b — on the Neural Engine

Face LoRA + LCM-LoRA folded together (exact: 1.8e-5), converted with `--unet-batch-one` (no CFG)
and a matching batch-1 ControlNet (`stage6_coreml/convert.py` fix 4), UNet palettized to 8-bit.

| Phone path, M4 Neural Engine | UNet | Per image | Identity | CLIP | Compliance | Head cut |
|---|---|---|---|---|---|---|
| stage 7: 8-bit, 25 steps, CFG 6 | 862 MB | 16.9 s | 0.49 | 21.7 | 11.4 | 0/8 |
| **stage 8: 8-bit, LCM 4, CFG 1** | 862 MB | **1.34 s** | **0.60** | 21.8 | **11.8** | 0/8 |

Per image: 4 × (UNet 0.185 s + ControlNet 0.074 s) + VAE 0.225 s + text 0.004 s.

## 8c — sustained load (50 in a row)

Median 1.30 s for the first 10, 1.50 s for the last 10 (+15%), worst 2.0 s, 70 s total;
macOS recorded no thermal warning. Mild drift, nothing like the GPU's 17 → 22 s in stage 6.
A phone has far less cooling than a Mac: the iPhone number is still the one to measure.

`./generate_headshot.sh "<prompt with ohwx man>" N` now runs this configuration by default.
