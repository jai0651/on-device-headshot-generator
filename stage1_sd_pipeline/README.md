# Stage 1 — drive SD 1.5, then take the pipeline apart

Section 01 of the build notes, on a real model. The goal is to be able to debug the pipeline, not
just call it: every tensor between prompt and pixels printed, and a proof that the hand-written loop
is the same thing the library runs.

| File | What's in it |
|---|---|
| `common.py` | Loads SD 1.5 as its three separate networks (text encoder, UNet, VAE) + the six samplers |
| `baseline.py` | The black box: one `StableDiffusionPipeline` call |
| `manual_loop.py` | The same thing as ~60 lines of explicit loop, printing every shape; `--compare` diffs it against diffusers |
| `sweeps.py` | CFG 1→15 grid, and 6 samplers × {10, 20, 30} steps |

Weights: `stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16 only (~2 GB), cached by Hugging Face.

## Run it

```bash
uv run python -m stage1_sd_pipeline.baseline                    # one image, the easy way
uv run python -m stage1_sd_pipeline.manual_loop --compare       # explicit loop + pixel diff vs diffusers
uv run python -m stage1_sd_pipeline.manual_loop --my-ddim --compare   # stage 0's DDIM math inside SD
uv run python -m stage1_sd_pipeline.sweeps                      # both grids (~5 min on an M4)
```

Add `--fp32` to any of them if fp16 produces black or NaN images.

## What to look for

- **The shape printout** is section 01's table, live: 77×768 text embedding, a 1×4×64×64 latent standing
  in for 512×512×3 pixels (48× fewer numbers), a 2×4×64×64 UNet input because CFG batches the
  conditional and unconditional passes, and one VAE decode at the end.
- **`--compare` says MATCH.** If it does, there is nothing left in the pipeline that isn't in
  `generate()`. That's the whole point of this stage.
- **`--my-ddim` also matches.** SD's DDIM step is exactly the update written in stage 0, just on
  latents and with SD's `scaled_linear` beta schedule. Same equation, bigger network.
- **`sweep_cfg.png`** — cfg 1 ignores most of the prompt; 5–7.5 is the usable band for faces; by 10–15
  skin goes waxy and contrast blows out. That's the "CFG 4–7 for photoreal faces" rule in section 03.
  Also note in `sweep_timings.csv` that cfg 1 costs half the UNet calls of every other column.
- **`sweep_sampler.png`** — DPM++ 2M Karras and UniPC are already clean at 10–20 steps; DDIM and PNDM
  need more; `euler_a` keeps changing the image as steps increase because it injects noise every step.

## Done when

- [ ] `manual_loop --compare` and `--my-ddim --compare` both print MATCH
- [ ] Can point at the line in `generate()` responsible for each of: the 2× CFG cost, the 48× latent
      compression, `init_noise_sigma` (1.0 for DPM++, ~14.6 for Euler), `scale_model_input`, the 0.18215 VAE scaling factor
- [ ] Have a baseline seconds-per-image number on this Mac for 25 steps — the figure every later stage
      gets compared against
