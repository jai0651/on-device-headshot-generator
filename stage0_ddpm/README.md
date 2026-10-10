# Stage 0 — a diffusion model from scratch

Section 01 of the build notes, as ~300 lines of PyTorch. No diffusers, no pretrained weights.

| File | What's in it |
|---|---|
| `model.py` | Small UNet (6.5M params): ResBlocks + timestep embedding, attention at 7×7, skip connections |
| `diffusion.py` | Linear beta schedule, closed-form forward noising, ε-prediction MSE loss, DDPM and DDIM samplers |
| `train.py` | MNIST / CIFAR-10 training loop with EMA weights; saves a sample grid every epoch |
| `sample.py` | One checkpoint through DDPM-1000 and DDIM-100/50/20/10/5, plus a denoising trajectory |

## Run it

```bash
uv run python -m stage0_ddpm.train                  # MNIST, 20 epochs
uv run python -m stage0_ddpm.sample                 # sampler comparison on that checkpoint
uv run python -m stage0_ddpm.train --dataset cifar10 --epochs 100   # harder; expect blurry objects
```

Outputs land in `runs/stage0/<dataset>/`.

## What to look for

- **`samples_epoch*.png`** — every grid starts from the same noise, so you watch the *same* 64 samples
  sharpen epoch by epoch. Loss flattens early; the images keep improving after it does.
- **`steps_comparison.png`** — DDIM rows share structure: same noise, deterministic sampler, so
  DDIM-100 and DDIM-20 produce mostly the *same digits*. DDPM draws fresh noise every step, so its row
  is different digits entirely. Watch where DDIM falls apart as steps drop — that's the gap
  DPM-Solver and LCM close later.
- **`trajectory.png`** — the "predicted x_0" row is the key picture. At high t the network only commits
  to a blob of roughly the right shape; the digit's identity settles in the middle; the last steps just
  clean edges. That's "coarse decisions at high noise, eyelashes at low noise" from section 01.

## Break it on purpose

Each takes one edit and one short run (`--epochs 3` is enough to see it):

1. **Remove the timestep.** In `ResBlock.forward`, drop the `h + self.temb(...)` line. The network no
   longer knows how noisy its input is. Samples turn to mush.
2. **Predict x₀ instead of ε.** Change the loss target to `x0` and adapt the samplers. Compare quality
   at the same epoch count — this is the "why predict noise" argument.
3. **Cosine schedule.** Replace the linear betas with the cosine schedule from Nichol & Dhariwal
   (improved DDPM). Look at how few-step DDIM behaves afterwards.
4. **DDIM with `eta=1`.** Stochastic again — confirm it behaves like DDPM with fewer steps.

## Done when

- [ ] MNIST samples are clean, recognisable digits
- [ ] Can explain every line of `ddpm_sample` and `ddim_sample` from the equations
- [ ] Ran at least two of the break-it exercises and can say *why* each failure looks the way it does
