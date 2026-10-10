"""Turn a rough scribble into a clean handwritten digit with the stage 0 DDPM (SDEdit).

    uv run python -m stage0_ddpm.sdedit                         # draws a crude "7" itself
    uv run python -m stage0_ddpm.sdedit --image my_scribble.png # your drawing (white on black, any size)

The model is unconditional: it can't be told "draw a 7". But diffusion can start from anywhere,
not just pure noise. Take the scribble, jump it forward to timestep t with the closed-form
forward process (x_t = √ᾱ·x0 + √(1−ᾱ)·ε), then run the normal reverse process from t down to 0.
Small t keeps your strokes and only cleans them; large t keeps just the rough layout and lets
the model redraw. At t = 999 it's ordinary sampling and the scribble is forgotten entirely.
Same idea as stage 5c's img2img "harmonize" pass, on a 28×28 digit.

Writes runs/stage0/sdedit.png: scribble, then one row per t, one column per seed.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

from .common import load_checkpoint, pick_device
from .diffusion import Diffusion


def crude_seven(size=28):
    img = np.zeros((size, size), np.uint8)
    cv2.line(img, (6, 6), (21, 6), 255, 3)     # top bar
    cv2.line(img, (21, 6), (11, 23), 255, 3)   # diagonal
    return img


@torch.no_grad()
def sdedit(model, diff, x0, t_start, generator):
    """Forward-noise x0 to t_start in one jump, then reverse DDPM steps back to 0."""
    t = torch.full((x0.shape[0],), t_start, device=x0.device, dtype=torch.long)
    x = diff.q_sample(x0, t, diff._randn(x0.shape, generator))
    for i in reversed(range(t_start + 1)):
        tt = torch.full((x0.shape[0],), i, device=x0.device, dtype=torch.long)
        eps = model(x, tt)
        beta, alpha, abar = (diff._at(a, tt) for a in (diff.betas, diff.alphas, diff.abar))
        mean = (x - beta / (1 - abar).sqrt() * eps) / alpha.sqrt()
        x = mean + diff._at(diff.posterior_var, tt).sqrt() * diff._randn(x0.shape, generator) if i > 0 else mean
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage0/mnist/ckpt.pt")
    ap.add_argument("--image", default=None, help="white-on-black drawing; resized to 28x28")
    ap.add_argument("--ts", type=int, nargs="+", default=[200, 400, 600, 800])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", default="runs/stage0/sdedit.png")
    args = ap.parse_args()

    device = pick_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    diff = Diffusion(device=device)
    raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE) if args.image else crude_seven()
    raw = cv2.resize(raw, (28, 28), interpolation=cv2.INTER_AREA)
    x0 = torch.from_numpy(raw).float().div(127.5).sub(1)[None, None].repeat(args.seeds, 1, 1, 1).to(device)
    print(f"checkpoint epoch {ckpt['epoch']} · scribble {'from ' + args.image if args.image else '= crude 7'}")

    rows = [x0.cpu()]
    for t in args.ts:
        g = torch.Generator().manual_seed(0)
        rows.append(sdedit(model, diff, x0, t, g).cpu())
        print(f"  t = {t}: done")
    grid = torch.cat(rows)  # (1 + len(ts)) rows of `seeds` images
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image((grid + 1) / 2, args.out, nrow=args.seeds, padding=2, pad_value=0.5)
    print(f"-> {args.out}  (row 1: scribble; rows 2..: t = {args.ts}, left to right: seeds)")


if __name__ == "__main__":
    main()
