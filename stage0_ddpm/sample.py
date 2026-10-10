"""Run one trained checkpoint through different samplers and step counts.

    uv run python -m stage0_ddpm.sample --ckpt runs/stage0/mnist/ckpt.pt

Writes next to the checkpoint:
  steps_comparison.png   same starting noise, DDPM-1000 vs DDIM at 100/50/20/10/5 steps
  trajectory.png         x_t and the network's guess at x_0, over one DDIM run
and prints wall-clock time per sampler.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from .common import load_checkpoint, pick_device
from .diffusion import Diffusion

RUNS = [("DDPM", 1000), ("DDIM", 100), ("DDIM", 50), ("DDIM", 20), ("DDIM", 10), ("DDIM", 5)]


def to_img(x, nrow):
    grid = make_grid((x + 1) / 2, nrow=nrow, padding=1, pad_value=1).clamp(0, 1)
    return grid.permute(1, 2, 0).numpy()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage0/mnist/ckpt.pt")
    ap.add_argument("--n", type=int, default=10, help="images per row")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    diffusion = Diffusion(device=device)
    ch, size = ckpt["model_cfg"]["in_ch"], ckpt["image_size"]
    shape = (args.n, ch, size, size)
    out = Path(args.ckpt).parent

    # 1. Step-count comparison. Every row starts from the identical noise tensor.
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(args.n * 0.9, len(RUNS) * 1.1))
    print(f"{'sampler':<8}{'steps':>6}{'seconds':>10}")
    for ax, (name, steps) in zip(axes, RUNS):
        gen = torch.Generator().manual_seed(args.seed)
        sync(device)
        t0 = time.time()
        if name == "DDPM":
            x, _ = diffusion.ddpm_sample(model, shape, generator=gen)
        else:
            x, _ = diffusion.ddim_sample(model, shape, steps=steps, generator=gen)
        sync(device)
        dt = time.time() - t0
        print(f"{name:<8}{steps:>6}{dt:>10.2f}")
        ax.imshow(to_img(x.cpu(), args.n), cmap="gray")
        ax.set_axis_off()
        ax.set_title(f"{name} · {steps} steps · {dt:.1f}s", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "steps_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Trajectory: what the image looks like, and what the network thinks the
    #    final image will be, at points along a 50-step DDIM run.
    gen = torch.Generator().manual_seed(args.seed)
    _, traj = diffusion.ddim_sample(model, (4, ch, size, size), steps=50, generator=gen, record_every=5)
    fig, axes = plt.subplots(2, len(traj), figsize=(len(traj) * 1.1, 2 * 4.6))
    for j, (t, xt, x0) in enumerate(traj):
        axes[0, j].imshow(to_img(xt.clamp(-1, 1), 1), cmap="gray")
        axes[1, j].imshow(to_img(x0, 1), cmap="gray")
        axes[0, j].set_title(f"t={t}", fontsize=8)
        for a in axes[:, j]:
            a.set_axis_off()
    axes[0, 0].text(-0.3, 0.5, "x_t", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.3, 0.5, "predicted x_0", transform=axes[1, 0].transAxes, rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=150)
    plt.close(fig)

    print(f"wrote {out / 'steps_comparison.png'} and {out / 'trajectory.png'}")


if __name__ == "__main__":
    main()
