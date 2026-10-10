"""Train a DDPM on MNIST (default) or CIFAR-10.

    uv run python -m stage0_ddpm.train                       # MNIST, ~20 epochs
    uv run python -m stage0_ddpm.train --max-steps 200       # smoke test
    uv run python -m stage0_ddpm.train --dataset cifar10 --epochs 100
    python3 -m stage0_ddpm.train --out outputs/stage0 --resume   # on a RunPod GPU (see README)

Writes to runs/stage0/<dataset>/: ckpt.pt, losses.csv, and a sample grid per epoch.
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from .common import pick_device
from .diffusion import Diffusion
from .model import UNet

DATASETS = {
    # name: (torchvision class, channels, image size)
    "mnist": (datasets.MNIST, 1, 28),
    "cifar10": (datasets.CIFAR10, 3, 32),
}


def get_loader(name, root, batch_size, workers):
    cls, ch, _ = DATASETS[name]
    tf = [transforms.ToTensor(), transforms.Normalize([0.5] * ch, [0.5] * ch)]  # pixels -> [-1, 1]
    if name == "cifar10":
        tf.insert(0, transforms.RandomHorizontalFlip())
    ds = cls(root, train=True, download=True, transform=transforms.Compose(tf))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def update_ema(ema, model, step, max_decay=0.9995):
    # Ramp the decay up from ~0 so the EMA isn't dominated by random init early on.
    decay = min(max_decay, (1 + step) / (10 + step))
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early after this many steps (0 = off)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64, help="UNet base channel width")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/stage0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if it exists (spot pods)")
    ap.add_argument("--conditional", action="store_true", help="learn to draw a requested digit (label + CFG)")
    ap.add_argument("--p-uncond", type=float, default=0.1, help="label dropout, so the model also learns 'any digit'")
    args = ap.parse_args()

    device = pick_device()
    # bf16 autocast on CUDA roughly doubles throughput; MPS stays in fp32.
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    _, ch, size = DATASETS[args.dataset]

    model_cfg = dict(in_ch=ch, base=args.base, num_classes=10 if args.conditional else 0)
    model = UNet(**model_cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))  # short warmup
    loader = get_loader(args.dataset, args.data, args.batch_size, args.workers)

    step, start_epoch = 0, 0
    if args.resume and (out / "ckpt.pt").exists():
        ckpt = torch.load(out / "ckpt.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, start_epoch = ckpt["step"], ckpt["epoch"] + 1
        print(f"resumed from epoch {ckpt['epoch']} (step {step})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params / 1e6:.2f}M  steps/epoch={len(loader)}")

    log = open(out / "losses.csv", "a" if start_epoch else "w", newline="")
    writer = csv.writer(log)
    if not start_epoch:
        writer.writerow(["step", "epoch", "loss"])
    sample_noise = torch.Generator().manual_seed(0)  # same starting noise every epoch

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        bar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for x, y in bar:
            with amp:
                if args.conditional:
                    y = y.to(device)
                    y = torch.where(torch.rand(y.shape, device=device) < args.p_uncond, torch.full_like(y, 10), y)
                    loss = diffusion.loss(model, x.to(device, non_blocking=True), y)
                else:
                    loss = diffusion.loss(model, x.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            update_ema(ema, model, step)
            step += 1
            if step % 50 == 0:
                writer.writerow([step, epoch, f"{loss.item():.5f}"])
                bar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break

        torch.save(
            {"model": model.state_dict(), "ema": ema.state_dict(), "model_cfg": model_cfg,
             "opt": opt.state_dict(), "sched": sched.state_dict(),
             "image_size": size, "dataset": args.dataset, "step": step, "epoch": epoch},
            out / "ckpt.pt",
        )
        torch.save({"ema": {k: v.half() for k, v in ema.state_dict().items()}, "model_cfg": model_cfg,
                    "image_size": size, "dataset": args.dataset, "step": step, "epoch": epoch},
                   out / "ema_fp16.pt")  # ~13 MB: all that generation needs
        sample_noise.manual_seed(0)
        if args.conditional:  # 10 rows: digits 0..9, guidance 3
            from .diffusion import guided
            labels = torch.arange(10, device=device).repeat_interleave(8)
            imgs, _ = diffusion.ddim_sample(guided(ema, labels, 3.0), (80, ch, size, size), steps=50, generator=sample_noise)
        else:
            imgs, _ = diffusion.ddim_sample(ema, (64, ch, size, size), steps=50, generator=sample_noise)
        save_image((imgs + 1) / 2, out / f"samples_epoch{epoch:03d}.png", nrow=8)
        log.flush()
        print(f"epoch {epoch}  step {step}  loss {loss.item():.4f}  {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break

    log.close()
    print(f"done — checkpoint and samples in {out}/")


if __name__ == "__main__":
    main()
