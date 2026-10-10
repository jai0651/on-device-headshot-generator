"""Draw the digit you ask for with the class-conditional DDPM.

    uv run python -m stage0_ddpm.draw 7                 # eight 7s
    uv run python -m stage0_ddpm.draw 3 --n 16 --guidance 5
    uv run python -m stage0_ddpm.draw all               # a 10×8 grid, 0..9

Classifier-free guidance, the same formula SD uses for prompts: noise = uncond + w·(cond − uncond).
w = 1 is the plain conditional model; higher w commits harder to the requested digit at some cost
to variety. If a trained classifier exists (stage0_ddpm.classifier), every generated digit is
checked: "asked for a 7, classifier sees a 7" is the accuracy printed.
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .common import load_checkpoint, pick_device
from .diffusion import Diffusion, guided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("digit", help="0-9, or 'all'")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/stage0/mnist_cond/ema_fp16.pt")
    ap.add_argument("--classifier", default="runs/stage0/mnist_classifier.pt")
    ap.add_argument("--scale", type=int, default=4, help="upscale the 28×28 output for viewing")
    args = ap.parse_args()

    device = pick_device()
    model, _ = load_checkpoint(args.ckpt, device)
    if not model.num_classes:
        raise SystemExit(f"{args.ckpt} is unconditional: train with --conditional to choose digits")
    diff = Diffusion(device=device)
    digits = list(range(10)) if args.digit == "all" else [int(args.digit)]
    labels = torch.tensor(digits, device=device).repeat_interleave(args.n)
    g = torch.Generator().manual_seed(args.seed)
    x, _ = diff.ddim_sample(guided(model, labels, args.guidance), (len(labels), 1, 28, 28), steps=args.steps, generator=g)

    out = Path("outputs/stage0/draw")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"digit_{args.digit}_w{args.guidance:g}_s{args.seed}.png"
    img = torch.nn.functional.interpolate((x + 1) / 2, scale_factor=args.scale, mode="nearest")
    save_image(img, path, nrow=args.n, padding=2, pad_value=0.4)

    if Path(args.classifier).exists():
        from .classifier import Classifier
        clf = Classifier().to(device).eval()
        clf.load_state_dict(torch.load(args.classifier, map_location=device))
        with torch.no_grad():
            pred = clf(x).argmax(1)
        acc = (pred == labels).float().mean().item()
        print(f"asked for {args.digit}: classifier agrees on {acc:.0%} of {len(labels)}  (predicted: {pred.tolist()[:16]}{'…' if len(labels) > 16 else ''})")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
