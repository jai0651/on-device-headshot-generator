"""Step 1c: sweep the knobs from section 01 and look at what they do.

    uv run python -m stage1_sd_pipeline.sweeps --sweep cfg       # guidance 1 -> 15, watch skin go plastic
    uv run python -m stage1_sd_pipeline.sweeps --sweep sampler   # 6 samplers x {10, 20, 30} steps
    uv run python -m stage1_sd_pipeline.sweeps --sweep all

Same seed everywhere, so every difference in a grid comes from the one knob being swept.
Grids and a timings CSV go to runs/stage1/.
"""

import argparse
import csv
from pathlib import Path

from .common import DEFAULT_NEGATIVE, DEFAULT_PROMPT, SCHEDULERS, labeled_grid, load_components
from .manual_loop import generate

CFG_VALUES = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
SEEDS = [0, 1]
SAMPLER_STEPS = [10, 20, 30]


def sweep_cfg(c, args, out, log):
    images = []
    for seed in SEEDS:
        for cfg in CFG_VALUES:
            img, s = generate(c, args.prompt, args.negative, steps=25, cfg=cfg, seed=seed, scheduler="dpmpp_2m_karras")
            images.append(img)
            log.writerow(["cfg", "dpmpp_2m_karras", 25, cfg, seed, s["unet_calls"], f"{s['seconds']:.2f}"])
            print(f"  cfg={cfg:<5} seed={seed}  {s['unet_calls']} unet calls  {s['seconds']:.1f}s")
    labeled_grid(images, len(SEEDS), len(CFG_VALUES), out / "sweep_cfg.png",
                 row_labels=[f"seed {s}" for s in SEEDS], col_labels=[f"cfg {v}" for v in CFG_VALUES])


def sweep_sampler(c, args, out, log):
    names = list(SCHEDULERS)
    images = []
    for steps in SAMPLER_STEPS:
        for name in names:
            img, s = generate(c, args.prompt, args.negative, steps=steps, cfg=7.0, seed=0, scheduler=name)
            images.append(img)
            log.writerow(["sampler", name, steps, 7.0, 0, s["unet_calls"], f"{s['seconds']:.2f}"])
            print(f"  {name:<16} {steps:>2} steps  {s['seconds']:.1f}s")
    labeled_grid(images, len(SAMPLER_STEPS), len(names), out / "sweep_sampler.png",
                 row_labels=[f"{n} steps" for n in SAMPLER_STEPS], col_labels=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["cfg", "sampler", "all"], default="all")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default="runs/stage1")
    args = ap.parse_args()

    c = load_components(args.fp32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep_timings.csv", "a", newline="") as f:
        log = csv.writer(f)
        if f.tell() == 0:
            log.writerow(["sweep", "scheduler", "steps", "cfg", "seed", "unet_calls", "seconds"])
        if args.sweep in ("cfg", "all"):
            print("CFG sweep")
            sweep_cfg(c, args, out, log)
        if args.sweep in ("sampler", "all"):
            print("sampler sweep")
            sweep_sampler(c, args, out, log)
    print(f"grids in {out}/")


if __name__ == "__main__":
    main()
