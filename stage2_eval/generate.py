"""Generate the fixed prompt set with plain SD 1.5, with no identity conditioning at all.

    python -m stage2_eval.generate --out outputs_raw/stage2/baseline

This is the floor every later method is measured against: whatever identity score a
stranger gets, and how compliant SD 1.5 is on its own before ControlNet. Writes PNGs
plus manifest.json (prompt, group, seed per file) for stage2_eval.score.
"""

import argparse
import json
from pathlib import Path

from stage1_sd_pipeline.common import load_components
from stage1_sd_pipeline.manual_loop import generate

from .prompts import NEGATIVE, expand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", default="a person")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke tests)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = expand(args.subject)[: args.limit or None]
    c = load_components()
    for it in items:
        img, stats = generate(c, it["prompt"], NEGATIVE, steps=args.steps, cfg=args.cfg, seed=it["seed"])
        img.save(out / it["file"])
        it.update(steps=args.steps, cfg=args.cfg, seconds=round(stats["seconds"], 2))
        print(f"  {it['file']:<24} {stats['seconds']:.1f}s")
    (out / "manifest.json").write_text(json.dumps(items, indent=2))
    print(f"-> {len(items)} images + manifest.json in {out}")


if __name__ == "__main__":
    main()
