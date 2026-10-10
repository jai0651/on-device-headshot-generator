"""Score every LoRA checkpoint's samples and pick one by the curve. CPU: runs on the Mac.

    uv run python -m stage3_lora.score_checkpoints --run r16_lr1e-4 --refs photos/jai

For each outputs/stage3/<run>/samples/step_XXXX: identity (both recognizers, against the
reference photos), CLIP prompt score and compliance, per prompt group. Then:

  curve.png          identity rising vs flexibility falling, by step: section 03's
                     "likeness climbs, plateaus, then flexibility falls off a cliff"
  grid.jpg           rows = steps, columns = prompts, scores under each image
  checkpoints.csv    one row per step
  pick.json          the chosen step and why
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage2_eval.clip_score import ClipScorer
from stage2_eval.common import list_images
from stage2_eval.compliance import ComplianceChecker
from stage2_eval.contact_sheet import render
from stage2_eval.identity import IdentityScorer

ID_THRESHOLD = 0.30  # from stage 2's LFW validation: the gap between same- and different-person


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--root", default="outputs/stage3")
    args = ap.parse_args()
    run = Path(args.root) / args.run
    # training checkpoints are samples/step_XXXX; scale sweeps are samples/scale_X.XX
    steps = sorted(run.glob("samples/*_*"), key=lambda p: float(p.name.split("_")[1]))
    is_scale = steps[0].name.startswith("scale")

    ident = IdentityScorer(list_images(args.refs))
    clip, comp = ClipScorer(), ComplianceChecker()
    per_step, sheet = [], []
    for sd in steps:
        step = float(sd.name.split("_")[1]) if is_scale else int(sd.name.split("_")[1])
        manifest = json.loads((sd / "manifest.json").read_text())
        rows = []
        for it in manifest:
            p = sd / it["file"]
            r = {**it, **ident.score(p), "clip": clip.score(p, it["prompt"])}
            c = comp.check(p)
            r.update(comp_passed=c["passed"], comp_total=c["total"], comp_all_pass=c["all_pass"])
            rows.append(r)
            sheet.append({**r, "image": p, "label": f"{step} {it['name']}"})
        g = lambda group, key: nanmean([r[key] for r in rows if r["group"] == group])
        s = {"step": step,
             "id_identity": g("identity", "id_primary"), "id_identity_2nd": g("identity", "id_secondary"),
             "id_flex": g("flexibility", "id_primary"), "id_passport": g("compliance", "id_primary"),
             "clip_identity": g("identity", "clip"), "clip_flex": g("flexibility", "clip"),
             "id_pass_rate": float(np.mean([r["id_primary"] >= ID_THRESHOLD for r in rows if r["id_primary"] == r["id_primary"]] or [0])),
             "no_face": sum(r["id_primary"] != r["id_primary"] for r in rows),
             "comp_passport": g("compliance", "comp_passed")}
        per_step.append(s)
        print(f"  {'scale' if is_scale else 'step'} {step:>5}  id {s['id_identity']:.3f}/{s['id_identity_2nd']:.3f}  id-flex {s['id_flex']:.3f}  "
              f"clip-flex {s['clip_flex']:.1f}  pass {s['id_pass_rate']:.0%}  no-face {s['no_face']}")

    with open(run / "checkpoints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_step[0]))
        w.writeheader()
        w.writerows(per_step)

    # Pick: highest identity among checkpoints whose flexibility CLIP hasn't dropped more than
    # 10% below the untrained model's. Identity alone would pick the most overcooked one.
    if is_scale:
        # scale sweep: the lowest scale that keeps >= 95% of full-strength identity on headshots
        full = max(per_step, key=lambda s: s["step"])
        ok = [s for s in per_step if s["id_identity"] >= 0.95 * full["id_identity"]]
        pick = min(ok, key=lambda s: s["step"]) if ok else full  # NaN identity (no faces) -> nothing qualifies
        why = (f"lowest scale keeping >= 95% of scale-{full['step']} identity "
               f"({pick['id_identity']:.3f} vs {full['id_identity']:.3f}); flexibility CLIP {pick['clip_flex']:.1f}")
    else:
        base_flex = per_step[0]["clip_flex"]
        ok = [s for s in per_step if s["step"] > 0 and s["clip_flex"] >= 0.9 * base_flex]
        pick = max(ok or per_step[1:] or per_step, key=lambda s: s["id_identity"])
        why = (f"highest identity ({pick['id_identity']:.3f}) among checkpoints keeping flexibility CLIP >= 90% "
               f"of step 0 ({0.9 * base_flex:.1f})") if ok else "no checkpoint kept flexibility; picked max identity"
    (run / "pick.json").write_text(json.dumps({"step": pick["step"], "why": why, "scores": pick}, indent=2))

    x = [s["step"] for s in per_step]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [s["id_identity"] for s in per_step], "o-", c="#2e8b57", label="identity: identity prompts (primary)")
    ax.plot(x, [s["id_identity_2nd"] for s in per_step], "o--", c="#2e8b57", alpha=0.5, label="identity (secondary)")
    ax.plot(x, [s["id_flex"] for s in per_step], "s-", c="#7aa2d8", label="identity: flexibility prompts")
    ax.axhline(ID_THRESHOLD, c="k", ls=":", lw=1, label="same-person threshold")
    ax.set_xlabel("LoRA scale" if is_scale else "training step")
    ax.set_ylabel("identity cosine")
    ax2 = ax.twinx()
    ax2.plot(x, [s["clip_flex"] for s in per_step], "^-", c="#c0392b", label="CLIP: flexibility prompts")
    ax2.set_ylabel("CLIP score", color="#c0392b")
    ax.axvline(pick["step"], c="#f0b429", lw=2, alpha=0.6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.set_title(f"{args.run}: picked {'scale' if is_scale else 'step'} {pick['step']}")
    fig.tight_layout()
    fig.savefig(run / "curve.png", dpi=120)

    cols = len(json.loads((steps[0] / "manifest.json").read_text()))
    render(sheet, run / "grid.jpg", id_threshold=ID_THRESHOLD, cols=cols, tile=200)
    print(f"picked {'scale' if is_scale else 'step'} {pick['step']}: {why}")
    print(f"-> {run}/checkpoints.csv, curve.png, grid.jpg, pick.json")


if __name__ == "__main__":
    main()
