"""Score every stage 5 method with every instrument and put them in one table. CPU: Mac.

    uv run python -m stage5_zeroshot.score_methods --refs photos/jai

Honesty rule from stage 2: FaceID conditions on buffalo_l (our PRIMARY recognizer), so its
primary identity score is circular. The comparison column is the SECONDARY recognizer
(antelopev2: different backbone, different training data) for every method.

Writes outputs/stage5/comparison.csv, comparison.md, and per-method contact sheets.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stage2_eval.clip_score import ClipScorer
from stage2_eval.common import list_images
from stage2_eval.compliance import ComplianceChecker
from stage2_eval.contact_sheet import render
from stage2_eval.identity import IdentityScorer

ID_THRESHOLD = 0.30


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/stage5")
    ap.add_argument("--refs", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    ident = IdentityScorer(list_images(args.refs))
    clip, comp = ClipScorer(), ComplianceChecker()

    table = []
    for mdir in sorted(p for p in root.iterdir() if (p / "samples" / "m" / "manifest.json").exists()):
        sd = mdir / "samples" / "m"
        rows = []
        for it in json.loads((sd / "manifest.json").read_text()):
            p = sd / it["file"]
            c = comp.check(p)
            rows.append({**it, **ident.score(p), "clip": clip.score(p, it["prompt"]),
                         "comp_passed": c["passed"], "comp_total": c["total"], "comp_all_pass": c["all_pass"]})
        g = lambda group, key: mean([r[key] for r in rows if group is None or r["group"] == group])
        pose = mdir.name.endswith("_pose")
        t = {
            "method": mdir.name, "n": len(rows),
            "id2_headshot": g("identity", "id_secondary") if not pose else mean([r["id_secondary"] for r in rows if r["name"] == "corporate_suit"]),
            "id2_passport": g("compliance", "id_secondary"),
            "id2_flex": g("flexibility", "id_secondary"),
            "id1_headshot": g("identity", "id_primary") if not pose else mean([r["id_primary"] for r in rows if r["name"] == "corporate_suit"]),
            "id2_pass_rate": mean([float(r["id_secondary"] >= ID_THRESHOLD) for r in rows if r["id_secondary"] == r["id_secondary"]]),
            "clip_flex": g("flexibility", "clip"), "clip_all": g(None, "clip"),
            "comp_passport": g("compliance", "comp_passed"),
            "passport_full_pass": sum(r["comp_all_pass"] for r in rows if r["group"] == "compliance"),
            "no_face": sum(r["id_secondary"] != r["id_secondary"] for r in rows),
        }
        table.append(t)
        render([{**r, "image": sd / r["file"], "label": r["file"][:-4],
                 "id_primary": r["id_secondary"], "id_secondary": r["id_primary"]} for r in rows],
               mdir / "contact_sheet.jpg", id_threshold=ID_THRESHOLD, cols=6, tile=200)
        print(f"  {t['method']:<22} id2 headshot {t['id2_headshot']:.3f}  passport {t['id2_passport']:.3f}  "
              f"flex {t['id2_flex']:.3f}  clip-flex {t['clip_flex']:.1f}  passport comp {t['comp_passport']:.1f}/13")

    with open(root / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    fmt = lambda v: "—" if v != v else (f"{v:.3f}" if isinstance(v, float) else str(v))
    cols = ["method", "id2_headshot", "id2_passport", "id2_flex", "id1_headshot", "id2_pass_rate", "clip_flex",
            "comp_passport", "passport_full_pass"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(t[c]) for c in cols) + " |" for t in table]
    (root / "comparison.md").write_text(
        "id2 = secondary recognizer (antelopev2), the honest identity number for every method; "
        "id1 = primary (buffalo_l), circular for FaceID.\n\n" + "\n".join(md) + "\n")
    print(f"-> {root}/comparison.csv, comparison.md, */contact_sheet.jpg")


if __name__ == "__main__":
    main()
