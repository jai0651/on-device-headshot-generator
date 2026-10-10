"""Score a folder of generated images with every instrument.

    python -m stage2_eval.score --images outputs_raw/stage2/baseline --out outputs/stage2/baseline
    python -m stage2_eval.score --images DIR --refs photos/me --out DIR   # adds identity

--images must contain manifest.json (written by stage2_eval.generate): one entry per
image with its prompt, group and seed. Writes scores.csv, summary.json and a contact
sheet to --out.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .common import list_images
from .compliance import ComplianceChecker
from .contact_sheet import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--refs", help="folder of reference photos of the target person")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--calibration", default="outputs/stage2/identity_calibration.json",
                    help="threshold file from validate_identity, used to colour the contact sheet")
    args = ap.parse_args()

    images = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((images / "manifest.json").read_text())

    comp = ComplianceChecker()
    clip = None if args.no_clip else __import__("stage2_eval.clip_score", fromlist=["ClipScorer"]).ClipScorer()
    ident = None
    if args.refs:
        from .identity import IdentityScorer
        ident = IdentityScorer(list_images(args.refs))

    rows = []
    for item in manifest:
        path = images / item["file"]
        row = {**item}
        c = comp.check(path)
        row.update(comp_passed=c["passed"], comp_total=c["total"], comp_all_pass=c["all_pass"])
        row.update({f"c_{k}": v for k, (v, _) in c["checks"].items()})
        row.update({f"ok_{k}": ok for k, (_, ok) in c["checks"].items()})
        if clip:
            row["clip"] = clip.score(path, item["prompt"])
        if ident:
            row.update(ident.score(path))
        rows.append(row)
        print(f"  {item['file']:<24} comp {c['passed']}/{c['total']}"
              + (f"  clip {row['clip']:.1f}" if clip else "")
              + (f"  id {row['id_primary']:.2f}/{row['id_secondary']:.2f}" if ident else ""))

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Per-group means: identity groups, flexibility groups and compliance behave differently
    # and averaging across them hides exactly the trade-off being measured.
    summary = defaultdict(dict)
    metrics = [m for m in ("id_primary", "id_secondary", "clip", "comp_passed") if m in keys]
    for group in sorted({r["group"] for r in rows}) + ["all"]:
        sel = [r for r in rows if group == "all" or r["group"] == group]
        for m in metrics:
            vals = [r[m] for r in sel if r[m] == r[m]]
            summary[group][m] = sum(vals) / len(vals) if vals else math.nan
        summary[group]["comp_all_pass_rate"] = sum(r["comp_all_pass"] for r in sel) / len(sel)
        summary[group]["n"] = len(sel)
    per_check = {k[3:]: sum(r[k] for r in rows) / len(rows) for k in keys if k.startswith("ok_")}
    (out / "summary.json").write_text(json.dumps({"groups": summary, "check_pass_rate": per_check}, indent=2))

    threshold = None
    cal = Path(args.calibration)
    if ident and cal.exists():
        threshold = json.loads(cal.read_text())["primary"]["threshold_eer"]
    render([{**r, "image": images / r["file"], "label": r["file"][:-4]} for r in rows],
           out / "contact_sheet.jpg", id_threshold=threshold)
    print(json.dumps(summary["all"], indent=2))
    print(f"-> {out}/scores.csv, summary.json, contact_sheet.jpg")


if __name__ == "__main__":
    main()
