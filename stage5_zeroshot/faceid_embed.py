"""Face-recognition embeddings for IP-Adapter-FaceID. CPU: runs on the Mac, so the GPU pod
never needs insightface and the photos themselves never have to leave for this method.

    uv run python -m stage5_zeroshot.faceid_embed photos/jai --out lora_data/faceid/jai.npz

IP-Adapter-FaceID conditions on insightface buffalo_l's ArcFace embedding, which is stage 2's
PRIMARY recognizer. Scores from the primary recognizer are therefore circular for this
method; the secondary (antelopev2, different backbone and training data) is the honest one.

Saves every photo's 512-d unit embedding plus which photo is the most frontal and sharpest,
so the generator can condition on one photo (the product's "one selfie" case) or all of them.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from stage2_eval.common import list_images, load_rgb
from stage2_eval.identity import RECOGNIZERS, Recognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=None, help="audit.csv from stage3_lora.audit_dataset, to pick the single best photo")
    args = ap.parse_args()
    rec = Recognizer(RECOGNIZERS["primary"])  # must be buffalo_l: it's what FaceID was trained on

    names, embs = [], []
    for p in list_images(args.folder):
        e = rec.embed(load_rgb(p))
        if e is not None:
            names.append(p.name)
            embs.append(e)
    embs = np.stack(embs).astype(np.float32)

    best = 0
    audit = Path(args.audit or f"runs/stage3/audit/{Path(args.folder).name}/audit.csv")
    if audit.exists():
        rows = {r["file"]: r for r in csv.DictReader(open(audit))}
        # most frontal, open-eyed, sharp: the product's single-selfie case at its best
        score = lambda n: (abs(float(rows[n]["yaw"] or 99)) < 8, float(rows[n]["sharpness"] or 0))
        best = max(range(len(names)), key=lambda i: score(names[i]) if names[i] in rows else (False, 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=embs, names=np.array(names), best=best)
    c = embs.mean(0)
    print(f"{len(names)} embeddings ({embs.shape}), single-photo pick: {names[best]}")
    print(f"mean pairwise cosine {float((embs @ embs.T)[np.triu_indices(len(embs), 1)].mean()):.3f}, "
          f"centroid norm {np.linalg.norm(c):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
