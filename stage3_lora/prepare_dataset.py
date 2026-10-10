"""Turn audited photos into LoRA training images: 512x512 head-and-shoulders crops.

    uv run python -m stage3_lora.prepare_dataset photos/jai --out lora_data/jai

SD 1.5 was trained at 512, so that's the training resolution. Each crop is a square
centred on the face (nudged down so shoulders are included), sized so the face box is
~40% of the crop height: close enough for identity detail, loose enough that clothes
and background exist to be captioned and varied. Photos are EXIF-rotated first.

Writes <out>/<stem>.png plus <out>/preview.jpg. Captions go in <out>/captions.json,
written separately (see README): the captions are the half of the dataset that decides
what the token absorbs.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stage2_eval.common import list_images, load_rgb
from stage2_eval.identity import RECOGNIZERS, Recognizer

FACE_FRACTION = 0.40
SIZE = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="file names to leave out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recognizer(RECOGNIZERS["primary"])

    made = []
    for path in list_images(args.folder):
        if path.name in args.exclude:
            continue
        full = load_rgb(path)
        H, W = full.shape[:2]
        s = min(1.0, 1600 / max(H, W))
        small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full
        faces = rec._faces(small)
        if not faces:
            print(f"  skip {path.name}: no face")
            continue
        x0, y0, x1, y1 = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).bbox / s
        side = (y1 - y0) / FACE_FRACTION
        side = min(side, W, H)                      # can't crop wider than the photo
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.12 * side  # face sits a bit above centre
        left = int(np.clip(cx - side / 2, 0, W - side))
        top = int(np.clip(cy - side / 2, 0, H - side))
        crop = full[top:top + int(side), left:left + int(side)]
        img = Image.fromarray(crop).resize((SIZE, SIZE), Image.LANCZOS)
        name = path.stem.replace(" ", "_").replace(".", "_") + ".png"
        img.save(out / name)
        upscale = SIZE / side
        made.append((name, img, upscale))
        print(f"  {path.name:<44} -> {name:<40} crop {int(side)}px{'  (upscaled x%.1f)' % upscale if upscale > 1 else ''}")

    cols, tile = 6, 170
    sheet = Image.new("RGB", (cols * tile, math.ceil(len(made) / cols) * tile), "white")
    for i, (_, img, _) in enumerate(made):
        sheet.paste(img.resize((tile, tile)), ((i % cols) * tile, (i // cols) * tile))
    sheet.save(out / "preview.jpg", quality=90)
    print(f"-> {len(made)} crops in {out}/ (+ preview.jpg)")


if __name__ == "__main__":
    main()
