"""Stage 5c, step 1 (Mac, CPU): turn a selfie into an inpainting job.

    uv run python -m stage5_zeroshot.edit_prepare photos/jai/IMG_0747.jpeg --out lora_data/edit

The edit path keeps the real face instead of generating one. So:
  1. Reframe the selfie onto a 512x512 canvas at passport geometry: scale so the head
     (crown to chin) is 60% of the height, eye line 62% up from the bottom, face centred.
     Anything the photo doesn't cover becomes empty canvas.
  2. Segment it. Face skin + hair = KEEP (these pixels are you). Everything else,
     clothes, background, neck, the empty canvas, is REPAINT.
  3. Feather the boundary so the inpainting model blends into the face edge.

The geometry matches stage5_zeroshot.pose_template's skeleton, so the same ControlNet pose
image lines up with the reframed head. Writes <out>/<stem>/framed.png, keep.png, repaint.png
and preview.jpg.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from stage2_eval.common import load_rgb
from stage2_eval.compliance import CHIN, FACE_SKIN, HAIR, IRIS_A, IRIS_B, ComplianceChecker
from stage2_eval.identity import RECOGNIZERS, Recognizer

SIZE = 512
HEAD_FRAC, EYE_FROM_BOTTOM = 0.60, 0.62   # inside the US passport ranges (0.50-0.69, 0.56-0.69)
FEATHER = 9                               # px of soft edge between kept face and repainted area


def segment(checker, rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    mask = checker.segmenter.segment(img).category_mask.numpy_view().squeeze()
    return cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", default="lora_data/edit")
    args = ap.parse_args()
    checker = ComplianceChecker()
    detector = Recognizer(RECOGNIZERS["primary"])

    for path in args.photos:
        full = load_rgb(path)
        # MediaPipe's face model is short-range: a small face in a big frame is invisible to
        # it. Find the face with insightface first and work on a crop ~4 face-heights across.
        s = min(1.0, 1600 / max(full.shape[:2]))
        small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full
        faces = detector._faces(small)
        if not faces:
            print(f"  skip {path}: no face")
            continue
        x0, y0, x1, y1 = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).bbox / s
        side = 4 * (y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.5 * (y1 - y0)
        L, T = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        R, B = int(min(cx + side / 2, full.shape[1])), int(min(cy + side / 2, full.shape[0]))
        rgb = full[T:B, L:R]
        k0 = min(1.0, 1600 / max(rgb.shape[:2]))
        rgb = cv2.resize(rgb, None, fx=k0, fy=k0, interpolation=cv2.INTER_AREA) if k0 < 1 else rgb
        h, w = rgb.shape[:2]
        res = checker.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            print(f"  skip {path}: no face")
            continue
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
        mask = segment(checker, rgb)
        eye = pts[[IRIS_A, IRIS_B]].mean(axis=0)
        chin_y = pts[CHIN, 1]
        cols = slice(int(pts[:, 0].min()), int(pts[:, 0].max()) + 1)
        head_rows = np.where(np.isin(mask[:, cols], (HAIR, FACE_SKIN)).any(axis=1))[0]
        crown_y = head_rows[0] if len(head_rows) else pts[:, 1].min()
        if crown_y <= 1:
            print(f"  warning {path}: hair touches the top edge, head height is underestimated")

        # similarity transform: scale to the target head height, then put the eye midpoint
        # at (centre, 38% from the top). The canvas outside the photo is left mid-grey.
        k = HEAD_FRAC * SIZE / (chin_y - crown_y)
        tx = SIZE / 2 - k * eye[0]
        ty = (1 - EYE_FROM_BOTTOM) * SIZE - k * eye[1]
        M = np.float32([[k, 0, tx], [0, k, ty]])
        framed = cv2.warpAffine(rgb, M, (SIZE, SIZE), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        covered = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)

        # re-segment the framed image (cleaner than warping the old mask) and keep face + hair
        seg = segment(checker, framed)
        keep = (np.isin(seg, (HAIR, FACE_SKIN)) & (covered > 0)).astype(np.uint8) * 255
        # largest connected blob only: stray "hair" pixels in the background shouldn't survive
        n, lab, stats, _ = cv2.connectedComponentsWithStats(keep)
        if n > 1:
            keep = np.where(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        soft = cv2.GaussianBlur(cv2.erode(keep, np.ones((3, 3), np.uint8)), (0, 0), FEATHER / 2)
        repaint = 255 - soft

        out = Path(args.out) / Path(path).stem.replace(" ", "_").replace(".", "_")
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(framed).save(out / "framed.png")
        Image.fromarray(soft).save(out / "keep.png")
        Image.fromarray(repaint).save(out / "repaint.png")
        viz = framed.copy()
        viz[repaint > 127] = (0.35 * viz[repaint > 127] + [0, 90, 160]).astype(np.uint8)
        Image.fromarray(np.hstack([framed, viz])).save(out / "preview.jpg", quality=90)
        print(f"  {Path(path).name:<44} scale {k:.3f}  keep {100 * (keep > 0).mean():4.1f}% of canvas  "
              f"photo covers {100 * (covered > 0).mean():4.1f}%  -> {out}/")


if __name__ == "__main__":
    main()
