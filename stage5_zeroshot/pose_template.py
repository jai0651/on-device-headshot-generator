"""An OpenPose control image drawn straight from the passport geometry spec.

    uv run python -m stage5_zeroshot.pose_template --out lora_data/control/passport_pose.png

ControlNet decides *where* things go; identity is someone else's job (section 02). Rather
than detect a pose from some photo, the keypoints are placed where stage 2's compliance
checks want them: eye line 62% up from the bottom (spec 56-69%), head centred, chin at
~68% so head height is ~0.6 of the frame (spec 0.50-0.69), shoulders square and level.

Drawing follows the OpenPose body format the openpose ControlNet was trained on: COCO-18
keypoints, 17 limbs, the standard colour table, limbs at 60% intensity (same scheme as
controlnet_aux's draw_bodypose, reimplemented to avoid its import-time dependencies).
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

# COCO-18 order. "R"/"L" are the subject's right/left, i.e. image left/right for a frontal photo.
NOSE, NECK, R_SHO, R_ELB, R_WRI, L_SHO, L_ELB, L_WRI = range(8)
R_EYE, L_EYE, R_EAR, L_EAR = 14, 15, 16, 17

LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

# normalised (x, y), y down
PASSPORT = {
    R_EYE: (0.43, 0.38), L_EYE: (0.57, 0.38),     # eye line 0.62 from the bottom
    NOSE: (0.50, 0.46),
    R_EAR: (0.33, 0.41), L_EAR: (0.67, 0.41),
    NECK: (0.50, 0.77),                           # chin ~0.68 -> head height ~0.60 with crown ~0.08
    R_SHO: (0.22, 0.87), L_SHO: (0.78, 0.87),     # square, level shoulders
}


def draw(keypoints, size=512):
    canvas = np.zeros((size, size, 3), np.uint8)
    pts = {k: (x * size, y * size) for k, (x, y) in keypoints.items()}
    for (a, b), color in zip(LIMBS, COLORS):
        if a in pts and b in pts:
            (x0, y0), (x1, y1) = pts[a], pts[b]
            length = math.hypot(x1 - x0, y1 - y0)
            angle = math.degrees(math.atan2(y0 - y1, x0 - x1))
            poly = cv2.ellipse2Poly((int((x0 + x1) / 2), int((y0 + y1) / 2)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in color])
    for k, (x, y) in pts.items():
        cv2.circle(canvas, (int(x), int(y)), 4, COLORS[k], thickness=-1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lora_data/control/passport_pose.png")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True) if out.suffix == "" else out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(draw(PASSPORT), cv2.COLOR_RGB2BGR))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
