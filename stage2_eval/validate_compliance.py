"""Does each compliance check detect the thing it claims to? A mutation test.

    python -m stage2_eval.validate_compliance --base outputs_raw/stage2/baseline/passport_s0.png

Take a base image, apply one known defect at a time (tilt, blur, off-centre, busy
background, one-sided shadow, zoom that cuts off the head), and check that the
matching check moves the right way and ideally flips from pass to fail. A check that
doesn't react to its own defect is measuring nothing.
"""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .common import load_rgb
from .compliance import BG, FACE_SKIN, ComplianceChecker
from .contact_sheet import render
from PIL import Image


def mutations(rgb, mask):
    h, w = rgb.shape[:2]
    yield "tilt_12deg", "roll_deg", cv2.warpAffine(rgb, cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0), (w, h),
                                                   borderMode=cv2.BORDER_REPLICATE)
    yield "blur", "sharpness", cv2.GaussianBlur(rgb, (0, 0), 3.0)
    yield "shift_right_20pct", "center_offset", cv2.warpAffine(rgb, np.float32([[1, 0, 0.2 * w], [0, 1, 0]]), (w, h),
                                                               borderMode=cv2.BORDER_REPLICATE)
    z = 1.5  # zoom into the upper-middle: head grows and the crown leaves the frame
    crop = rgb[: int(h / z), int(w * (1 - 1 / z) / 2): int(w * (1 + 1 / z) / 2)]
    yield "zoom_1.5x", "crown_margin", cv2.resize(crop, (w, h))

    busy = rgb.copy()
    xs = np.linspace(0, 1, w)[None, :, None]
    stripes = (127 + 110 * np.sin(xs * 40 * np.pi)) * np.array([1.0, 0.6, 0.3])
    busy[mask == BG] = np.broadcast_to(stripes, rgb.shape)[mask == BG].astype(np.uint8)
    yield "busy_background", "background_std", busy

    shadow = rgb.astype(np.float32)
    face_cols = np.where((mask == FACE_SKIN).any(axis=0))[0]
    if len(face_cols):
        cx = int(face_cols.mean())
        face = mask == FACE_SKIN
        left, right = face & (np.arange(w)[None, :] < cx), face & (np.arange(w)[None, :] >= cx)
        # darken whichever half is already darker, so asymmetry can only grow
        darker = left if rgb[left].mean() <= rgb[right].mean() else right
        shadow[darker] *= 0.55
    yield "shadow_one_side", "shadow_asymmetry", shadow.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="one or more base images")
    ap.add_argument("--out", default="outputs/stage2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checker = ComplianceChecker()

    results, sheet = [], []
    for base_path in args.base:
        rgb = load_rgb(base_path)
        base = checker.check(rgb)
        mask = checker.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                                         ).category_mask.numpy_view().squeeze()
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        sheet.append({"image": Image.fromarray(rgb), "label": f"base {Path(base_path).stem}",
                      "comp_passed": base["passed"], "comp_total": base["total"], "comp_all_pass": base["all_pass"]})
        print(f"\nbase {base_path}: {base['passed']}/{base['total']} checks pass")
        for name, target, img in mutations(rgb, mask):
            r = checker.check(img)
            b_val, b_ok = base["checks"][target]
            m_val, m_ok = r["checks"][target]
            flipped = b_ok and not m_ok
            verdict = "FLIPPED" if flipped else ("already failing" if not b_ok else "MISSED")
            results.append({"base": Path(base_path).name, "mutation": name, "check": target,
                            "base_value": round(float(b_val), 4), "mutated_value": round(float(m_val), 4),
                            "base_pass": b_ok, "mutated_pass": m_ok, "verdict": verdict})
            print(f"  {name:<20} {target:<18} {b_val:9.3f} -> {m_val:9.3f}   {verdict}")
            sheet.append({"image": Image.fromarray(img), "label": f"{name} [{target}]",
                          "comp_passed": r["passed"], "comp_total": r["total"], "comp_all_pass": r["all_pass"]})

    with open(out / "compliance_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    render(sheet, out / "compliance_validation.jpg", cols=7)
    caught = sum(r["verdict"] == "FLIPPED" for r in results)
    testable = sum(r["base_pass"] for r in results)
    print(f"\n{caught}/{testable} testable mutations caught -> {out}/compliance_validation.csv, .jpg")


if __name__ == "__main__":
    main()
