"""Audit a folder of training photos against the section 03 dataset rules, before any
GPU time is spent training on them. CPU only: runs on the Mac.

    uv run python -m stage3_lora.audit_dataset photos/jai

Per photo: face count (a second person is a hard no), identity consistency against the
other photos, face resolution, face sharpness, head pose, expression, lens and EXIF time.
Across the set: burst duplicates, shooting sessions that dominate the set, near-identical
frames, and coverage of angles and expressions.

Never modifies the photos. Writes runs/stage3/audit/<name>/audit.csv and audit_sheet.jpg
(face crops with verdicts).
"""

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
from scipy.spatial.transform import Rotation

from stage2_eval.common import list_images, load_rgb
from stage2_eval.compliance import ComplianceChecker
from stage2_eval.identity import RECOGNIZERS, Recognizer

TAGS = {v: k for k, v in ExifTags.TAGS.items()}
WORK_MAX_SIDE = 1600      # analyse a downscaled copy; plenty for detection, 10x faster
MIN_FACE_PX = 200         # face box short side in the original: a 512 training crop at ~40% face
SAME_PERSON_COS = 0.45    # leave-one-out cosine; LFW same-person mean was 0.64, impostors 0.01
SHARPNESS_MIN = 60        # same measure and scale as the compliance check
BURST_SECONDS = 10
SESSION_GAP_MIN = 15
SESSION_MAX_SHARE = 0.30


def exif_info(path):
    im = Image.open(path)
    ex = im.getexif()
    sub = ex.get_ifd(0x8769)
    t = sub.get(TAGS["DateTimeOriginal"])
    return {
        "time": datetime.strptime(t, "%Y:%m:%d %H:%M:%S") if t else None,
        "focal35": sub.get(TAGS["FocalLengthIn35mmFilm"]),
        "lens": str(sub.get(TAGS["LensModel"], "")),
    }


def dhash(rgb, size=8):
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (size + 1, size))
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="runs/stage3/audit")
    args = ap.parse_args()
    folder = Path(args.folder)
    out = Path(args.out) / folder.name
    out.mkdir(parents=True, exist_ok=True)

    rec = Recognizer(RECOGNIZERS["primary"])
    lm = ComplianceChecker().landmarker
    rows = []
    for path in list_images(folder):
        full = load_rgb(path)                       # EXIF-rotated
        H, W = full.shape[:2]
        s = min(1.0, WORK_MAX_SIDE / max(H, W))
        rgb = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else full
        r = {"file": path.name, "width": W, "height": H, **exif_info(path)}

        faces = sorted(rec._faces(rgb), key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        faces = [f for f in faces if f.det_score > 0.5]
        r["n_faces"] = len(faces)
        if faces:
            f = faces[0]
            x0, y0, x1, y1 = f.bbox / s
            area = lambda g: (g.bbox[2] - g.bbox[0]) * (g.bbox[3] - g.bbox[1])
            r["second_face_ratio"] = area(faces[1]) / area(f) if len(faces) > 1 else 0.0
            r["face_px"] = min(x1 - x0, y1 - y0)
            r["face_frac"] = (y1 - y0) / H
            r["emb"] = f.normed_embedding
            pad = 0.35 * (x1 - x0)
            cx0, cy0 = int(max(x0 - pad, 0)), int(max(y0 - pad, 0))
            cx1, cy1 = int(min(x1 + pad, W)), int(min(y1 + pad, H))
            r["crop"] = full[cy0:cy1, cx0:cx1]
            face = cv2.cvtColor(full[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)], cv2.COLOR_RGB2GRAY)
            face = cv2.resize(face, (256, max(1, round(256 * face.shape[0] / max(face.shape[1], 1)))))
            r["sharpness"] = float(cv2.Laplacian(face, cv2.CV_64F).var())
            r["face_lum"] = float(cv2.cvtColor(r["crop"], cv2.COLOR_RGB2LAB)[..., 0].mean())

        # MediaPipe's face model is short-range: on a full-frame photo where the face is
        # ~10% of the width it finds nothing. Run it on the padded crop insightface found.
        res = None
        if "crop" in r:
            c = r["crop"]
            k = 512 / max(c.shape[:2])
            c = cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(c)))
        if res and res.face_landmarks:
            pitch, yaw, _ = Rotation.from_matrix(np.asarray(res.facial_transformation_matrixes[0])[:3, :3]).as_euler("xyz", degrees=True)
            b = {c.category_name: c.score for c in res.face_blendshapes[0]}
            r.update(yaw=yaw, pitch=pitch,
                     smile=(b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2,
                     eyes_closed=max(b.get("eyeBlinkLeft", 0), b.get("eyeBlinkRight", 0)))
        r["hash"] = dhash(rgb)
        rows.append(r)
        print(f"  {path.name:<44} faces {r['n_faces']}  face {r.get('face_px', 0):5.0f}px  "
              f"sharp {r.get('sharpness', 0):6.0f}  yaw {r.get('yaw', float('nan')):+5.0f}  smile {r.get('smile', 0):.2f}")

    # ---- identity consistency: each photo against the centroid of all the others ----
    embs = {i: r["emb"] for i, r in enumerate(rows) if "emb" in r}
    for i, e in embs.items():
        others = np.stack([v for j, v in embs.items() if j != i])
        c = others.mean(0)
        rows[i]["loo_cos"] = float(np.dot(e, c / np.linalg.norm(c)))

    # ---- sessions and bursts from EXIF time ----
    timed = sorted((r["time"], i) for i, r in enumerate(rows) if r["time"])
    session, prev = 0, None
    for t, i in timed:
        if prev and (t - prev).total_seconds() > SESSION_GAP_MIN * 60:
            session += 1
        rows[i]["session"] = session
        prev = t
    for (t0, i0), (t1, i1) in zip(timed, timed[1:]):
        if (t1 - t0).total_seconds() <= BURST_SECONDS:
            # keep the sharper of a burst pair; the other is "one photo wearing two hats"
            a, b = rows[i0], rows[i1]
            weaker = a if a.get("sharpness", 0) < b.get("sharpness", 0) else b
            stronger = b if weaker is a else a
            weaker.setdefault("burst_of", stronger["file"])
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if np.count_nonzero(a["hash"] != b["hash"]) <= 6:
                a.setdefault("near_duplicate_of", b["file"])

    n_sessions = session + 1 if timed else 0
    sess_count = {}
    for r in rows:
        if "session" in r:
            sess_count[r["session"]] = sess_count.get(r["session"], 0) + 1

    # ---- verdicts ----
    for r in rows:
        drop, review = [], []
        if r["n_faces"] == 0:
            drop.append("no face found")
        if r.get("second_face_ratio", 0) > 0.2:
            drop.append("another person in frame")
        if "loo_cos" in r and r["loo_cos"] < SAME_PERSON_COS:
            drop.append(f"doesn't match the others (cos {r['loo_cos']:.2f})")
        if r.get("face_px", 1e9) < MIN_FACE_PX:
            drop.append(f"face too small ({r['face_px']:.0f}px)")
        if r.get("sharpness", 1e9) < SHARPNESS_MIN:
            review.append(f"soft focus ({r['sharpness']:.0f})")
        if r.get("eyes_closed", 0) > 0.5:
            review.append("eyes closed")
        if "burst_of" in r:
            review.append(f"burst duplicate of {r['burst_of']}")
        if "near_duplicate_of" in r:
            review.append(f"near-identical to {r['near_duplicate_of']}")
        if r.get("focal35") and r["focal35"] <= 30 and r.get("face_frac", 0) > 0.35:
            review.append(f"close wide-angle ({r['focal35']}mm): nose distortion risk")
        if "session" in r and sess_count[r["session"]] / len(rows) > SESSION_MAX_SHARE:
            review.append(f"session {r['session']} is {sess_count[r['session']]}/{len(rows)} of the set")
        r["verdict"] = "DROP" if drop else ("REVIEW" if review else "KEEP")
        r["reasons"] = "; ".join(drop + review)

    # ---- coverage ----
    keep = [r for r in rows if r["verdict"] != "DROP" and "yaw" in r]
    yaw = np.array([r["yaw"] for r in keep])
    smile = np.array([r["smile"] for r in keep])
    lum = np.array([r["face_lum"] for r in keep if "face_lum" in r])
    coverage = {
        "frontal (|yaw|<12)": int((np.abs(yaw) < 12).sum()),
        "turned one way (yaw<=-12)": int((yaw <= -12).sum()),
        "turned other way (yaw>=12)": int((yaw >= 12).sum()),
        "smiling (>0.4)": int((smile > 0.4).sum()),
        "not smiling": int((smile <= 0.4).sum()),
        "face brightness spread (std)": round(float(lum.std()), 1) if len(lum) else 0,
        "shooting sessions": n_sessions,
    }

    fields = ["file", "verdict", "reasons", "width", "height", "n_faces", "face_px", "sharpness", "loo_cos",
              "yaw", "pitch", "smile", "eyes_closed", "focal35", "lens", "time", "session"]
    with open(out / "audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in fields})

    # ---- sheet of face crops with verdicts ----
    tile, cap, cols = 200, 64, 6
    font = ImageFont.load_default(size=13)
    sheet = Image.new("RGB", (cols * tile, math.ceil(len(rows) / cols) * (tile + cap)), "white")
    d = ImageDraw.Draw(sheet)
    colour = {"KEEP": (40, 140, 70), "REVIEW": (200, 120, 0), "DROP": (190, 50, 40)}
    for i, r in enumerate(rows):
        x, y = (i % cols) * tile, (i // cols) * (tile + cap)
        if "crop" in r:
            sheet.paste(Image.fromarray(r["crop"]).resize((tile, tile)), (x, y))
        d.text((x + 4, y + tile + 2), f"{r['file'][:24]}  {r['verdict']}", fill=colour[r["verdict"]], font=font)
        d.text((x + 4, y + tile + 18), f"cos {r.get('loo_cos', float('nan')):.2f}  yaw {r.get('yaw', float('nan')):+.0f}  "
               f"sharp {r.get('sharpness', 0):.0f}", fill=(60, 60, 60), font=font)
        d.text((x + 4, y + tile + 34), r["reasons"][:34], fill=(90, 90, 90), font=font)
        d.text((x + 4, y + tile + 48), r["reasons"][34:68], fill=(90, 90, 90), font=font)
    sheet.save(out / "audit_sheet.jpg", quality=88)

    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("KEEP", "REVIEW", "DROP")}
    print(f"\n{len(rows)} photos: {counts}")
    for r in rows:
        if r["verdict"] != "KEEP":
            print(f"  {r['verdict']:<7}{r['file']:<44}{r['reasons']}")
    print("coverage:", coverage)
    print(f"-> {out}/audit.csv, audit_sheet.jpg")


if __name__ == "__main__":
    main()
