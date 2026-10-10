"""Instrument 3: is it a usable ID-style headshot? Objective checks, no opinions.

MediaPipe Face Landmarker gives 478 landmarks, blendshapes (eyes, jaw) and a head-pose
matrix; the multiclass selfie segmenter gives background / hair / face-skin masks.
Everything below is plain geometry and pixel statistics on those.

Geometry thresholds follow the US passport 2x2 in spec (head 1-1 3/8 in, eyes
1 1/8-1 3/8 in from the bottom). ICAO 9303 differs in detail. Pixel-statistic
thresholds (background, shadow, sharpness) are starting points to calibrate, not
standards: they're marked as such.
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.spatial.transform import Rotation

from .common import load_rgb, mediapipe_model

# landmark indices in the MediaPipe 478-point face mesh
CHIN, CHEEK_L, CHEEK_R = 152, 234, 454
IRIS_A, IRIS_B = 468, 473
EYE_A, EYE_B = 33, 263  # outer corners, used if iris points are missing

# selfie_multiclass_256x256 categories
BG, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)

SPEC = {
    # name: (low, high, source)
    "head_height_ratio": (0.50, 0.69, "US passport: head 1-1 3/8 in of 2 in"),
    "eye_line_from_bottom": (0.56, 0.69, "US passport: eyes 1 1/8-1 3/8 in from bottom"),
    "center_offset": (0.0, 0.05, "face centre within 5% of image width of the middle"),
    "roll_deg": (0.0, 5.0, "eye line within 5 deg of horizontal"),
    "yaw_deg": (0.0, 10.0, "facing the camera"),
    "pitch_deg": (0.0, 12.0, "not looking up or down"),
    "crown_margin": (0.02, 1.0, "top of head inside the frame"),
    "background_std": (0.0, 14.0, "calibrate: plain backdrop, luminance std (0-255)"),
    "shadow_asymmetry": (0.0, 0.15, "calibrate: left/right face-luminance difference"),
    "sharpness": (60.0, float("inf"), "calibrate: variance of Laplacian on the face"),
    "eye_blink": (0.0, 0.5, "eyes open: max(eyeBlinkLeft, eyeBlinkRight)"),
    "jaw_open": (0.0, 0.25, "mouth closed: jawOpen blendshape"),
}


def _cpu(path):
    # CPU delegate explicitly: on macOS the default tries Metal and aborts when there's
    # no graphics service (headless shells, and there's no GPU delegate on Linux pods).
    return BaseOptions(model_asset_path=path, delegate=BaseOptions.Delegate.CPU)


class ComplianceChecker:
    def __init__(self):
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=_cpu(mediapipe_model("face_landmarker.task")),
            num_faces=2, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=_cpu(mediapipe_model("selfie_multiclass_256x256.tflite")),
            output_category_mask=True, output_confidence_masks=False))

    def measure(self, image):
        """-> dict of raw measurements (pixels are y-down, fractions of image size)."""
        rgb = np.ascontiguousarray(load_rgb(image))
        h, w = rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        lm_res = self.landmarker.detect(mp_img)
        m = {"n_faces": len(lm_res.face_landmarks)}
        if m["n_faces"] == 0:
            return m

        pts = np.array([(p.x * w, p.y * h) for p in lm_res.face_landmarks[0]])
        blend = {c.category_name: c.score for c in lm_res.face_blendshapes[0]}
        mask = self.segmenter.segment(mp_img).category_mask.numpy_view().squeeze()  # comes back HxWx1
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)

        # Crown: topmost hair/face-skin pixel within the face's horizontal extent. The
        # mesh stops at the forehead, so the segmentation mask is the only way to find it.
        x0, x1 = int(max(pts[CHEEK_L, 0], 0)), int(min(pts[CHEEK_R, 0], w - 1))
        x0, x1 = min(x0, x1), max(x0, x1) + 1
        head_rows = np.where(np.isin(mask[:, x0:x1], (HAIR, FACE_SKIN)).any(axis=1))[0]
        crown_y = float(head_rows[0]) if len(head_rows) else float(pts[:, 1].min())
        chin_y = pts[CHIN, 1]
        eyes = pts[[IRIS_A, IRIS_B]] if len(pts) > IRIS_B else pts[[EYE_A, EYE_B]]
        eye_y = eyes[:, 1].mean()
        face_cx = (pts[CHEEK_L, 0] + pts[CHEEK_R, 0]) / 2

        m["head_height_ratio"] = (chin_y - crown_y) / h
        m["eye_line_from_bottom"] = (h - eye_y) / h
        m["center_offset"] = abs(face_cx / w - 0.5)
        dx, dy = eyes[1] - eyes[0]
        m["roll_deg"] = abs(np.degrees(np.arctan2(dy, dx)))
        m["roll_deg"] = min(m["roll_deg"], 180 - m["roll_deg"])  # eye order doesn't matter
        pitch, yaw, _ = Rotation.from_matrix(np.asarray(lm_res.facial_transformation_matrixes[0])[:3, :3]).as_euler("xyz", degrees=True)
        m["yaw_deg"], m["pitch_deg"] = abs(yaw), abs(pitch)
        m["crown_margin"] = crown_y / h

        bg = lum[mask == BG]
        m["background_fraction"] = bg.size / (h * w)
        m["background_std"] = float(bg.std()) if bg.size > 0.02 * h * w else float("inf")

        # Erode first: the 256px mask is upsampled, so its edge pixels are often hair or
        # background, which is dark and would read as "shadow" on whichever side has more of it.
        face = cv2.erode((mask == FACE_SKIN).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        cols = np.arange(w)[None, :]
        left, right = lum[face & (cols < face_cx)], lum[face & (cols >= face_cx)]
        if left.size and right.size:
            m["shadow_asymmetry"] = abs(left.mean() - right.mean()) / max((left.mean() + right.mean()) / 2, 1)
        else:
            m["shadow_asymmetry"] = float("inf")

        fx0, fy0 = np.maximum(pts.min(axis=0).astype(int), 0)
        fx1, fy1 = np.minimum(pts.max(axis=0).astype(int), (w - 1, h - 1))
        crop = cv2.cvtColor(rgb[fy0:fy1 + 1, fx0:fx1 + 1], cv2.COLOR_RGB2GRAY)
        crop = cv2.resize(crop, (256, max(1, round(256 * crop.shape[0] / max(crop.shape[1], 1)))))  # scale-invariant
        m["sharpness"] = float(cv2.Laplacian(crop, cv2.CV_64F).var())

        m["eye_blink"] = max(blend.get("eyeBlinkLeft", 0), blend.get("eyeBlinkRight", 0))
        m["jaw_open"] = blend.get("jawOpen", 0)
        return m

    def check(self, image):
        """-> {"checks": {name: (value, passed)}, "passed": int, "total": int, "all_pass": bool}"""
        m = self.measure(image)
        checks = {"one_face": (m["n_faces"], m["n_faces"] == 1)}
        for name, (lo, hi, _) in SPEC.items():
            v = m.get(name, float("nan"))
            checks[name] = (v, bool(lo <= v <= hi))  # NaN fails every comparison
        passed = sum(ok for _, ok in checks.values())
        return {"checks": checks, "passed": passed, "total": len(checks), "all_pass": passed == len(checks),
                "measurements": m}
