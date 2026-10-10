"""Shared bits: model files, image loading."""

import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODELS = Path(os.environ.get("DML_MODELS", "models"))

MEDIAPIPE_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "selfie_multiclass_256x256.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def mediapipe_model(name):
    """Path to a MediaPipe model file, downloading it on first use (models/ isn't synced to pods)."""
    path = MODELS / "mediapipe" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MEDIAPIPE_URLS[name], path)
    return str(path)


def load_rgb(path_or_img):
    """path, PIL image or HxWx3 array -> HxWx3 uint8 RGB array."""
    if isinstance(path_or_img, np.ndarray):
        return path_or_img.astype(np.uint8)
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    # Phone photos store rotation/mirroring in EXIF rather than in the pixels; without this
    # a portrait selfie arrives sideways and face detection quietly fails or mis-aligns.
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert("RGB"))


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
