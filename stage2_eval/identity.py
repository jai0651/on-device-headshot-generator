"""Instrument 1: is this the same person?

Two recognizers with different backbones AND different training data, so that a
method conditioned on one (IP-Adapter-FaceID uses buffalo_l's ArcFace) can be
scored by the other without the check being circular:

  primary    buffalo_l   ArcFace R50,  trained on WebFace600K
  secondary  antelopev2  ArcFace R100, trained on Glint360K

Both are insightface model packs: non-commercial research licence. Fine for this
study; a shipping product needs a commercially licensed recognizer (section 10).
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .common import load_rgb

RECOGNIZERS = {"primary": "buffalo_l", "secondary": "antelopev2"}


def _flatten_nested_pack(pack, root="~/.insightface"):
    """antelopev2.zip unpacks to antelopev2/antelopev2/*.onnx, where FaceAnalysis finds
    nothing. Move the files up one level."""
    d = Path(root).expanduser() / "models" / pack
    inner = d / pack
    if inner.is_dir():
        for f in inner.iterdir():
            shutil.move(str(f), d / f.name)
        inner.rmdir()


class Recognizer:
    def __init__(self, pack, det_size=640):
        self.pack = pack
        kw = dict(name=pack, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
        try:
            self.app = FaceAnalysis(**kw)
        except AssertionError:  # no detection model found: the nested-zip layout
            _flatten_nested_pack(pack)
            self.app = FaceAnalysis(**kw)
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def _faces(self, rgb):
        return self.app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def embed(self, image):
        """Unit-norm embedding of the largest face, or None if no face is found.

        Tight crops (e.g. LFW's 125x94) often defeat the detector because there is no
        context around the face, so on a miss retry once with a padded, upscaled copy."""
        rgb = load_rgb(image)
        faces = self._faces(rgb)
        if not faces:
            h, w = rgb.shape[:2]
            padded = cv2.copyMakeBorder(rgb, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_REPLICATE)
            faces = self._faces(cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding.astype(np.float32)


def centroid(embeddings):
    """Reference identity = normalised mean of the reference photos' embeddings."""
    e = np.stack([x for x in embeddings if x is not None])
    c = e.mean(axis=0)
    return c / np.linalg.norm(c)


def cosine(a, b):
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b))


class IdentityScorer:
    """Scores images against a reference identity with both recognizers."""

    def __init__(self, reference_images=None):
        self.recognizers = {role: Recognizer(pack) for role, pack in RECOGNIZERS.items()}
        self.refs = {}
        if reference_images:
            self.set_reference(reference_images)

    def set_reference(self, images):
        for role, rec in self.recognizers.items():
            embs = [rec.embed(im) for im in images]
            if not any(e is not None for e in embs):
                raise ValueError(f"{rec.pack}: no face found in any reference image")
            self.refs[role] = centroid(embs)

    def score(self, image):
        """-> {"id_primary": cos, "id_secondary": cos}; NaN where no face was found."""
        return {f"id_{role}": cosine(rec.embed(image), self.refs[role])
                for role, rec in self.recognizers.items()}
