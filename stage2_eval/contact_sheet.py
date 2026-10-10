"""A grid of images with their scores printed underneath, so a decision can be made
from numbers and pictures at the same time rather than either alone."""

import math

from PIL import Image, ImageDraw, ImageFont

GREEN, RED, GREY, INK = (40, 140, 70), (190, 50, 40), (110, 110, 110), (20, 20, 20)


def _fmt(v, spec="{:.2f}"):
    return "—" if v is None or v != v else spec.format(v)  # v != v catches NaN


def render(rows, path, id_threshold=None, cols=6, tile=256):
    """rows: dicts with "image" (path or PIL) and any of: label, id_primary, id_secondary,
    clip, comp_passed, comp_total, comp_all_pass."""
    font = ImageFont.load_default(size=14)
    cap = 58
    n = len(rows)
    cols = min(cols, n)
    sheet = Image.new("RGB", (cols * tile, math.ceil(n / cols) * (tile + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x, y = (i % cols) * tile, (i // cols) * (tile + cap)
        img = r["image"] if isinstance(r["image"], Image.Image) else Image.open(r["image"])
        sheet.paste(img.convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y))
        ty = y + tile + 3
        draw.text((x + 6, ty), str(r.get("label", ""))[:34], fill=INK, font=font)
        if "id_primary" in r:
            ok = id_threshold is not None and r["id_primary"] == r["id_primary"] and r["id_primary"] >= id_threshold
            draw.text((x + 6, ty + 18), f"id {_fmt(r['id_primary'])} / {_fmt(r.get('id_secondary'))}",
                      fill=GREEN if ok else (RED if id_threshold is not None else GREY), font=font)
        if "clip" in r:
            draw.text((x + 150, ty + 18), f"clip {_fmt(r['clip'], '{:.1f}')}", fill=GREY, font=font)
        if "comp_passed" in r:
            draw.text((x + 6, ty + 36), f"compliance {r['comp_passed']}/{r['comp_total']}",
                      fill=GREEN if r.get("comp_all_pass") else RED, font=font)
    sheet.save(path, quality=88)
    return path
