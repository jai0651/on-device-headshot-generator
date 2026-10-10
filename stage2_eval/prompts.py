"""The fixed prompt set: written once, seeds frozen. Every checkpoint, method and
quantization level generates exactly these, which is what makes numbers comparable.

`{subject}` is filled per method: "sks person" for a trained LoRA, "a person" for
zero-shot FaceID or the no-identity baseline.
"""

PROMPT_SET_VERSION = 2  # bump whenever a prompt changes: scores are only comparable within a version
SEEDS = (0, 1, 2)

PROMPTS = [
    # identity under variation: should all still be the same person
    ("identity", "corporate_suit", "corporate headshot of {subject}, navy suit, plain grey backdrop, softbox lighting"),
    ("identity", "outdoor", "{subject}, outdoors, overcast daylight, smiling"),
    # "portrait" alone made SD 1.5 produce antique engravings and oval-framed 1800s photos,
    # which tested prompt interpretation, not identity at an angle. Say "photo" explicitly.
    ("identity", "three_quarter", "three-quarter view photo of {subject}, head turned slightly to the side, modern DSLR"),
    ("identity", "glasses", "{subject} wearing glasses, office background"),
    # flexibility: does the model still obey the prompt at all
    # v2 (2026-09-28): the v1 astronaut wore a helmet and the v1 hoodie figure faced the sea, so
    # identity had no face to measure. Flexibility = "new context, face still visible".
    ("flexibility", "astronaut", "portrait of {subject} as an astronaut, helmet off, face clearly visible, on the moon"),
    ("flexibility", "oil_painting", "oil painting portrait of {subject}, renaissance style"),
    ("flexibility", "red_hoodie", "{subject} in a red hoodie on a beach at sunset, facing the camera"),
    # the target use case
    ("compliance", "passport", "modern color ID photo of {subject}, plain white background, neutral expression, "
                               "front facing, head and shoulders, centered, even lighting"),  # "passport photo" gave vintage scans
]

NEGATIVE = "blurry, lowres, deformed, cartoon, oversaturated, cropped head, out of frame"


def expand(subject):
    """-> list of dicts: one per (prompt, seed)."""
    return [{"group": g, "name": n, "prompt": p.format(subject=subject), "seed": s,
             "file": f"{n}_s{s}.png", "prompt_set": PROMPT_SET_VERSION}
            for g, n, p in PROMPTS for s in SEEDS]
