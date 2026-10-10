# Stage 5 — zero-shot identity, pose control

The comparison the project is organised around: IP-Adapter-FaceID (no per-person training) against
the stage 3 LoRA, plus stacking and ControlNet pose framing. Full write-up with images:
`docs/results.html#s5`.

| File | Runs on | What it does |
|---|---|---|
| `faceid_embed.py` | Mac (CPU) | buffalo_l ArcFace embeddings of the reference photos → `lora_data/faceid/*.npz`. The pod gets numbers, not photos |
| `pose_template.py` | Mac | OpenPose skeleton placed from the passport geometry spec → `lora_data/control/passport_pose.png` |
| `validate_paths.py` | Mac | Proves the hand-written loop's FaceID and ControlNet paths match diffusers' pipelines pixel for pixel (both: max diff 0) |
| `generate_methods.py` | GPU pod | 9 methods × the fixed prompt set, one session |
| `score_methods.py` | Mac | All instruments per method → `outputs/stage5/comparison.{csv,md}` + contact sheets |
| `edit_prepare.py` | Mac | 5c: reframe a selfie to passport geometry; keep = face + hair, repaint = everything else |
| `edit_generate.py` | GPU pod | 5c: inpaint (SD 1.5 inpainting + pose ControlNet), paste the real face back, harmonize at 0 / 0.2 / 0.35 |

## Result (RTX A4500, prompt set v2, secondary recognizer)

| Method | Identity, headshots | Other styles | Flexibility CLIP | Passport compliance |
|---|---|---|---|---|
| base | −0.05 | −0.07 | 27.0 | 9.0 |
| lora | 0.49 | 0.17 | 25.5 | 9.7 |
| **faceid, 1 selfie** | **0.67** | **0.70** | 25.5 | 9.3 |
| faceid, 17 photos | 0.77 | 0.81 | 23.9 | 10.3 |
| lora + faceid | 0.78 | 0.75 | 25.1 | 9.7 |
| faceid 17 + pose | 0.82 | — | — | 11.3 |

One-selfie FaceID beats the trained LoRA by the independent metric, with no training, which is the
property the on-device product needs. Costs: unprompted black-and-white/desaturated outputs
(flexibility CLIP 23.9), a non-commercial licence, and an unconfirmed human judgement. Pose control
adds 1–1.7 compliance checks but no passport image passes all 13.

## Open

- [ ] Blind A/B test: `outputs/stage5/blind_test.jpg` (answer key in `runs/stage5_blind_key.json`)
- [x] 5c edit path: identity **0.89**, passport compliance 11.4/13, **8 of 27 pass all 13 checks** (first in the project). Harmonize costs identity fast (0.81 at 0.2, 0.69 at 0.35). Bug: garbled text on plain t-shirts; add "text, letters, logo" to the negative
- [x] FaceID colour fix: greyscale outputs 6/24 → 0/24, identity 0.67 → 0.68, saturation 53 → 106 (vivid, a bit punchy)
