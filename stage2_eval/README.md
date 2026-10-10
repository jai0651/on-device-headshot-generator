# Stage 2 — the eval harness

Section 04 of the build notes, built before any training. Every later stage (the LoRA, FaceID,
quantization, LCM) is judged by these numbers, so the instruments are validated too.

| File | What's in it |
|---|---|
| `identity.py` | Two face recognizers: **primary** `buffalo_l` (ArcFace R50, WebFace600K), **secondary** `antelopev2` (R100, Glint360K). Different backbones *and* data, so scoring FaceID output isn't circular |
| `clip_score.py` | Prompt adherence: 100 × cos(CLIP ViT-L/14 image, text) |
| `compliance.py` | 13 objective checks from MediaPipe landmarks, blendshapes, head pose and a face/hair/background mask |
| `prompts.py` | The fixed prompt set: 8 prompts (identity / flexibility / compliance) × 3 frozen seeds |
| `contact_sheet.py` | Grid of images with scores printed underneath |
| `generate.py` | The fixed set with plain SD 1.5 and no identity: the **floor** every method must beat |
| `score.py` | Runs all instruments over a folder → `scores.csv`, `summary.json` (per group), `contact_sheet.jpg` |
| `validate_identity.py` | Tests the identity instrument on 1000 LFW pairs → AUC, EER threshold, agreement |
| `validate_compliance.py` | Mutation test: inject one defect at a time, check the matching check catches it |

## Run it

Heavy runs go to a RunPod GPU via `scripts/pod_stage2.sh`. Locally, smoke tests only:

```bash
uv run python -m stage2_eval.validate_identity --limit 20 --out runs/stage2_smoke
uv run python -m stage2_eval.validate_compliance --base some_headshot.png --out runs/stage2_smoke
uv run python -m stage2_eval.generate --out runs/stage2_smoke/gen --limit 1 --steps 3
uv run python -m stage2_eval.score --images DIR --refs photos_of_target --out DIR
```

## Compliance checks

| Check | Pass range | Source |
|---|---|---|
| one_face | exactly 1 | — |
| head_height_ratio | 0.50–0.69 of image height | US passport 2×2 in |
| eye_line_from_bottom | 0.56–0.69 | US passport |
| center_offset | ≤ 0.05 of width | — |
| roll / yaw / pitch | ≤ 5° / 10° / 12° | — |
| crown_margin | top of head inside frame | — |
| background_std | ≤ 14 luminance std | *calibrate* |
| shadow_asymmetry | ≤ 0.15 | *calibrate* |
| sharpness | ≥ 60 var-of-Laplacian | *calibrate* |
| eye_blink / jaw_open | ≤ 0.5 / ≤ 0.25 | blendshapes |

## Things learned building it

- **CPU work doesn't belong on a GPU pod.** The pod reported 48 CPUs but was capped at 5.1
  (`/sys/fs/cgroup/cpu.max`). onnxruntime started 288 threads for identity validation, which
  throttled the whole container and starved the GPU. The same validation ran in 9.5 min on the M4.

- **Same seed ≠ same image across hardware.** Seed 0 gave a blonde woman on the M4 and a completely
  different woman on the RTX 3090, even though the noise is drawn on CPU. fp16 kernels differ, and
  the trajectory diverges. Every comparison has to run on one machine type.
- **Validate instruments before trusting them.** The first compliance run flagged a 0.6 shadow
  asymmetry. The image really was side-lit, but the upsampled segmentation mask was also leaking
  dark hair into "face skin". Fixed by eroding the mask.
- **Library versions moved under the code:** mediapipe 1.0 aborts on macOS (Metal graph service),
  so it's pinned to 0.10.x. insightface's `antelopev2` zip unpacks one folder too deep, so
  `identity.py` flattens it. transformers 5 returns an output object from `get_image_features`.

## Done when

- [x] Each instrument runs end to end locally (smoke)
- [x] Identity: 1000 LFW pairs. AUC 0.993 / 0.992, EER 1.5% / 2.2%, 98.7% same decision. Use **cos ≥ 0.30** as the guard threshold (the gap between the distributions); the EER point, 0.15, sits on the impostor tail
- [~] Compliance: 8/9 testable mutations caught on the baseline passport images. Background, shadow and sharpness were already failing on every base, so those stay untested until a truly compliant base exists
- [x] Baseline scored: CLIP 24.7, compliance ~6/13, **0/24 fully compliant**. Identity floor pending reference photos
- [ ] 5–10 consented reference identities collected (needed for stage 3+)
