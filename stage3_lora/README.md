# Stage 3 — the first face LoRA

Sections 02–03 of the build notes: teach SD 1.5 one face with a low-rank adapter, and pick the
checkpoint by measurement instead of by eye.

| File | Runs on | What it does |
|---|---|---|
| `audit_dataset.py` | Mac (CPU) | Checks photos against section 03's rules: one face, same person as the rest (leave-one-out cosine), face size, sharpness, pose, bursts, dominant sessions, close wide-angle selfies. Never modifies photos |
| `prepare_dataset.py` | Mac (CPU) | 512×512 head-and-shoulders crops, EXIF-rotated, face ≈ 40% of the height |
| `lora_data/jai/captions.json` | — | One caption per crop, describing only what should stay changeable |
| `train_lora.py` | GPU pod | UNet LoRA (+ optional text-encoder LoRA). Samples the fixed prompt set at every checkpoint |
| `scale_sweep.py` | GPU pod | One LoRA at several strengths |
| `score_checkpoints.py` | Mac (CPU) | Identity (2 recognizers), CLIP, compliance per checkpoint → curve, grid, pick |

## Dataset: 17 photos of one person

Audit-driven: the first set of 21 had 7 soft photos from one night (same shirt, same backdrop),
burst pairs, and almost no turned heads. After two rounds of edits: 17 photos, all the same person
(leave-one-out cosine 0.78–0.91), sharp, six outfits. Remaining gaps: no photo turned to the
subject's left, and nearly all smiling.

Trigger phrase **`ohwx man`**, not `sks`: `sks` already means a rifle to SD 1.5.

## Run 1 vs run 2 (rank 16, lr 1e-4, 2000 steps, RTX 3090, ~13.5 min each, $0.13 total)

| | Run 1: UNet LoRA only | Run 2: + text-encoder LoRA 5e-5 |
|---|---|---|
| Picked step | 1500 | 2000 |
| Identity, headshot prompts (primary / secondary) | **0.47 / 0.49** | 0.45 / 0.48 |
| Identity, passport prompt | **0.57** | 0.50 |
| Identity, oil painting | 0.16 | **0.35** |
| Prompts with identity ≥ 0.30 | 62% | **75%** |
| Flexibility CLIP | **26.9** | 26.4 |

Section 03's prediction, measured: the text-encoder LoRA spreads the face into other styles and
learns faster, at a small cost in prompt adherence and headshot likeness. **For headshots, run 1
step 1500 is the one to use.** Identity plateaus around 0.45–0.49 in both, which is above the 0.30
same-person line but below real-photo pairs (0.64 on LFW, 0.75–0.9 within the training set), so
the stage 4 sweep goes after capacity and learning rate.

## Things learned

- **Captions don't stop every leak.** The red tilak (captioned) still appeared on an early
  passport image, and grey/maroon tops recur without being asked for. The text encoder absorbs
  more of this.
- **A metric can't measure what the image doesn't show.** The v1 flexibility prompts put the
  astronaut in a helmet and the hoodie figure facing away, so identity there was ~0 regardless of
  the model. Prompt set v2 asks for the face to be visible, and `PROMPT_SET_VERSION` marks when
  scores stop being comparable.
- **Checkpoint scoring overlaps GPU time.** Run 1's samples were fetched and scored on the Mac
  while run 2 trained.

## Done when

- [x] Dataset audited and captioned
- [x] First LoRA trained, every checkpoint scored, pick made by the curve
- [x] Text encoder on vs off, measured
- [ ] Deliberately overcooked (6000 steps) and uncaptioned runs, to learn the failure signatures
- [ ] Stage 4: rank / LR / scale sweep
