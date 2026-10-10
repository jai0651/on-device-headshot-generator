# Stage 7 — quantization: how small before it stops being you?

The UNet (face LoRA folded in, pose-ControlNet inputs) palettized after training, no retraining.
Everything on the Mac, in `.venv-coreml/`. Scored with stage 2's instruments.

| File | What it does |
|---|---|
| `palettize.py` | k-means palettization to N bits (one bit-width per process: coremltools 9.0's worker pool dies after one call) |
| `evaluate.py` | One UNet step per variant vs the FP16 reference, with real ControlNet residuals: PSNR, speed |
| `run_e2e.sh` | The 8 fixed prompts with pose control per variant, on the Neural Engine, with peak memory |

## The table (M4 Neural Engine, 512², 25 steps, pose control on)

| UNet | Size | Per image | Peak RAM* | Identity | Image PSNR vs FP16 | Prompt CLIP | Compliance | Head cut |
|---|---|---|---|---|---|---|---|---|
| FP16 | 1,721 MB | 16.9 s | 1.96 GB | 0.49 | — | 24.3 | 11.4 | 0/8 |
| **8-bit** | **862 MB** | 14.3 s | 1.37 GB | **0.49** | **31.7 dB** | 24.3 | 11.4 | 0/8 |
| 6-bit | 647 MB | 15.6 s | 1.16 GB | 0.46 | 21.3 dB | 24.2 | 11.4 | 0/8 |
| 4-bit (per tensor) | 432 MB | 14.8 s | 1.24 GB | **0.33** | **12.2 dB** | 23.9 | 11.8 | 0/8 |

\* process RSS on macOS; Neural Engine buffers live in a system process, so this is a lower bound.
Identity uses 4 headshot images per variant: differences under ~0.03 are within noise.

## Findings

- **8-bit is free**: half the size, same identity, same prompt adherence, visually the same image.
  The default for shipping.
- **6-bit: small, probably acceptable loss** (identity −0.03, details shift). 62% smaller.
- **4-bit per-tensor breaks the face and the image**: identity 0.33 (the same-person line is 0.30),
  noisy crinkled backgrounds, harsh faces. It confirms open question 1: **identity degrades before
  "image quality" as CLIP and compliance measure it**, and those two barely moved.
- **Harness gap found**: CLIP and the compliance count are blind to texture artifacts. Image PSNR
  against the FP16 output at the same seed catches them (31.7 / 21.3 / 12.2 dB). It now belongs in
  every compression comparison.
- **Correction**: the single-step test showed 4-bit 65% slower; over 25 real steps it isn't (0.385 s).
  All 3,070 ops stay on the Neural Engine at every bit-width.

## Next

- 4-bit done properly: per-group tables (needs `--min-deployment-target iOS18` at conversion), or
  mixed-bit, keeping the most sensitive layers at 6–8 bits (Apple's own recipe for SD).
- ControlNet to 8-bit (690 → ~345 MB). Full 8-bit bundle ≈ text encoder 246 + UNet 862 + ControlNet ~345
  + VAE 99 ≈ **1.55 GB**.
- Peak memory measured on an actual iPhone.
