# Stage 6 — Core ML: onto Apple's Neural Engine

Section 05 of the build notes. Everything here runs on the Mac (Core ML is macOS-only), in an
isolated environment: `.venv-coreml/` (Python 3.11, coremltools 9.0, Apple's
`python_coreml_stable_diffusion` with its pinned diffusers 0.30.2 / transformers 4.44.2 / numpy 1.23).
No GPU pod.

| File | What it does |
|---|---|
| `fold_lora.py` | W' = W + B·A written into the UNet; proves live-adapter vs folded agree (max diff 4.7e-6) |
| `verify_unet.py` | Core ML UNet vs PyTorch UNet on identical inputs, per compute unit: PSNR, load, per-step time |
| `coreml_generate.py` | Full generation in Core ML, each network on its own compute unit, with timing |
| `compute_plan.py` | Per-op device placement (Xcode's performance report, scripted) |
| `convert.py` | Apple's converter with 3 runtime fixes (ControlNet base-model check, fp16 weights, auth kwarg) |

Conversion (the heavy step, 77 s, 8.0 GB peak RSS on the 16 GB M4):
```bash
.venv-coreml/bin/python -m python_coreml_stable_diffusion.torch2coreml --model-version models/fused/sd15_ohwx \
    --convert-unet --attention-implementation SPLIT_EINSUM --compute-unit CPU_AND_NE -o models/coreml/ours_ohwx
```
Only the UNet is converted: the LoRA never touched the text encoder or VAE, so Apple's official
conversions of those are identical to ours.

## Results (M4, FP16, 512×512, 25 steps, CFG batch 2)

| UNet | Compute unit | Agreement with PyTorch | Per step |
|---|---|---|---|
| Apple's (base SD 1.5) | CPU only | **100% NaN** | 1.84 s |
| Apple's | GPU | 78.1 dB | 0.63 s |
| Apple's | Neural Engine | 35.5 dB | 0.37 s |
| **ours (face folded in)** | GPU | 75.3 dB | 0.62 s |
| **ours** | **Neural Engine** | **36.0 dB** | **0.37 s** |

| Per image, all Core ML | Neural Engine UNet | GPU UNet |
|---|---|---|
| Identity (secondary), 4 headshot prompts | 0.44 | 0.47 (PyTorch 3090: 0.49) |
| Time | **9.9 s**, flat | 18.5 s, drifting 17 → 22 s (thermal) |

Operation placement on the Neural Engine setting: UNet 3,057 / 3,057 ops on ANE, 0 switches.
VAE decoder 397 / 397 on ANE. Text encoder 399 / 408 (9 token-bookkeeping ops on CPU at the ends).
VAE decoder on ANE vs GPU: 0.23 s vs 0.64 s at 44 dB. **Every network wants the Neural Engine**:
≈ 9.5 s per image end to end on this M4.

## Things learned

- **Your face survives conversion.** Identity 0.44–0.47 vs 0.49 in PyTorch; prompt adherence unchanged.
- **The Neural Engine is less exact but not worse.** 36 dB per step compounds into a *different image
  of the same person* (ANE vs GPU image PSNR 18.8 dB, identity within 0.02). Validate ANE output with
  distribution scores, never by pixel-diffing against a GPU render.
- **CPU_ONLY is not a safe fallback for an FP16 model.** Apple's FP16 UNet gives 100% NaN on the CPU.
- **The GPU throttles, the ANE doesn't** (over 8 images): 17 → 22 s per image vs a flat 9.9 s.
- Version traps: diffusers loads `*.fp16.safetensors` only with `variant="fp16"` (else it quietly
  downloads 4 GB of fp32); transformers 5 saves tokenizers in a format 4.44 can't read; a Core ML
  model's compiled `.mlmodelc` is a temp dir that vanishes with the Python object.

## Pose ControlNet on the Neural Engine (fixes head-cropping)

`convert.py --convert-unet --unet-support-controlnet --convert-controlnet lllyasviel/control_v11p_sd15_openpose`:
115 s, 7.7 GB peak. ControlNet 690 MB (13 residual outputs), control-UNet 1.6 GB (16 inputs).

| Core ML, Neural Engine | Head cut off | Identity (headshots) | Compliance (headshots) | Per image |
|---|---|---|---|---|
| UNet alone | 5 / 6 | 0.44 | 7.4 / 13 | 9.9 s |
| **+ pose ControlNet** | **0 / 8** | **0.49** | **11.4 / 13** | 14.6 s (ControlNet 0.15 s/step) |

`./generate_headshot.sh "<prompt with ohwx man>" N` now uses pose control by default (`NOPOSE=1` to skip).

## Next (stage 7)

Palettize the UNet to 8 / 6 / 4 bits and re-run exactly this: size, per-step time, identity,
compliance. The table that answers "how small before it stops being me".
