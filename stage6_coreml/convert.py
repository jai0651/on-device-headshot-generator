"""Run Apple's torch2coreml with three local fixes, without editing its code.

    .venv-coreml/bin/python -m stage6_coreml.convert --model-version models/fused/sd15_ohwx \\
        --convert-unet --unet-support-controlnet --convert-controlnet lllyasviel/control_v11p_sd15_openpose \\
        --attention-implementation SPLIT_EINSUM --compute-unit CPU_AND_NE -o models/coreml/ours_ohwx_cn

  1. Base-model check: the converter refuses a ControlNet whose model card says "trained on
     runwayml/stable-diffusion-v1-5" unless --model-version is exactly that string. Ours is SD 1.5
     with a LoRA folded in, so the check is a false alarm here: skipped.
  2. fp16 weights: it loads ControlNetModel without variant="fp16", which would download ~1.4 GB
     of fp32 that the cache doesn't have. Patched to use the cached fp16 file (upcast to fp32 for
     tracing, as the converter expects).
  3. use_auth_token: deprecated keyword, dropped.
  4. ControlNet batch follows --unet-batch-one (see below).
All arguments are passed straight through to the converter.
"""

import functools
import sys

import torch
from diffusers import ControlNetModel
from python_coreml_stable_diffusion import torch2coreml

torch2coreml._get_controlnet_base_model = lambda version: None          # fix 1

_orig = ControlNetModel.from_pretrained.__func__


@classmethod
@functools.wraps(_orig)
def _from_pretrained(cls, name, *a, **kw):                                # fixes 2 and 3
    kw.pop("use_auth_token", None)
    kw.setdefault("variant", "fp16")
    kw.setdefault("torch_dtype", torch.float32)
    return _orig(cls, name, *a, **kw)


ControlNetModel.from_pretrained = _from_pretrained

# fix 4 (stage 8): with --unet-batch-one (no CFG, e.g. LCM at guidance 1) the UNet gets batch 1
# but convert_controlnet hard-codes batch 2, so the two wouldn't fit together. Rewrite that one
# line of the function's source at runtime; Apple's files stay untouched.
import inspect
import textwrap

_src = textwrap.dedent(inspect.getsource(torch2coreml.convert_controlnet))
_old = "batch_size = 2  # for classifier-free guidance"
assert _src.count(_old) == 1, "converter changed: re-check the ControlNet batch patch"
_src = _src.replace(_old, "batch_size = 1 if args.unet_batch_one else 2  # patched: follow the UNet")
exec(compile(_src, torch2coreml.__file__, "exec"), torch2coreml.__dict__)

if __name__ == "__main__":
    parser = torch2coreml.parser_spec()
    torch2coreml.main(parser.parse_args(sys.argv[1:]))
