import torch

from .model import UNet


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint(path, device, use_ema=True):
    """Returns (model, checkpoint dict). EMA weights sample noticeably better."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UNet(**ckpt["model_cfg"])
    model.load_state_dict({k: v.float() for k, v in ckpt["ema" if use_ema or "model" not in ckpt else "model"].items()})
    return model.to(device).eval(), ckpt
