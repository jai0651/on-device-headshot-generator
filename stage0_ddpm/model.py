"""A small UNet that predicts the noise in a noisy image, given the timestep.

Same shape as the DDPM paper's network, scaled down to run on a laptop:
ResBlocks with a timestep embedding added in, self-attention at the lowest
resolution, and skip connections from each encoder level to its decoder level.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer timesteps, as in the Transformer and DDPM papers."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        # The timestep enters here: one learned bias per channel. This is the only
        # way the network knows how noisy its input is.
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        # (b, c, hw) -> (b, 1 head, hw tokens, c)
        q, k, v = (z.transpose(1, 2)[:, None] for z in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out[:, 0].transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """in_ch: image channels. base * mults: channel width per resolution level.
    attn_at: which levels get self-attention (0 = full resolution)."""

    def __init__(self, in_ch=1, base=64, mults=(1, 2, 2), attn_at=(2,), num_res=2, dropout=0.1, num_classes=0):
        super().__init__()
        self.base = base
        tdim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        # Class conditioning, SD's text prompt in miniature: one learned vector per digit, plus
        # one extra "null" label (index num_classes) meaning "any digit", used for CFG.
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes + 1, tdim) if num_classes else None
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)

        # Encoder. `chans` records the width of every activation we'll skip across.
        self.down = nn.ModuleList()
        chans, c = [base], base
        for i, m in enumerate(mults):
            for _ in range(num_res):
                blocks = [ResBlock(c, base * m, tdim, dropout)]
                c = base * m
                if i in attn_at:
                    blocks.append(SelfAttention(c))
                self.down.append(nn.ModuleList(blocks))
                chans.append(c)
            if i != len(mults) - 1:
                self.down.append(Downsample(c))
                chans.append(c)

        self.mid = nn.ModuleList([ResBlock(c, c, tdim, dropout), SelfAttention(c), ResBlock(c, c, tdim, dropout)])

        # Decoder. Each block concatenates one skip activation, popped in reverse.
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res + 1):
                blocks = [ResBlock(c + chans.pop(), base * m, tdim, dropout)]
                c = base * m
                if i in attn_at:
                    blocks.append(SelfAttention(c))
                self.up.append(nn.ModuleList(blocks))
            if i != 0:
                self.up.append(Upsample(c))

        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))

    @staticmethod
    def _run(blocks, h, temb):
        for m in blocks:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
        return h

    def forward(self, x, t, y=None):
        temb = self.time_mlp(timestep_embedding(t, self.base))
        if self.class_emb is not None:
            if y is None:  # unlabelled call = the null label
                y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
            temb = temb + self.class_emb(y)
        h = self.inc(x)
        hs = [h]
        for layer in self.down:
            h = layer(h) if isinstance(layer, Downsample) else self._run(layer, h, temb)
            hs.append(h)
        h = self._run(self.mid, h, temb)
        for layer in self.up:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                h = self._run(layer, torch.cat([h, hs.pop()], dim=1), temb)
        return self.out(h)
