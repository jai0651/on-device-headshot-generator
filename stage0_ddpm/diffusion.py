"""The diffusion process itself: noise schedule, forward noising, training loss,
and two samplers (DDPM and DDIM) that run the same trained network.

Notation follows Ho et al. 2020:
  beta_t        noise added at step t
  alpha_t       1 - beta_t
  abar_t        product of alpha_1..alpha_t — how much signal survives to step t
"""

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        # Built in float64 on CPU (MPS has no float64), then cast down.
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        posterior_var = betas * (1 - abar_prev) / (1 - abar)

        def to(x):
            return x.float().to(device)

        self.betas, self.alphas, self.abar, self.posterior_var = map(to, (betas, alphas, abar, posterior_var))

    def _randn(self, shape, generator):
        # Draw on CPU so a seeded generator gives the same noise on any device.
        return torch.randn(shape, generator=generator).to(self.device)

    @staticmethod
    def _at(a, t):
        return a[t].view(-1, 1, 1, 1)

    # --- forward process: no learning, closed form straight to step t ---

    def q_sample(self, x0, t, noise):
        abar = self._at(self.abar, t)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    def loss(self, model, x0, y=None):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t) if y is None else model(self.q_sample(x0, t, noise), t, y)
        return F.mse_loss(pred, noise)


    # --- reverse process ---

    def _predict_x0(self, x, t, eps):
        abar = self._at(self.abar, t)
        return (x - (1 - abar).sqrt() * eps) / abar.sqrt()

    @torch.no_grad()
    def ddpm_sample(self, model, shape, generator=None, record_every=0):
        """Ancestral sampling: one network call per timestep, all T of them, with fresh
        noise injected at every step. Returns (images, trajectory)."""
        x = self._randn(shape, generator)
        trajectory = []
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            beta, alpha, abar = (self._at(a, t) for a in (self.betas, self.alphas, self.abar))
            mean = (x - beta / (1 - abar).sqrt() * eps) / alpha.sqrt()
            if record_every and (i % record_every == 0 or i == self.T - 1):
                trajectory.append((i, x.cpu(), self._predict_x0(x, t, eps).clamp(-1, 1).cpu()))
            if i > 0:
                x = mean + self._at(self.posterior_var, t).sqrt() * self._randn(shape, generator)
            else:
                x = mean
        return x.clamp(-1, 1), trajectory

    @torch.no_grad()
    def ddim_sample(self, model, shape, steps=50, eta=0.0, generator=None, record_every=0):
        """DDIM: same network, but jumps between a subset of timesteps. eta=0 makes it
        deterministic — the same starting noise always gives the same image."""
        x = self._randn(shape, generator)
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        trajectory = []
        for k, i in enumerate(ts):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            abar = self.abar[i]
            abar_prev = self.abar[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=self.device)

            x0 = self._predict_x0(x, t, eps).clamp(-1, 1)
            # Recompute eps from the clipped x0 so the two stay consistent.
            eps = (x - abar.sqrt() * x0) / (1 - abar).sqrt()
            if record_every and (k % record_every == 0 or k == len(ts) - 1):
                trajectory.append((i, x.cpu(), x0.cpu()))

            sigma = eta * ((1 - abar_prev) / (1 - abar) * (1 - abar / abar_prev)).sqrt()
            x = abar_prev.sqrt() * x0 + (1 - abar_prev - sigma**2).sqrt() * eps
            if eta > 0 and k + 1 < len(ts):
                x = x + sigma * self._randn(shape, generator)
        return x, trajectory


def guided(model, y, w):
    """Classifier-free guidance, exactly SD's formula: eps_u + w·(eps_c − eps_u). Returns a
    model(x, t) callable, so the samplers above don't need to know about labels at all."""
    null = torch.full_like(y, model.num_classes)

    def f(x, t):
        eps_u, eps_c = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([null, y])).chunk(2)
        return eps_u + w * (eps_c - eps_u)
    return f
