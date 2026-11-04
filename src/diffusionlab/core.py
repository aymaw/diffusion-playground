"""Conditional epsilon prediction and deterministic DDIM sampling in two dimensions."""

import math

import numpy as np
import torch
from torch import nn


def shapes(n, seed):
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.full(n, 1.6)
    spiral = y == 1
    angle[spiral] *= 1.5
    radius[spiral] = 0.25 + 1.55 * angle[spiral] / (3 * np.pi)
    blobs = y == 2
    angle[blobs] = rng.integers(0, 8, blobs.sum()) * np.pi / 4
    x = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    x += rng.normal(0, 0.075, x.shape)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


def schedule(steps=128):
    if steps < 2:
        raise ValueError("at least two diffusion timesteps required")
    return torch.cumprod(1 - torch.linspace(0.0001, 0.08, steps), 0)


def noisy(x, t, noise, abar):
    a = abar[t, None]
    return a.sqrt() * x + (1 - a).sqrt() * noise


class Denoiser(nn.Module):
    def __init__(self, width=96, steps=128):
        super().__init__()
        self.steps = steps
        self.label = nn.Embedding(4, 12)  # index 3 means dropped conditioning
        self.net = nn.Sequential(
            nn.Linear(30, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2)
        )

    def forward(self, x, t, labels):
        freq = 2 ** torch.arange(8, device=x.device) * math.pi
        phase = t[:, None] / (self.steps - 1) * freq
        return self.net(torch.cat([x, phase.sin(), phase.cos(), self.label(labels)], -1))


def ddim_step(x, eps, a, prev_a):
    clean = (x - (1 - a).sqrt() * eps) / a.sqrt()
    return prev_a.sqrt() * clean + (1 - prev_a).sqrt() * eps


@torch.inference_mode()
def sample(model, labels, seed, steps=32, guidance=1.0):
    if not 2 <= steps <= model.steps or not np.isfinite(guidance) or guidance < 0:
        raise ValueError("invalid steps or guidance")
    if labels.ndim != 1 or len(labels) == 0 or torch.any((labels < 0) | (labels > 2)):
        raise ValueError("labels must be a nonempty vector in 0..2")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(len(labels), 2, generator=generator)
    abar = schedule(model.steps)
    indices = torch.linspace(model.steps - 1, 0, steps).long().tolist()
    trajectory = [x.clone()]
    for pos, t in enumerate(indices):
        tv = torch.full_like(labels, t)
        unconditional = model(x, tv, torch.full_like(labels, 3))
        conditional = model(x, tv, labels)
        eps = unconditional + guidance * (conditional - unconditional)
        previous = indices[pos + 1] if pos + 1 < len(indices) else -1
        pa = abar[previous] if previous >= 0 else torch.tensor(1.0)
        x = ddim_step(x, eps, abar[t], pa)
        trajectory.append(x.clone())
    return x, trajectory


def train(seed, updates=4000):
    if updates < 1:
        raise ValueError("updates must be positive")
    torch.manual_seed(seed)
    model = Denoiser()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x, y = shapes(8192, seed + 100)
    abar = schedule(model.steps)
    loss_trace = []
    for step in range(updates):
        idx = torch.randint(len(x), (256,))
        t = torch.randint(model.steps, (len(idx),))
        noise = torch.randn_like(x[idx])
        labels = y[idx].clone()
        labels[torch.rand(len(idx)) < 0.15] = 3
        loss = (model(noisy(x[idx], t, noise, abar), t, labels) - noise).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == updates - 1:
            loss_trace.append({"update": step + 1, "mse": float(loss.detach())})
    return model, loss_trace


def sliced_wasserstein(x, y, seed=301, projections=128):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
        raise ValueError("equal nonempty [n,2] arrays required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or projections < 1:
        raise ValueError("finite arrays and positive projections required")
    direction = np.random.default_rng(seed).normal(size=(2, projections))
    direction /= np.linalg.norm(direction, axis=0)
    return float(np.abs(np.sort(x @ direction, axis=0) - np.sort(y @ direction, axis=0)).mean())
