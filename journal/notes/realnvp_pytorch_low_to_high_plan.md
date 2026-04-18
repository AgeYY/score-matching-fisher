# Plan: RealNVP as a fixed low-to-high synthetic generator (Option B only)

## 1) Goal

Use a PyTorch RealNVP-style transform only as a **synthetic generator**:

- sample low-dimensional latent `z in R^m` (`m << n`)
- zero-pad to `R^n`
- pass through a RealNVP in `R^n`
- obtain high-dimensional synthetic embedding `h in R^n`
- optionally transform to positive rates and sample observations

This plan intentionally uses **no training**.

---

## 2) Scope constraints

- Keep RealNVP parameters at initialization (random fixed mapping).
- Do not define optimizer, loss, backward pass, or fitting loop.
- Use this only for controlled synthetic data generation, not representation learning.

---

## 3) Setup

Install dependencies:

```bash
pip install torch glasflow
```

Core construction:

```text
z (m-dim) -> zero pad to n-dim -> RealNVP_n (fixed weights) -> h (n-dim)
```

---

## 4) Minimal generator module (no training)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from glasflow import RealNVP


class LowToHighRealNVPGenerator(nn.Module):
    def __init__(self, z_dim=2, h_dim=64, n_transforms=6, n_neurons=128,
                 positive_output=False, seed=0):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.positive_output = positive_output

        # Fix random initialization for reproducible synthetic mapping
        torch.manual_seed(seed)

        self.flow = RealNVP(
            n_inputs=h_dim,
            n_transforms=n_transforms,
            n_neurons=n_neurons,
            batch_norm_between_transforms=True,
        )

        # Explicitly freeze parameters: generator is fixed, not trainable
        for p in self.flow.parameters():
            p.requires_grad = False

    def pad_latent(self, z):
        if self.h_dim < self.z_dim:
            raise ValueError("h_dim must be >= z_dim")
        pad = torch.zeros(z.shape[0], self.h_dim - self.z_dim,
                          device=z.device, dtype=z.dtype)
        return torch.cat([z, pad], dim=-1)

    @torch.no_grad()
    def forward(self, z):
        x0 = self.pad_latent(z)
        h, _ = self.flow.forward(x0)
        if self.positive_output:
            h = F.softplus(h)
        return h
```

---

## 5) Sampling workflow

```python
import torch

# Build fixed generator once
gen = LowToHighRealNVPGenerator(
    z_dim=4,
    h_dim=128,
    n_transforms=6,
    n_neurons=128,
    positive_output=False,
    seed=123,
)

# 1) sample latent
z = torch.randn(2000, 4)

# 2) generate high-dimensional synthetic embeddings
h = gen(z)  # shape: [2000, 128]

# 3) optional: generate synthetic observations from h
# Gaussian example
x_gauss = h + 0.1 * torch.randn_like(h)

# Poisson-like example (if using positive outputs)
gen_pos = LowToHighRealNVPGenerator(
    z_dim=4, h_dim=128, positive_output=True, seed=123
)
rate = gen_pos(z)
x_pois = torch.poisson(rate)
```

---

## 6) Reproducibility checklist

- Fix `seed` for deterministic initialization.
- Save sampled `z`, generated `h`, and final observations to disk.
- Record RealNVP hyperparameters (`z_dim`, `h_dim`, `n_transforms`, `n_neurons`).

---

## 7) Caveat

Because `z` is zero-padded before applying an `R^n -> R^n` bijection, outputs lie on a transformed lower-dimensional manifold in `R^n`. This is fine for synthetic generation, but it is not an exact full-dimensional density model of ambient `R^n` data.
