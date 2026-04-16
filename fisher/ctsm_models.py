"""
Tiny MLPs for CTSM-v toy experiments (time-conditioned score / vector field).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToyFullTimeScoreNet(nn.Module):
    """
    Minimal version of models/toy_networks.toy_full_time_scorenet.

    forward_full(x, t): vector in R^dim
    forward(x, t): scalar sum over dims — integrated quantity for log-ratio ODE.
    """

    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward_full(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

    def forward(self, x, t):
        return self.forward_full(x, t).sum(dim=-1, keepdim=True)


class ToyPairConditionedTimeScoreNet(nn.Module):
    """
    Pair-conditioned CTSM-v network: f_phi(x, t, m, Delta) in R^dim.

    Endpoints (a, b) are encoded as midpoint m = (a+b)/2 and displacement Delta = b-a.
    Scalar time score is 1^T f_phi (see forward).

    ``m_scale`` and ``delta_scale`` multiply (m, Delta) before concatenation so typical
    parameter ranges (e.g. theta in [-1,1], |Delta| <= 2) sit near unit scale with x, t.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 128,
        *,
        m_scale: float = 1.0,
        delta_scale: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.m_scale = float(m_scale)
        self.delta_scale = float(delta_scale)
        # x (dim) + t + m + delta
        self.net = nn.Sequential(
            nn.Linear(dim + 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward_full(self, x, t, m, delta):
        """x,t,m,delta: (B, dim) and (B,1) each for the scalar channels."""
        if m.dim() == 1:
            m = m.unsqueeze(-1)
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        m = m * self.m_scale
        delta = delta * self.delta_scale
        xt = torch.cat([x, t, m, delta], dim=-1)
        return self.net(xt)

    def forward(self, x, t, m, delta):
        return self.forward_full(x, t, m, delta).sum(dim=-1, keepdim=True)
