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
