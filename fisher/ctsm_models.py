"""
Tiny MLPs for CTSM-v toy experiments (time-conditioned score / vector field).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairConditionedTimeScoreNetBase(nn.Module):
    """Shared interface for pair-conditioned CTSM-v score networks (H-matrix / objectives)."""

    dim: int


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


class ToyPairConditionedTimeScoreNet(PairConditionedTimeScoreNetBase):
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


class ToyPairConditionedTimeScoreNetFiLM(PairConditionedTimeScoreNetBase):
    """
    Pair-conditioned CTSM-v with an **x-input trunk** and **per-layer FiLM** from
    ``(logit(t), m, Delta)`` (same style as ``ConditionalXFlowVelocityFiLMPerLayer``).

    Conditioning does not concatenate with ``x``; it modulates hidden features after
    ``silu(in_proj(x))``.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        *,
        m_scale: float = 1.0,
        delta_scale: float = 0.5,
        use_logit_time: bool = True,
        gated_film: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.m_scale = float(m_scale)
        self.delta_scale = float(delta_scale)
        self.use_logit_time = bool(use_logit_time)
        self.gated_film = bool(gated_film)
        cond_dim = 3  # t_feat, m, delta
        self.in_proj = nn.Linear(self.dim, self.hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(self.hidden_dim, self.hidden_dim),
                        "gamma": nn.Linear(cond_dim, self.hidden_dim),
                        "beta": nn.Linear(cond_dim, self.hidden_dim),
                    }
                )
            )
        self.out = nn.Linear(self.hidden_dim, self.dim)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def _cond(self, t: torch.Tensor, m: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if m.dim() == 1:
            m = m.unsqueeze(-1)
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        m = m * self.m_scale
        delta = delta * self.delta_scale
        t_feat = self._t_feat(t)
        return torch.cat([t_feat, m, delta], dim=-1)

    def forward_full(self, x: torch.Tensor, t: torch.Tensor, m: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        cond = self._cond(t, m, delta)
        h = torch.nn.functional.silu(self.in_proj(x))
        for blk in self.blocks:
            h = blk["lin"](h)
            gamma = blk["gamma"](cond)
            beta = blk["beta"](cond)
            if self.gated_film:
                h = (1.0 + 0.5 * torch.tanh(gamma)) * h + beta
            else:
                h = gamma * h + beta
            h = torch.nn.functional.silu(h)
        return self.out(h)

    def forward(self, x: torch.Tensor, t: torch.Tensor, m: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.forward_full(x, t, m, delta).sum(dim=-1, keepdim=True)
