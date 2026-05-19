"""
Tiny MLPs for CTSM-v toy experiments (time-conditioned score / vector field).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairConditionedTimeScoreNetBase(nn.Module):
    """Shared interface for pair-conditioned CTSM-v score networks (H-matrix / objectives)."""

    dim: int
    theta_dim: int


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


class ToyBinaryTimeScoreNet(ToyFullTimeScoreNet):
    """
    Unconditioned CTSM-v network for a fixed binary distribution pair.

    The scalar forward output integrates to ``log p_1(x) - log p_0(x)`` when
    trained with class-0 samples as ``x0`` and class-1 samples as ``x1``.
    """


class AdditiveDiagonalGaussianPosterior(nn.Module):
    """
    Additive-evidence diagonal Gaussian posterior in natural parameters.

    q(z | x, t) has diagonal precision
        lambda(x,t) = lambda_0(t) + sum_i lambda_i(x_i,t)
    and natural mean
        eta(x,t) = eta_0(t) + sum_i eta_i(x_i,t).
    """

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int = 128, eps: float = 1e-4):
        super().__init__()
        self.x_dim = int(x_dim)
        self.z_dim = int(z_dim)
        self.eps = float(eps)
        if self.x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        if self.z_dim < 1:
            raise ValueError("z_dim must be >= 1.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.prior_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * self.z_dim),
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * self.z_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("x must have shape (B, D).")
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("t must have shape (B, 1).")
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x last dim {self.x_dim}, got {x.shape[-1]}.")
        if x.shape[0] != t.shape[0]:
            raise ValueError("x and t batch sizes must match.")

        prior_raw_prec, prior_eta = self.prior_net(t).chunk(2, dim=-1)
        precision = F.softplus(prior_raw_prec) + self.eps
        eta = prior_eta

        t_expand = t.unsqueeze(1).expand(-1, self.x_dim, -1)
        coord_in = torch.cat([x.unsqueeze(-1), t_expand], dim=-1).reshape(-1, 2)
        raw_prec_i, eta_i = self.evidence_net(coord_in).chunk(2, dim=-1)
        precision_i = F.softplus(raw_prec_i).reshape(x.shape[0], self.x_dim, self.z_dim)
        eta_i = eta_i.reshape(x.shape[0], self.x_dim, self.z_dim)

        precision = precision + precision_i.sum(dim=1)
        eta = eta + eta_i.sum(dim=1)
        mean = eta / precision
        std = torch.rsqrt(precision)
        return mean, std

    def rsample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(x, t)
        return mean + std * torch.randn_like(mean)

    def rsample_pair(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rsample(x, t), self.rsample(x, t)


class ToyLatentBeliefBinaryTimeScoreNet(nn.Module):
    """
    Latent-belief CTSM-v network for a fixed binary distribution pair.

    The posterior q_phi(z | x,t) is additive-evidence diagonal Gaussian; the
    readout B_psi(z,t) returns a vector in R^dim. Inference integrates the sum
    of the posterior-mean vector readout.
    """

    def __init__(
        self,
        dim: int = 2,
        h_dim: int = 4,
        hidden_dim: int = 128,
        *,
        precision_eps: float = 1e-4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.h_dim = int(h_dim)
        self.posterior = AdditiveDiagonalGaussianPosterior(
            x_dim=self.dim,
            z_dim=self.h_dim,
            hidden_dim=int(hidden_dim),
            eps=float(precision_eps),
        )
        self.readout_net = nn.Sequential(
            nn.Linear(self.h_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def readout(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if z.ndim != 2:
            raise ValueError("z must have shape (B, h_dim).")
        if t.shape[0] != z.shape[0]:
            raise ValueError("z and t batch sizes must match.")
        return self.readout_net(torch.cat([z, t], dim=-1))

    def sample_two_readouts(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = self.posterior.rsample_pair(x, t)
        return self.readout(z1, t), self.readout(z2, t)

    def sample_readouts(self, x: torch.Tensor, t: torch.Tensor, n_mc: int = 1) -> torch.Tensor:
        n = max(1, int(n_mc))
        return torch.stack([self.readout(self.posterior.rsample(x, t), t) for _ in range(n)], dim=0)

    def posterior_mean_vector(self, x: torch.Tensor, t: torch.Tensor, n_mc: int = 16) -> torch.Tensor:
        return self.sample_readouts(x, t, n_mc=n_mc).mean(dim=0)

    def forward_full(self, x: torch.Tensor, t: torch.Tensor, n_mc: int = 16) -> torch.Tensor:
        return self.posterior_mean_vector(x, t, n_mc=n_mc)

    def forward(self, x: torch.Tensor, t: torch.Tensor, n_mc: int = 16) -> torch.Tensor:
        return self.forward_full(x, t, n_mc=n_mc).sum(dim=-1, keepdim=True)


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
        theta_dim: int = 1,
        m_scale: float = 1.0,
        delta_scale: float = 0.5,
    ):
        super().__init__()
        self.dim = int(dim)
        self.theta_dim = int(theta_dim)
        if self.theta_dim < 1:
            raise ValueError("theta_dim must be >= 1.")
        self.m_scale = float(m_scale)
        self.delta_scale = float(delta_scale)
        # x (dim) + t + m (theta_dim) + delta (theta_dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim + 1 + 2 * self.theta_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward_full(self, x, t, m, delta):
        """x,t,m,delta: x is (B, dim), t is (B,1), m/delta are (B, theta_dim)."""
        if m.dim() == 1:
            m = m.unsqueeze(-1)
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if m.shape[-1] != self.theta_dim or delta.shape[-1] != self.theta_dim:
            raise ValueError(f"Expected m/delta last dim {self.theta_dim}, got {m.shape[-1]} and {delta.shape[-1]}.")
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
        theta_dim: int = 1,
        m_scale: float = 1.0,
        delta_scale: float = 0.5,
        use_logit_time: bool = True,
        gated_film: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.theta_dim = int(theta_dim)
        if self.theta_dim < 1:
            raise ValueError("theta_dim must be >= 1.")
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.m_scale = float(m_scale)
        self.delta_scale = float(delta_scale)
        self.use_logit_time = bool(use_logit_time)
        self.gated_film = bool(gated_film)
        cond_dim = 1 + 2 * self.theta_dim  # t_feat, m, delta
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
        if m.shape[-1] != self.theta_dim or delta.shape[-1] != self.theta_dim:
            raise ValueError(f"Expected m/delta last dim {self.theta_dim}, got {m.shape[-1]} and {delta.shape[-1]}.")
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
