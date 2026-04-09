from __future__ import annotations

import torch
from torch import nn


class ConditionalScore1D(nn.Module):
    """Score model for s(theta_tilde, x, sigma), with scalar theta."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = False,
        use_layer_norm: bool = False,
        zero_out_init: bool = False,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.use_log_sigma = use_log_sigma
        in_dim = 1 + x_dim + 1  # theta_tilde, x, sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        out = nn.Linear(hidden_dim, 1)
        if zero_out_init:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, theta_tilde: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        sigma_feat = torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma
        feats = torch.cat([theta_tilde, x, sigma_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, x, sigma)


class ConditionalScore1DFiLMPerLayer(nn.Module):
    """Posterior DSM score s(theta_tilde, x, sigma) with x-input trunk and residual FiLM blocks.

    The main stream starts from ``silu(in_proj(x))`` plus an **additive** ``cond_residual([theta_tilde, sigma])``
    into ``hidden_dim``. Stacked FiLM blocks then refine ``h``. Each block applies a linear map on ``h``,
    FiLM modulation from ``(theta_tilde, sigma_feat)``, SiLU, and a residual add back to ``h``.
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = False,
        use_layer_norm: bool = False,
        gated_film: bool = False,
        zero_out_init: bool = False,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_log_sigma = bool(use_log_sigma)
        self.gated_film = bool(gated_film)
        cond_dim = 2  # theta_tilde, sigma_feat (scalars)
        self.in_proj = nn.Linear(int(x_dim), int(hidden_dim))
        self.in_norm = nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()
        # Additive residual from (noisy theta, sigma) into the x trunk (same hidden_dim).
        self.cond_residual = nn.Linear(cond_dim, int(hidden_dim))
        nn.init.zeros_(self.cond_residual.weight)
        nn.init.zeros_(self.cond_residual.bias)
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(int(hidden_dim), int(hidden_dim)),
                        "gamma": nn.Linear(cond_dim, int(hidden_dim)),
                        "beta": nn.Linear(cond_dim, int(hidden_dim)),
                        "norm": (nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()),
                    }
                )
            )
        self.out = nn.Linear(int(hidden_dim), 1)
        if zero_out_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _sigma_feat(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        return torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma

    def forward(self, theta_tilde: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if theta_tilde.ndim == 1:
            theta_tilde = theta_tilde.unsqueeze(-1)
        sigma_feat = self._sigma_feat(sigma)
        cond = torch.cat([theta_tilde, sigma_feat], dim=-1)
        h = self.in_norm(self.in_proj(x))
        h = torch.nn.functional.silu(h) + self.cond_residual(cond)
        for blk in self.blocks:
            y = blk["lin"](h)
            gamma = blk["gamma"](cond)
            beta = blk["beta"](cond)
            if self.gated_film:
                # Bounded multiplicative modulation for stability.
                y = (1.0 + 0.5 * torch.tanh(gamma)) * y + beta
            else:
                y = gamma * y + beta
            y = blk["norm"](y)
            h = h + torch.nn.functional.silu(y)
        return self.out(h)

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, x, sigma)


class ConditionalThetaFlowVelocity(nn.Module):
    """Conditional theta-velocity model v(theta_t, x, t) with scalar output."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.use_logit_time = bool(use_logit_time)
        in_dim = 1 + x_dim + 1  # theta_t, x, t
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            t_feat = torch.log(t_clip) - torch.log1p(-t_clip)
        else:
            t_feat = t
        feats = torch.cat([theta_t, x, t_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_velocity(self, theta: torch.Tensor, x: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, x, t)


class PriorThetaFlowVelocity(nn.Module):
    """Unconditional theta-velocity model v(theta_t, t) with scalar output."""

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        self.use_logit_time = bool(use_logit_time)
        in_dim = 1 + 1  # theta_t, t
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            t_feat = torch.log(t_clip) - torch.log1p(-t_clip)
        else:
            t_feat = t
        feats = torch.cat([theta_t, t_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_velocity(self, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, t)


class ConditionalXScore(nn.Module):
    """Conditional x-score model for s(x_tilde, theta, sigma) with vector output in R^{x_dim}."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.use_log_sigma = bool(use_log_sigma)
        in_dim = x_dim + 1 + 1  # x_tilde, theta, sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_tilde: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        sigma_feat = torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma
        feats = torch.cat([x_tilde, theta, sigma_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor, theta: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((x.shape[0], 1), float(sigma_eval), device=x.device)
        return self.forward(x, theta, sigma)


class UnconditionalXScore(nn.Module):
    """Denoising score model s(x_tilde, sigma) in R^{x_dim} with no parameter conditioning."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.use_log_sigma = bool(use_log_sigma)
        in_dim = x_dim + 1  # x_tilde, sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_tilde: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        sigma_feat = torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma
        feats = torch.cat([x_tilde, sigma_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((x.shape[0], 1), float(sigma_eval), device=x.device)
        return self.forward(x, sigma)


class ConditionalXFlowVelocity(nn.Module):
    """Conditional velocity model v(x_t, theta, t) with output in R^{x_dim}."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.use_logit_time = bool(use_logit_time)
        in_dim = x_dim + 1 + 1  # x_t, theta, t
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            t_feat = torch.log(t_clip) - torch.log1p(-t_clip)
        else:
            t_feat = t
        feats = torch.cat([x_t, theta, t_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, theta, t)


class ConditionalXFlowVelocityFiLMPerLayer(nn.Module):
    """Conditional velocity v(x_t, theta, t) with per-layer FiLM from (theta, logit_time)."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        cond_dim = 2
        self.in_proj = nn.Linear(int(x_dim), int(hidden_dim))
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(int(hidden_dim), int(hidden_dim)),
                        "gamma": nn.Linear(cond_dim, int(hidden_dim)),
                        "beta": nn.Linear(cond_dim, int(hidden_dim)),
                    }
                )
            )
        self.out = nn.Linear(int(hidden_dim), int(x_dim))

    def _theta_feat(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return theta

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def _cond(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._theta_feat(theta), self._t_feat(t)], dim=-1)

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self._cond(theta, t)
        h = torch.nn.functional.silu(self.in_proj(x_t))
        for blk in self.blocks:
            h = blk["lin"](h)
            gamma = blk["gamma"](cond)
            beta = blk["beta"](cond)
            h = gamma * h + beta
            h = torch.nn.functional.silu(h)
        return self.out(h)

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, theta, t)


class UnconditionalXFlowVelocity(nn.Module):
    """Unconditional velocity v(x_t, t) in R^{x_dim} (no parameter conditioning)."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.use_logit_time = bool(use_logit_time)
        in_dim = x_dim + 1  # x_t, t
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            t_feat = torch.log(t_clip) - torch.log1p(-t_clip)
        else:
            t_feat = t
        feats = torch.cat([x_t, t_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, t)


class UnconditionalXFlowVelocityFiLMPerLayer(nn.Module):
    """Unconditional velocity v(x_t, t) with per-layer FiLM from logit_time only."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        cond_dim = 1
        self.in_proj = nn.Linear(int(x_dim), int(hidden_dim))
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(int(hidden_dim), int(hidden_dim)),
                        "gamma": nn.Linear(cond_dim, int(hidden_dim)),
                        "beta": nn.Linear(cond_dim, int(hidden_dim)),
                    }
                )
            )
        self.out = nn.Linear(int(hidden_dim), int(x_dim))

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self._t_feat(t)
        h = torch.nn.functional.silu(self.in_proj(x_t))
        for blk in self.blocks:
            h = blk["lin"](h)
            gamma = blk["gamma"](cond)
            beta = blk["beta"](cond)
            h = gamma * h + beta
            h = torch.nn.functional.silu(h)
        return self.out(h)

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, t)


class PriorScore1D(nn.Module):
    """Unconditional score model for s(theta_tilde, sigma) approximating prior score over scalar theta."""

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = False,
        use_layer_norm: bool = False,
        zero_out_init: bool = False,
    ) -> None:
        super().__init__()
        self.use_log_sigma = use_log_sigma
        in_dim = 1 + 1  # theta_tilde, sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        out = nn.Linear(hidden_dim, 1)
        if zero_out_init:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, theta_tilde: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        sigma_feat = torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma
        feats = torch.cat([theta_tilde, sigma_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, sigma)


class PriorScore1DFiLMPerLayer(nn.Module):
    """Prior DSM score s(theta_tilde, sigma) with theta_tilde trunk and residual FiLM blocks.

    There is no observation ``x``; the main stream starts from a projection of
    ``theta_tilde`` to ``hidden_dim`` (analogous to the x-trunk in the posterior FiLM).
    FiLM modulation uses ``(theta_tilde, sigma_feat)`` with residual connections.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = False,
        use_layer_norm: bool = False,
        gated_film: bool = False,
        zero_out_init: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_log_sigma = bool(use_log_sigma)
        self.gated_film = bool(gated_film)
        cond_dim = 2  # theta_tilde, sigma_feat
        self.in_proj = nn.Linear(1, int(hidden_dim))
        self.in_norm = nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(int(hidden_dim), int(hidden_dim)),
                        "gamma": nn.Linear(cond_dim, int(hidden_dim)),
                        "beta": nn.Linear(cond_dim, int(hidden_dim)),
                        "norm": (nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()),
                    }
                )
            )
        self.out = nn.Linear(int(hidden_dim), 1)
        if zero_out_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _sigma_feat(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        return torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma

    def forward(self, theta_tilde: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if theta_tilde.ndim == 1:
            theta_tilde = theta_tilde.unsqueeze(-1)
        sigma_feat = self._sigma_feat(sigma)
        cond = torch.cat([theta_tilde, sigma_feat], dim=-1)
        h = self.in_norm(self.in_proj(theta_tilde))
        h = torch.nn.functional.silu(h)
        for blk in self.blocks:
            y = blk["lin"](h)
            gamma = blk["gamma"](cond)
            beta = blk["beta"](cond)
            if self.gated_film:
                y = (1.0 + 0.5 * torch.tanh(gamma)) * y + beta
            else:
                y = gamma * y + beta
            y = blk["norm"](y)
            h = h + torch.nn.functional.silu(y)
        return self.out(h)

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, sigma)


class LocalDecoderLogit(nn.Module):
    """Binary decoder that outputs logit for class y in {0,1}."""

    def __init__(self, x_dim: int = 2, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        in_dim = x_dim
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
