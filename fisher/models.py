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
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.use_log_sigma = use_log_sigma
        in_dim = 1 + x_dim + 1  # theta_tilde, x, sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        theta_tilde: torch.Tensor,
        x: torch.Tensor,
        sigma: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = theta  # unused; optional for API parity with FiLM variant
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
    """Scalar score s(theta_tilde, x, sigma) with per-layer FiLM from (theta, log(sigma)).

    Backbone sees [theta_tilde, x] only; FiLM uses clean batch theta and log(sigma).
    Training calls forward(..., theta=tb); predict_score passes clean theta.
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = False,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_log_sigma = bool(use_log_sigma)
        cond_dim = 2  # theta, log(sigma)

        self._stem = nn.Linear(1 + x_dim, hidden_dim)
        self._blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self._blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self._head = nn.Linear(hidden_dim, 1)

        self._film_layers = nn.ModuleList()
        for _ in range(depth):
            self._film_layers.append(
                nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                )
            )

    def _sigma_feat(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        return torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma

    def _film_params(self, layer_idx: int, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gb = self._film_layers[layer_idx](cond)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return gamma, beta

    def _film_activate(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu((1.0 + gamma) * z + beta)

    def forward(
        self,
        theta_tilde: torch.Tensor,
        x: torch.Tensor,
        sigma: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sigma_feat = self._sigma_feat(sigma)
        theta_feat = theta if theta is not None else theta_tilde
        cond = torch.cat([theta_feat, sigma_feat], dim=-1)

        z0 = self._stem(torch.cat([theta_tilde, x], dim=-1))
        gamma0, beta0 = self._film_params(0, cond)
        h = self._film_activate(z0, gamma0, beta0)

        for i, blk in enumerate(self._blocks, start=1):
            zi = blk(h)
            gamma, beta = self._film_params(i, cond)
            h = self._film_activate(zi, gamma, beta)

        return self._head(h)

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, x, sigma, theta=theta)


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
    """Unconditional x-score model for s(x_tilde, sigma) with vector output in R^{x_dim}."""

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


class UnconditionalXScoreFiLMPerLayer(nn.Module):
    """Unconditional x-score with per-layer FiLM from log(sigma) only (no theta).

    Same backbone idea as ConditionalXScoreFiLMPerLayer, but condition c = [log sigma].
    """

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
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_log_sigma = bool(use_log_sigma)
        cond_dim = 1  # log(sigma) only

        self._stem = nn.Linear(x_dim, hidden_dim)
        self._blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self._blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self._head = nn.Linear(hidden_dim, x_dim)

        self._film_layers = nn.ModuleList()
        for _ in range(depth):
            self._film_layers.append(
                nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                )
            )

    def _sigma_feat(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        return torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma

    def _film_params(self, layer_idx: int, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gb = self._film_layers[layer_idx](cond)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return gamma, beta

    def _film_activate(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu((1.0 + gamma) * z + beta)

    def forward(self, x_tilde: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        cond = self._sigma_feat(sigma)

        z0 = self._stem(x_tilde)
        gamma0, beta0 = self._film_params(0, cond)
        h = self._film_activate(z0, gamma0, beta0)

        for i, blk in enumerate(self._blocks, start=1):
            zi = blk(h)
            gamma, beta = self._film_params(i, cond)
            h = self._film_activate(zi, gamma, beta)

        return self._head(h)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((x.shape[0], 1), float(sigma_eval), device=x.device)
        return self.forward(x, sigma)


class ConditionalXScoreResidualConcat(nn.Module):
    """Conditional x-score with θ and log(σ) concatenated into every block; residual updates in hidden space.

    First block: h = SiLU(W [x̃, θ, log σ]). Further blocks: h ← h + SiLU(W [h, θ, log σ]).
    Output: W_out [h, θ, log σ] → R^{x_dim}. Conditioning cannot be ``forgotten'' by depth alone.
    """

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
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_log_sigma = bool(use_log_sigma)
        cond_dim = 2  # theta, sigma feature
        self._stem = nn.Linear(x_dim + cond_dim, hidden_dim)
        self._blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self._blocks.append(nn.Linear(hidden_dim + cond_dim, hidden_dim))
        self._head = nn.Linear(hidden_dim + cond_dim, x_dim)

    def _sigma_feat(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        return torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma

    def forward(self, x_tilde: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_feat = self._sigma_feat(sigma)
        c = torch.cat([theta, sigma_feat], dim=-1)
        h = torch.nn.functional.silu(self._stem(torch.cat([x_tilde, c], dim=-1)))
        for blk in self._blocks:
            h = h + torch.nn.functional.silu(blk(torch.cat([h, c], dim=-1)))
        return self._head(torch.cat([h, c], dim=-1))

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor, theta: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((x.shape[0], 1), float(sigma_eval), device=x.device)
        return self.forward(x, theta, sigma)


class ConditionalXScoreFiLMPerLayer(nn.Module):
    """Conditional x-score using per-layer FiLM from (theta, log(sigma)).

    Hidden backbone processes x_tilde only. Each hidden layer l is modulated by
    FiLM parameters (gamma_l, beta_l) predicted from condition c=[theta, log(sigma)]:
    h_l = SiLU((1 + gamma_l(c)) * z_l + beta_l(c)).
    """

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
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_log_sigma = bool(use_log_sigma)
        cond_dim = 2  # theta, sigma feature

        self._stem = nn.Linear(x_dim, hidden_dim)
        self._blocks = nn.ModuleList()
        for _ in range(depth - 1):
            self._blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self._head = nn.Linear(hidden_dim, x_dim)

        # One dedicated FiLM conditioner per hidden layer.
        self._film_layers = nn.ModuleList()
        for _ in range(depth):
            self._film_layers.append(
                nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                )
            )

    def _sigma_feat(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        return torch.log(torch.clamp(sigma, min=1e-8)) if self.use_log_sigma else sigma

    def _film_params(self, layer_idx: int, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gb = self._film_layers[layer_idx](cond)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return gamma, beta

    def _film_activate(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu((1.0 + gamma) * z + beta)

    def forward(self, x_tilde: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_feat = self._sigma_feat(sigma)
        cond = torch.cat([theta, sigma_feat], dim=-1)

        z0 = self._stem(x_tilde)
        gamma0, beta0 = self._film_params(0, cond)
        h = self._film_activate(z0, gamma0, beta0)

        for i, blk in enumerate(self._blocks, start=1):
            zi = blk(h)
            gamma, beta = self._film_params(i, cond)
            h = self._film_activate(zi, gamma, beta)

        return self._head(h)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor, theta: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((x.shape[0], 1), float(sigma_eval), device=x.device)
        return self.forward(x, theta, sigma)


class PriorScore1D(nn.Module):
    """Unconditional score model for s(theta_tilde, sigma) approximating prior score over scalar theta."""

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_log_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.use_log_sigma = use_log_sigma
        in_dim = 1 + 1  # theta_tilde, sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
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
