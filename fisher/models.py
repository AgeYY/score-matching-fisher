from __future__ import annotations

import torch
from torch import nn


class ConditionalScore1D(nn.Module):
    """Gaussian-parameterized score model for s(theta_tilde, x, sigma), scalar theta."""

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
        self.min_log_std = -8.0
        self.max_log_std = 8.0
        self.eps = 1e-8
        self.mu_net = self._build_x_net(x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)
        self.log_std_net = self._build_x_net(x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)

    @staticmethod
    def _build_x_net(x_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
        in_dim = x_dim
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, theta_tilde: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        mu = self.mu_net(x)
        log_std = torch.clamp(self.log_std_net(x), min=self.min_log_std, max=self.max_log_std)
        std_sq = torch.exp(2.0 * log_std)
        noise_var = sigma**2
        total_var = torch.clamp(std_sq + noise_var, min=self.eps)
        return -(theta_tilde - mu) / total_var

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, x, sigma)


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
