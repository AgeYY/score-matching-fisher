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
