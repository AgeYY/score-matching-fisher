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
