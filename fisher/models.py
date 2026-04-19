from __future__ import annotations

import torch
from torch import nn


def _activation_from_str(name: str) -> type[nn.Module]:
    n = str(name).strip().lower()
    if n == "silu":
        return nn.SiLU
    if n == "relu":
        return nn.ReLU
    if n == "tanh":
        return nn.Tanh
    raise ValueError(f"Unknown activation '{name}'. Supported: silu, relu, tanh.")


def _scalar_embedding_mlp(out_dim: int, depth: int, act: str = "silu") -> nn.Sequential:
    """Map scalar (B, 1) to (B, out_dim) with ``depth`` linear layers and SiLU between (except last)."""
    if out_dim < 1:
        raise ValueError("out_dim must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")
    act_cls = _activation_from_str(act)
    layers: list[nn.Module] = []
    in_dim = 1
    for i in range(depth):
        layers.append(nn.Linear(in_dim, out_dim))
        in_dim = out_dim
        if i < depth - 1:
            layers.append(act_cls())
    return nn.Sequential(*layers)


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


class ConditionalThetaFlowVelocityIIDSoft(nn.Module):
    """Conditional theta-flow velocity: mean over observation coordinates + interaction (velocity in R^M).

    Shared small MLP ``phi`` maps ``(theta~, x_i, t)`` to a vector in ``R^M`` (``M = theta_dim``).
    A second MLP ``psi`` maps the full ``(theta~, x, t)`` to ``R^M``. Output is::

        v(theta~, x, t) = (1/D_x) * sum_i phi(theta~, x_i, t) + alpha * psi(theta~, x, t)

    where ``D_x = x_dim`` is the observation dimension and ``theta~`` is the full ``theta_t`` vector.
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_dim: int = 1,
        interaction_hidden_dim: int | None = None,
        interaction_depth: int | None = None,
        alpha_init: float = 0.001,
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.alpha_learnable = bool(alpha_learnable)
        int_h = int(interaction_hidden_dim) if interaction_hidden_dim is not None else self.hidden_dim
        int_d = int(interaction_depth) if interaction_depth is not None else self.depth
        if int_h < 1:
            raise ValueError("interaction_hidden_dim must be >= 1.")
        if int_d < 1:
            raise ValueError("interaction_depth must be >= 1.")

        # phi(theta~, x_i, t_feat) -> R^{theta_dim}
        in_phi = int(self.theta_dim) + 1 + 1
        d_in = in_phi
        phi_layers: list[nn.Module] = []
        for _ in range(self.depth):
            phi_layers.append(nn.Linear(d_in, self.hidden_dim))
            phi_layers.append(nn.SiLU())
            d_in = self.hidden_dim
        phi_layers.append(nn.Linear(d_in, int(self.theta_dim)))
        self.phi_net = nn.Sequential(*phi_layers)

        # psi(theta~, x, t_feat) -> R^{theta_dim}
        in_psi = int(self.theta_dim) + int(x_dim) + 1
        d_psi = in_psi
        psi_layers: list[nn.Module] = []
        for _ in range(int_d):
            psi_layers.append(nn.Linear(d_psi, int_h))
            psi_layers.append(nn.SiLU())
            d_psi = int_h
        psi_layers.append(nn.Linear(d_psi, int(self.theta_dim)))
        self.psi_net = nn.Sequential(*psi_layers)

        a0 = float(alpha_init)
        if self.alpha_learnable:
            self.register_parameter("alpha", nn.Parameter(torch.tensor([a0], dtype=torch.float32)))
        else:
            self.register_buffer("alpha", torch.tensor([a0], dtype=torch.float32))

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta_t.shape[-1] != self.theta_dim:
            raise ValueError(f"theta_t last dim {theta_t.shape[-1]} != theta_dim={self.theta_dim}")
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"x last dim {x.shape[-1]} != x_dim={self.x_dim}")
        t_feat = self._t_feat(t)
        parts: list[torch.Tensor] = []
        for i in range(self.x_dim):
            xi = x[:, i : i + 1]
            feats = torch.cat([theta_t, xi, t_feat], dim=-1)
            parts.append(self.phi_net(feats))
        # (B, D_x, M) then mean over x coordinates -> (B, M)
        phi_stack = torch.stack(parts, dim=1)
        v_add = phi_stack.mean(dim=1)
        feats_psi = torch.cat([theta_t, x, t_feat], dim=-1)
        v_int = self.psi_net(feats_psi)
        alpha = self.alpha.view(1, 1).to(dtype=v_add.dtype, device=v_add.device)
        return v_add + alpha * v_int

    @torch.no_grad()
    def predict_velocity(self, theta: torch.Tensor, x: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, x, t)


class PriorThetaFlowVelocityIIDSoft(nn.Module):
    """Prior theta-flow velocity: same pooling structure as the conditional model, without ``x``.

    ``v(theta, t) = (1/D) * sum_k phi(theta_k, t) * 1_D + alpha * psi(theta, t)`` with ``D = theta_dim``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_dim: int = 1,
        interaction_hidden_dim: int | None = None,
        interaction_depth: int | None = None,
        alpha_init: float = 0.001,
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.alpha_learnable = bool(alpha_learnable)
        int_h = int(interaction_hidden_dim) if interaction_hidden_dim is not None else self.hidden_dim
        int_d = int(interaction_depth) if interaction_depth is not None else self.depth
        if int_h < 1:
            raise ValueError("interaction_hidden_dim must be >= 1.")
        if int_d < 1:
            raise ValueError("interaction_depth must be >= 1.")

        d_phi = 1 + 1  # theta_k, t_feat
        phi_layers: list[nn.Module] = []
        for _ in range(self.depth):
            phi_layers.append(nn.Linear(d_phi, self.hidden_dim))
            phi_layers.append(nn.SiLU())
            d_phi = self.hidden_dim
        phi_layers.append(nn.Linear(d_phi, 1))
        self.phi_net = nn.Sequential(*phi_layers)

        d_psi = int(self.theta_dim) + 1
        psi_layers: list[nn.Module] = []
        for _ in range(int_d):
            psi_layers.append(nn.Linear(d_psi, int_h))
            psi_layers.append(nn.SiLU())
            d_psi = int_h
        psi_layers.append(nn.Linear(d_psi, int(self.theta_dim)))
        self.psi_net = nn.Sequential(*psi_layers)

        a0 = float(alpha_init)
        if self.alpha_learnable:
            self.register_parameter("alpha", nn.Parameter(torch.tensor([a0], dtype=torch.float32)))
        else:
            self.register_buffer("alpha", torch.tensor([a0], dtype=torch.float32))

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta_t.shape[-1] != self.theta_dim:
            raise ValueError(f"theta_t last dim {theta_t.shape[-1]} != theta_dim={self.theta_dim}")
        t_feat = self._t_feat(t)
        parts: list[torch.Tensor] = []
        for k in range(self.theta_dim):
            tk = theta_t[:, k : k + 1]
            feats = torch.cat([tk, t_feat], dim=-1)
            parts.append(self.phi_net(feats))
        phi_stack = torch.cat(parts, dim=-1)
        v_add = phi_stack.mean(dim=-1, keepdim=True).expand_as(theta_t)
        feats_psi = torch.cat([theta_t, t_feat], dim=-1)
        v_int = self.psi_net(feats_psi)
        alpha = self.alpha.view(1, 1).to(dtype=v_add.dtype, device=v_add.device)
        return v_add + alpha * v_int

    @torch.no_grad()
    def predict_velocity(self, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, t)


class ConditionalThetaFlowVelocityThetaFourierMLP(nn.Module):
    """Conditional theta-flow velocity v(theta_t, x, t) with theta features [1, theta] plus sin/cos harmonics.

    Same scalar theta encoding as ``ConditionalXFlowVelocityThetaFourierMLP``, but the MLP input is
    ``[theta_feats, x, t_feat]`` with scalar output (theta dimension 1).
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.theta_fourier_k = int(theta_fourier_k)
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        in_dim = self._theta_feat_dim + int(x_dim) + 1  # theta_feats, x, t
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta_feat = self._theta_features(theta_t)
        t_feat = self._t_feat(t)
        feats = torch.cat([theta_feat, x, t_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_velocity(self, theta: torch.Tensor, x: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, x, t)


class PriorThetaFlowVelocityThetaFourierMLP(nn.Module):
    """Prior theta-flow velocity v(theta_t, t) with theta features [1, theta] plus sin/cos harmonics."""

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.theta_fourier_k = int(theta_fourier_k)
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        in_dim = self._theta_feat_dim + 1  # theta_feats, t
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta_feat = self._theta_features(theta_t)
        t_feat = self._t_feat(t)
        feats = torch.cat([theta_feat, t_feat], dim=-1)
        return self.net(feats)

    @torch.no_grad()
    def predict_velocity(self, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, t)


class ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer(nn.Module):
    """Theta-flow velocity v(theta_t, x, t): x-trunk + residual FiLM from (Fourier(theta_t), logit t)."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        use_layer_norm: bool = False,
        gated_film: bool = False,
        zero_out_init: bool = False,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.gated_film = bool(gated_film)
        self.theta_fourier_k = int(theta_fourier_k)
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        cond_dim = self._theta_feat_dim + 1
        self.in_proj = nn.Linear(int(x_dim), int(hidden_dim))
        self.in_norm = nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()
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

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def _cond(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._theta_features(theta), self._t_feat(t)], dim=-1)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self._cond(theta_t, t)
        h = self.in_norm(self.in_proj(x))
        h = torch.nn.functional.silu(h) + self.cond_residual(cond)
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
    def predict_velocity(self, theta: torch.Tensor, x: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, x, t)


class PriorThetaFlowVelocityThetaFourierFiLMPerLayer(nn.Module):
    """Prior theta-flow velocity v(theta_t, t): theta trunk + residual FiLM from (Fourier(theta_t), logit t)."""

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        use_layer_norm: bool = False,
        gated_film: bool = False,
        zero_out_init: bool = False,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.gated_film = bool(gated_film)
        self.theta_fourier_k = int(theta_fourier_k)
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        cond_dim = self._theta_feat_dim + 1
        self.in_proj = nn.Linear(1, int(hidden_dim))
        self.in_norm = nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()
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

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def _cond(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._theta_features(theta), self._t_feat(t)], dim=-1)

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta_t.ndim == 1:
            theta_t = theta_t.unsqueeze(-1)
        cond = self._cond(theta_t, t)
        h = self.in_norm(self.in_proj(theta_t))
        h = torch.nn.functional.silu(h) + self.cond_residual(cond)
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
    def predict_velocity(self, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, t)


class ConditionalThetaFlowVelocityFiLMPerLayer(nn.Module):
    """Theta-flow velocity v(theta_t, x, t): x-trunk + residual FiLM from embedded (theta_t, logit t)."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        use_layer_norm: bool = False,
        gated_film: bool = False,
        zero_out_init: bool = False,
        cond_embed_dim: int = 16,
        cond_embed_depth: int = 1,
        cond_embed_act: str = "silu",
    ) -> None:
        super().__init__()
        if x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.gated_film = bool(gated_film)
        self.cond_embed_dim = int(cond_embed_dim)
        self.cond_embed_depth = int(cond_embed_depth)
        self.cond_embed_act = str(cond_embed_act)
        self.theta_embed = _scalar_embedding_mlp(self.cond_embed_dim, self.cond_embed_depth, self.cond_embed_act)
        self.time_embed = _scalar_embedding_mlp(self.cond_embed_dim, self.cond_embed_depth, self.cond_embed_act)
        cond_dim = 2 * self.cond_embed_dim
        self.in_proj = nn.Linear(int(x_dim), int(hidden_dim))
        self.in_norm = nn.LayerNorm(int(hidden_dim)) if use_layer_norm else nn.Identity()
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

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta_t.ndim == 1:
            theta_t = theta_t.unsqueeze(-1)
        t_feat = self._t_feat(t)
        theta_e = self.theta_embed(theta_t)
        time_e = self.time_embed(t_feat)
        cond = torch.cat([theta_e, time_e], dim=-1)
        h = self.in_norm(self.in_proj(x))
        h = torch.nn.functional.silu(h) + self.cond_residual(cond)
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
    def predict_velocity(self, theta: torch.Tensor, x: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((theta.shape[0], 1), float(t_eval), device=theta.device)
        return self.forward(theta, x, t)


class PriorThetaFlowVelocityFiLMPerLayer(nn.Module):
    """Prior theta-flow velocity v(theta_t, t): theta-trunk + residual FiLM from embedded (theta_t, logit t)."""

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        use_layer_norm: bool = False,
        gated_film: bool = False,
        zero_out_init: bool = False,
        cond_embed_dim: int = 16,
        cond_embed_depth: int = 1,
        cond_embed_act: str = "silu",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.gated_film = bool(gated_film)
        self.cond_embed_dim = int(cond_embed_dim)
        self.cond_embed_depth = int(cond_embed_depth)
        self.cond_embed_act = str(cond_embed_act)
        self.theta_embed = _scalar_embedding_mlp(self.cond_embed_dim, self.cond_embed_depth, self.cond_embed_act)
        self.time_embed = _scalar_embedding_mlp(self.cond_embed_dim, self.cond_embed_depth, self.cond_embed_act)
        cond_dim = 2 * self.cond_embed_dim
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

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta_t.ndim == 1:
            theta_t = theta_t.unsqueeze(-1)
        t_feat = self._t_feat(t)
        theta_e = self.theta_embed(theta_t)
        time_e = self.time_embed(t_feat)
        cond = torch.cat([theta_e, time_e], dim=-1)
        h = self.in_norm(self.in_proj(theta_t))
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
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
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


class ConditionalXFlowVelocityIndependentMLP(nn.Module):
    """Conditional velocity v(x_t, theta, t) with no cross-dimension coupling.

    Each output dimension k is produced by a separate MLP that only sees
    ``(x_t[..., k], theta, t)``. There is no mixing of different x coordinates
    inside the network (factorized / independent-dimension inductive bias).
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        in_dim = 1 + 1 + 1  # x_k, theta, t_feat
        self.nets = nn.ModuleList()
        for _ in range(self.x_dim):
            layers: list[nn.Module] = []
            d_in = in_dim
            for _ in range(depth):
                layers.append(nn.Linear(d_in, hidden_dim))
                layers.append(nn.SiLU())
                d_in = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            self.nets.append(nn.Sequential(*layers))

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.shape[-1] != self.x_dim:
            raise ValueError(f"x_t last dim {x_t.shape[-1]} != x_dim={self.x_dim}")
        t_feat = self._t_feat(t)
        parts: list[torch.Tensor] = []
        for k in range(self.x_dim):
            xk = x_t[:, k : k + 1]
            feats = torch.cat([xk, theta, t_feat], dim=-1)
            parts.append(self.nets[k](feats))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, theta, t)


class ConditionalXFlowVelocityIndependentThetaFourierMLP(nn.Module):
    """Per-dimension independent MLPs with Fourier(theta) features (no cross-dimension coupling).

    Same theta encoding as :class:`ConditionalXFlowVelocityThetaFourierMLP`, but each output
    dimension k uses a separate subnetwork on ``(x_t[...,k], theta_feats, t_feat)`` only.
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.theta_fourier_k = int(theta_fourier_k)
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        in_dim = 1 + self._theta_feat_dim + 1  # x_k, theta_feats, t_feat
        self.nets = nn.ModuleList()
        for _ in range(self.x_dim):
            layers: list[nn.Module] = []
            d_in = in_dim
            for _ in range(int(depth)):
                layers.append(nn.Linear(d_in, hidden_dim))
                layers.append(nn.SiLU())
                d_in = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            self.nets.append(nn.Sequential(*layers))

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.shape[-1] != self.x_dim:
            raise ValueError(f"x_t last dim {x_t.shape[-1]} != x_dim={self.x_dim}")
        theta_feat = self._theta_features(theta)
        t_feat = self._t_feat(t)
        parts: list[torch.Tensor] = []
        for k in range(self.x_dim):
            xk = x_t[:, k : k + 1]
            feats = torch.cat([xk, theta_feat, t_feat], dim=-1)
            parts.append(self.nets[k](feats))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, theta, t)


class ConditionalXFlowVelocityIIDSoft(nn.Module):
    """Conditional x-flow with soft iid-x bias: mean-field additive branch + interaction residual.

    A shared small MLP ``phi`` maps each coordinate to a scalar, then the additive velocity is the
    coordinate-wise mean (pooled evidence) broadcast to every output dimension. A second small MLP
    ``psi`` maps the full state. Output is::

        v(x, theta, t) = (1/D) * sum_i phi(x_i, theta, t) * 1_D + alpha * psi(x, theta, t)

    where ``1_D`` is the all-ones vector in ``R^D``, ``alpha`` is learnable or fixed, and
    ``phi``, ``psi`` are the usual coordinate / full-vector branches from the ``iid-x`` idea.
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        interaction_hidden_dim: int | None = None,
        interaction_depth: int | None = None,
        alpha_init: float = 0.001,
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.alpha_learnable = bool(alpha_learnable)
        int_h = int(interaction_hidden_dim) if interaction_hidden_dim is not None else self.hidden_dim
        int_d = int(interaction_depth) if interaction_depth is not None else self.depth
        if int_h < 1:
            raise ValueError("interaction_hidden_dim must be >= 1.")
        if int_d < 1:
            raise ValueError("interaction_depth must be >= 1.")

        # Shared phi: scalar on each coordinate (inputs: x_k, theta, t_feat); pooled by mean over k.
        in_phi = 1 + 1 + 1
        d_in = in_phi
        phi_layers: list[nn.Module] = []
        for _ in range(self.depth):
            phi_layers.append(nn.Linear(d_in, self.hidden_dim))
            phi_layers.append(nn.SiLU())
            d_in = self.hidden_dim
        phi_layers.append(nn.Linear(d_in, 1))
        self.phi_net = nn.Sequential(*phi_layers)

        # Full interaction psi: R^{x_dim} -> R^{x_dim}.
        in_psi = int(x_dim) + 1 + 1
        d_psi = in_psi
        psi_layers: list[nn.Module] = []
        for _ in range(int_d):
            psi_layers.append(nn.Linear(d_psi, int_h))
            psi_layers.append(nn.SiLU())
            d_psi = int_h
        psi_layers.append(nn.Linear(d_psi, int(x_dim)))
        self.psi_net = nn.Sequential(*psi_layers)

        a0 = float(alpha_init)
        if self.alpha_learnable:
            self.register_parameter("alpha", nn.Parameter(torch.tensor([a0], dtype=torch.float32)))
        else:
            self.register_buffer("alpha", torch.tensor([a0], dtype=torch.float32))

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.shape[-1] != self.x_dim:
            raise ValueError(f"x_t last dim {x_t.shape[-1]} != x_dim={self.x_dim}")
        t_feat = self._t_feat(t)
        parts: list[torch.Tensor] = []
        for k in range(self.x_dim):
            xk = x_t[:, k : k + 1]
            feats = torch.cat([xk, theta, t_feat], dim=-1)
            parts.append(self.phi_net(feats))
        # (1/D) * sum_i phi(x_i, theta, t), broadcast to R^D (same scalar in every coordinate).
        phi_stack = torch.cat(parts, dim=-1)
        v_add_mean = phi_stack.mean(dim=-1, keepdim=True)
        v_add = v_add_mean.expand_as(x_t)
        feats_psi = torch.cat([x_t, theta, t_feat], dim=-1)
        v_int = self.psi_net(feats_psi)
        alpha = self.alpha.view(1, 1).to(dtype=v_add.dtype, device=v_add.device)
        return v_add + alpha * v_int

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, theta: torch.Tensor, t_eval: float) -> torch.Tensor:
        self.eval()
        t = torch.full((x_t.shape[0], 1), float(t_eval), device=x_t.device)
        return self.forward(x_t, theta, t)


class ConditionalXFlowVelocityThetaFourierMLP(nn.Module):
    """Conditional velocity v(x_t, theta, t) with theta features [1, theta] plus sin/cos harmonics.

    The scalar theta is encoded as an optional constant 1, optional linear theta, and for k = 1..K
    pairs ``sin(k * omega * theta)``, ``cos(k * omega * theta)``. This keeps a non-periodic path while
    giving the MLP periodic inductive bias for circular structure in theta.
    """

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.theta_fourier_k = int(theta_fourier_k)
        # Scalar omega in sin(k*omega*theta); training code may set this from (2*pi/span)*mult.
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        in_dim = int(x_dim) + self._theta_feat_dim + 1  # x_t, theta_feats, t
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, int(x_dim)))
        self.net = nn.Sequential(*layers)

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta_feat = self._theta_features(theta)
        t_feat = self._t_feat(t)
        feats = torch.cat([x_t, theta_feat, t_feat], dim=-1)
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
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
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


class ConditionalXFlowVelocityThetaFourierFiLMPerLayer(nn.Module):
    """Conditional velocity v(x_t, theta, t): x-trunk + per-layer FiLM from (Fourier(theta), logit t)."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        use_logit_time: bool = True,
        *,
        theta_fourier_k: int = 4,
        theta_fourier_omega: float = 1.0,
        theta_fourier_include_linear: bool = True,
        theta_fourier_include_bias: bool = True,
    ) -> None:
        super().__init__()
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_logit_time = bool(use_logit_time)
        self.theta_fourier_k = int(theta_fourier_k)
        self.theta_fourier_omega = float(theta_fourier_omega)
        self.theta_fourier_include_linear = bool(theta_fourier_include_linear)
        self.theta_fourier_include_bias = bool(theta_fourier_include_bias)
        if self.theta_fourier_k < 0:
            raise ValueError("theta_fourier_k must be >= 0.")
        if self.theta_fourier_k > 0 and abs(self.theta_fourier_omega) < 1e-12:
            raise ValueError("theta_fourier_omega must be non-zero when theta_fourier_k > 0.")
        theta_feat_dim = 0
        if self.theta_fourier_include_bias:
            theta_feat_dim += 1
        if self.theta_fourier_include_linear:
            theta_feat_dim += 1
        theta_feat_dim += 2 * self.theta_fourier_k
        if theta_feat_dim < 1:
            raise ValueError(
                "Theta feature dim is 0: enable bias and/or linear term, or set theta_fourier_k >= 1."
            )
        self._theta_feat_dim = int(theta_feat_dim)
        cond_dim = self._theta_feat_dim + 1
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

    def _theta_features(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        parts: list[torch.Tensor] = []
        if self.theta_fourier_include_bias:
            parts.append(torch.ones_like(theta))
        if self.theta_fourier_include_linear:
            parts.append(theta)
        w = float(self.theta_fourier_omega)
        for k in range(1, self.theta_fourier_k + 1):
            ang = float(k) * w * theta
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=-1)

    def _t_feat(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_logit_time:
            t_clip = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
            return torch.log(t_clip) - torch.log1p(-t_clip)
        return t

    def _cond(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._theta_features(theta), self._t_feat(t)], dim=-1)

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
