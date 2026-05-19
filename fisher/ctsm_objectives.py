"""
CTSM-v loss (two-sample, full vector target) and log q/p estimation by integrating scalar time score.
"""

from __future__ import annotations

import torch
from torchdiffeq import odeint

from fisher.ctsm_models import PairConditionedTimeScoreNetBase, ToyFullTimeScoreNet, ToyLatentBeliefBinaryTimeScoreNet
from fisher.ctsm_paths import TwoEndpointBridge


def ctsm_v_two_sample_loss(
    model: ToyFullTimeScoreNet,
    prob_path: TwoEndpointBridge,
    x0: torch.Tensor,
    x1: torch.Tensor,
    factor: float = 1.0,
    t_eps: float = 1e-5,
) -> torch.Tensor:
    """
    full=True branch of get_toy_c_timewise_score_estimation, specialized to the TwoSB path.
    """
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, 1, device=x0.device) * (1.0 - 2.0 * t_eps) + t_eps

    mean, std, _ = prob_path.marginal_prob(x0, x1, t)
    epsilon = torch.randn_like(x0)
    x_t = mean + std * epsilon

    lambda_t, targets = prob_path.full_epsilon_target(
        epsilon=epsilon,
        x0=x0,
        x1=x1,
        t=t,
        factor=factor,
    )

    pred = lambda_t * model.forward_full(x_t, t)
    loss_per_sample = torch.mean((targets - pred) ** 2, dim=-1)
    return loss_per_sample.mean()


def ctsm_v_pair_conditioned_loss(
    model: PairConditionedTimeScoreNetBase,
    prob_path: TwoEndpointBridge,
    x0: torch.Tensor,
    x1: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    factor: float = 1.0,
    t_eps: float = 1e-5,
) -> torch.Tensor:
    """
    CTSM-v loss with pair conditioning (m, Delta) = ((a+b)/2, b-a).

    Same TwoSB bridge and vector target as two-sample CTSM-v; only the network input changes.
    """
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, 1, device=x0.device) * (1.0 - 2.0 * t_eps) + t_eps

    if a.dim() == 1:
        a = a.unsqueeze(-1)
    if b.dim() == 1:
        b = b.unsqueeze(-1)

    m = 0.5 * (a + b)
    delta = b - a

    mean, std, _ = prob_path.marginal_prob(x0, x1, t)
    epsilon = torch.randn_like(x0)
    x_t = mean + std * epsilon

    lambda_t, targets = prob_path.full_epsilon_target(
        epsilon=epsilon,
        x0=x0,
        x1=x1,
        t=t,
        factor=factor,
    )

    pred = lambda_t * model.forward_full(x_t, t, m, delta)
    loss_per_sample = torch.mean((targets - pred) ** 2, dim=-1)
    return loss_per_sample.mean()


def _two_sample_path_batch(
    prob_path: TwoEndpointBridge,
    x0: torch.Tensor,
    x1: torch.Tensor,
    factor: float,
    t_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, 1, device=x0.device) * (1.0 - 2.0 * t_eps) + t_eps
    mean, std, _ = prob_path.marginal_prob(x0, x1, t)
    epsilon = torch.randn_like(x0)
    x_t = mean + std * epsilon
    lambda_t, targets = prob_path.full_epsilon_target(
        epsilon=epsilon,
        x0=x0,
        x1=x1,
        t=t,
        factor=factor,
    )
    return x_t, t, lambda_t, targets


def latent_belief_ctsm_v_two_sample_loss(
    model: ToyLatentBeliefBinaryTimeScoreNet,
    prob_path: TwoEndpointBridge,
    x0: torch.Tensor,
    x1: torch.Tensor,
    factor: float = 1.0,
    t_eps: float = 1e-5,
    n_posterior_pairs: int = 1,
) -> torch.Tensor:
    """
    Latent-belief CTSM-v posterior-outer loss.

    Uses two independent posterior samples per pair and the same lambda-scaled
    convention as ctsm_v_two_sample_loss.
    """
    x_t, t, lambda_t, targets = _two_sample_path_batch(prob_path, x0, x1, factor, t_eps)
    losses = []
    for _ in range(max(1, int(n_posterior_pairs))):
        b1, b2 = model.sample_two_readouts(x_t, t)
        res1 = targets - lambda_t * b1
        res2 = targets - lambda_t * b2
        losses.append(torch.mean(res1 * res2, dim=-1))
    return torch.stack(losses, dim=0).mean()


def latent_belief_ctsm_v_posterior_mean_loss(
    model: ToyLatentBeliefBinaryTimeScoreNet,
    prob_path: TwoEndpointBridge,
    x0: torch.Tensor,
    x1: torch.Tensor,
    factor: float = 1.0,
    t_eps: float = 1e-5,
    n_mc: int = 8,
) -> torch.Tensor:
    """Stable nonnegative validation loss based on posterior-mean readout."""
    x_t, t, lambda_t, targets = _two_sample_path_batch(prob_path, x0, x1, factor, t_eps)
    pred = lambda_t * model.posterior_mean_vector(x_t, t, n_mc=max(1, int(n_mc)))
    loss_per_sample = torch.mean((targets - pred) ** 2, dim=-1)
    return loss_per_sample.mean()


def latent_belief_ctsm_v_inner_posterior_loss(
    model: ToyLatentBeliefBinaryTimeScoreNet,
    prob_path: TwoEndpointBridge,
    x0: torch.Tensor,
    x1: torch.Tensor,
    factor: float = 1.0,
    t_eps: float = 1e-5,
    nh: int = 32,
) -> torch.Tensor:
    """
    Latent-belief CTSM-v loss with posterior expectation inside the square.

    The posterior mean readout is estimated by Monte Carlo samples, then used
    in the same lambda-scaled CTSM-v residual as the deterministic binary loss.
    """
    x_t, t, lambda_t, targets = _two_sample_path_batch(prob_path, x0, x1, factor, t_eps)
    pred = lambda_t * model.posterior_mean_vector(x_t, t, n_mc=max(1, int(nh)))
    loss_per_sample = torch.mean((targets - pred) ** 2, dim=-1)
    return loss_per_sample.mean()


@torch.no_grad()
def estimate_log_ratio_trapz(
    model: ToyFullTimeScoreNet,
    x: torch.Tensor,
    eps1: float = 1e-5,
    eps2: float = 1e-5,
    n_time: int = 300,
) -> torch.Tensor:
    """
    Trapezoidal integration of scalar model(x, t) over t (replacement for solve_ivp in demos).
    """
    ts = torch.linspace(eps1, 1.0 - eps2, n_time, device=x.device)
    values = []
    for t in ts:
        t_batch = torch.full((x.shape[0], 1), float(t), device=x.device)
        values.append(model(x, t_batch).squeeze(-1))
    values = torch.stack(values, dim=0)
    return torch.trapz(values, ts, dim=0)


@torch.no_grad()
def estimate_log_ratio_trapz_pair(
    model: PairConditionedTimeScoreNetBase,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps1: float = 1e-5,
    eps2: float = 1e-5,
    n_time: int = 300,
) -> torch.Tensor:
    """
    Trapezoidal integration of scalar s_hat(x,t,m,Delta) over t, with
    m = (a+b)/2, Delta = b-a, estimating log p(x|b) - log p(x|a).
    """
    if a.dim() == 1:
        a = a.unsqueeze(-1)
    if b.dim() == 1:
        b = b.unsqueeze(-1)
    m = 0.5 * (a + b)
    delta = b - a

    ts = torch.linspace(eps1, 1.0 - eps2, n_time, device=x.device)
    values = []
    for t in ts:
        t_batch = torch.full((x.shape[0], 1), float(t), device=x.device)
        values.append(model(x, t_batch, m, delta).squeeze(-1))
    values = torch.stack(values, dim=0)
    return torch.trapz(values, ts, dim=0)


@torch.no_grad()
def estimate_log_ratio_torch(
    model: ToyFullTimeScoreNet,
    x: torch.Tensor,
    eps1: float = 1e-5,
    eps2: float = 1e-5,
    n_steps: int = 256,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Fixed-step torchdiffeq RK4 integration of scalar model(x, t) over t.

    If `device` is None, uses the device of model parameters.
    """
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1.")
    if eps1 < 0.0 or eps2 < 0.0:
        raise ValueError("eps1 and eps2 must be >= 0.")
    t0 = float(eps1)
    t1 = float(1.0 - eps2)
    if not t0 < t1:
        raise ValueError("Require eps1 < 1 - eps2 for a valid integration interval.")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = x.device

    x_dev = x.detach().to(device)
    step_size = (t1 - t0) / float(n_steps)
    t_span = torch.tensor([t0, t1], device=device, dtype=x_dev.dtype)
    y0 = torch.zeros((x_dev.shape[0],), device=device, dtype=x_dev.dtype)

    def ode_func(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _ = y
        t_batch = torch.full((x_dev.shape[0], 1), float(t), device=device, dtype=x_dev.dtype)
        return model(x_dev, t_batch).squeeze(-1)

    y_path = odeint(
        ode_func,
        y0,
        t_span,
        method="rk4",
        options={"step_size": step_size},
    )
    return y_path[-1].to(device=x.device, dtype=x.dtype)
