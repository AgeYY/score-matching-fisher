"""
CTSM-v loss (two-sample, full vector target) and log q/p estimation by integrating scalar time score.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import integrate

from fisher.ctsm_models import ToyFullTimeScoreNet
from fisher.ctsm_paths import TwoSB


def ctsm_v_two_sample_loss(
    model: ToyFullTimeScoreNet,
    prob_path: TwoSB,
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
def estimate_log_ratio_scipy(
    model: ToyFullTimeScoreNet,
    x: torch.Tensor,
    eps1: float = 1e-5,
    eps2: float = 1e-5,
    method: str = "RK45",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    scipy.integrate.solve_ivp analogue of density_ratios.get_toy_density_ratio_fn.

    If `device` is None, uses the device of model parameters.
    """
    if device is None:
        device = next(model.parameters()).device

    x_cpu = x.detach().cpu()

    def ode_func(t, y):
        t_batch = torch.full((x_cpu.shape[0], 1), float(t), dtype=x_cpu.dtype)
        out = model(x_cpu.to(device), t_batch.to(device)).squeeze(-1)
        return out.detach().cpu().numpy()

    solution = integrate.solve_ivp(
        ode_func,
        (eps1, 1.0 - eps2),
        np.zeros((x_cpu.shape[0],), dtype=np.float64),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return torch.tensor(solution.y[:, -1], dtype=x.dtype, device=x.device)
