#!/usr/bin/env python3
"""Conditional flow matching with a KNN diagonal Gaussian velocity prior.

This is a tests-only standalone experiment for the conditional toy problem
described in ``journal/notes/conditional_fm_knn_gaussian_velocity_prior_plan.md``.
All implementation code intentionally lives under ``tests/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except ImportError:  # pragma: no cover
    plt = None
    Ellipse = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATAROOT

try:
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, LinearVPScheduler, VPScheduler
    from flow_matching.solver.ode_solver import ODESolver
except ImportError as _e:  # pragma: no cover
    raise SystemExit(
        "The `flow_matching` package is required. Run inside the geo_diffusion environment:\n"
        "  mamba run -n geo_diffusion python tests/conditional_fm_knn_gaussian_velocity_prior.py --device cuda\n"
        f"Import error: {_e}"
    ) from _e


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class ExperimentConfig:
    rho: float
    beta: float
    train_size: int
    val_size: int
    test_size: int
    n_gen_per_theta: int
    train_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    depth: int
    time_frequencies: int
    theta_frequencies: int
    lambda_prior: float
    knn_k: int
    bandwidth_floor: float
    weighted_var_correction: bool
    variance_floor: float
    t_epsilon: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    ode_steps: int
    seed: int
    device: str
    scheduler: str


def default_output_dir() -> Path:
    return Path(DATAROOT) / "tests" / "conditional_fm_knn_gaussian_velocity_prior"


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=str, default=str(default_output_dir()))
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["condot", "cosine", "linear_vp", "vp"],
        help="Affine flow-matching scheduler used by data FM and the analytical Gaussian prior.",
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed; ignored when --seeds is set.")
    p.add_argument("--seeds", type=str, default="0", help="Whitespace/comma-separated seed list.")
    p.add_argument(
        "--train-sizes",
        type=str,
        default="16 32 128 512",
        help="Whitespace/comma-separated N list.",
    )
    p.add_argument("--lambda-priors", type=str, default="0 0.01", help="Whitespace/comma-separated lambda list.")
    p.add_argument(
        "--eval-thetas",
        type=str,
        default="0 0.7853981633974483 1.5707963267948966 3.141592653589793 4.71238898038469",
        help="Whitespace/comma-separated fixed theta values for conditional evaluation.",
    )
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument(
        "--val-size",
        type=int,
        default=32,
        help="Number of samples held out from each --train-sizes value for validation.",
    )
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--n-gen-per-theta", type=int, default=512)
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--time-frequencies", type=int, default=8)
    p.add_argument("--theta-frequencies", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=64)
    p.add_argument("--bandwidth-floor", type=float, default=1e-3)
    p.add_argument("--no-weighted-var-correction", action="store_true")
    p.add_argument("--variance-floor", type=float, default=1e-6)
    p.add_argument("--t-epsilon", type=float, default=1e-4)
    p.add_argument("--early-stopping-patience", type=int, default=1000)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument("--ode-steps", type=int, default=100)
    p.add_argument("--n-mmd", type=int, default=512)
    p.add_argument("--n-sliced", type=int, default=64)
    p.add_argument("--plot-max-points", type=int, default=1200)
    return p.parse_args()


def require_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable. Per AGENTS.md, not falling back to CPU.")
    return torch.device(device_name)


def set_seed(seed: int, device: torch.device) -> torch.Generator:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def make_scheduler(name: str) -> Any:
    scheduler_lookup = {
        "condot": CondOTScheduler,
        "cosine": CosineScheduler,
        "linear_vp": LinearVPScheduler,
        "vp": VPScheduler,
    }
    key = str(name).strip().lower()
    if key not in scheduler_lookup:
        supported = ", ".join(sorted(scheduler_lookup))
        raise ValueError(f"Unknown scheduler {name!r}. Supported: {supported}.")
    return scheduler_lookup[key]()


def circular_distance(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(theta_a - theta_b + math.pi, TWO_PI) - math.pi
    return diff.abs()


def conditional_mean(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.reshape(-1)
    return torch.stack([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=-1)


def conditional_scales(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta = theta.reshape(-1)
    sigma1 = 0.4 + 0.2 * torch.sin(theta)
    sigma2 = 0.3 + 0.1 * torch.cos(theta)
    return sigma1, sigma2


def conditional_latent_cov(
    theta: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    sigma1, sigma2 = conditional_scales(theta)
    cov = torch.zeros((theta.reshape(-1).shape[0], 2, 2), device=theta.device, dtype=theta.dtype)
    cov[:, 0, 0] = sigma1.square()
    cov[:, 1, 1] = sigma2.square()
    cov[:, 0, 1] = float(rho) * sigma1 * sigma2
    cov[:, 1, 0] = cov[:, 0, 1]
    return cov


def sample_conditional_banana_given_theta(
    theta: torch.Tensor,
    *,
    rho: float,
    beta: float,
    generator: torch.Generator,
) -> torch.Tensor:
    theta = theta.reshape(-1)
    mean = conditional_mean(theta)
    cov = conditional_latent_cov(theta, rho=float(rho))
    chol = torch.linalg.cholesky(cov)
    eps = torch.randn((theta.shape[0], 2), generator=generator, device=theta.device, dtype=theta.dtype)
    z = torch.einsum("bij,bj->bi", chol, eps)
    sigma1, _ = conditional_scales(theta)
    x = mean + z
    x[:, 1] = x[:, 1] + float(beta) * (z[:, 0].square() - sigma1.square())
    return x


def sample_conditional_banana(
    n: int,
    *,
    rho: float,
    beta: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = TWO_PI * torch.rand((int(n),), generator=generator, device=device, dtype=dtype)
    x = sample_conditional_banana_given_theta(theta, rho=rho, beta=beta, generator=generator)
    return x, theta


def true_conditional_cov(theta: torch.Tensor, *, rho: float, beta: float) -> torch.Tensor:
    theta = theta.reshape(-1)
    sigma1, sigma2 = conditional_scales(theta)
    cov = torch.zeros((theta.shape[0], 2, 2), device=theta.device, dtype=theta.dtype)
    cov[:, 0, 0] = sigma1.square()
    cov[:, 0, 1] = float(rho) * sigma1 * sigma2
    cov[:, 1, 0] = cov[:, 0, 1]
    cov[:, 1, 1] = sigma2.square() + 2.0 * float(beta) ** 2 * sigma1.pow(4)
    return cov


class KnnDiagGaussianConditionalPrior:
    def __init__(
        self,
        theta_train: torch.Tensor,
        x_train: torch.Tensor,
        *,
        k: int,
        bandwidth_floor: float = 1e-3,
        variance_floor: float = 1e-6,
        weighted_var_correction: bool = True,
    ) -> None:
        if theta_train.ndim != 1:
            theta_train = theta_train.reshape(-1)
        if theta_train.shape[0] != x_train.shape[0]:
            raise ValueError("theta_train and x_train must have the same leading dimension.")
        self.theta_train = theta_train
        self.x_train = x_train
        self.k = max(1, min(int(k), int(theta_train.shape[0])))
        self.bandwidth_floor = float(bandwidth_floor)
        self.variance_floor = float(variance_floor)
        self.weighted_var_correction = bool(weighted_var_correction)

    def query(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = theta.reshape(-1).to(device=self.theta_train.device, dtype=self.theta_train.dtype)
        d = circular_distance(theta[:, None], self.theta_train[None, :])
        neighbor_d, neighbor_idx = torch.topk(d, k=self.k, dim=1, largest=False, sorted=True)
        h = torch.clamp(neighbor_d[:, -1:], min=self.bandwidth_floor)
        weights = torch.exp(-0.5 * (neighbor_d / h).square())
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        x_neighbors = self.x_train[neighbor_idx]
        mu = torch.sum(weights[..., None] * x_neighbors, dim=1)
        centered = x_neighbors - mu[:, None, :]
        var = torch.sum(weights[..., None] * centered.square(), dim=1)
        if self.weighted_var_correction:
            denom = torch.clamp(1.0 - weights.square().sum(dim=1, keepdim=True), min=1e-6)
            var = var / denom
        var = torch.clamp(var, min=self.variance_floor)
        return mu, var


def analytical_gaussian_prior_velocity(
    x: torch.Tensor,
    t: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    scheduler: Any,
) -> torch.Tensor:
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    schedule = scheduler(t)
    alpha = schedule.alpha_t
    sigma = schedule.sigma_t
    d_alpha = schedule.d_alpha_t
    d_sigma = schedule.d_sigma_t
    mu = mu.to(device=x.device, dtype=x.dtype)
    var = var.to(device=x.device, dtype=x.dtype)
    denom = sigma.square() + alpha.square() * var
    gain = (sigma * d_sigma + alpha * d_alpha * var) / denom
    return d_alpha * mu + gain * (x - alpha * mu)


def sample_gaussian_prior_path(
    t: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    scheduler: Any,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    schedule = scheduler(t)
    alpha = schedule.alpha_t
    sigma = schedule.sigma_t
    mu = mu.to(device=t.device, dtype=t.dtype)
    var = var.to(device=t.device, dtype=t.dtype)
    std_t = torch.sqrt(sigma.square() + alpha.square() * var)
    noise = torch.randn(mu.shape, generator=generator, device=t.device, dtype=t.dtype)
    return alpha * mu + std_t * noise


class ConditionalVelocityMLP(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
        time_frequencies: int = 8,
        theta_frequencies: int = 4,
    ) -> None:
        super().__init__()
        self.time_frequencies = int(time_frequencies)
        self.theta_frequencies = int(theta_frequencies)
        if self.time_frequencies > 0:
            t_freq = torch.logspace(0.0, math.log10(1000.0), self.time_frequencies)
        else:
            t_freq = torch.empty(0)
        if self.theta_frequencies > 0:
            theta_freq = torch.arange(1, self.theta_frequencies + 1, dtype=torch.float32)
        else:
            theta_freq = torch.empty(0)
        self.register_buffer("t_freq", t_freq)
        self.register_buffer("theta_freq", theta_freq)
        in_dim = int(x_dim) + 1 + 2 * self.time_frequencies + 2 * self.theta_frequencies
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.SiLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(x_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        theta = theta.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
        if self.time_frequencies > 0:
            phase = 2.0 * math.pi * t * self.t_freq.reshape(1, -1)
            t_feat = torch.cat([t, torch.sin(phase), torch.cos(phase)], dim=-1)
        else:
            t_feat = t
        if self.theta_frequencies > 0:
            theta_phase = theta * self.theta_freq.reshape(1, -1)
            theta_feat = torch.cat([torch.sin(theta_phase), torch.cos(theta_phase)], dim=-1)
        else:
            theta_feat = theta.new_empty((theta.shape[0], 0))
        return self.net(torch.cat([x, t_feat, theta_feat], dim=-1))


class ConditionalSolverVelocityWrapper(nn.Module):
    def __init__(self, model: nn.Module, theta: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("theta", theta.reshape(-1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_: Any) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0]).unsqueeze(-1)
        elif t.ndim == 1:
            if t.numel() == 1:
                t = t.expand(x.shape[0]).unsqueeze(-1)
            else:
                t = t.unsqueeze(-1)
        theta = self.theta
        if theta.numel() == 1:
            theta = theta.expand(x.shape[0])
        return self.model(x, t, theta)


@torch.no_grad()
def validation_fm_loss(
    model: nn.Module,
    path: AffineProbPath,
    x_val: torch.Tensor,
    theta_val: torch.Tensor,
    x0_val: torch.Tensor,
    t_val: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    n = x_val.shape[0]
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        sample = path.sample(x_0=x0_val[start:end], x_1=x_val[start:end], t=t_val[start:end])
        pred = model(sample.x_t, sample.t, theta_val[start:end])
        chunk_loss = F.mse_loss(pred, sample.dx_t, reduction="mean")
        losses.append(chunk_loss * float(end - start))
    return torch.stack(losses).sum() / float(n)


def train_one_model(
    x_train: torch.Tensor,
    theta_train: torch.Tensor,
    x_val: torch.Tensor,
    theta_val: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    prior: KnnDiagGaussianConditionalPrior,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[ConditionalVelocityMLP, list[dict[str, float]], dict[str, Any]]:
    model = ConditionalVelocityMLP(
        x_dim=x_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        time_frequencies=cfg.time_frequencies,
        theta_frequencies=cfg.theta_frequencies,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(cfg.scheduler)
    path = AffineProbPath(scheduler=scheduler)
    logs: list[dict[str, float]] = []
    n = x_train.shape[0]
    batch_size = min(int(cfg.batch_size), n)
    val_batch_size = min(int(cfg.batch_size), int(x_val.shape[0]))
    x0_val = torch.randn(x_val.shape, generator=generator, device=device, dtype=x_val.dtype)
    t_val = torch.rand((x_val.shape[0],), generator=generator, device=device, dtype=x_val.dtype)
    t_val = cfg.t_epsilon + (1.0 - 2.0 * cfg.t_epsilon) * t_val
    best_val_loss = float("inf")
    best_step = 0
    stopped_step = int(cfg.train_steps)
    patience_counter = 0
    early_stopped = False
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for step in range(1, int(cfg.train_steps) + 1):
        idx = torch.randint(n, (batch_size,), generator=generator, device=device)
        x1 = x_train[idx]
        theta = theta_train[idx]
        x0 = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
        t = torch.rand((batch_size,), generator=generator, device=device)
        t = cfg.t_epsilon + (1.0 - 2.0 * cfg.t_epsilon) * t
        sample = path.sample(x_0=x0, x_1=x1, t=t)
        pred = model(sample.x_t, sample.t, theta)
        fm_loss = F.mse_loss(pred, sample.dx_t, reduction="mean")

        prior_loss = torch.zeros((), device=device)
        if cfg.lambda_prior > 0.0:
            t_prior = torch.rand((batch_size, 1), generator=generator, device=device)
            t_prior = cfg.t_epsilon + (1.0 - 2.0 * cfg.t_epsilon) * t_prior
            prior_mu, prior_var = prior.query(theta)
            x_prior_t = sample_gaussian_prior_path(t_prior, prior_mu, prior_var, scheduler, generator=generator)
            target_prior = analytical_gaussian_prior_velocity(x_prior_t, t_prior, prior_mu, prior_var, scheduler)
            prior_loss = F.mse_loss(model(x_prior_t, t_prior, theta), target_prior, reduction="mean")

        loss = fm_loss + float(cfg.lambda_prior) * prior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        val_loss_t = validation_fm_loss(
            model,
            path,
            x_val,
            theta_val,
            x0_val,
            t_val,
            batch_size=val_batch_size,
        )
        val_loss = float(val_loss_t.detach().cpu().item())
        if val_loss < best_val_loss - float(cfg.early_stopping_min_delta):
            best_val_loss = val_loss
            best_step = step
            patience_counter = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1

        should_log = step == 1 or step == cfg.train_steps or step % max(1, cfg.train_steps // 20) == 0
        if should_log:
            logs.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu().item()),
                    "fm_loss": float(fm_loss.detach().cpu().item()),
                    "prior_loss": float(prior_loss.detach().cpu().item()),
                    "val_fm_loss": val_loss,
                    "best_val_fm_loss": float(best_val_loss),
                    "patience_counter": float(patience_counter),
                }
            )
        if patience_counter >= int(cfg.early_stopping_patience):
            stopped_step = step
            early_stopped = True
            if not should_log:
                logs.append(
                    {
                        "step": float(step),
                        "loss": float(loss.detach().cpu().item()),
                        "fm_loss": float(fm_loss.detach().cpu().item()),
                        "prior_loss": float(prior_loss.detach().cpu().item()),
                        "val_fm_loss": val_loss,
                        "best_val_fm_loss": float(best_val_loss),
                        "patience_counter": float(patience_counter),
                    }
                )
            break
    model.load_state_dict(best_state)
    training_summary = {
        "best_step": int(best_step),
        "stopped_step": int(stopped_step),
        "early_stopped": bool(early_stopped),
        "best_val_fm_loss": float(best_val_loss),
    }
    return model, logs, training_summary


@torch.no_grad()
def sample_model_at_theta(
    model: nn.Module,
    theta_value: float,
    n: int,
    ode_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    theta = torch.full((int(n),), float(theta_value), device=device)
    solver = ODESolver(velocity_model=ConditionalSolverVelocityWrapper(model, theta))
    x0 = torch.randn((int(n), 2), generator=generator, device=device)
    time_grid = torch.linspace(0.0, 1.0, int(ode_steps) + 1, device=device)
    out = solver.sample(x_init=x0, step_size=None, method="euler", time_grid=time_grid)
    return out[-1] if isinstance(out, (list, tuple)) else out


def covariance_np(x: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(x, dtype=np.float64), rowvar=False)


def conditional_banana_quadratic_coef_np(x: np.ndarray, theta_value: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    m1 = 2.0 * math.cos(float(theta_value))
    m2 = 2.0 * math.sin(float(theta_value))
    sigma1 = 0.4 + 0.2 * math.sin(float(theta_value))
    x1c = x[:, 0] - m1
    y = x[:, 1] - m2
    design = np.stack([np.ones(x.shape[0]), x1c, x1c**2 - sigma1**2], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[2])


def rbf_mmd2_np(x: np.ndarray, y: np.ndarray, *, max_n: int = 1024) -> float:
    rng = np.random.default_rng(12345)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]
    z = np.concatenate([x, y], axis=0)
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    positive = d2[d2 > 0.0]
    med = np.median(positive) if positive.size else 1.0
    gamma = 1.0 / max(float(med), 1e-8)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def sliced_wasserstein_np(x: np.ndarray, y: np.ndarray, *, n_proj: int = 64) -> float:
    rng = np.random.default_rng(54321)
    m = min(x.shape[0], y.shape[0])
    if x.shape[0] > m:
        x = x[rng.choice(x.shape[0], m, replace=False)]
    if y.shape[0] > m:
        y = y[rng.choice(y.shape[0], m, replace=False)]
    dirs = rng.normal(size=(int(n_proj), x.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = []
    for direction in dirs:
        px = np.sort(x @ direction)
        py = np.sort(y @ direction)
        vals.append(np.mean(np.abs(px - py)))
    return float(np.mean(vals))


def energy_distance_np(x: np.ndarray, y: np.ndarray, *, max_n: int = 1024) -> float:
    rng = np.random.default_rng(24680)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]
    xy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    xx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean()
    yy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1).mean()
    return float(2.0 * xy - xx - yy)


def evaluate_at_theta(
    generated: np.ndarray,
    true_samples: np.ndarray,
    theta_value: float,
    *,
    rho: float,
    beta: float,
    n_mmd: int,
    n_sliced: int,
) -> dict[str, float]:
    theta_t = torch.tensor([float(theta_value)], dtype=torch.float64)
    true_mean = conditional_mean(theta_t).cpu().numpy()[0]
    true_cov = true_conditional_cov(theta_t, rho=rho, beta=beta).cpu().numpy()[0]
    cov_gen = covariance_np(generated)
    corr_true = true_cov[0, 1] / math.sqrt(true_cov[0, 0] * true_cov[1, 1])
    corr_gen = cov_gen[0, 1] / math.sqrt(max(cov_gen[0, 0] * cov_gen[1, 1], 1e-12))
    return {
        "mean_error": float(np.linalg.norm(generated.mean(axis=0) - true_mean)),
        "cov_fro_error": float(np.linalg.norm(cov_gen - true_cov)),
        "corr_error": float(abs(corr_gen - corr_true)),
        "banana_coef_error": float(abs(conditional_banana_quadratic_coef_np(generated, theta_value) - float(beta))),
        "mmd2": rbf_mmd2_np(generated, true_samples, max_n=int(n_mmd)),
        "sliced_wasserstein": sliced_wasserstein_np(generated, true_samples, n_proj=int(n_sliced)),
        "energy_distance": energy_distance_np(generated, true_samples, max_n=int(n_mmd)),
    }


def average_metrics(per_theta: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(per_theta[0].keys()) if per_theta else []
    return {f"avg_{key}": float(np.mean([row[key] for row in per_theta])) for key in keys}


def save_conditional_figure(
    path: Path,
    eval_thetas: list[float],
    true_by_theta: dict[float, np.ndarray],
    generated_by_lambda: dict[float, dict[float, np.ndarray]],
    metrics_by_lambda: dict[float, dict[float, dict[str, float]]],
    prior_mu_by_theta: dict[float, np.ndarray],
    prior_var_by_theta: dict[float, np.ndarray],
    max_points: int,
) -> None:
    if plt is None or Ellipse is None:
        return
    lambdas = sorted(generated_by_lambda)
    nrows = len(eval_thetas)
    ncols = 1 + len(lambdas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
    rng = np.random.default_rng(777)

    def _sub(x: np.ndarray) -> np.ndarray:
        if x.shape[0] <= max_points:
            return x
        return x[rng.choice(x.shape[0], int(max_points), replace=False)]

    def _add_prior(ax: Any, theta_value: float) -> None:
        mu = prior_mu_by_theta[theta_value]
        var = prior_var_by_theta[theta_value]
        ell = Ellipse(
            xy=(float(mu[0]), float(mu[1])),
            width=4.0 * math.sqrt(float(var[0])),
            height=4.0 * math.sqrt(float(var[1])),
            angle=0.0,
            fill=False,
            lw=1.2,
            color="black",
            alpha=0.8,
        )
        ax.add_patch(ell)

    for row_idx, theta_value in enumerate(eval_thetas):
        true_s = _sub(true_by_theta[theta_value])
        axes[row_idx, 0].scatter(true_s[:, 0], true_s[:, 1], s=4, alpha=0.45)
        axes[row_idx, 0].set_title(f"true theta={theta_value:.3g}")
        _add_prior(axes[row_idx, 0], theta_value)
        for col_idx, lam in enumerate(lambdas, start=1):
            arr_s = _sub(generated_by_lambda[lam][theta_value])
            axes[row_idx, col_idx].scatter(arr_s[:, 0], arr_s[:, 1], s=4, alpha=0.45)
            metrics = metrics_by_lambda[lam][theta_value]
            axes[row_idx, col_idx].set_title(
                f"lambda={lam:g}\n"
                f"MMD2={metrics['mmd2']:.3g}, SW={metrics['sliced_wasserstein']:.3g}"
            )
            _add_prior(axes[row_idx, col_idx], theta_value)
    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_single_setting(
    cfg: ExperimentConfig,
    *,
    eval_thetas: list[float],
    output_dir: Path,
    n_mmd: int,
    n_sliced: int,
) -> dict[str, Any]:
    device = require_device(cfg.device)
    g = set_seed(cfg.seed, device)
    if cfg.train_size < 2:
        raise ValueError("--train-sizes entries must be at least 2 when using a validation holdout.")
    # Cap val at ~half the data so small n (e.g. 16) still leave enough points to train.
    val_size = max(1, min(int(cfg.val_size), int(cfg.train_size) - 1, int(cfg.train_size) // 2))
    fit_size = int(cfg.train_size) - val_size
    x_all, theta_all = sample_conditional_banana(
        cfg.train_size,
        rho=cfg.rho,
        beta=cfg.beta,
        generator=g,
        device=device,
    )
    x_train, theta_train = x_all[:fit_size], theta_all[:fit_size]
    x_val, theta_val = x_all[fit_size:], theta_all[fit_size:]
    prior = KnnDiagGaussianConditionalPrior(
        theta_train,
        x_train,
        k=cfg.knn_k,
        bandwidth_floor=cfg.bandwidth_floor,
        variance_floor=cfg.variance_floor,
        weighted_var_correction=cfg.weighted_var_correction,
    )
    model, train_logs, training_summary = train_one_model(
        x_train,
        theta_train,
        x_val,
        theta_val,
        cfg,
        prior=prior,
        generator=g,
        device=device,
    )

    true_samples: list[np.ndarray] = []
    generated_samples: list[np.ndarray] = []
    prior_mus: list[np.ndarray] = []
    prior_vars: list[np.ndarray] = []
    per_theta_metrics: list[dict[str, float]] = []
    for theta_value in eval_thetas:
        theta_eval = torch.full((cfg.test_size,), float(theta_value), device=device)
        x_true = sample_conditional_banana_given_theta(theta_eval, rho=cfg.rho, beta=cfg.beta, generator=g)
        x_gen = sample_model_at_theta(model, theta_value, cfg.n_gen_per_theta, cfg.ode_steps, g, device)
        prior_mu, prior_var = prior.query(torch.tensor([float(theta_value)], device=device))

        true_np = x_true.detach().cpu().numpy()
        gen_np = x_gen.detach().cpu().numpy()
        true_samples.append(true_np)
        generated_samples.append(gen_np)
        prior_mus.append(prior_mu.detach().cpu().numpy()[0])
        prior_vars.append(prior_var.detach().cpu().numpy()[0])
        theta_metrics = evaluate_at_theta(
            gen_np,
            true_np,
            theta_value,
            rho=cfg.rho,
            beta=cfg.beta,
            n_mmd=n_mmd,
            n_sliced=n_sliced,
        )
        theta_metrics["theta"] = float(theta_value)
        per_theta_metrics.append(theta_metrics)

    avg_metrics = average_metrics([{k: v for k, v in row.items() if k != "theta"} for row in per_theta_metrics])
    run_dir = output_dir / f"n_{cfg.train_size}" / f"seed_{cfg.seed}" / f"lambda_{cfg.lambda_prior:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "samples.npz",
        train=x_train.detach().cpu().numpy(),
        theta_train=theta_train.detach().cpu().numpy(),
        val=x_val.detach().cpu().numpy(),
        theta_val=theta_val.detach().cpu().numpy(),
        observed=x_all.detach().cpu().numpy(),
        theta_observed=theta_all.detach().cpu().numpy(),
        eval_thetas=np.asarray(eval_thetas, dtype=np.float64),
        true_samples=np.stack(true_samples, axis=0),
        generated_samples=np.stack(generated_samples, axis=0),
        prior_mu=np.stack(prior_mus, axis=0),
        prior_var=np.stack(prior_vars, axis=0),
    )
    payload = {
        "config": asdict(cfg),
        "metrics": avg_metrics,
        "per_theta_metrics": per_theta_metrics,
        "train_logs": train_logs,
        "training_summary": training_summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "metrics": avg_metrics,
        "per_theta_metrics": per_theta_metrics,
        "train_logs": train_logs,
        "training_summary": training_summary,
    }


def write_aggregate(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    metric_keys = sorted(rows[0]["metrics"].keys()) if rows else []
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["train_size", "seed", "lambda_prior", "run_dir", *metric_keys],
        )
        writer.writeheader()
        for row in rows:
            cfg = row["config"]
            writer.writerow(
                {
                    "train_size": cfg["train_size"],
                    "seed": cfg["seed"],
                    "lambda_prior": cfg["lambda_prior"],
                    "run_dir": row["run_dir"],
                    **row["metrics"],
                }
            )
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds) if str(args.seeds).strip() else [int(args.seed)]
    train_sizes = parse_int_list(args.train_sizes)
    lambda_priors = parse_float_list(args.lambda_priors)
    eval_thetas = parse_float_list(args.eval_thetas)

    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        for seed in seeds:
            generated_by_lambda: dict[float, dict[float, np.ndarray]] = {}
            metrics_by_lambda: dict[float, dict[float, dict[str, float]]] = {}
            true_by_theta: dict[float, np.ndarray] | None = None
            prior_mu_by_theta: dict[float, np.ndarray] | None = None
            prior_var_by_theta: dict[float, np.ndarray] | None = None
            for lam in lambda_priors:
                cfg = ExperimentConfig(
                    rho=float(args.rho),
                    beta=float(args.beta),
                    train_size=int(train_size),
                    val_size=int(args.val_size),
                    test_size=int(args.test_size),
                    n_gen_per_theta=int(args.n_gen_per_theta),
                    train_steps=int(args.train_steps),
                    batch_size=int(args.batch_size),
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                    hidden_dim=int(args.hidden_dim),
                    depth=int(args.depth),
                    time_frequencies=int(args.time_frequencies),
                    theta_frequencies=int(args.theta_frequencies),
                    lambda_prior=float(lam),
                    knn_k=int(args.knn_k),
                    bandwidth_floor=float(args.bandwidth_floor),
                    weighted_var_correction=not bool(args.no_weighted_var_correction),
                    variance_floor=float(args.variance_floor),
                    t_epsilon=float(args.t_epsilon),
                    early_stopping_patience=int(args.early_stopping_patience),
                    early_stopping_min_delta=float(args.early_stopping_min_delta),
                    ode_steps=int(args.ode_steps),
                    seed=int(seed),
                    device=str(args.device),
                    scheduler=str(args.scheduler),
                )
                row = run_single_setting(
                    cfg,
                    eval_thetas=eval_thetas,
                    output_dir=output_dir,
                    n_mmd=int(args.n_mmd),
                    n_sliced=int(args.n_sliced),
                )
                rows.append(row)
                samples = np.load(Path(row["run_dir"]) / "samples.npz")
                gen_map = {
                    float(theta): np.asarray(samples["generated_samples"][i], dtype=np.float64)
                    for i, theta in enumerate(samples["eval_thetas"])
                }
                generated_by_lambda[float(lam)] = gen_map
                metrics_by_lambda[float(lam)] = {
                    float(metric_row["theta"]): dict(metric_row) for metric_row in row["per_theta_metrics"]
                }
                if true_by_theta is None:
                    true_by_theta = {
                        float(theta): np.asarray(samples["true_samples"][i], dtype=np.float64)
                        for i, theta in enumerate(samples["eval_thetas"])
                    }
                    prior_mu_by_theta = {
                        float(theta): np.asarray(samples["prior_mu"][i], dtype=np.float64)
                        for i, theta in enumerate(samples["eval_thetas"])
                    }
                    prior_var_by_theta = {
                        float(theta): np.asarray(samples["prior_var"][i], dtype=np.float64)
                        for i, theta in enumerate(samples["eval_thetas"])
                    }
                print(
                    "[conditional_fm_knn_gaussian_velocity_prior] "
                    f"n={train_size} seed={seed} lambda={lam:g} "
                    f"avg_mmd2={row['metrics']['avg_mmd2']:.4g} "
                    f"avg_sw={row['metrics']['avg_sliced_wasserstein']:.4g} "
                    f"best_step={row['training_summary']['best_step']} "
                    f"stopped_step={row['training_summary']['stopped_step']} "
                    f"run_dir={row['run_dir']}",
                    flush=True,
                )
            if true_by_theta is not None and prior_mu_by_theta is not None and prior_var_by_theta is not None:
                fig_path = output_dir / f"conditional_samples_n_{train_size}_seed_{seed}.png"
                save_conditional_figure(
                    fig_path,
                    eval_thetas,
                    true_by_theta,
                    generated_by_lambda,
                    metrics_by_lambda,
                    prior_mu_by_theta,
                    prior_var_by_theta,
                    int(args.plot_max_points),
                )

    write_aggregate(output_dir, rows)
    print(f"[conditional_fm_knn_gaussian_velocity_prior] Wrote outputs under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
