#!/usr/bin/env python3
"""Flow matching with an analytical diagonal Gaussian velocity prior.

This is a tests-only standalone experiment for the 2D correlated banana
benchmark described in ``journal/notes/toy_experiment_plan_fm_gaussian_velocity_prior.md``.
It intentionally keeps all implementation code under ``tests/``.
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
except ImportError:  # pragma: no cover
    plt = None

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
        "  mamba run -n geo_diffusion python tests/fm_gaussian_velocity_prior.py --device cuda\n"
        f"Import error: {_e}"
    ) from _e


@dataclass(frozen=True)
class ExperimentConfig:
    rho: float
    beta: float
    train_size: int
    test_size: int
    n_gen: int
    train_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    depth: int
    time_frequencies: int
    lambda_prior: float
    variance_floor: float
    t_epsilon: float
    ode_steps: int
    seed: int
    device: str
    scheduler: str


def default_output_dir() -> Path:
    return Path(DATAROOT) / "tests" / "fm_gaussian_velocity_prior"


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
        help="Affine flow-matching scheduler used by both data FM and the analytical Gaussian prior.",
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed; ignored when --seeds is set.")
    p.add_argument("--seeds", type=str, default="0", help="Whitespace/comma-separated seed list.")
    p.add_argument("--train-sizes", type=str, default="64 256", help="Whitespace/comma-separated N list.")
    p.add_argument("--lambda-priors", type=str, default="0 0.1", help="Whitespace/comma-separated lambda list.")
    p.add_argument("--rho", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--n-gen", type=int, default=1024)
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--time-frequencies", type=int, default=8)
    p.add_argument("--variance-floor", type=float, default=1e-6)
    p.add_argument("--t-epsilon", type=float, default=1e-4)
    p.add_argument("--ode-steps", type=int, default=100)
    p.add_argument("--n-mmd", type=int, default=1024)
    p.add_argument("--n-sliced", type=int, default=64)
    p.add_argument("--plot-max-points", type=int, default=3000)
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


def banana_latent_cov(rho: float, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor([[1.0, float(rho)], [float(rho), 1.0]], device=device, dtype=dtype)


def sample_banana(
    n: int,
    *,
    rho: float,
    beta: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cov = banana_latent_cov(rho, device=device, dtype=dtype)
    chol = torch.linalg.cholesky(cov)
    z = torch.randn((int(n), 2), generator=generator, device=device, dtype=dtype) @ chol.T
    x = z.clone()
    x[:, 1] = z[:, 1] + float(beta) * (z[:, 0].square() - 1.0)
    return x


def banana_inverse(x: torch.Tensor, beta: float) -> torch.Tensor:
    z = x.clone()
    z[:, 1] = x[:, 1] - float(beta) * (x[:, 0].square() - 1.0)
    return z


def banana_log_prob(x: torch.Tensor, *, rho: float, beta: float) -> torch.Tensor:
    z = banana_inverse(x, beta=beta)
    cov = banana_latent_cov(rho, device=x.device, dtype=x.dtype)
    inv_cov = torch.linalg.inv(cov)
    q = torch.einsum("bi,ij,bj->b", z, inv_cov, z)
    log_det = torch.logdet(cov)
    return -0.5 * (q + 2.0 * math.log(2.0 * math.pi) + log_det)


def estimate_diag_gaussian_prior(x_train: torch.Tensor, variance_floor: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mu = x_train.mean(dim=0)
    var = x_train.var(dim=0, unbiased=True)
    var = torch.clamp(var, min=float(variance_floor))
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
    mu_b = mu.reshape(1, -1).to(device=x.device, dtype=x.dtype)
    var_b = var.reshape(1, -1).to(device=x.device, dtype=x.dtype)
    denom = sigma.square() + alpha.square() * var_b
    gain = (sigma * d_sigma + alpha * d_alpha * var_b) / denom
    return d_alpha * mu_b + gain * (x - alpha * mu_b)


def sample_gaussian_prior_path(
    n: int,
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
    mu_b = mu.reshape(1, -1).to(device=t.device, dtype=t.dtype)
    var_b = var.reshape(1, -1).to(device=t.device, dtype=t.dtype)
    std_t = torch.sqrt(sigma.square() + alpha.square() * var_b)
    noise = torch.randn((int(n), mu_b.shape[1]), generator=generator, device=t.device, dtype=t.dtype)
    return alpha * mu_b + std_t * noise


class VelocityMLP(nn.Module):
    def __init__(self, x_dim: int = 2, hidden_dim: int = 128, depth: int = 3, time_frequencies: int = 8) -> None:
        super().__init__()
        self.time_frequencies = int(time_frequencies)
        if self.time_frequencies > 0:
            freq = torch.logspace(0.0, math.log10(1000.0), self.time_frequencies)
        else:
            freq = torch.empty(0)
        self.register_buffer("freq", freq)
        in_dim = int(x_dim) + 1 + 2 * self.time_frequencies
        layers: list[nn.Module] = []
        for _ in range(int(depth)):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.SiLU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(x_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.time_frequencies > 0:
            phase = 2.0 * math.pi * t * self.freq.reshape(1, -1)
            t_feat = torch.cat([t, torch.sin(phase), torch.cos(phase)], dim=-1)
        else:
            t_feat = t
        return self.net(torch.cat([x, t_feat], dim=-1))


class SolverVelocityWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_: Any) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0]).unsqueeze(-1)
        elif t.ndim == 1:
            if t.numel() == 1:
                t = t.expand(x.shape[0]).unsqueeze(-1)
            else:
                t = t.unsqueeze(-1)
        return self.model(x, t)


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    return -0.5 * (flat.square().sum(dim=1) + flat.shape[1] * math.log(2.0 * math.pi))


def train_one_model(
    x_train: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    prior_mu: torch.Tensor,
    prior_var: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[VelocityMLP, list[dict[str, float]]]:
    model = VelocityMLP(
        x_dim=x_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        time_frequencies=cfg.time_frequencies,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(cfg.scheduler)
    path = AffineProbPath(scheduler=scheduler)
    logs: list[dict[str, float]] = []
    n = x_train.shape[0]
    batch_size = min(int(cfg.batch_size), n)
    for step in range(1, int(cfg.train_steps) + 1):
        idx = torch.randint(n, (batch_size,), generator=generator, device=device)
        x1 = x_train[idx]
        x0 = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
        t = torch.rand((batch_size,), generator=generator, device=device)
        t = cfg.t_epsilon + (1.0 - 2.0 * cfg.t_epsilon) * t
        sample = path.sample(x_0=x0, x_1=x1, t=t)
        pred = model(sample.x_t, sample.t)
        fm_loss = F.mse_loss(pred, sample.dx_t, reduction="mean")

        prior_loss = torch.zeros((), device=device)
        if cfg.lambda_prior > 0.0:
            t_prior = torch.rand((batch_size, 1), generator=generator, device=device)
            t_prior = cfg.t_epsilon + (1.0 - 2.0 * cfg.t_epsilon) * t_prior
            x_prior_t = sample_gaussian_prior_path(
                batch_size,
                t_prior,
                prior_mu,
                prior_var,
                scheduler,
                generator=generator,
            )
            target_prior = analytical_gaussian_prior_velocity(x_prior_t, t_prior, prior_mu, prior_var, scheduler)
            prior_loss = F.mse_loss(model(x_prior_t, t_prior), target_prior, reduction="mean")

        loss = fm_loss + float(cfg.lambda_prior) * prior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == cfg.train_steps or step % max(1, cfg.train_steps // 20) == 0:
            logs.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu().item()),
                    "fm_loss": float(fm_loss.detach().cpu().item()),
                    "prior_loss": float(prior_loss.detach().cpu().item()),
                }
            )
    return model, logs


@torch.no_grad()
def sample_model(model: nn.Module, n: int, ode_steps: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    solver = ODESolver(velocity_model=SolverVelocityWrapper(model))
    x0 = torch.randn((int(n), 2), generator=generator, device=device)
    time_grid = torch.linspace(0.0, 1.0, int(ode_steps) + 1, device=device)
    out = solver.sample(x_init=x0, step_size=None, method="euler", time_grid=time_grid)
    return out[-1] if isinstance(out, (list, tuple)) else out


def model_log_prob(model: nn.Module, x: torch.Tensor, ode_steps: int) -> torch.Tensor:
    solver = ODESolver(velocity_model=SolverVelocityWrapper(model))
    time_grid = torch.linspace(1.0, 0.0, int(ode_steps) + 1, device=x.device)
    _, logp = solver.compute_likelihood(
        x_1=x,
        log_p0=standard_normal_log_prob,
        step_size=None,
        method="euler",
        time_grid=time_grid,
        exact_divergence=True,
    )
    return logp


def covariance_np(x: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(x, dtype=np.float64), rowvar=False)


def banana_quadratic_coef_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    design = np.stack([np.ones(x.shape[0]), x[:, 0], x[:, 0] ** 2 - 1.0], axis=1)
    coef, *_ = np.linalg.lstsq(design, x[:, 1], rcond=None)
    return float(coef[2])


def rbf_mmd2_np(x: np.ndarray, y: np.ndarray, *, max_n: int = 2048) -> float:
    rng = np.random.default_rng(12345)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]
    z = np.concatenate([x, y], axis=0)
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0.0])
    gamma = 1.0 / max(float(med), 1e-8)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def sliced_wasserstein_np(x: np.ndarray, y: np.ndarray, *, n_proj: int = 128) -> float:
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


def evaluate_model(
    model: nn.Module,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    x_gen: torch.Tensor,
    *,
    cfg: ExperimentConfig,
    n_mmd: int,
    n_sliced: int,
) -> dict[str, float]:
    train_lp = model_log_prob(model, x_train, cfg.ode_steps).detach()
    test_lp = model_log_prob(model, x_test, cfg.ode_steps).detach()
    true_test_lp = banana_log_prob(x_test, rho=cfg.rho, beta=cfg.beta)

    train_np = x_train.detach().cpu().numpy()
    test_np = x_test.detach().cpu().numpy()
    gen_np = x_gen.detach().cpu().numpy()
    cov_test = covariance_np(test_np)
    cov_gen = covariance_np(gen_np)
    corr_test = cov_test[0, 1] / math.sqrt(cov_test[0, 0] * cov_test[1, 1])
    corr_gen = cov_gen[0, 1] / math.sqrt(cov_gen[0, 0] * cov_gen[1, 1])
    return {
        "train_logp_mean": float(train_lp.mean().cpu().item()),
        "test_logp_mean": float(test_lp.mean().cpu().item()),
        "true_test_logp_mean": float(true_test_lp.mean().cpu().item()),
        "delta_nll": float((-test_lp.mean() + true_test_lp.mean()).cpu().item()),
        "train_test_nll_gap": float((train_lp.mean() - test_lp.mean()).cpu().item()),
        "mmd2": rbf_mmd2_np(gen_np, test_np, max_n=int(n_mmd)),
        "sliced_wasserstein": sliced_wasserstein_np(gen_np, test_np, n_proj=int(n_sliced)),
        "mean_error": float(np.linalg.norm(gen_np.mean(axis=0) - test_np.mean(axis=0))),
        "diag_var_error": float(np.linalg.norm(np.diag(cov_gen) - np.diag(cov_test))),
        "cov_fro_error": float(np.linalg.norm(cov_gen - cov_test)),
        "corr_error": float(abs(corr_gen - corr_test)),
        "banana_coef_error": float(abs(banana_quadratic_coef_np(gen_np) - cfg.beta)),
        "train_banana_coef": banana_quadratic_coef_np(train_np),
        "gen_banana_coef": banana_quadratic_coef_np(gen_np),
        "test_banana_coef": banana_quadratic_coef_np(test_np),
    }


def save_scatter_figure(
    path: Path,
    test: np.ndarray,
    generated_by_lambda: dict[float, np.ndarray],
    metrics_by_lambda: dict[float, dict[str, float]],
    max_points: int,
) -> None:
    if plt is None:
        return
    ncols = 1 + len(generated_by_lambda)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.5), squeeze=False)
    axes = axes[0]
    rng = np.random.default_rng(777)

    def _sub(x: np.ndarray) -> np.ndarray:
        if x.shape[0] <= max_points:
            return x
        return x[rng.choice(x.shape[0], int(max_points), replace=False)]

    test_s = _sub(test)
    axes[0].scatter(test_s[:, 0], test_s[:, 1], s=4, alpha=0.5)
    axes[0].set_title("true test")
    for ax, (lam, arr) in zip(axes[1:], sorted(generated_by_lambda.items())):
        arr_s = _sub(arr)
        ax.scatter(arr_s[:, 0], arr_s[:, 1], s=4, alpha=0.5)
        metrics = metrics_by_lambda.get(float(lam), {})
        if metrics:
            ax.set_title(
                f"lambda={lam:g}\n"
                f"MMD2={metrics.get('mmd2', float('nan')):.3g}, "
                f"SW={metrics.get('sliced_wasserstein', float('nan')):.3g}"
            )
        else:
            ax.set_title(f"lambda={lam:g}")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_single_setting(
    cfg: ExperimentConfig,
    *,
    output_dir: Path,
    n_mmd: int,
    n_sliced: int,
    plot_max_points: int,
) -> dict[str, Any]:
    device = require_device(cfg.device)
    g = set_seed(cfg.seed, device)
    x_train = sample_banana(cfg.train_size, rho=cfg.rho, beta=cfg.beta, generator=g, device=device)
    x_test = sample_banana(cfg.test_size, rho=cfg.rho, beta=cfg.beta, generator=g, device=device)
    prior_mu, prior_var = estimate_diag_gaussian_prior(x_train, variance_floor=cfg.variance_floor)
    model, train_logs = train_one_model(
        x_train,
        cfg,
        prior_mu=prior_mu,
        prior_var=prior_var,
        generator=g,
        device=device,
    )
    x_gen = sample_model(model, cfg.n_gen, cfg.ode_steps, g, device)
    metrics = evaluate_model(model, x_train, x_test, x_gen, cfg=cfg, n_mmd=n_mmd, n_sliced=n_sliced)

    run_dir = output_dir / f"n_{cfg.train_size}" / f"seed_{cfg.seed}" / f"lambda_{cfg.lambda_prior:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "samples.npz",
        train=x_train.detach().cpu().numpy(),
        test=x_test.detach().cpu().numpy(),
        generated=x_gen.detach().cpu().numpy(),
        prior_mu=prior_mu.detach().cpu().numpy(),
        prior_var=prior_var.detach().cpu().numpy(),
    )
    payload = {"config": asdict(cfg), "metrics": metrics, "train_logs": train_logs}
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"run_dir": str(run_dir), "config": asdict(cfg), "metrics": metrics, "train_logs": train_logs}


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

    rows: list[dict[str, Any]] = []
    for train_size in train_sizes:
        for seed in seeds:
            generated_by_lambda: dict[float, np.ndarray] = {}
            metrics_by_lambda: dict[float, dict[str, float]] = {}
            test_for_plot: np.ndarray | None = None
            for lam in lambda_priors:
                cfg = ExperimentConfig(
                    rho=float(args.rho),
                    beta=float(args.beta),
                    train_size=int(train_size),
                    test_size=int(args.test_size),
                    n_gen=int(args.n_gen),
                    train_steps=int(args.train_steps),
                    batch_size=int(args.batch_size),
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                    hidden_dim=int(args.hidden_dim),
                    depth=int(args.depth),
                    time_frequencies=int(args.time_frequencies),
                    lambda_prior=float(lam),
                    variance_floor=float(args.variance_floor),
                    t_epsilon=float(args.t_epsilon),
                    ode_steps=int(args.ode_steps),
                    seed=int(seed),
                    device=str(args.device),
                    scheduler=str(args.scheduler),
                )
                row = run_single_setting(
                    cfg,
                    output_dir=output_dir,
                    n_mmd=int(args.n_mmd),
                    n_sliced=int(args.n_sliced),
                    plot_max_points=int(args.plot_max_points),
                )
                rows.append(row)
                samples = np.load(Path(row["run_dir"]) / "samples.npz")
                generated_by_lambda[float(lam)] = np.asarray(samples["generated"], dtype=np.float64)
                metrics_by_lambda[float(lam)] = dict(row["metrics"])
                test_for_plot = np.asarray(samples["test"], dtype=np.float64)
                print(
                    "[fm_gaussian_velocity_prior] "
                    f"n={train_size} seed={seed} lambda={lam:g} "
                    f"test_logp={row['metrics']['test_logp_mean']:.4f} "
                    f"mmd2={row['metrics']['mmd2']:.4g} "
                    f"run_dir={row['run_dir']}",
                    flush=True,
                )
            if test_for_plot is not None:
                fig_path = output_dir / f"samples_n_{train_size}_seed_{seed}.png"
                save_scatter_figure(
                    fig_path,
                    test_for_plot,
                    generated_by_lambda,
                    metrics_by_lambda,
                    int(args.plot_max_points),
                )

    write_aggregate(output_dir, rows)
    print(f"[fm_gaussian_velocity_prior] Wrote outputs under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
