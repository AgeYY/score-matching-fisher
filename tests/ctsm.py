"""
CTSM-v two-sample toy: bimodal GMM p vs q, TwoSB path, log-ratio by trapezoid integration.

Core math lives in fisher.ctsm_paths, fisher.ctsm_models, fisher.ctsm_objectives.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow `python tests/ctsm.py` without PYTHONPATH (repo root must be on sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from fisher.ctsm_models import ToyFullTimeScoreNet
from fisher.ctsm_objectives import (
    ctsm_v_two_sample_loss,
    estimate_log_ratio_trapz,
)
from fisher.ctsm_paths import TwoSB

# =========================
# Reproducibility / device
# =========================
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================
# Two unknown distributions
# =========================
# The METHOD only uses samples from p and q.
# For evaluation only, we also keep exact log_prob here so we can measure error.
DIM = 2
SIGMA = 0.7


def make_gmm(means, sigma=0.7):
    means = torch.tensor(means, dtype=torch.float32, device=device)
    mix = Categorical(torch.ones(len(means), device=device) / len(means))
    comp = Independent(
        Normal(means, torch.full_like(means, sigma)),
        1,
    )
    return MixtureSameFamily(mix, comp)


# p and q are both non-Gaussian, multi-modal, and treated as unknown by training.
p_dist = make_gmm(
    [
        [-3.0, 0.0],
        [0.0, 3.0],
    ]
)

q_dist = make_gmm(
    [
        [3.0, 0.0],
        [0.0, -3.0],
    ]
)


def sample_p(n: int) -> torch.Tensor:
    return p_dist.sample((n,))


def sample_q(n: int) -> torch.Tensor:
    return q_dist.sample((n,))


def sample_two_unknown(n: int):
    x0 = sample_p(n)
    x1 = sample_q(n)
    return x0, x1


def true_log_ratio(x: torch.Tensor) -> torch.Tensor:
    return q_dist.log_prob(x) - p_dist.log_prob(x)


@torch.no_grad()
def make_grid(xmin=-6, xmax=6, ymin=-6, ymax=6, n=140, dev=None):
    dev = dev if dev is not None else device
    xs = torch.linspace(xmin, xmax, n)
    ys = torch.linspace(ymin, ymax, n)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).to(dev)
    return xx, yy, grid


def main():
    hidden_dim = 128
    batch_size = 512
    num_steps = 2500
    lr = 2e-3
    factor = 1.0  # corresponds to unit_factor=True in the repo
    two_sb_var = 2.0

    prob_path = TwoSB(dim=DIM, var=two_sb_var)
    model = ToyFullTimeScoreNet(dim=DIM, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    progress = tqdm(range(num_steps), desc="training CTSM-v")

    for step in progress:
        x0, x1 = sample_two_unknown(batch_size)
        loss = ctsm_v_two_sample_loss(
            model=model,
            prob_path=prob_path,
            x0=x0,
            x1=x1,
            factor=factor,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % 50 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    plt.figure(figsize=(5, 3))
    plt.plot(loss_history)
    plt.xlabel("step")
    plt.ylabel("CTSM-v loss")
    plt.title("Training curve")
    plt.tight_layout()
    plt.show()

    # Held-out evaluation
    n_eval = 2000
    x_eval = torch.cat([sample_p(n_eval // 2), sample_q(n_eval // 2)], dim=0)
    perm = torch.randperm(x_eval.shape[0], device=device)
    x_eval = x_eval[perm]

    ratio_true = true_log_ratio(x_eval)
    ratio_hat = estimate_log_ratio_trapz(model, x_eval, n_time=300)

    mse = torch.mean((ratio_hat - ratio_true) ** 2).item()
    corr = torch.corrcoef(torch.stack([ratio_true, ratio_hat]))[0, 1].item()

    print(f"MSE  : {mse:.6f}")
    print(f"Corr : {corr:.6f}")

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(
        ratio_true.detach().cpu().numpy(),
        ratio_hat.detach().cpu().numpy(),
        s=8,
        alpha=0.5,
    )
    lo = min(ratio_true.min().item(), ratio_hat.min().item())
    hi = max(ratio_true.max().item(), ratio_hat.max().item())
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("true log q/p")
    plt.ylabel("estimated log q/p")
    plt.title("Two unknown distributions")
    plt.tight_layout()
    plt.show()

    xx, yy, grid = make_grid(dev=device)
    grid_true = true_log_ratio(grid).reshape(xx.shape).detach().cpu().numpy()
    grid_hat = estimate_log_ratio_trapz(model, grid, n_time=250).reshape(xx.shape).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axes[0].contourf(xx.numpy(), yy.numpy(), grid_true, levels=40)
    axes[0].set_title("True log q/p")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(xx.numpy(), yy.numpy(), grid_hat, levels=40)
    axes[1].set_title("Estimated log q/p")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
