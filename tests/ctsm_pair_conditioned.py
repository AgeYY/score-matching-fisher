"""
Pair-conditioned CTSM-v toy: conditional GMM p(x|theta) with theta indexing variance (discrete grid).

Trains ToyPairConditionedTimeScoreNet with ctsm_v_pair_conditioned_loss; evaluates
log p(x|b) - log p(x|a) via estimate_log_ratio_trapz_pair.

Core math: fisher.ctsm_paths, fisher.ctsm_models, fisher.ctsm_objectives.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from fisher.ctsm_models import ToyPairConditionedTimeScoreNet
from fisher.ctsm_objectives import ctsm_v_pair_conditioned_loss, estimate_log_ratio_trapz_pair
from fisher.ctsm_paths import TwoSB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pair-conditioned CTSM-v GMM toy (discrete theta grid).")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (repo default is cuda; use cpu only if CUDA unavailable).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--two-sb-var", type=float, default=2.0)
    p.add_argument("--n-theta", type=int, default=9, help="Number of discrete theta values (variance levels).")
    p.add_argument("--n-eval", type=int, default=4000)
    p.add_argument("--n-time", type=int, default=320)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, save loss curve and scatter PNG here (default: repo data/ctsm_pair_conditioned_toy).",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Conditional GMM: two isotropic components, shared means, variance indexed by theta
# -----------------------------------------------------------------------------

DIM = 2


def sigma_from_theta(theta: torch.Tensor, theta_lo: float, theta_hi: float, sigma_lo: float, sigma_hi: float) -> torch.Tensor:
    """Scalar or batched theta -> component std (isotropic)."""
    t = (theta - theta_lo) / (theta_hi - theta_lo)
    return sigma_lo + (sigma_hi - sigma_lo) * t.clamp(0.0, 1.0)


def _mixture_log_prob(
    x: torch.Tensor,
    theta: torch.Tensor,
    means: torch.Tensor,
    theta_lo: float,
    theta_hi: float,
    sigma_lo: float,
    sigma_hi: float,
    log_pi: float,
) -> torch.Tensor:
    """log p(x|theta) for 50/50 isotropic Gaussian mixture; x (B,d), theta (B,) or (B,1)."""
    if theta.dim() > 1:
        theta = theta.squeeze(-1)
    B = x.shape[0]
    d = float(DIM)
    sig = sigma_from_theta(theta, theta_lo, theta_hi, sigma_lo, sigma_hi)  # (B,)
    # xc[b,k,:] = x[b] - means[k]
    xc = x.unsqueeze(1) - means.unsqueeze(0)  # (B, K, d)
    sig2 = (sig**2).unsqueeze(-1).unsqueeze(-1)  # (B,1,1) for broadcast with (B,K,d) - need (B,1,1)
    quad = (xc**2).sum(dim=-1)  # (B, K)
    log_comp = (
        -0.5 * d * math.log(2.0 * math.pi)
        - d * torch.log(sig).unsqueeze(-1)
        - 0.5 * quad / (sig**2).unsqueeze(-1)
    )
    return torch.logsumexp(log_pi + log_comp, dim=-1)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu or use a GPU machine.")

    device = torch.device(args.device)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Theta grid in [theta_lo, theta_hi]; larger theta -> larger component std (more diffuse)
    theta_lo, theta_hi = -1.0, 1.0
    sigma_lo, sigma_hi = 0.35, 0.95
    theta_grid = torch.linspace(theta_lo, theta_hi, int(args.n_theta), device=device)

    means = torch.tensor(
        [[-2.5, 0.0], [2.5, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    log_pi = math.log(0.5)

    def log_prob(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return _mixture_log_prob(x, theta, means, theta_lo, theta_hi, sigma_lo, sigma_hi, log_pi)

    def true_log_ratio(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return log_prob(x, b) - log_prob(x, a)

    # Training: sample random ordered pairs (a,b) from grid, x0~p(.|a), x1~p(.|b)
    n_theta = theta_grid.numel()

    def sample_training_batch(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ia = torch.randint(0, n_theta, (batch,), device=device)
        ib = torch.randint(0, n_theta, (batch,), device=device)
        a = theta_grid[ia].unsqueeze(-1)
        b = theta_grid[ib].unsqueeze(-1)
        sig_a = sigma_from_theta(a.squeeze(-1), theta_lo, theta_hi, sigma_lo, sigma_hi)
        sig_b = sigma_from_theta(b.squeeze(-1), theta_lo, theta_hi, sigma_lo, sigma_hi)
        k0 = torch.randint(0, means.shape[0], (batch,), device=device)
        k1 = torch.randint(0, means.shape[0], (batch,), device=device)
        mu0 = means[k0]
        mu1 = means[k1]
        x0 = mu0 + sig_a.unsqueeze(-1) * torch.randn(batch, DIM, device=device)
        x1 = mu1 + sig_b.unsqueeze(-1) * torch.randn(batch, DIM, device=device)
        return x0, x1, a, b

    prob_path = TwoSB(dim=DIM, var=float(args.two_sb_var))
    model = ToyPairConditionedTimeScoreNet(
        dim=DIM,
        hidden_dim=int(args.hidden_dim),
        m_scale=1.0,
        delta_scale=0.5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    loss_history: list[float] = []
    progress = tqdm(range(int(args.num_steps)), desc="pair-conditioned CTSM-v")
    for step in progress:
        x0, x1, a, b = sample_training_batch(int(args.batch_size))
        loss = ctsm_v_pair_conditioned_loss(
            model=model,
            prob_path=prob_path,
            x0=x0,
            x1=x1,
            a=a,
            b=b,
            factor=1.0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        if step % 50 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    # Evaluation: random (a,b) pairs; x ~ p(.|a) (held-out locations for ratio at x)
    n_eval = int(args.n_eval)
    ia = torch.randint(0, n_theta, (n_eval,), device=device)
    ib = torch.randint(0, n_theta, (n_eval,), device=device)
    a_eval = theta_grid[ia].unsqueeze(-1)
    b_eval = theta_grid[ib].unsqueeze(-1)
    sig_a = sigma_from_theta(a_eval.squeeze(-1), theta_lo, theta_hi, sigma_lo, sigma_hi)
    k0 = torch.randint(0, means.shape[0], (n_eval,), device=device)
    mu0 = means[k0]
    x_eval = mu0 + sig_a.unsqueeze(-1) * torch.randn(n_eval, DIM, device=device)

    ratio_true = true_log_ratio(x_eval, a_eval, b_eval)
    ratio_hat = estimate_log_ratio_trapz_pair(
        model,
        x_eval,
        a_eval,
        b_eval,
        n_time=int(args.n_time),
    )

    mse = torch.mean((ratio_hat - ratio_true) ** 2).item()
    corr_tensor = torch.corrcoef(torch.stack([ratio_true, ratio_hat]))
    corr = corr_tensor[0, 1].item() if ratio_true.numel() > 1 else float("nan")

    abs_gap = (b_eval - a_eval).abs().squeeze(-1)

    print(f"MSE  : {mse:.6f}")
    print(f"Corr : {corr:.6f}")
    # Often easier when |b-a| is small (closer to local ratio / less extrapolation in Delta).
    small_gap = abs_gap < 0.75
    if small_gap.any():
        mse_s = torch.mean((ratio_hat[small_gap] - ratio_true[small_gap]) ** 2).item()
        c_tensor = torch.corrcoef(torch.stack([ratio_true[small_gap], ratio_hat[small_gap]]))
        corr_s = c_tensor[0, 1].item() if int(small_gap.sum()) > 1 else float("nan")
        print(f"MSE  (|b-a|<0.75): {mse_s:.6f}")
        print(f"Corr (|b-a|<0.75): {corr_s:.6f}")

    out_root = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "data" / "ctsm_pair_conditioned_toy"
    out_root.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 3))
    plt.plot(loss_history)
    plt.xlabel("step")
    plt.ylabel("CTSM-v loss")
    plt.title("Pair-conditioned training")
    plt.tight_layout()
    plt.savefig(out_root / "pair_ctsm_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(
        ratio_true.detach().cpu().numpy(),
        ratio_hat.detach().cpu().numpy(),
        s=8,
        alpha=0.45,
    )
    lo = min(ratio_true.min().item(), ratio_hat.min().item())
    hi = max(ratio_true.max().item(), ratio_hat.max().item())
    plt.plot([lo, hi], [lo, hi], "--", color="gray")
    plt.xlabel(r"true $\log p(x|b)-\log p(x|a)$")
    plt.ylabel(r"estimated (pair-conditioned)")
    plt.title("Discrete-theta GMM (variance indexed by $\\theta$)")
    plt.tight_layout()
    plt.savefig(out_root / "pair_ctsm_scatter.png", dpi=150)
    plt.close()

    print("MSE by |b-a| (coarse bins):")
    for lo_g, hi_g in [(0.0, 0.5), (0.5, 1.0), (1.0, 3.0)]:
        mask = (abs_gap >= lo_g) & (abs_gap < hi_g)
        if mask.sum() == 0:
            continue
        mse_g = torch.mean((ratio_hat[mask] - ratio_true[mask]) ** 2).item()
        print(f"  |b-a| in [{lo_g}, {hi_g}): n={int(mask.sum().item())} MSE={mse_g:.6f}")

    print(f"Saved figures under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
