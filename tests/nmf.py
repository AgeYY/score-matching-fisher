# pip install zuko torch

import math
from pathlib import Path
import torch
import torch.nn as nn
import zuko
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1) Synthetic conditional dataset: theta | x
#    theta is 1D, x has configurable dimension x_dim
# ------------------------------------------------------------
def generate_batch(batch_size: int, x_dim: int, device: str):
    x = torch.randn(batch_size, x_dim, device=device)

    # Build a nonlinear summary of x so p(theta | x) is nontrivial
    w1 = torch.linspace(0.5, 1.5, x_dim, device=device)
    w2 = torch.cos(torch.linspace(0.0, math.pi, x_dim, device=device))

    s1 = (x * w1).sum(dim=-1, keepdim=True) / math.sqrt(x_dim)
    s2 = (x * w2).sum(dim=-1, keepdim=True) / math.sqrt(x_dim)

    # Two conditional modes
    mean1 = 0.9 * torch.sin(1.2 * s1) + 0.35 * s2
    mean2 = -0.7 + 0.8 * torch.cos(1.5 * s1 - 0.4 * s2)

    # Mixing weight depends on x
    gate = torch.sigmoid(1.4 * s1 - 0.8 * s2)
    choose1 = torch.bernoulli(gate)

    mean = choose1 * mean1 + (1.0 - choose1) * mean2
    sigma = 0.10 + 0.12 * torch.sigmoid(s2)

    theta = mean + sigma * torch.randn_like(mean)   # shape [B, 1]
    return x, theta


def true_log_prob(theta: torch.Tensor, x: torch.Tensor):
    # Exact log p(theta | x) under the synthetic mixture used in generate_batch.
    x_dim = x.shape[-1]
    w1 = torch.linspace(0.5, 1.5, x_dim, device=x.device)
    w2 = torch.cos(torch.linspace(0.0, math.pi, x_dim, device=x.device))

    s1 = (x * w1).sum(dim=-1, keepdim=True) / math.sqrt(x_dim)
    s2 = (x * w2).sum(dim=-1, keepdim=True) / math.sqrt(x_dim)

    mean1 = 0.9 * torch.sin(1.2 * s1) + 0.35 * s2
    mean2 = -0.7 + 0.8 * torch.cos(1.5 * s1 - 0.4 * s2)
    gate = torch.sigmoid(1.4 * s1 - 0.8 * s2).clamp(1e-6, 1 - 1e-6)
    sigma = 0.10 + 0.12 * torch.sigmoid(s2)

    normalizer = -0.5 * math.log(2.0 * math.pi)
    z1 = (theta - mean1) / sigma
    z2 = (theta - mean2) / sigma
    log_n1 = normalizer - torch.log(sigma) - 0.5 * z1.square()
    log_n2 = normalizer - torch.log(sigma) - 0.5 * z2.square()

    return torch.logaddexp(torch.log(gate) + log_n1, torch.log1p(-gate) + log_n2).squeeze(-1)


# ------------------------------------------------------------
# 2) Conditional discrete flow model for p(theta | x)
#    Raw x -> encoder -> context c -> NSF flow over theta
# ------------------------------------------------------------
class ConditionalThetaFlow(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int = 16,
        encoder_hidden: int = 64,
        flow_hidden: int = 64,
        transforms: int = 4,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, encoder_hidden),
            nn.Tanh(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.Tanh(),
            nn.Linear(encoder_hidden, context_dim),
        )

        # features=1 because theta is 1D
        self.flow = zuko.flows.NSF(
            features=1,
            context=context_dim,
            transforms=transforms,
            hidden_features=[flow_hidden, flow_hidden],
        )

    def distribution(self, x: torch.Tensor):
        c = self.encoder(x)          # [B, context_dim]
        return self.flow(c)          # conditional dist p(theta | x)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor):
        theta = theta.view(-1, 1)    # ensure shape [B, 1]
        return self.distribution(x).log_prob(theta)

    def sample(self, x: torch.Tensor, n_samples: int = 1):
        dist = self.distribution(x)
        # returns shape [n_samples, B, 1]
        samples = dist.sample((n_samples,))
        return samples.squeeze(-1)   # [n_samples, B]


# ------------------------------------------------------------
# 3) Train by maximizing conditional log likelihood
# ------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Change this freely. Everything else still works.
    x_dim = 20

    model = ConditionalThetaFlow(
        x_dim=x_dim,
        context_dim=16,
        encoder_hidden=64,
        flow_hidden=64,
        transforms=5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed validation set
    x_val, theta_val = generate_batch(batch_size=4000, x_dim=x_dim, device=device)
    train_steps = []
    train_nlls = []
    val_steps = []
    val_nlls = []

    print("Training...")
    for step in range(1, 2001):
        x_batch, theta_batch = generate_batch(batch_size=256, x_dim=x_dim, device=device)

        # Negative conditional log likelihood
        loss = -model.log_prob(theta_batch, x_batch).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_steps.append(step)
        train_nlls.append(loss.item())

        if step % 200 == 0:
            with torch.no_grad():
                val_nll = -model.log_prob(theta_val, x_val).mean().item()
            val_steps.append(step)
            val_nlls.append(val_nll)
            print(f"step={step:4d}  train_nll={loss.item():.4f}  val_nll={val_nll:.4f}")

    # --------------------------------------------------------
    # 4) Figure: loss vs training step
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_steps, train_nlls, linewidth=1.2, alpha=0.85, label="Train NLL")
    ax.plot(val_steps, val_nlls, marker="o", linewidth=1.6, label="Validation NLL")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Negative log likelihood")
    ax.set_title("NMF Training Curve")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    train_curve_path = Path("data/tests/nmf_loss_vs_step.png")
    train_curve_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(train_curve_path, dpi=180)
    plt.close(fig)

    # --------------------------------------------------------
    # 5) Estimate log likelihood on new data
    # --------------------------------------------------------
    x_test, theta_test = generate_batch(batch_size=5, x_dim=x_dim, device=device)

    with torch.no_grad():
        logp = model.log_prob(theta_test, x_test)
        logp_gt = true_log_prob(theta_test, x_test)

    print("\nExample conditional log likelihoods log p(theta | x):")
    for i in range(5):
        print(
            f"sample {i}: theta={theta_test[i,0].item(): .4f}, "
            f"logp={logp[i].item(): .4f}"
        )

    # --------------------------------------------------------
    # 6) Sample from p(theta | x) for a chosen x
    # --------------------------------------------------------
    x_star = x_test[:1]  # one conditioning vector
    with torch.no_grad():
        theta_samples = model.sample(x_star, n_samples=20)  # [20, 1]

    print("\n20 samples from p(theta | x_star):")
    print(theta_samples[:, 0].cpu())

    # --------------------------------------------------------
    # 7) Figure: estimated vs. ground-truth log likelihood
    # --------------------------------------------------------
    x_cmp, theta_cmp = generate_batch(batch_size=2000, x_dim=x_dim, device=device)
    with torch.no_grad():
        logp_est_cmp = model.log_prob(theta_cmp, x_cmp).detach().cpu()
        logp_gt_cmp = true_log_prob(theta_cmp, x_cmp).detach().cpu()

    lo = min(logp_est_cmp.min().item(), logp_gt_cmp.min().item())
    hi = max(logp_est_cmp.max().item(), logp_gt_cmp.max().item())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(logp_gt_cmp.numpy(), logp_est_cmp.numpy(), s=10, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Ground-truth log p(theta | x)")
    ax.set_ylabel("Estimated log p(theta | x)")
    ax.set_title("Conditional Log-Likelihood: Estimated vs Ground Truth")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = Path("data/tests/nmf_loglik_est_vs_gt.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    repo_root = Path(__file__).resolve().parents[1]
    print(f"\nSaved figure: {(repo_root / train_curve_path)}")
    print(f"\nSaved figure: {(repo_root / out_path)}")


if __name__ == "__main__":
    main()
