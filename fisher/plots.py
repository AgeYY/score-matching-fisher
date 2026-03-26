from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from fisher.evaluation import BinnedFisher


def plot_training_loss(losses: list[float], out_path: str, title: str) -> None:
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(losses, linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_fisher_curve(
    curves: BinnedFisher,
    out_path: str,
    model_label: str,
    title: str,
) -> None:
    x = curves.centers
    m = curves.fisher_model
    f = curves.fisher_fd
    sm = curves.se_model
    sf = curves.se_fd
    valid = curves.valid

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(x[valid], m[valid], color="#1f77b4", linewidth=2.4, label=model_label)
    plt.plot(x[valid], f[valid], color="#d62728", linewidth=2.1, label="Finite-difference baseline")
    if np.any(np.isfinite(sm[valid])):
        plt.fill_between(
            x[valid],
            m[valid] - 1.96 * sm[valid],
            m[valid] + 1.96 * sm[valid],
            color="#1f77b4",
            alpha=0.15,
            linewidth=0.0,
        )
    if np.any(np.isfinite(sf[valid])):
        plt.fill_between(
            x[valid],
            f[valid] - 1.96 * sf[valid],
            f[valid] + 1.96 * sf[valid],
            color="#d62728",
            alpha=0.12,
            linewidth=0.0,
        )
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fisher information")
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_extrapolation_diagnostics(
    centers: np.ndarray,
    sigma_values: np.ndarray,
    fisher_per_sigma: np.ndarray,
    intercept: np.ndarray,
    slope: np.ndarray,
    valid_bins: np.ndarray,
    out_path: str,
) -> None:
    sigma2 = sigma_values**2
    valid_idx = np.where(valid_bins)[0]
    if valid_idx.size == 0:
        return

    pick_n = min(6, valid_idx.size)
    pick = np.linspace(0, valid_idx.size - 1, pick_n).round().astype(int)
    chosen = valid_idx[pick]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[pick_n:]:
        ax.axis("off")

    for k, b in enumerate(chosen):
        ax = axes_flat[k]
        yy = fisher_per_sigma[:, b]
        ax.scatter(sigma2, yy, color="#1f77b4", s=30, alpha=0.85, label="bin means")
        xline = np.linspace(0.0, sigma2.max() * 1.05, 100)
        yline = intercept[b] + slope[b] * xline
        ax.plot(xline, yline, color="#d62728", linewidth=1.8, label="linear fit")
        ax.set_title(rf"$\theta\approx{centers[b]:.2f}$")
        ax.grid(alpha=0.25, linestyle=":")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\sigma^2$")
    axes[0, 0].set_ylabel(r"$I_\sigma(\theta)$")
    axes[1, 0].set_ylabel(r"$I_\sigma(\theta)$")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(r"Extrapolation Diagnostics: $I_\sigma(\theta)$ vs $\sigma^2$", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_calibration_examples(
    centers: np.ndarray,
    logits_pos_map: dict[int, np.ndarray],
    logits_neg_map: dict[int, np.ndarray],
    out_path: str,
) -> None:
    chosen = sorted(logits_pos_map.keys())
    cols = 2
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.0 * rows), sharex=False, sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(rows, cols).ravel()

    for ax in axes_arr[len(chosen) :]:
        ax.axis("off")

    for k, idx in enumerate(chosen):
        ax = axes_arr[k]
        lp = logits_pos_map[idx]
        ln = logits_neg_map[idx]
        lo = min(float(np.min(lp)), float(np.min(ln)))
        hi = max(float(np.max(lp)), float(np.max(ln)))
        bins = np.linspace(lo, hi, 40, dtype=np.float64)
        ax.hist(lp, bins=bins, density=True, alpha=0.5, color="#1f77b4", label=r"class 1: $\theta_+$")
        ax.hist(ln, bins=bins, density=True, alpha=0.5, color="#d62728", label=r"class 0: $\theta_-$")
        ax.set_title(rf"$\theta_0 \approx {centers[idx]:.2f}$")
        ax.set_xlabel("Decoder logit")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2, linestyle=":")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Decoder Logit Separation at Representative $\\theta_0$", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_loss_examples(centers: np.ndarray, losses: dict[int, list[float]], out_path: str) -> None:
    plt.figure(figsize=(8.0, 5.0))
    for idx, trace in losses.items():
        arr = np.asarray(trace, dtype=np.float64)
        plt.plot(arr, linewidth=2.0, label=rf"$\theta_0\approx{centers[idx]:.2f}$")
    plt.xlabel("Epoch")
    plt.ylabel("Train BCE")
    plt.title("Local Decoder Training Loss (Representative $\\theta_0$)")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
