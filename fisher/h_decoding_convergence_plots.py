#!/usr/bin/env python3
"""Plotting, cache loading, and summaries for H-decoding convergence."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, TypedDict, cast

try:
    from typing import NotRequired
except ImportError:  # Python <3.11
    from typing_extensions import NotRequired

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

# Matplotlib rcParams (tick size, spines) apply when ``global_setting`` is imported — before pyplot.
from global_setting import DATA_DIR

import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher import h_binned_visualization as vhb
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.hellinger_gt import (
    bin_centers_from_edges,
    estimate_hellinger_sq_grid_centers_analytic,
    estimate_hellinger_sq_one_sided_mc,
    estimate_mean_llr_one_sided_mc,
    theta_centers_for_analytic_gt,
)
from fisher.gaussian_network import (
    ConditionalDiagonalGaussianPrecisionMLP,
    ConditionalGaussianPrecisionMLP,
    ConditionalLowRankGaussianCovarianceMLP,
    ObservationAutoencoder,
    compute_gaussian_network_c_matrix,
    encode_observations,
    train_gaussian_network,
    train_observation_autoencoder,
)
from fisher.gaussian_x_flow import (
    ConditionalDiagonalGaussianCovarianceFMMLP,
    ConditionalGaussianCovarianceFMMLP,
    compute_gaussian_x_flow_c_matrix,
    path_schedule_from_name,
    train_gaussian_x_flow,
)
from fisher.linear_x_flow import (
    ConditionalTimeDiagonalLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeDiagonalLinearXFlowMLP,
    ConditionalTimeLinearXFlowMLP,
    ConditionalTimeLowRankCorrectionLinearXFlowMLP,
    ConditionalTimePureConditionalLowRankLinearXFlowMLP,
    ConditionalTimePureLowRankLinearXFlowMLP,
    ConditionalTimeScalarLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeThetaOnlyBLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeThetaScalarLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeThetaDiagonalLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeRandomBasisLowRankLinearXFlowMLP,
    ConditionalTimeScalarLinearXFlowMLP,
    ConditionalTimeThetaDiagonalLinearXFlowMLP,
    compute_ode_time_linear_x_flow_c_matrix,
    compute_time_linear_x_flow_c_matrix,
    compute_linear_x_flow_analytic_hellinger_matrix,
    train_low_rank_t_warmup_then_full,
    train_low_rank_t_theta_only_b_mean_regression_pretrain_then_freeze_b,
    train_time_linear_x_flow_schedule,
)
from fisher.linear_theta_flow import (
    ConditionalLinearThetaFlowMixtureMLP,
    compute_linear_theta_flow_c_matrix,
    train_linear_theta_flow,
)
from fisher.contrastive_llr import (
    ContrastiveAdditiveIndependentScorer,
    ContrastiveNormalizedDotScorer,
    compute_contrastive_soft_c_matrix,
    dot_scorer_augmented_theta_dim,
    h_directed_from_delta_l as compute_h_directed_contrastive,
    train_contrastive_soft_llr,
)

_TIME_LXF_METHODS = {
    "linear_x_flow_t",
    "linear_x_flow_scalar_t",
    "linear_x_flow_diagonal_t",
    "linear_x_flow_diagonal_theta_t",
    "linear_x_flow_low_rank_t",
    "linear_x_flow_pure_low_rank_t",
    "linear_x_flow_pure_cond_low_rank_t",
    "linear_x_flow_lr_t_ts",
    "linear_x_flow_low_rank_randb_t",
    "xflow_sir_lrank",
    "xflow_sir_lrank_dia",
    "xflow_sir_lrank_dia_theta",
    "xflow_sir_lrank_scalar",
    "xflow_sir_lrank_scalar_theta",
    "xflow_sir_pure_lrank",
}
from fisher.lxf_bin_likelihood_hellinger import lxf_bin_likelihood_hellinger
from fisher.nf_hellinger import (
    ConditionalThetaNF,
    PriorThetaNF,
    compute_c_matrix_nf,
    compute_delta_l as compute_delta_l_nf,
    compute_h_directed as compute_h_directed_nf,
    compute_log_p_theta_prior_nf,
    compute_ratio_matrix_posterior_minus_prior,
    require_zuko_for_nf,
    symmetrize as symmetrize_nf,
    train_conditional_nf,
    train_prior_nf,
)
from fisher.nf_reduction import (
    NFReductionModel,
    compute_nf_reduction_c_matrix,
    train_nf_reduction,
)
from fisher.pi_nf import PiNFModel, compute_pi_nf_c_matrix, pi_nf_diagnostics, train_pi_nf
from fisher.evaluation import log_p_x_given_theta
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    validate_estimation_args,
)


# Isolated ``h_decoding_convergence.{png,svg}`` size; combined figure uses the same width/height
# ratio for the right-hand curve column (column height matches the matrix panel height).
_H_DECODING_CURVE_FIGSIZE_IN: tuple[float, float] = (3.5, 3.5)

def _sqrt_h_like(mat: np.ndarray) -> np.ndarray:
    """Elementwise sqrt for H^2- or H_sym-like matrices in [0, 1]; preserves NaN positions."""
    h = np.asarray(mat, dtype=np.float64)
    out = np.full_like(h, np.nan, dtype=np.float64)
    finite = np.isfinite(h)
    out[finite] = np.sqrt(np.clip(h[finite], 0.0, 1.0))
    return out


def _offdiag_gt_xy(
    est: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Off-diagonal pairs (x=GT, y=estimated) for scatter; same mask as ``matrix_corr_offdiag_pearson``."""
    aa = np.asarray(est, dtype=np.float64)
    bb = np.asarray(gt, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2 or aa.shape[0] != aa.shape[1]:
        raise ValueError("offdiag_gt_xy requires equal square matrices.")
    n = aa.shape[0]
    if n < 2:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            0,
        )
    off = ~np.eye(n, dtype=bool)
    mask = off & np.isfinite(aa) & np.isfinite(bb)
    n_pts = int(np.sum(mask))
    return bb[mask], aa[mask], n_pts


def _plot_estimated_vs_gt_h_scatter(
    ax: plt.Axes,
    *,
    est: np.ndarray,
    gt: np.ndarray,
    n: int,
    r_offdiag: float,
) -> None:
    """Scatter of off-diagonal estimated sqrt(H) vs MC GT (x=GT, y=est); y=x reference."""
    x_gt, y_est, n_pts = _offdiag_gt_xy(est, gt)
    if n_pts < 1:
        ax.text(
            0.5,
            0.5,
            f"n={n}\nno off-diagonal pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()
        return
    ax.scatter(x_gt, y_est, s=8, alpha=0.5, c="#1f77b4", edgecolors="none")
    lo = float(np.min([np.min(x_gt), np.min(y_est)]))
    hi = float(np.max([np.max(x_gt), np.max(y_est)]))
    if hi <= lo:
        hi = lo + 1e-9
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", linestyle="--", linewidth=1.0, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"GT off-diag: $\sqrt{H^2}$ (MC)", fontsize=8)
    ax.set_ylabel(r"Est. off-diag: $\sqrt{H_{\mathrm{sym}}^2}$ binned", fontsize=8)
    ax.set_title(
        f"n={n}  Pearson $r$={float(r_offdiag):.4f}  (N={n_pts})",
        fontsize=9,
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=7)


def _h_matrix_results_npz_basename(*, dataset_family: str) -> str:
    suf = "_non_gauss" if str(dataset_family) == "cosine_gmm" else "_theta_cov"
    return f"h_matrix_results{suf}.npz"


def _load_delta_l_from_run_dir(run_dir: str, *, dataset_family: str) -> np.ndarray:
    p = os.path.join(run_dir, _h_matrix_results_npz_basename(dataset_family=dataset_family))
    if not os.path.isfile(p):
        p_cov = os.path.join(run_dir, "h_matrix_results_theta_cov.npz")
        if os.path.isfile(p_cov):
            p = p_cov
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"expected H npz for LLR: {p} (also tried h_matrix_results_theta_cov.npz in {run_dir})"
        )
    z = np.load(p, allow_pickle=True)
    if "delta_l_matrix" not in z.files:
        raise KeyError(
            f"{p} has no delta_l_matrix; re-run with h_save_intermediates or use a method that stores ΔL (e.g. nf)."
        )
    return np.asarray(z["delta_l_matrix"], dtype=np.float64)


def _metrics_delta_l_binned(
    delta_l: np.ndarray,
    subset: SweepSubset,
    n_bins: int,
) -> np.ndarray:
    """Bin-average directed ΔL (same contract as binned h_sym)."""
    dl_binned, _ = vhb.average_matrix_by_bins(
        np.asarray(delta_l, dtype=np.float64),
        subset.bin_all,
        n_bins,
    )
    return dl_binned


def _plot_estimated_vs_gt_llr_scatter(
    ax: plt.Axes,
    *,
    est: np.ndarray,
    gt: np.ndarray,
    n: int,
    r_offdiag: float,
) -> None:
    """Scatter of off-diagonal binned model ΔL vs generative one-sided mean log p(x|θ')-log p(x|θ)."""
    x_gt, y_est, n_pts = _offdiag_gt_xy(est, gt)
    if n_pts < 1:
        ax.text(
            0.5,
            0.5,
            f"n={n}\nno off-diagonal LLR pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()
        return
    ax.scatter(x_gt, y_est, s=8, alpha=0.5, c="#2ca02c", edgecolors="none")
    lo = float(np.min([np.min(x_gt), np.min(y_est)]))
    hi = float(np.max([np.max(x_gt), np.max(y_est)]))
    if hi <= lo:
        hi = lo + 1e-9
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", linestyle="--", linewidth=1.0, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        "GT off-diag: " + r"$E_x[\log p(x|\theta_j)-\log p(x|\theta_i)]$ (MC, gen.)",
        fontsize=7,
    )
    ax.set_ylabel("Est. off-diag: binned " + r"$\Delta L$ (model)", fontsize=7)
    ax.set_title(
        f"n={n}  LLR  Pearson $r$={float(r_offdiag):.4f}  (N={n_pts})",
        fontsize=9,
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=7)

def _save_figure_png_svg(fig: plt.Figure, path_png: str, *, dpi: int = 160) -> str:
    """Write PNG and a sibling .svg (vector) for the same figure."""
    fig.savefig(path_png, dpi=dpi)
    path_svg = str(Path(path_png).with_suffix(".svg"))
    fig.savefig(path_svg)
    return path_svg


def _log_std_normal_scalar(theta_flat: np.ndarray) -> np.ndarray:
    t = np.asarray(theta_flat, dtype=np.float64).reshape(-1)
    return -0.5 * t * t - 0.5 * float(np.log(2.0 * np.pi))


def _stable_softmax_log(logp: np.ndarray) -> np.ndarray:
    z = np.asarray(logp, dtype=np.float64).reshape(-1)
    z = z - float(np.max(z))
    e = np.exp(np.clip(z, -700.0, 700.0))
    s = float(np.sum(e))
    if (not np.isfinite(s)) or s <= 0.0:
        u = np.full_like(e, 1.0 / max(e.size, 1))
        return u
    return e / s


def _npz_row_or_vector(
    z: Any | None,
    key: str,
    *,
    row: int,
    n: int,
) -> np.ndarray | None:
    if z is None or key not in getattr(z, "files", ()):
        return None
    arr = np.asarray(z[key], dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] > int(row) and arr.shape[1] == int(n):
        return np.asarray(arr[int(row), :], dtype=np.float64).reshape(-1)
    if arr.ndim == 1 and arr.size == int(n):
        return np.asarray(arr, dtype=np.float64).reshape(-1)
    return None


def _model_posterior_log_weights_for_fixed_x(
    *,
    hfm: str,
    c_row: np.ndarray,
    h_npz: Any | None,
    row: int,
) -> tuple[np.ndarray, str]:
    c = np.asarray(c_row, dtype=np.float64).reshape(-1)
    method = str(hfm).strip().lower()
    if method in ("theta_flow", "theta_flow_autoencoder"):
        log_post = _npz_row_or_vector(h_npz, "theta_flow_log_post_matrix", row=int(row), n=c.size)
        if log_post is not None:
            suffix = " over encoded z" if method == "theta_flow_autoencoder" else ""
            return log_post, f"learned posterior log-density{suffix}"
        log_prior = _npz_row_or_vector(h_npz, "theta_flow_log_prior_matrix", row=int(row), n=c.size)
        if log_prior is not None:
            suffix = " over encoded z" if method == "theta_flow_autoencoder" else ""
            return c + log_prior, f"ratio + learned prior log-density{suffix}"
        print(
            "[convergence] fixed-x diagnostic: theta_flow artifact lacks learned posterior/prior "
            "log-density fields; using ratio row as a uniform-prior fallback.",
            flush=True,
        )
        return c, "ratio-only fallback"
    if method == "nf":
        return c, "posterior log-density"
    if method == "nf_reduction":
        return c, r"NF-reduction log $p(z|\theta)$"
    if method == "pi_nf":
        return c, r"pi-NF log $p(z|\theta)$"
    if method == "gaussian_network":
        return c, r"Gaussian-network log $p(x|\theta)$"
    if method == "gaussian_network_diagonal":
        return c, r"Gaussian-network diagonal log $p(x|\theta)$"
    if method == "gaussian_network_diagonal_binned_pca":
        return c, r"Gaussian-network binned-PCA diagonal log $p(z|\theta)$"
    if method == "gaussian_network_low_rank":
        return c, r"Gaussian-network low-rank log $p(x|\theta)$"
    if method == "gaussian_network_autoencoder":
        return c, r"Gaussian-network AE log $p(z|\theta)$"
    if method == "gaussian_network_diagonal_autoencoder":
        return c, r"Gaussian-network diagonal AE log $p(z|\theta)$"
    if method == "x_flow_pca":
        return c, r"X-flow PCA log $p(z|\theta)$"
    if method == "sir_xflow_lrank_t":
        return c, r"SIR + linear X-flow low-rank log $p(z|\theta)$"
    if method == "sir_xflow":
        return c, r"SIR + X-flow log $p(z|\theta)$"
    if method == "sir_thetaflow":
        return c, r"SIR + theta-flow Bayes ratio on $z$"
    if method == "gaussian_x_flow":
        return c, r"Gaussian X-flow log $p(x|\theta)$"
    if method == "gaussian_x_flow_diagonal":
        return c, r"Gaussian X-flow (diagonal cov.) log $p(x|\theta)$"
    if method == "linear_x_flow":
        return c, r"Linear X-flow log $p(x|\theta)$"
    if method == "linear_x_flow_t":
        return c, r"Linear X-flow time-dependent $A(t)$ log $p(x|\theta)$"
    if method == "linear_x_flow_scalar_t":
        return c, r"Linear X-flow scalar $A(t)$ log $p(x|\theta)$"
    if method == "linear_x_flow_diagonal_theta_t":
        return c, r"Linear X-flow diagonal $A(t,\theta)$ log $p(x|\theta)$"
    if method == "linear_x_flow_low_rank_t":
        return c, r"Linear X-flow full $A(t)$ + orthonormal low-rank correction log $p(x|\theta)$"
    if method == "xflow_sir_lrank":
        return c, r"X-flow SIR low-rank correction log $p(x|\theta)$"
    if method == "xflow_sir_lrank_dia":
        return c, r"X-flow SIR low-rank correction, diagonal $A(t)$ log $p(x|\theta)$"
    if method == "xflow_sir_lrank_dia_theta":
        return c, r"X-flow SIR low-rank correction, diagonal $A(t,\theta)$ log $p(x|\theta)$"
    if method == "xflow_sir_lrank_scalar":
        return c, r"X-flow SIR low-rank correction, scalar $A(t)$ log $p(x|\theta)$"
    if method == "xflow_sir_lrank_scalar_theta":
        return c, r"X-flow SIR low-rank correction, scalar $A(t,\theta)$ log $p(x|\theta)$"
    if method == "xflow_sir_pure_lrank":
        return c, r"X-flow SIR pure low-rank $U h(U^{\mathsf T}x)$ log $p(x|\theta)$"
    if method == "linear_x_flow_pure_low_rank_t":
        return c, r"Linear X-flow pure orthonormal low-rank $U h(U^{\mathsf T}x)$ log $p(x|\theta)$"
    if method == "linear_x_flow_pure_cond_low_rank_t":
        return c, r"Linear X-flow pure conditional $U(\theta,t) h(U^{\mathsf T}x)$ log $p(x|\theta)$"
    if method == "linear_x_flow_lr_t_ts":
        return c, r"Linear X-flow full $A(t)$ + low-rank correction with $b(\theta)$ log $p(x|\theta)$"
    if method == "linear_x_flow_low_rank_randb_t":
        return c, r"Linear X-flow random-basis low-rank $A(t)$ log $p(x|\theta)$"
    if method == "linear_x_flow_diagonal_t":
        return c, r"Linear X-flow diagonal $A(t)$ log $p(x|\theta)$"
    if method == "linear_theta_flow":
        return c, r"Linear theta-flow GMM log $p(\theta|x)$"
    return c, "matrix row"


def _normalize_density_trapz(theta_grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    t = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    q = np.asarray(density, dtype=np.float64).reshape(-1)
    if t.size != q.size or t.size < 2:
        return np.zeros_like(t, dtype=np.float64)
    q = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    z = float(np.trapezoid(q, t))
    if (not np.isfinite(z)) or z <= 0.0:
        span = float(np.max(t) - np.min(t))
        if span <= 0.0:
            return np.full_like(t, 1.0 / max(t.size, 1), dtype=np.float64)
        return np.full_like(t, 1.0 / span, dtype=np.float64)
    return q / z


def _weighted_kde_density(
    theta_samples: np.ndarray,
    weights: np.ndarray,
    theta_dense: np.ndarray,
) -> np.ndarray:
    th = np.asarray(theta_samples, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    td = np.asarray(theta_dense, dtype=np.float64).reshape(-1)
    if th.size < 2 or w.size != th.size or td.size < 2:
        return np.zeros_like(td, dtype=np.float64)
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    sw = float(np.sum(w))
    if sw <= 0.0 or (not np.isfinite(sw)):
        w = np.full_like(w, 1.0 / max(w.size, 1), dtype=np.float64)
    else:
        w = w / sw

    mu = float(np.sum(w * th))
    var = float(np.sum(w * (th - mu) ** 2))
    sigma = float(np.sqrt(max(var, 1e-12)))
    n_eff = float(1.0 / max(np.sum(w * w), 1e-12))
    # Weighted Silverman rule with range/spacing floor for robustness.
    h = 1.06 * sigma * max(n_eff, 2.0) ** (-0.2)
    span = float(np.max(th) - np.min(th))
    diffs = np.diff(np.sort(np.unique(th)))
    min_step = float(np.median(diffs)) if diffs.size > 0 else (span / max(th.size - 1, 1))
    floor_h = max(span / 200.0, min_step * 0.5, 1e-3)
    h = float(max(h, floor_h))

    z = (td[:, None] - th[None, :]) / h
    kern = np.exp(-0.5 * np.clip(z * z, 0.0, 1e6)) / (np.sqrt(2.0 * np.pi) * h)
    q = kern @ w
    return _normalize_density_trapz(td, q)


def _obs_dim_matches_generative_likelihood(x_fixed: np.ndarray, dataset: Any) -> bool:
    """True iff ``x_fixed`` can be evaluated under ``dataset.log_p_x_given_theta``.

    PR-embedded NPZs store observations in ``h_dim`` while :func:`build_dataset_from_meta`
    builds the toy likelihood in ``generative_x_dim_from_meta`` (e.g. z-space). Fixed-$x$
    GT posterior curves require the same dimension as that likelihood.
    """
    obs = int(np.asarray(x_fixed, dtype=np.float64).reshape(-1).size)
    gen = int(getattr(dataset, "x_dim", obs))
    return obs == gen


def _approx_gt_posterior_density(
    *,
    dataset: Any,
    x_fixed: np.ndarray,
    theta_dense: np.ndarray,
    theta_low: float,
    theta_high: float,
) -> np.ndarray | None:
    td = np.asarray(theta_dense, dtype=np.float64).reshape(-1)
    if not _obs_dim_matches_generative_likelihood(x_fixed, dataset):
        return None
    x1 = np.asarray(x_fixed, dtype=np.float64).reshape(1, -1)
    if td.size < 2:
        return np.zeros_like(td, dtype=np.float64)
    x_rep = np.repeat(x1, repeats=td.size, axis=0)
    tcol = td.reshape(-1, 1)
    ll = np.asarray(log_p_x_given_theta(x_rep, tcol, dataset), dtype=np.float64).reshape(-1)
    if ll.size != td.size:
        raise ValueError(f"log_p_x_given_theta returned {ll.size}, expected {td.size}.")
    width = float(theta_high - theta_low)
    if width <= 0.0:
        raise ValueError(f"theta range must satisfy high>low, got [{theta_low}, {theta_high}].")
    log_prior = -np.log(width)
    log_post = ll + float(log_prior)
    z = log_post - float(np.max(log_post))
    q = np.exp(np.clip(z, -700.0, 700.0))
    return _normalize_density_trapz(td, q)


def _select_two_fixed_x_indices(n: int, *, perm_seed: int, n_subset: int) -> tuple[int, int]:
    i_a = (int(perm_seed) * 1_000_003 + 17 * int(n_subset)) % int(n)
    if int(n) <= 1:
        return int(i_a), int(i_a)
    hop = max(1, int(n) // 2)
    i_b = (int(i_a) + int(hop)) % int(n)
    if int(i_b) == int(i_a):
        i_b = (int(i_a) + 1) % int(n)
    return int(i_a), int(i_b)


def _plot_fixed_x_column(
    *,
    ax_top: plt.Axes,
    ax_bot: plt.Axes,
    i_fix: int,
    hfm: str,
    c: np.ndarray,
    th_flat: np.ndarray,
    xa: np.ndarray,
    th_grid: np.ndarray,
    mu: np.ndarray,
    dataset: Any,
    lo: float,
    hi: float,
    h_npz: Any | None = None,
) -> None:
    c_row = np.asarray(c[int(i_fix), :], dtype=np.float64).reshape(-1)
    logp, logp_source = _model_posterior_log_weights_for_fixed_x(
        hfm=hfm,
        c_row=c_row,
        h_npz=h_npz,
        row=int(i_fix),
    )
    w = _stable_softmax_log(logp)
    order = np.argsort(th_flat, kind="mergesort")
    th_s = th_flat[order]
    w_s = w[order]
    q_model = _weighted_kde_density(th_s, w_s, th_grid)

    x_fixed = np.asarray(xa[int(i_fix)], dtype=np.float64).reshape(-1)
    x0 = float(x_fixed[0]) if x_fixed.size else float("nan")
    xn = float(np.linalg.norm(x_fixed))
    q_gt = _approx_gt_posterior_density(
        dataset=dataset,
        x_fixed=x_fixed,
        theta_dense=th_grid,
        theta_low=lo,
        theta_high=hi,
    )
    j_map = int(np.argmax(w))
    theta_map = float(th_flat[j_map])

    ax_top.fill_between(th_s, 0.0, w_s, color="#1f77b4", alpha=0.22, step="mid")
    ax_top.plot(th_s, w_s, "o", color="#1f77b4", ms=2, alpha=0.55, label="Posterior mass on θ samples")
    ax_top.plot(th_grid, q_model, color="#1f77b4", lw=1.6, label=f"Model posterior (approx; {logp_source})")
    if q_gt is not None:
        ax_top.plot(th_grid, q_gt, color="#d62728", lw=1.5, ls="--", label="GT posterior (approx)")
    ax_top.set_ylabel("density")
    ax_top.set_title(
        f"Fixed-$x$ posterior diagnostics  (row $i$={int(i_fix)},  method={hfm})",
        fontsize=9,
    )
    ax_top.legend(loc="upper right", fontsize=7)
    ann = f"x[0]={x0:.3g}  ||x||={xn:.3g}  theta_map={theta_map:.3g}"
    if q_gt is None:
        ann += "  (GT posterior n/a: embedded obs. dim ≠ generative likelihood dim)"
    ax_top.annotate(
        ann,
        xy=(0.5, 1.12),
        xycoords="axes fraction",
        ha="center",
        fontsize=7,
    )

    gen_match = _obs_dim_matches_generative_likelihood(x_fixed, dataset)
    d_dim = int(mu.shape[1]) if mu.ndim == 2 else 1
    d_plot = int(min(3, max(d_dim, 1)))
    for dd in range(d_plot):
        col = "black" if dd == 0 else f"C{dd + 1}"
        mu_d = np.asarray(mu[:, dd], dtype=np.float64).reshape(-1)
        ax_bot.plot(
            th_grid,
            mu_d,
            color=col,
            lw=1.0,
            label=fr"GT $\mu_{{{dd}}}(\theta)$" if d_plot > 1 else r"GT $\mu_0(\theta)$ (1st dim)",
        )
        if gen_match and dd < int(x_fixed.size):
            x_dd = float(x_fixed[dd])
            ax_bot.axhline(
                x_dd,
                color=col,
                ls="--",
                lw=0.8,
                alpha=0.65,
                label=fr"$x_{{{dd}}}$ fixed={x_dd:.3g}",
            )
    tmn = float(np.clip(theta_map, lo, hi))
    jmap = int(np.argmin(np.abs(th_grid - tmn))) if th_grid.size else 0
    y_mark = float(np.asarray(mu[jmap, 0], dtype=np.float64)) if mu.size else float("nan")
    ax_bot.axvline(tmn, color="#1f77b4", alpha=0.45, ls="--", lw=0.9)
    ax_bot.scatter([tmn], [y_mark], s=22, zorder=4, color="#1f77b4")
    ax_bot.set_ylabel("mean / fixed-x")
    ax_bot.set_xlabel(r"$\theta$")
    ax_bot.set_title(r"Generative mean $\mu(\theta)$, fixed-$x$ component guides, and grid MAP $\theta$", fontsize=8)
    ax_bot.legend(loc="best", fontsize=6, ncol=2)


def _write_fixed_x_posterior_diagnostic(
    *,
    run_dir: str,
    persistent_diagnostics_dir: str,
    meta: dict[str, Any],
    perm_seed: int,
    n_subset: int,
    x_aligned: np.ndarray,
) -> str | None:
    """Write ``theta_flow_single_x_posterior_hist`` PNG+SVG for embedding in the combined figure.

    Uses ``c_matrix`` from ``h_matrix_results*.npz`` (requires ``h_save_intermediates``) and
    a deterministic row index ``i`` derived from ``perm_seed`` and ``n_subset``.

    - ``theta_flow``: ``C[i,j] = log p(θ_j|x_i) - log p(θ_j)`` (std-normal base); we add
      ``log p(θ_j)`` back for a softmax "posterior mass" on the training θ grid.
    - ``nf``: ``C[i,j] = log p(θ_j|x_i)`` directly.
    - Other H-field methods: soft-max the C row (scale may be method-specific) for a coarse view.
    """
    os.makedirs(persistent_diagnostics_dir, exist_ok=True)
    out_base = os.path.join(persistent_diagnostics_dir, "theta_flow_single_x_posterior_hist")
    out_png = out_base + ".png"

    ds_fam = str(meta.get("dataset_family", ""))
    suffix = "_non_gauss" if ds_fam == "cosine_gmm" else "_theta_cov"
    h_path = os.path.join(run_dir, f"h_matrix_results{suffix}.npz")
    if not os.path.isfile(h_path):
        print(f"[convergence] fixed-x diagnostic: missing {h_path}", flush=True)
        return None
    z = np.load(h_path, allow_pickle=True)
    if "c_matrix" not in z.files:
        analytic_lxf = (
            bool(np.asarray(z["lxf_analytic_gaussian_hellinger"]).reshape(-1)[0])
            if "lxf_analytic_gaussian_hellinger" in z.files
            else False
        )
        msg = (
            "c_matrix is not available for analytic Gaussian linear-X-flow Hellinger mode; "
            "re-run with --lxf-save-c-matrix to save this diagnostic."
            if analytic_lxf
            else "c_matrix not in H-matrix npz (need h_save_intermediates=True)."
        )
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 2.2), dpi=120, layout="tight")
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _save_figure_png_svg(fig, out_png, dpi=120)
        plt.close(fig)
        return out_png

    c = np.asarray(z["c_matrix"], dtype=np.float64)
    theta_u = np.asarray(z["theta_used"], dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        print(f"[convergence] fixed-x diagnostic: bad c_matrix shape {getattr(c, 'shape', None)}", flush=True)
        return None
    n = int(c.shape[0])
    if n < 2:
        return None
    if theta_u.ndim == 2 and int(theta_u.shape[1]) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 2.2), dpi=120, layout="tight")
        ax.text(
            0.5,
            0.5,
            "Fixed-x posterior diagnostic: requires scalar theta (N×1) for C-row softmax.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _save_figure_png_svg(fig, out_png, dpi=120)
        plt.close(fig)
        return out_png

    th_flat = np.asarray(theta_u, dtype=np.float64).reshape(-1)
    xa = np.asarray(x_aligned, dtype=np.float64)
    if int(xa.shape[0]) != n:
        print(
            f"[convergence] fixed-x diagnostic: x length {xa.shape[0]} != c rows {n}",
            flush=True,
        )
        return None

    hfm_raw = "theta_flow"
    if "h_field_method" in z.files:
        raw = z["h_field_method"]
        try:
            hfm_raw = str(np.asarray(raw).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            hfm_raw = str(hfm_raw)
    hfm = str(hfm_raw).strip().lower()
    i_fix_a, i_fix_b = _select_two_fixed_x_indices(
        n,
        perm_seed=int(perm_seed),
        n_subset=int(n_subset),
    )
    try:
        dataset = build_dataset_from_meta(meta)
        lo = float(meta["theta_low"])
        hi = float(meta["theta_high"])
        th_grid = np.linspace(lo, hi, 400, dtype=np.float64)
        tcol = th_grid.reshape(-1, 1)
        mu = np.asarray(dataset.tuning_curve(tcol), dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.2), dpi=120, sharey=True, layout="tight")
        for ax, i_fix in zip(np.asarray(axes).reshape(-1), (i_fix_a, i_fix_b)):
            c_row = np.asarray(c[int(i_fix), :], dtype=np.float64).reshape(-1)
            logp, logp_source = _model_posterior_log_weights_for_fixed_x(
                hfm=hfm,
                c_row=c_row,
                h_npz=z,
                row=int(i_fix),
            )
            w = _stable_softmax_log(logp)
            order = np.argsort(th_flat, kind="mergesort")
            th_s = th_flat[order]
            w_s = w[order]
            x_fixed = np.asarray(xa[int(i_fix)], dtype=np.float64).reshape(-1)
            x0 = float(x_fixed[0]) if x_fixed.size else float("nan")
            xn = float(np.linalg.norm(x_fixed))
            j_map = int(np.argmax(w))
            theta_map = float(th_flat[j_map])
            ax.fill_between(th_s, 0.0, w_s, color="#1f77b4", alpha=0.35, step="mid")
            ax.plot(
                th_s,
                w_s,
                "o-",
                color="#1f77b4",
                ms=2,
                lw=0.8,
                label=f"softmax weights ({logp_source})",
            )
            ax.set_ylabel("mass")
            ax.set_xlabel(r"$\theta$")
            ax.set_title(
                f"fixed $x$  i={int(i_fix)}  method={hfm}\n(posterior overlay failed: {e!s})",
                fontsize=8,
            )
            ax.annotate(
                f"x[0]={x0:.3g}  ||x||={xn:.3g}  theta_map={theta_map:.3g}",
                xy=(0.5, 1.02),
                xycoords="axes fraction",
                ha="center",
                fontsize=7,
            )
        _save_figure_png_svg(fig, out_png, dpi=120)
        plt.close(fig)
        print(f"[convergence] fixed-x diagnostic -> {out_png}", flush=True)
        return out_png

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 5.0), dpi=120, sharex="col", layout="tight")
    ax00 = cast(plt.Axes, axs[0, 0])
    ax01 = cast(plt.Axes, axs[0, 1])
    ax10 = cast(plt.Axes, axs[1, 0])
    ax11 = cast(plt.Axes, axs[1, 1])
    _plot_fixed_x_column(
        ax_top=ax00,
        ax_bot=ax10,
        i_fix=int(i_fix_a),
        hfm=hfm,
        c=c,
        th_flat=th_flat,
        xa=xa,
        th_grid=th_grid,
        mu=mu,
        dataset=dataset,
        lo=lo,
        hi=hi,
        h_npz=z,
    )
    _plot_fixed_x_column(
        ax_top=ax01,
        ax_bot=ax11,
        i_fix=int(i_fix_b),
        hfm=hfm,
        c=c,
        th_flat=th_flat,
        xa=xa,
        th_grid=th_grid,
        mu=mu,
        dataset=dataset,
        lo=lo,
        hi=hi,
        h_npz=z,
    )

    _save_figure_png_svg(fig, out_png, dpi=120)
    plt.close(fig)
    print(f"[convergence] fixed-x diagnostic -> {out_png}", flush=True)
    return out_png


def _backfill_fixed_x_posterior_diagnostic_if_missing(
    *,
    output_dir: str,
    bundle: SharedDatasetBundle,
    meta: dict[str, Any],
    ns: list[int],
    perm_seed: int,
    n_pool: int,
) -> None:
    """Backfill per-n diagnostic PNGs when old runs have H-matrix NPZs but no diagnostic images."""
    rng = np.random.default_rng(int(perm_seed))
    perm = rng.permutation(int(n_pool))
    ds_fam = str(meta.get("dataset_family", ""))
    suffix = "_non_gauss" if ds_fam == "cosine_gmm" else "_theta_cov"
    for n_raw in ns:
        n = int(n_raw)
        diag_png = os.path.join(
            output_dir,
            "sweep_runs",
            f"n_{n:06d}",
            "diagnostics",
            "theta_flow_single_x_posterior_hist.png",
        )
        if os.path.isfile(diag_png):
            continue
        run_dir = os.path.join(output_dir, "sweep_runs", f"n_{n:06d}")
        h_path = os.path.join(run_dir, f"h_matrix_results{suffix}.npz")
        if not os.path.isfile(h_path):
            continue
        sub = perm[:n]
        x_aligned = np.asarray(bundle.x_all[sub], dtype=np.float64)
        per_diag = os.path.join(output_dir, "sweep_runs", f"n_{n:06d}", "diagnostics")
        _write_fixed_x_posterior_diagnostic(
            run_dir=run_dir,
            persistent_diagnostics_dir=per_diag,
            meta=meta,
            perm_seed=int(perm_seed),
            n_subset=int(n),
            x_aligned=x_aligned,
        )


def _load_per_n_training_loss_npz(path: str) -> dict[str, Any]:
    """Load arrays from ``score_prior_training_losses.npz`` (DSM or flow)."""
    # Object arrays (e.g. ``theta_field_method``) require allow_pickle=True.
    z = np.load(path, allow_pickle=True)

    def _arr(name: str) -> np.ndarray:
        if name not in z.files:
            return np.asarray([], dtype=np.float64)
        return np.asarray(z[name], dtype=np.float64).ravel()

    prior_enable = True
    if "prior_enable" in z.files:
        pv = z["prior_enable"]
        prior_enable = bool(np.asarray(pv).reshape(-1)[0])

    tfm = "theta_flow"
    if "theta_field_method" in z.files:
        raw = z["theta_field_method"]
        if raw.dtype == object or raw.dtype.kind in ("U", "S"):
            tfm = str(np.asarray(raw).reshape(-1)[0])
        else:
            tfm = str(raw)

    return {
        "theta_field_method": tfm,
        "prior_enable": prior_enable,
        "score_train_losses": _arr("score_train_losses"),
        "score_val_losses": _arr("score_val_losses"),
        "score_val_monitor_losses": _arr("score_val_monitor_losses"),
        "score_likelihood_finetune_train_losses": _arr("score_likelihood_finetune_train_losses"),
        "score_likelihood_finetune_val_losses": _arr("score_likelihood_finetune_val_losses"),
        "score_likelihood_finetune_val_monitor_losses": _arr("score_likelihood_finetune_val_monitor_losses"),
        "prior_train_losses": _arr("prior_train_losses"),
        "prior_val_losses": _arr("prior_val_losses"),
        "prior_val_monitor_losses": _arr("prior_val_monitor_losses"),
        "prior_likelihood_finetune_train_losses": _arr("prior_likelihood_finetune_train_losses"),
        "prior_likelihood_finetune_val_losses": _arr("prior_likelihood_finetune_val_losses"),
        "prior_likelihood_finetune_val_monitor_losses": _arr("prior_likelihood_finetune_val_monitor_losses"),
        "gn_pretrain_train_losses": _arr("gn_pretrain_train_losses"),
        "gn_pretrain_val_losses": _arr("gn_pretrain_val_losses"),
        "gn_pretrain_val_monitor_losses": _arr("gn_pretrain_val_monitor_losses"),
    }


def _bundle_has_any_likelihood_finetune(bundle: dict[str, Any]) -> bool:
    """True if any likelihood fine-tune loss array in ``bundle`` is non-empty."""
    keys = (
        "score_likelihood_finetune_train_losses",
        "score_likelihood_finetune_val_losses",
        "score_likelihood_finetune_val_monitor_losses",
        "prior_likelihood_finetune_train_losses",
        "prior_likelihood_finetune_val_losses",
        "prior_likelihood_finetune_val_monitor_losses",
    )
    for k in keys:
        a = bundle.get(k)
        if a is None:
            continue
        if np.asarray(a, dtype=np.float64).size > 0:
            return True
    return False


def _loss_dir_has_any_likelihood_finetune(ns: list[int], loss_dir: str) -> bool:
    for n in ns:
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        if not os.path.isfile(path):
            continue
        try:
            bundle = _load_per_n_training_loss_npz(path)
        except Exception:
            continue
        if _bundle_has_any_likelihood_finetune(bundle):
            return True
    return False


def _bundle_has_gn_pretrain(bundle: dict[str, Any]) -> bool:
    """True if Gaussian-network MLE pretrain curves were saved (contrastive-soft-GN family)."""
    for k in ("gn_pretrain_train_losses", "gn_pretrain_val_losses", "gn_pretrain_val_monitor_losses"):
        a = bundle.get(k)
        if a is None:
            continue
        if np.asarray(a, dtype=np.float64).size > 0:
            return True
    return False


def _loss_dir_has_gn_pretrain(ns: list[int], loss_dir: str) -> bool:
    for n in ns:
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        if not os.path.isfile(path):
            continue
        try:
            bundle = _load_per_n_training_loss_npz(path)
        except Exception:
            continue
        if _bundle_has_gn_pretrain(bundle):
            return True
    return False


def _plot_loss_triplet(
    ax: plt.Axes,
    train: np.ndarray,
    val: np.ndarray,
    ema: np.ndarray,
    *,
    ylabel: str,
    title: str | None,
    show_legend: bool,
    score_like: bool,
) -> None:
    """Plot train / val / EMA monitor curves on one axis (single training phase)."""
    train = np.asarray(train, dtype=np.float64).ravel()
    val = np.asarray(val, dtype=np.float64).ravel()
    ema = np.asarray(ema, dtype=np.float64).ravel()
    lengths = [s.size for s in (train, val, ema) if s.size > 0]
    if not lengths:
        ax.text(0.5, 0.5, "no loss data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return
    n_epoch = int(max(lengths))
    epochs = np.arange(1, n_epoch + 1, dtype=np.float64)
    if train.size > 0:
        m = min(train.size, n_epoch)
        ax.plot(epochs[:m], train[:m], color="#1f77b4", linewidth=1.6, label="train")
    if val.size > 0 and np.any(np.isfinite(val)):
        m = min(val.size, n_epoch)
        ax.plot(epochs[:m], val[:m], color="#d62728", linewidth=1.6, label="val")
    if ema.size > 0 and np.any(np.isfinite(ema)):
        m = min(ema.size, n_epoch)
        label = "val EMA (early-stop)" if score_like else "prior val EMA"
        ax.plot(epochs[:m], ema[:m], color="#ff7f0e", linewidth=1.4, linestyle="--", label=label)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.35)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def _render_training_losses_panel(
    *,
    ns: list[int],
    loss_dir: str,
    out_png_path: str,
    dpi: int = 160,
) -> str:
    """Two or four rows (FM pretrain; optional NLL fine-tune), one column per ``n``; save PNG + SVG."""
    n_cols = len(ns)
    if n_cols < 1:
        raise ValueError("n-list must be non-empty for training loss panel.")
    want_ft = _loss_dir_has_any_likelihood_finetune(ns, loss_dir)
    n_loss_rows = 4 if want_ft else 2
    w = max(2.6 * n_cols, 6.0)
    h = 5.8 * (n_loss_rows / 2.0)
    sharex_kw: dict[str, Any] = {"sharex": "col"} if n_loss_rows == 2 else {"sharex": False}
    fig, axes = plt.subplots(n_loss_rows, n_cols, figsize=(w, h), squeeze=False, **sharex_kw)

    row0_ylabel = "score / posterior loss" if not want_ft else "posterior FM loss"
    row1_ylabel = "prior loss" if not want_ft else "prior FM loss"
    row2_ylabel = "posterior NLL fine-tune"
    row3_ylabel = "prior NLL fine-tune"

    for j, n in enumerate(ns):
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        if not os.path.isfile(path):
            for r in range(n_loss_rows):
                axes[r, j].text(
                    0.5,
                    0.5,
                    f"missing\n{path}",
                    ha="center",
                    va="center",
                    transform=axes[r, j].transAxes,
                    fontsize=8,
                    color="crimson",
                )
                axes[r, j].set_axis_off()
            continue

        try:
            bundle = _load_per_n_training_loss_npz(path)
        except Exception as e:
            for r in range(n_loss_rows):
                axes[r, j].text(
                    0.5,
                    0.5,
                    f"load error:\n{e!s}"[:200],
                    ha="center",
                    va="center",
                    transform=axes[r, j].transAxes,
                    fontsize=7,
                    color="crimson",
                )
                axes[r, j].set_axis_off()
            continue

        tfm = str(bundle.get("theta_field_method", "theta_flow")).strip().lower()
        if tfm == "theta_flow":
            post_lab = "theta-flow ODE Bayes-ratio"
        elif tfm == "theta_flow_autoencoder":
            post_lab = "theta-flow AE Bayes-ratio"
        elif tfm == "theta_path_integral":
            post_lab = "theta-path-integral score"
        elif tfm == "x_flow":
            post_lab = "x-flow direct likelihood"
        elif tfm == "x_flow_autoencoder":
            post_lab = "x-flow AE likelihood"
        elif tfm == "x_flow_pca":
            post_lab = "x-flow PCA likelihood"
        elif tfm == "sir_xflow_lrank_t":
            post_lab = "SIR + linear-x-flow low-rank likelihood"
        elif tfm == "sir_xflow":
            post_lab = "SIR + x-flow likelihood"
        elif tfm == "sir_thetaflow":
            post_lab = "SIR + theta-flow likelihood"
        elif tfm == "ctsm_v":
            post_lab = "pair-conditioned CTSM-v"
        elif tfm == "nf":
            post_lab = "normalizing-flow posterior"
        elif tfm == "nf_reduction":
            post_lab = "NF-reduction z likelihood"
        elif tfm == "pi_nf":
            post_lab = "pi-NF exact NLL"
        elif tfm == "gaussian_network":
            post_lab = "gaussian-network likelihood"
        elif tfm == "gaussian_network_diagonal":
            post_lab = "gaussian-network diagonal likelihood"
        elif tfm == "gaussian_network_diagonal_binned_pca":
            post_lab = "gaussian-network diagonal binned-PCA likelihood"
        elif tfm == "gaussian_network_low_rank":
            post_lab = "gaussian-network low-rank likelihood"
        elif tfm == "gaussian_network_autoencoder":
            post_lab = "gaussian-network AE likelihood"
        elif tfm == "gaussian_network_diagonal_autoencoder":
            post_lab = "gaussian-network diagonal AE likelihood"
        elif tfm == "gaussian_x_flow":
            post_lab = "gaussian-x-flow FM likelihood"
        elif tfm == "gaussian_x_flow_diagonal":
            post_lab = "gaussian-x-flow-diagonal FM likelihood"
        elif tfm == "linear_x_flow":
            post_lab = "linear-x-flow FM likelihood"
        elif tfm == "linear_x_flow_t":
            post_lab = "linear-x-flow time FM likelihood"
        elif tfm == "linear_x_flow_scalar_t":
            post_lab = "linear-x-flow scalar-t FM likelihood"
        elif tfm == "linear_x_flow_diagonal_theta_t":
            post_lab = "linear-x-flow diagonal-theta-t FM likelihood"
        elif tfm == "linear_x_flow_diagonal_t":
            post_lab = "linear-x-flow diagonal-t FM likelihood"
        elif tfm == "linear_x_flow_low_rank_t":
            post_lab = "linear-x-flow full-A(t) + low-rank correction FM likelihood"
        elif tfm == "xflow_sir_lrank":
            post_lab = "x-flow SIR low-rank correction FM likelihood"
        elif tfm == "xflow_sir_lrank_dia":
            post_lab = "x-flow SIR low-rank correction diagonal-A(t) FM likelihood"
        elif tfm == "xflow_sir_lrank_dia_theta":
            post_lab = "x-flow SIR low-rank correction diagonal-A(theta,t) FM likelihood"
        elif tfm == "xflow_sir_lrank_scalar":
            post_lab = "x-flow SIR low-rank correction scalar-A(t) FM likelihood"
        elif tfm == "xflow_sir_lrank_scalar_theta":
            post_lab = "x-flow SIR low-rank correction scalar-A(theta,t) FM likelihood"
        elif tfm == "xflow_sir_pure_lrank":
            post_lab = "x-flow SIR pure low-rank U h(U^T x) FM likelihood"
        elif tfm == "linear_x_flow_pure_low_rank_t":
            post_lab = "linear-x-flow pure low-rank U h(U^T x) FM likelihood"
        elif tfm == "linear_x_flow_pure_cond_low_rank_t":
            post_lab = "linear-x-flow pure conditional U(theta,t) h(U^T x) FM likelihood"
        elif tfm == "linear_x_flow_lr_t_ts":
            post_lab = "linear-x-flow full-A(t) + low-rank correction b(theta) FM likelihood"
        elif tfm == "linear_x_flow_low_rank_randb_t":
            post_lab = "linear-x-flow random-basis low-rank-t FM likelihood"
        elif tfm == "linear_theta_flow":
            post_lab = "linear-theta-flow mixture FM likelihood"
        else:
            post_lab = tfm
        _plot_loss_triplet(
            axes[0, j],
            bundle["score_train_losses"],
            bundle["score_val_losses"],
            bundle["score_val_monitor_losses"],
            ylabel=row0_ylabel if j == 0 else "",
            title=f"n={n} ({post_lab})",
            show_legend=(j == 0),
            score_like=True,
        )

        if bundle["prior_enable"]:
            _plot_loss_triplet(
                axes[1, j],
                bundle["prior_train_losses"],
                bundle["prior_val_losses"],
                bundle["prior_val_monitor_losses"],
                ylabel=row1_ylabel if j == 0 else "",
                title=None,
                show_legend=(j == 0),
                score_like=False,
            )
        else:
            axes[1, j].text(
                0.5,
                0.5,
                "prior disabled",
                ha="center",
                va="center",
                transform=axes[1, j].transAxes,
                fontsize=10,
            )
            axes[1, j].set_axis_off()

        if want_ft:
            if _bundle_has_any_likelihood_finetune(bundle):
                _plot_loss_triplet(
                    axes[2, j],
                    bundle["score_likelihood_finetune_train_losses"],
                    bundle["score_likelihood_finetune_val_losses"],
                    bundle["score_likelihood_finetune_val_monitor_losses"],
                    ylabel=row2_ylabel if j == 0 else "",
                    title=None,
                    show_legend=(j == 0),
                    score_like=True,
                )
            else:
                axes[2, j].text(
                    0.5,
                    0.5,
                    "no NLL fine-tune\nin this run",
                    ha="center",
                    va="center",
                    transform=axes[2, j].transAxes,
                    fontsize=9,
                    color="#555555",
                )
                axes[2, j].set_axis_off()

            if bundle["prior_enable"]:
                if (
                    np.asarray(bundle["prior_likelihood_finetune_train_losses"], dtype=np.float64).size > 0
                    or np.asarray(bundle["prior_likelihood_finetune_val_losses"], dtype=np.float64).size > 0
                    or np.asarray(bundle["prior_likelihood_finetune_val_monitor_losses"], dtype=np.float64).size
                    > 0
                ):
                    _plot_loss_triplet(
                        axes[3, j],
                        bundle["prior_likelihood_finetune_train_losses"],
                        bundle["prior_likelihood_finetune_val_losses"],
                        bundle["prior_likelihood_finetune_val_monitor_losses"],
                        ylabel=row3_ylabel if j == 0 else "",
                        title=None,
                        show_legend=(j == 0),
                        score_like=False,
                    )
                else:
                    axes[3, j].text(
                        0.5,
                        0.5,
                        "no prior NLL\nfine-tune data",
                        ha="center",
                        va="center",
                        transform=axes[3, j].transAxes,
                        fontsize=9,
                        color="#555555",
                    )
                    axes[3, j].set_axis_off()
            else:
                axes[3, j].text(
                    0.5,
                    0.5,
                    "prior disabled",
                    ha="center",
                    va="center",
                    transform=axes[3, j].transAxes,
                    fontsize=10,
                )
                axes[3, j].set_axis_off()

    if want_ft:
        fig.suptitle(
            "Training loss vs epoch (top two rows: FM pretrain; bottom two rows: NLL fine-tune). "
            "Columns: nested subset sizes n.",
            fontsize=11,
            y=1.01,
        )
    else:
        fig.suptitle(
            "Training loss vs epoch (top: posterior; bottom: prior). Columns: nested subset sizes n.",
            fontsize=11,
            y=1.02,
        )
    fig.tight_layout()
    svg = _save_figure_png_svg(fig, out_png_path, dpi=dpi)
    plt.close(fig)
    return svg


def _render_gn_pretrain_losses_panel(
    *,
    ns: list[int],
    loss_dir: str,
    out_png_path: str,
    dpi: int = 160,
) -> str:
    """Gaussian-network MLE pretrain NLL vs epoch (separate from main training-loss panel)."""
    n_cols = len(ns)
    if n_cols < 1:
        raise ValueError("n-list must be non-empty for GN pretrain loss panel.")
    w = max(2.6 * n_cols, 6.0)
    h = 3.4
    fig, axes = plt.subplots(1, n_cols, figsize=(w, h), squeeze=False)

    for j, n in enumerate(ns):
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        ax = axes[0, j]
        if not os.path.isfile(path):
            ax.text(
                0.5,
                0.5,
                f"missing\n{path}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="crimson",
            )
            ax.set_axis_off()
            continue
        try:
            bundle = _load_per_n_training_loss_npz(path)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"load error:\n{e!s}"[:200],
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
                color="crimson",
            )
            ax.set_axis_off()
            continue

        tr = np.asarray(bundle.get("gn_pretrain_train_losses", []), dtype=np.float64).ravel()
        va = np.asarray(bundle.get("gn_pretrain_val_losses", []), dtype=np.float64).ravel()
        ema = np.asarray(bundle.get("gn_pretrain_val_monitor_losses", []), dtype=np.float64).ravel()
        if tr.size == 0 and va.size == 0 and ema.size == 0:
            ax.text(
                0.5,
                0.5,
                "no gn_pretrain\nloss arrays",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#555555",
            )
            ax.set_axis_off()
            continue

        _plot_loss_triplet(
            ax,
            tr,
            va,
            ema,
            ylabel="NLL" if j == 0 else "",
            title=f"n={n} (GN MLE pretrain)",
            show_legend=(j == 0),
            score_like=True,
        )

    fig.suptitle(
        "Gaussian-network pretraining loss vs epoch (train/val/val EMA). "
        "Columns: nested subset sizes n.",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    svg = _save_figure_png_svg(fig, out_png_path, dpi=dpi)
    plt.close(fig)
    return svg


def _save_combined_convergence_figure(
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    n_bins: int,
    theta_centers: np.ndarray,
    ns: list[int],
    corr_h: np.ndarray,
    corr_clf: np.ndarray,
    loss_dir: str,
    diagnostic_png_paths: list[str | None],
    out_png_path: str,
    dpi: int = 160,
    llr_gt: np.ndarray | None = None,
    llr_est_mats: list[np.ndarray] | None = None,
    corr_llr: np.ndarray | None = None,
    binned_gaussian_corr_h: np.ndarray | None = None,
    extra_h_rows: list[tuple[str, list[np.ndarray]]] | None = None,
) -> str:
    """Single figure with matrix panel, correlation curves, H and LLR est-vs-GT scatters, losses, and optional diagnostic.

    PNG is raster as usual. SVG keeps the right-hand curve as vector paths (not a single
    embedded screenshot); heatmaps still use matplotlib's normal SVG image handling for ``imshow``.

    With LLR data: top row = matrix + curves; then H scatter row; then LLR scatter row (binned
    model ``ΔL`` vs generative one-sided mean LLR); then training losses; then diagnostic.
    If ``llr_gt``/``llr_est_mats``/``corr_llr`` are omitted, the LLR row is skipped (older runs).
    The training-loss strip uses two subplot rows by default, or four rows when any per-n
    ``training_losses`` NPZ contains non-empty likelihood fine-tune curves (separate y-scales).
    """
    crv_w, crv_h = _H_DECODING_CURVE_FIGSIZE_IN
    if crv_h <= 0:
        raise ValueError("_H_DECODING_CURVE_FIGSIZE_IN height must be > 0.")
    n_cols = len(h_mats)
    n_loss_cols = len(ns)
    want_ft = _loss_dir_has_any_likelihood_finetune(list(ns), loss_dir)
    n_loss_rows = 4 if want_ft else 2
    n_matrix_rows = 2 + len(extra_h_rows or [])
    m_w, m_h = 2.8 * n_cols, 2.5 * n_matrix_rows
    l_w, l_h = max(2.6 * n_loss_cols, 6.0), 5.8 * (n_loss_rows / 2.0)
    top_h = float(m_h)
    bot_h = float(l_h)
    fig_w = max(m_w + (top_h * (float(crv_w) / float(crv_h))), l_w)
    # Constrained layout: colorbars make tight_layout warn and mis-place panels.
    scatter_h = 2.8
    llr_h = 2.8
    diag_h = 5.2
    if len(h_mats) < 2 or len(h_mats) != len(ns) + 1:
        raise ValueError(
            "h_mats must have length len(ns)+1 (sweep columns + one GT / n_ref column) for "
            f"est-vs-GT scatter; got len(h_mats)={len(h_mats)} len(ns)={len(ns)}."
        )
    if int(np.asarray(corr_h, dtype=np.float64).ravel().size) != len(ns):
        raise ValueError("corr_h must have one entry per n in --n-list.")
    if len(diagnostic_png_paths) != len(ns):
        raise ValueError(
            "diagnostic_png_paths must have one entry per n in --n-list; "
            f"got {len(diagnostic_png_paths)} paths for {len(ns)} n values."
        )
    h_gt = h_mats[-1]
    use_llr = (
        llr_gt is not None
        and llr_est_mats is not None
        and corr_llr is not None
        and int(np.asarray(corr_llr, dtype=np.float64).ravel().size) == len(ns)
        and len(llr_est_mats) == len(ns)
    )
    if use_llr and np.asarray(llr_gt, dtype=np.float64).shape != (int(n_bins), int(n_bins)):
        raise ValueError("llr_gt must be (n_bins, n_bins).")
    n_grid_rows = 5 if use_llr else 4
    fig_h = top_h + scatter_h + (llr_h if use_llr else 0.0) + bot_h + diag_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    height_rows = [top_h, scatter_h, bot_h, diag_h]
    if use_llr:
        height_rows = [top_h, scatter_h, llr_h, bot_h, diag_h]
    gs0 = fig.add_gridspec(
        n_grid_rows,
        2,
        width_ratios=[m_w, top_h * (float(crv_w) / float(crv_h))],
        height_ratios=height_rows,
    )
    gs_left = gs0[0, 0].subgridspec(n_matrix_rows, n_cols)
    axes_m = np.empty((n_matrix_rows, n_cols), dtype=object)
    for r in range(n_matrix_rows):
        for c in range(n_cols):
            axes_m[r, c] = fig.add_subplot(gs_left[r, c])
    _populate_matrix_panel_axes(
        axes_m,
        h_mats=h_mats,
        clf_mats=clf_mats,
        col_labels=col_labels,
        n_bins=n_bins,
        theta_centers=theta_centers,
        extra_h_rows=extra_h_rows,
    )
    ax_c = fig.add_subplot(gs0[0, 1])
    _populate_convergence_curve_ax(
        ax_c,
        list(ns),
        corr_h,
        corr_clf,
        binned_gaussian_corr_h=binned_gaussian_corr_h,
        tick_labelsize=13.0,
        axis_labelsize=13.0,
        legend_fontsize=10.0,
    )
    n_scat = int(len(ns))
    gs_s = gs0[1, :].subgridspec(1, max(1, n_scat))
    for j, n in enumerate(ns):
        ax_s = fig.add_subplot(gs_s[0, j])
        _plot_estimated_vs_gt_h_scatter(
            ax_s,
            est=np.asarray(h_mats[j], dtype=np.float64),
            gt=np.asarray(h_gt, dtype=np.float64),
            n=int(n),
            r_offdiag=float(corr_h[j]),
        )
    if n_scat == 0:
        ax0 = fig.add_subplot(gs_s[0, 0])
        ax0.text(0.5, 0.5, "empty n-list", ha="center", va="center", transform=ax0.transAxes, fontsize=9)
        ax0.set_axis_off()

    loss_row = 3 if use_llr else 2
    if use_llr and llr_gt is not None and llr_est_mats is not None and corr_llr is not None:
        gs_llr = gs0[2, :].subgridspec(1, max(1, n_scat))
        for j, n in enumerate(ns):
            ax_l = fig.add_subplot(gs_llr[0, j])
            _plot_estimated_vs_gt_llr_scatter(
                ax_l,
                est=np.asarray(llr_est_mats[j], dtype=np.float64),
                gt=np.asarray(llr_gt, dtype=np.float64),
                n=int(n),
                r_offdiag=float(corr_llr[j]),
            )

    gs_loss = gs0[loss_row, :].subgridspec(n_loss_rows, n_loss_cols)
    axes_loss = np.empty((n_loss_rows, n_loss_cols), dtype=object)
    row0_ylabel = "score / posterior loss" if not want_ft else "posterior FM loss"
    row1_ylabel = "prior loss" if not want_ft else "prior FM loss"
    row2_ylabel = "posterior NLL fine-tune"
    row3_ylabel = "prior NLL fine-tune"
    for j, n in enumerate(ns):
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        for r in range(n_loss_rows):
            axes_loss[r, j] = fig.add_subplot(gs_loss[r, j])
        if not os.path.isfile(path):
            for r in range(n_loss_rows):
                axes_loss[r, j].text(
                    0.5,
                    0.5,
                    f"missing\n{path}",
                    ha="center",
                    va="center",
                    transform=axes_loss[r, j].transAxes,
                    fontsize=8,
                    color="crimson",
                )
                axes_loss[r, j].set_axis_off()
            continue
        try:
            bundle = _load_per_n_training_loss_npz(path)
        except Exception as e:
            for r in range(n_loss_rows):
                axes_loss[r, j].text(
                    0.5,
                    0.5,
                    f"load error:\n{e!s}"[:200],
                    ha="center",
                    va="center",
                    transform=axes_loss[r, j].transAxes,
                    fontsize=7,
                    color="crimson",
                )
                axes_loss[r, j].set_axis_off()
            continue
        tfm = str(bundle.get("theta_field_method", "theta_flow")).strip().lower()
        if tfm == "theta_flow":
            post_lab = "theta-flow ODE Bayes-ratio"
        elif tfm == "theta_flow_autoencoder":
            post_lab = "theta-flow AE Bayes-ratio"
        elif tfm == "theta_path_integral":
            post_lab = "theta-path-integral score"
        elif tfm == "x_flow":
            post_lab = "x-flow direct likelihood"
        elif tfm == "x_flow_autoencoder":
            post_lab = "x-flow AE likelihood"
        elif tfm == "x_flow_pca":
            post_lab = "x-flow PCA likelihood"
        elif tfm == "sir_xflow_lrank_t":
            post_lab = "SIR + linear-x-flow low-rank likelihood"
        elif tfm == "sir_xflow":
            post_lab = "SIR + x-flow likelihood"
        elif tfm == "sir_thetaflow":
            post_lab = "SIR + theta-flow likelihood"
        elif tfm == "ctsm_v":
            post_lab = "pair-conditioned CTSM-v"
        elif tfm == "nf":
            post_lab = "normalizing-flow posterior"
        elif tfm == "nf_reduction":
            post_lab = "NF-reduction z likelihood"
        elif tfm == "pi_nf":
            post_lab = "pi-NF exact NLL"
        elif tfm == "gaussian_network":
            post_lab = "gaussian-network likelihood"
        elif tfm == "gaussian_network_diagonal":
            post_lab = "gaussian-network diagonal likelihood"
        elif tfm == "gaussian_network_diagonal_binned_pca":
            post_lab = "gaussian-network diagonal binned-PCA likelihood"
        elif tfm == "gaussian_network_low_rank":
            post_lab = "gaussian-network low-rank likelihood"
        elif tfm == "gaussian_network_autoencoder":
            post_lab = "gaussian-network AE likelihood"
        elif tfm == "gaussian_network_diagonal_autoencoder":
            post_lab = "gaussian-network diagonal AE likelihood"
        elif tfm == "gaussian_x_flow":
            post_lab = "gaussian-x-flow FM likelihood"
        elif tfm == "gaussian_x_flow_diagonal":
            post_lab = "gaussian-x-flow-diagonal FM likelihood"
        elif tfm == "linear_x_flow":
            post_lab = "linear-x-flow FM likelihood"
        elif tfm == "linear_x_flow_t":
            post_lab = "linear-x-flow time FM likelihood"
        elif tfm == "linear_x_flow_scalar_t":
            post_lab = "linear-x-flow scalar-t FM likelihood"
        elif tfm == "linear_x_flow_diagonal_theta_t":
            post_lab = "linear-x-flow diagonal-theta-t FM likelihood"
        elif tfm == "linear_x_flow_diagonal_t":
            post_lab = "linear-x-flow diagonal-t FM likelihood"
        elif tfm == "linear_x_flow_low_rank_t":
            post_lab = "linear-x-flow full-A(t) + low-rank correction FM likelihood"
        elif tfm == "xflow_sir_lrank":
            post_lab = "x-flow SIR low-rank correction FM likelihood"
        elif tfm == "xflow_sir_lrank_dia":
            post_lab = "x-flow SIR low-rank correction diagonal-A(t) FM likelihood"
        elif tfm == "xflow_sir_lrank_dia_theta":
            post_lab = "x-flow SIR low-rank correction diagonal-A(theta,t) FM likelihood"
        elif tfm == "xflow_sir_lrank_scalar":
            post_lab = "x-flow SIR low-rank correction scalar-A(t) FM likelihood"
        elif tfm == "xflow_sir_lrank_scalar_theta":
            post_lab = "x-flow SIR low-rank correction scalar-A(theta,t) FM likelihood"
        elif tfm == "xflow_sir_pure_lrank":
            post_lab = "x-flow SIR pure low-rank U h(U^T x) FM likelihood"
        elif tfm == "linear_x_flow_pure_low_rank_t":
            post_lab = "linear-x-flow pure low-rank U h(U^T x) FM likelihood"
        elif tfm == "linear_x_flow_pure_cond_low_rank_t":
            post_lab = "linear-x-flow pure conditional U(theta,t) h(U^T x) FM likelihood"
        elif tfm == "linear_x_flow_lr_t_ts":
            post_lab = "linear-x-flow full-A(t) + low-rank correction b(theta) FM likelihood"
        elif tfm == "linear_x_flow_low_rank_randb_t":
            post_lab = "linear-x-flow random-basis low-rank-t FM likelihood"
        elif tfm == "linear_theta_flow":
            post_lab = "linear-theta-flow mixture FM likelihood"
        else:
            post_lab = tfm
        _plot_loss_triplet(
            axes_loss[0, j],
            bundle["score_train_losses"],
            bundle["score_val_losses"],
            bundle["score_val_monitor_losses"],
            ylabel=row0_ylabel if j == 0 else "",
            title=f"n={n} ({post_lab})",
            show_legend=(j == 0),
            score_like=True,
        )
        if bundle["prior_enable"]:
            _plot_loss_triplet(
                axes_loss[1, j],
                bundle["prior_train_losses"],
                bundle["prior_val_losses"],
                bundle["prior_val_monitor_losses"],
                ylabel=row1_ylabel if j == 0 else "",
                title=None,
                show_legend=(j == 0),
                score_like=False,
            )
        else:
            axes_loss[1, j].text(
                0.5,
                0.5,
                "prior disabled",
                ha="center",
                va="center",
                transform=axes_loss[1, j].transAxes,
                fontsize=10,
            )
            axes_loss[1, j].set_axis_off()

        if want_ft:
            if _bundle_has_any_likelihood_finetune(bundle):
                _plot_loss_triplet(
                    axes_loss[2, j],
                    bundle["score_likelihood_finetune_train_losses"],
                    bundle["score_likelihood_finetune_val_losses"],
                    bundle["score_likelihood_finetune_val_monitor_losses"],
                    ylabel=row2_ylabel if j == 0 else "",
                    title=None,
                    show_legend=(j == 0),
                    score_like=True,
                )
            else:
                axes_loss[2, j].text(
                    0.5,
                    0.5,
                    "no NLL fine-tune\nin this run",
                    ha="center",
                    va="center",
                    transform=axes_loss[2, j].transAxes,
                    fontsize=8,
                    color="#555555",
                )
                axes_loss[2, j].set_axis_off()

            if bundle["prior_enable"]:
                if (
                    np.asarray(bundle["prior_likelihood_finetune_train_losses"], dtype=np.float64).size > 0
                    or np.asarray(bundle["prior_likelihood_finetune_val_losses"], dtype=np.float64).size > 0
                    or np.asarray(bundle["prior_likelihood_finetune_val_monitor_losses"], dtype=np.float64).size
                    > 0
                ):
                    _plot_loss_triplet(
                        axes_loss[3, j],
                        bundle["prior_likelihood_finetune_train_losses"],
                        bundle["prior_likelihood_finetune_val_losses"],
                        bundle["prior_likelihood_finetune_val_monitor_losses"],
                        ylabel=row3_ylabel if j == 0 else "",
                        title=None,
                        show_legend=(j == 0),
                        score_like=False,
                    )
                else:
                    axes_loss[3, j].text(
                        0.5,
                        0.5,
                        "no prior NLL\nfine-tune data",
                        ha="center",
                        va="center",
                        transform=axes_loss[3, j].transAxes,
                        fontsize=8,
                        color="#555555",
                    )
                    axes_loss[3, j].set_axis_off()
            else:
                axes_loss[3, j].text(
                    0.5,
                    0.5,
                    "prior disabled",
                    ha="center",
                    va="center",
                    transform=axes_loss[3, j].transAxes,
                    fontsize=10,
                )
                axes_loss[3, j].set_axis_off()

    diag_row = 4 if use_llr else 3
    gs_diag = gs0[diag_row, :].subgridspec(1, max(1, len(ns)))
    for j, n in enumerate(ns):
        ax_diag = fig.add_subplot(gs_diag[0, j])
        diagnostic_png_path = diagnostic_png_paths[j]
        if diagnostic_png_path is None or not os.path.isfile(str(diagnostic_png_path)):
            ax_diag.text(
                0.5,
                0.5,
                "Fixed-x posterior+tuning diagnostic not found.\n"
                f"Expected: sweep_runs/n_{int(n):06d}/diagnostics/theta_flow_single_x_posterior_hist.png",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax_diag.set_title(f"n={int(n)}", fontsize=10)
            ax_diag.set_axis_off()
        else:
            try:
                img = plt.imread(str(diagnostic_png_path))
                ax_diag.imshow(img)
                ax_diag.set_title(f"Fixed-x posterior+tuning diagnostic, n={int(n)}", fontsize=10)
                ax_diag.axis("off")
            except Exception as e:
                ax_diag.text(
                    0.5,
                    0.5,
                    f"Failed to load diagnostic image:\n{e!s}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="crimson",
                )
                ax_diag.set_title(f"n={int(n)}", fontsize=10)
                ax_diag.set_axis_off()

    path_svg = _save_figure_png_svg(fig, out_png_path, dpi=dpi)
    plt.close(fig)
    return path_svg


def _finite_min_max(matrices: list[np.ndarray]) -> tuple[float, float]:
    """Shared color scale from stacked finite values (ignores NaN)."""
    parts: list[np.ndarray] = []
    for m in matrices:
        a = np.asarray(m, dtype=np.float64).ravel()
        parts.append(a[np.isfinite(a)])
    if not parts:
        return 0.0, 1.0
    v = np.concatenate(parts)
    if v.size == 0:
        return 0.0, 1.0
    return float(np.min(v)), float(np.max(v))


def _format_theta_tick_labels(theta_centers: np.ndarray) -> list[str]:
    """String tick labels from bin-center θ values (matches ``imshow`` index ticks 0..n_bins-1)."""
    tc = np.asarray(theta_centers, dtype=np.float64).ravel()
    return [f"{float(v):.1f}" for v in tc]


def _matrix_panel_tick_indices(n_bins: int, *, max_ticks: int = 5) -> np.ndarray:
    """Sparse integer bin indices (inclusive ends) so matrix panels show fewer ticks."""
    nb = int(n_bins)
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    k = min(max(2, int(max_ticks)), nb)
    if nb <= k:
        return np.arange(nb, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, nb - 1, k)).astype(np.int64))


def _matrix_axes_show_top_right_spines(ax: Any) -> None:
    """Override global_setting spine defaults for matrix heatmaps."""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def _populate_matrix_panel_axes(
    axes: np.ndarray,
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    n_bins: int,
    theta_centers: np.ndarray,
    extra_h_rows: list[tuple[str, list[np.ndarray]]] | None = None,
) -> None:
    """Draw H rows plus pairwise decoding on existing axes."""
    tc = np.asarray(theta_centers, dtype=np.float64).ravel()
    if int(tc.size) != int(n_bins):
        raise ValueError(f"theta_centers length {tc.size} must match n_bins={n_bins}.")
    tick_labs_full = _format_theta_tick_labels(tc)
    tick_idx = _matrix_panel_tick_indices(n_bins, max_ticks=5)
    tick_pos = tick_idx.tolist()
    tick_labs = [tick_labs_full[int(i)] for i in tick_idx]
    x_rot = 45 if len(tick_pos) > 6 else 0
    _matrix_tick_labelsize = 11
    _matrix_colorbar_tick_labelsize = 11
    n_cols = len(h_mats)
    if n_cols != len(clf_mats) or n_cols != len(col_labels):
        raise ValueError("h_mats, clf_mats, col_labels length mismatch.")
    extra_rows = list(extra_h_rows or [])
    n_h_rows = 1 + len(extra_rows)
    expected_shape = (n_h_rows + 1, n_cols)
    if axes.shape != expected_shape:
        raise ValueError(f"axes must be shape {expected_shape}; got {axes.shape}.")

    vmin_h, vmax_h = 0.0, 1.0
    vmin_c, vmax_c = _finite_min_max(clf_mats)
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12
    cmap = "viridis"

    for c in range(n_cols):
        ax0 = axes[0, c]
        im0 = ax0.imshow(
            h_mats[c],
            vmin=vmin_h,
            vmax=vmax_h,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        ax0.set_title(col_labels[c], fontsize=10)
        ax0.set_xticks(tick_pos)
        ax0.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=_matrix_tick_labelsize)
        ax0.set_yticks(tick_pos)
        ax0.set_yticklabels(tick_labs, fontsize=_matrix_tick_labelsize)
        ax0.tick_params(axis="both", labelsize=_matrix_tick_labelsize)
        _matrix_axes_show_top_right_spines(ax0)
        if c == 0:
            ax0.set_ylabel(r"$\sqrt{H^2}$", fontsize=11)
        _cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        _cb0.ax.tick_params(labelsize=_matrix_colorbar_tick_labelsize)

        for r_extra, (row_label, row_mats) in enumerate(extra_rows, start=1):
            if len(row_mats) != n_cols:
                raise ValueError(f"extra H row {row_label!r} has {len(row_mats)} mats; expected {n_cols}.")
            ax_extra = axes[r_extra, c]
            im_extra = ax_extra.imshow(
                row_mats[c],
                vmin=vmin_h,
                vmax=vmax_h,
                cmap=cmap,
                aspect="equal",
                origin="lower",
            )
            ax_extra.set_xticks(tick_pos)
            ax_extra.set_xticklabels(
                tick_labs,
                rotation=x_rot,
                ha="right" if x_rot else "center",
                fontsize=_matrix_tick_labelsize,
            )
            ax_extra.set_yticks(tick_pos)
            ax_extra.set_yticklabels(tick_labs, fontsize=_matrix_tick_labelsize)
            ax_extra.tick_params(axis="both", labelsize=_matrix_tick_labelsize)
            _matrix_axes_show_top_right_spines(ax_extra)
            if c == 0:
                ax_extra.set_ylabel(row_label, fontsize=11)
            _cb_extra = plt.colorbar(im_extra, ax=ax_extra, fraction=0.046, pad=0.04)
            _cb_extra.ax.tick_params(labelsize=_matrix_colorbar_tick_labelsize)

        ax1 = axes[n_h_rows, c]
        im1 = ax1.imshow(
            clf_mats[c],
            vmin=vmin_c,
            vmax=vmax_c,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=_matrix_tick_labelsize)
        ax1.set_yticks(tick_pos)
        ax1.set_yticklabels(tick_labs, fontsize=_matrix_tick_labelsize)
        ax1.tick_params(axis="both", labelsize=_matrix_tick_labelsize)
        _matrix_axes_show_top_right_spines(ax1)
        if c == 0:
            ax1.set_ylabel("Pairwise decoding", fontsize=11)
        ax1.set_xlabel(r"$\theta$", fontsize=11)
        _cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        _cb1.ax.tick_params(labelsize=_matrix_colorbar_tick_labelsize)


def _populate_convergence_curve_ax(
    ax: plt.Axes,
    ns: list[int],
    corr_h: np.ndarray,
    corr_clf: np.ndarray,
    *,
    binned_gaussian_corr_h: np.ndarray | None = None,
    tick_labelsize: float = 11.0,
    axis_labelsize: float = 12.0,
    legend_fontsize: float = 9.0,
) -> None:
    """Pearson r vs n on a single axis (shared with standalone curve figure + combined figure).

    Tick sizes default to ``global_setting``-style values (11 pt ticks, 12 pt axis labels).
    The combined figure passes larger values so the right panel stays readable at reduced width.
    """
    ns_list = [int(x) for x in ns]
    ch = np.asarray(corr_h, dtype=np.float64).ravel()
    cc = np.asarray(corr_clf, dtype=np.float64).ravel()
    ax.plot(
        ns_list,
        ch,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=6,
        label="H matrix",
    )
    ax.plot(
        ns_list,
        cc,
        color="#d62728",
        linewidth=1.8,
        marker="s",
        markersize=5,
        label="decoding acc matrix",
    )
    if binned_gaussian_corr_h is not None:
        cbg = np.asarray(binned_gaussian_corr_h, dtype=np.float64).ravel()
        if int(cbg.size) != len(ns_list):
            raise ValueError("binned_gaussian_corr_h must have one entry per n in --n-list.")
        if np.any(np.isfinite(cbg)):
            ax.plot(
                ns_list,
                cbg,
                color="#2ca02c",
                linewidth=1.8,
                linestyle="-.",
                marker="^",
                markersize=6,
                label="binned Gaussian H matrix",
            )
    ax.axhline(
        1.0,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
        clip_on=False,
    )
    ax.set_xlabel("dataset size n", fontsize=axis_labelsize)
    ax.set_ylabel("Pearson r (off-diagonal estimated vs off-diagonal approx GT)", fontsize=axis_labelsize)
    ax.set_xticks(ns_list)
    ax.tick_params(axis="both", labelsize=tick_labelsize)
    ax.legend(loc="best", fontsize=legend_fontsize)


def _render_matrix_panel(
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    out_path: str,
    n_bins: int,
    theta_centers: np.ndarray,
    extra_h_rows: list[tuple[str, list[np.ndarray]]] | None = None,
) -> None:
    """Rows: learned sqrt(H), optional auxiliary H rows, and pairwise decoding."""
    n_cols = len(h_mats)
    n_rows = 2 + len(extra_h_rows or [])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), squeeze=False)
    _populate_matrix_panel_axes(
        axes,
        h_mats=h_mats,
        clf_mats=clf_mats,
        col_labels=col_labels,
        n_bins=n_bins,
        theta_centers=theta_centers,
        extra_h_rows=extra_h_rows,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.12)
    _save_figure_png_svg(fig, out_path, dpi=160)
    plt.close(fig)


def _write_summary(
    path: str,
    args: argparse.Namespace,
    meta: dict,
    perm_seed: int,
    n_pool: int,
    ref_dir: str,
    paths_out: dict[str, str],
    per_n_loss_rows: list[dict[str, str]],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_convergence\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"dataset_family: {meta.get('dataset_family')}  # matches --dataset-family={args.dataset_family}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"n_ref: {args.n_ref}  n_list: {args.n_list}\n")
        f.write(f"num_theta_bins: {args.num_theta_bins}\n")
        f.write(f"theta_flow_onehot_state: {bool(getattr(args, 'theta_flow_onehot_state', False))}\n")
        f.write(f"theta_flow_fourier_state: {bool(getattr(args, 'theta_flow_fourier_state', False))}\n")
        if str(getattr(args, "theta_field_method", "")).strip().lower() in (
            "gaussian_network",
            "gaussian_network_diagonal",
            "gaussian_network_diagonal_binned_pca",
            "gaussian_network_low_rank",
            "gaussian_network_autoencoder",
            "gaussian_network_diagonal_autoencoder",
        ):
            f.write(f"gn_epochs: {int(getattr(args, 'gn_epochs', 0))}\n")
            f.write(f"gn_batch_size: {int(getattr(args, 'gn_batch_size', 0))}\n")
            f.write(f"gn_lr: {float(getattr(args, 'gn_lr', 0.0))}\n")
            f.write(f"gn_hidden_dim: {int(getattr(args, 'gn_hidden_dim', 0))}\n")
            f.write(f"gn_depth: {int(getattr(args, 'gn_depth', 0))}\n")
            f.write(f"gn_diag_floor: {float(getattr(args, 'gn_diag_floor', 0.0))}\n")
            f.write(f"gn_low_rank_dim: {int(getattr(args, 'gn_low_rank_dim', 0))}\n")
            f.write(f"gn_psi_floor: {float(getattr(args, 'gn_psi_floor', 0.0))}\n")
            f.write(f"gn_pca_dim: {int(getattr(args, 'gn_pca_dim', 0))}\n")
            f.write(f"gn_pca_num_bins: {getattr(args, 'gn_pca_num_bins', None)}\n")
            f.write(f"gn_early_patience: {int(getattr(args, 'gn_early_patience', 0))}\n")
            if str(getattr(args, "theta_field_method", "")).strip().lower() in (
                "gaussian_network_autoencoder",
                "gaussian_network_diagonal_autoencoder",
            ):
                f.write(f"gn_ae_latent_dim: {getattr(args, 'gn_ae_latent_dim', None)}\n")
                f.write(f"gn_ae_epochs: {int(getattr(args, 'gn_ae_epochs', 0))}\n")
                f.write(f"gn_ae_batch_size: {int(getattr(args, 'gn_ae_batch_size', 0))}\n")
                f.write(f"gn_ae_lr: {float(getattr(args, 'gn_ae_lr', 0.0))}\n")
                f.write(f"gn_ae_hidden_dim: {int(getattr(args, 'gn_ae_hidden_dim', 0))}\n")
                f.write(f"gn_ae_depth: {int(getattr(args, 'gn_ae_depth', 0))}\n")
                f.write(f"gn_ae_early_patience: {int(getattr(args, 'gn_ae_early_patience', 0))}\n")
        _tfm_sum = str(getattr(args, "theta_field_method", "")).strip().lower()
        if _tfm_sum in ("gaussian_x_flow", "gaussian_x_flow_diagonal"):
            f.write(f"gxf_epochs: {int(getattr(args, 'gxf_epochs', 0))}\n")
            f.write(f"gxf_batch_size: {int(getattr(args, 'gxf_batch_size', 0))}\n")
            f.write(f"gxf_lr: {float(getattr(args, 'gxf_lr', 0.0))}\n")
            f.write(f"gxf_hidden_dim: {int(getattr(args, 'gxf_hidden_dim', 0))}\n")
            f.write(f"gxf_depth: {int(getattr(args, 'gxf_depth', 0))}\n")
            f.write(f"gxf_weight_decay: {float(getattr(args, 'gxf_weight_decay', 0.0))}\n")
            f.write(f"gxf_diag_floor: {float(getattr(args, 'gxf_diag_floor', 0.0))}\n")
            f.write(f"gxf_cov_jitter: {float(getattr(args, 'gxf_cov_jitter', 0.0))}\n")
            f.write(f"gxf_t_eps: {float(getattr(args, 'gxf_t_eps', 0.0))}\n")
            f.write(f"gxf_path_schedule: {getattr(args, 'gxf_path_schedule', '')}\n")
            f.write(f"gxf_weight_ema_decay: {float(getattr(args, 'gxf_weight_ema_decay', 0.0))}\n")
            f.write(f"gxf_early_patience: {int(getattr(args, 'gxf_early_patience', 0))}\n")
            f.write(f"gxf_pair_batch_size: {int(getattr(args, 'gxf_pair_batch_size', 0))}\n")
            f.write(f"gxf_diagonal_covariance: {_tfm_sum == 'gaussian_x_flow_diagonal'}\n")
        if _tfm_sum in (
            "linear_x_flow_t",
            "linear_x_flow_scalar_t",
            "linear_x_flow_diagonal_theta_t",
            "linear_x_flow_diagonal_t",
            "linear_x_flow_low_rank_t",
            "xflow_sir_lrank",
            "xflow_sir_lrank_dia",
            "xflow_sir_lrank_dia_theta",
            "xflow_sir_lrank_scalar",
            "xflow_sir_lrank_scalar_theta",
            "xflow_sir_pure_lrank",
            "linear_x_flow_pure_low_rank_t",
            "linear_x_flow_pure_cond_low_rank_t",
            "linear_x_flow_lr_t_ts",
            "linear_x_flow_low_rank_randb_t",
            "sir_xflow_lrank_t",
        ):
            f.write(f"lxf_epochs: {int(getattr(args, 'lxf_epochs', 0))}\n")
            f.write(f"lxf_batch_size: {int(getattr(args, 'lxf_batch_size', 0))}\n")
            f.write(f"lxf_lr: {float(getattr(args, 'lxf_lr', 0.0))}\n")
            f.write(f"lxf_hidden_dim: {int(getattr(args, 'lxf_hidden_dim', 0))}\n")
            f.write(f"lxf_depth: {int(getattr(args, 'lxf_depth', 0))}\n")
            f.write(f"lxf_b_net: {getattr(args, 'lxf_b_net', 'mlp')}\n")
            f.write(f"lxf_low_rank_dim: {int(getattr(args, 'lxf_low_rank_dim', 0))}\n")
            f.write(f"lxf_randb_lambda_a: {float(getattr(args, 'lxf_randb_lambda_a', 0.0))}\n")
            f.write(f"lxf_randb_lambda_s: {float(getattr(args, 'lxf_randb_lambda_s', 0.0))}\n")
            f.write(f"lxf_weight_decay: {float(getattr(args, 'lxf_weight_decay', 0.0))}\n")
            f.write(f"lxf_t_eps: {float(getattr(args, 'lxf_t_eps', 0.0))}\n")
            f.write(f"lxf_solve_jitter: {float(getattr(args, 'lxf_solve_jitter', 0.0))}\n")
            f.write(f"lxf_weight_ema_decay: {float(getattr(args, 'lxf_weight_ema_decay', 0.0))}\n")
            f.write(f"lxf_early_patience: {int(getattr(args, 'lxf_early_patience', 0))}\n")
            f.write(f"lxf_restore_best: {bool(getattr(args, 'lxf_restore_best', True))}\n")
            f.write(f"lxf_pair_batch_size: {int(getattr(args, 'lxf_pair_batch_size', 0))}\n")
            f.write(f"lxf_nlpca_dim: {int(getattr(args, 'lxf_nlpca_dim', 0))}\n")
            f.write(f"lxf_nlpca_epochs: {int(getattr(args, 'lxf_nlpca_epochs', 0))}\n")
            f.write(f"lxf_nlpca_lr: {float(getattr(args, 'lxf_nlpca_lr', 0.0))}\n")
            f.write(f"lxf_nlpca_hidden_dim: {int(getattr(args, 'lxf_nlpca_hidden_dim', 0))}\n")
            f.write(f"lxf_nlpca_depth: {int(getattr(args, 'lxf_nlpca_depth', 0))}\n")
            f.write(f"lxf_nlpca_lambda_h: {float(getattr(args, 'lxf_nlpca_lambda_h', 0.0))}\n")
            f.write(f"lxf_nlpca_freeze_linear: {bool(getattr(args, 'lxf_nlpca_freeze_linear', False))}\n")
            f.write(f"lxf_nlpca_ode_steps: {int(getattr(args, 'lxf_nlpca_ode_steps', 0))}\n")
            if _tfm_sum in (
                "linear_x_flow_low_rank_t",
                "xflow_sir_lrank",
                "xflow_sir_lrank_dia",
                "xflow_sir_lrank_dia_theta",
                "xflow_sir_lrank_scalar",
                "xflow_sir_lrank_scalar_theta",
                "xflow_sir_pure_lrank",
                "linear_x_flow_pure_low_rank_t",
                "linear_x_flow_pure_cond_low_rank_t",
                "linear_x_flow_lr_t_ts",
                "sir_xflow_lrank_t",
            ):
                f.write(
                    f"lxf_low_rank_divergence_estimator: {str(getattr(args, 'lxf_low_rank_divergence_estimator', 'hutchinson')).strip().lower()}\n"
                )
                f.write(f"lxf_hutchinson_probes: {int(getattr(args, 'lxf_hutchinson_probes', 1))}\n")
        if _tfm_sum == "linear_theta_flow":
            f.write(f"ltf_num_components: {int(getattr(args, 'ltf_num_components', 0))}\n")
            f.write(f"ltf_epochs: {int(getattr(args, 'ltf_epochs', 0))}\n")
            f.write(f"ltf_batch_size: {int(getattr(args, 'ltf_batch_size', 0))}\n")
            f.write(f"ltf_lr: {float(getattr(args, 'ltf_lr', 0.0))}\n")
            f.write(f"ltf_hidden_dim: {int(getattr(args, 'ltf_hidden_dim', 0))}\n")
            f.write(f"ltf_depth: {int(getattr(args, 'ltf_depth', 0))}\n")
            f.write(f"ltf_weight_decay: {float(getattr(args, 'ltf_weight_decay', 0.0))}\n")
            f.write(f"ltf_t_eps: {float(getattr(args, 'ltf_t_eps', 0.0))}\n")
            f.write(f"ltf_solve_jitter: {float(getattr(args, 'ltf_solve_jitter', 0.0))}\n")
            f.write(f"ltf_weight_ema_decay: {float(getattr(args, 'ltf_weight_ema_decay', 0.0))}\n")
            f.write(f"ltf_early_patience: {int(getattr(args, 'ltf_early_patience', 0))}\n")
            f.write(f"ltf_pair_batch_size: {int(getattr(args, 'ltf_pair_batch_size', 0))}\n")
            f.write(f"lxfs_early_patience: {int(getattr(args, 'lxfs_early_patience', 0))}\n")
            f.write(f"lxfs_pair_batch_size: {int(getattr(args, 'lxfs_pair_batch_size', 0))}\n")
        if _tfm_sum in _TIME_LXF_METHODS or _tfm_sum == "sir_xflow_lrank_t":
            f.write(f"lxfs_path_schedule: {getattr(args, 'lxfs_path_schedule', '')}\n")
            f.write(f"lxfs_epochs: {int(getattr(args, 'lxfs_epochs', 0))}\n")
            f.write(f"lxfs_batch_size: {int(getattr(args, 'lxfs_batch_size', 0))}\n")
            f.write(f"lxfs_lr: {float(getattr(args, 'lxfs_lr', 0.0))}\n")
            f.write(f"lxfs_hidden_dim: {int(getattr(args, 'lxfs_hidden_dim', 0))}\n")
            f.write(f"lxfs_depth: {int(getattr(args, 'lxfs_depth', 0))}\n")
            f.write(f"lxfs_weight_decay: {float(getattr(args, 'lxfs_weight_decay', 0.0))}\n")
            f.write(f"lxfs_t_eps: {float(getattr(args, 'lxfs_t_eps', 0.0))}\n")
            f.write(f"lxfs_solve_jitter: {float(getattr(args, 'lxfs_solve_jitter', 0.0))}\n")
            f.write(f"lxfs_weight_ema_decay: {float(getattr(args, 'lxfs_weight_ema_decay', 0.0))}\n")
            f.write(f"lxfs_early_patience: {int(getattr(args, 'lxfs_early_patience', 0))}\n")
            f.write(f"lxfs_pair_batch_size: {int(getattr(args, 'lxfs_pair_batch_size', 0))}\n")
            f.write(f"lxfs_quadrature_steps: {int(getattr(args, 'lxfs_quadrature_steps', 0))}\n")
        if _tfm_sum == "x_flow_pca":
            f.write(f"flow_pca_dim: {int(getattr(args, 'flow_pca_dim', 0))}\n")
            f.write(f"flow_pca_num_bins: {getattr(args, 'flow_pca_num_bins', None)}\n")
        if _tfm_sum in (
            "sir_xflow_lrank_t",
            "sir_xflow",
            "sir_thetaflow",
            "xflow_sir_lrank",
            "xflow_sir_lrank_dia",
            "xflow_sir_lrank_dia_theta",
            "xflow_sir_lrank_scalar",
            "xflow_sir_lrank_scalar_theta",
            "xflow_sir_pure_lrank",
        ):
            _sir_dim_summary = (
                int(getattr(args, "lxf_low_rank_dim", 0))
                if _tfm_sum in (
                    "xflow_sir_lrank",
                    "xflow_sir_lrank_dia",
                    "xflow_sir_lrank_dia_theta",
                    "xflow_sir_lrank_scalar",
                    "xflow_sir_lrank_scalar_theta",
                    "xflow_sir_pure_lrank",
                )
                else int(getattr(args, "sir_dim", 0))
            )
            f.write(f"sir_dim: {_sir_dim_summary}\n")
            f.write(f"sir_num_bins: {int(getattr(args, 'sir_num_bins', 0))}\n")
            f.write(f"sir_ridge: {float(getattr(args, 'sir_ridge', 0.0))}\n")
        if _tfm_sum == "nf_reduction":
            f.write(f"nfr_latent_dim: {int(getattr(args, 'nfr_latent_dim', 0))}\n")
            f.write(f"nfr_epochs: {int(getattr(args, 'nfr_epochs', 0))}\n")
            f.write(f"nfr_batch_size: {int(getattr(args, 'nfr_batch_size', 0))}\n")
            f.write(f"nfr_lr: {float(getattr(args, 'nfr_lr', 0.0))}\n")
            f.write(f"nfr_hidden_dim: {int(getattr(args, 'nfr_hidden_dim', 0))}\n")
            f.write(f"nfr_context_dim: {int(getattr(args, 'nfr_context_dim', 0))}\n")
            f.write(f"nfr_transforms: {int(getattr(args, 'nfr_transforms', 0))}\n")
            f.write(f"nfr_early_patience: {int(getattr(args, 'nfr_early_patience', 0))}\n")
            f.write(f"nfr_pair_batch_size: {int(getattr(args, 'nfr_pair_batch_size', 0))}\n")
        if _tfm_sum == "pi_nf":
            f.write(f"pinf_latent_dim: {int(getattr(args, 'pinf_latent_dim', 0))}\n")
            f.write(f"pinf_epochs: {int(getattr(args, 'pinf_epochs', 0))}\n")
            f.write(f"pinf_batch_size: {int(getattr(args, 'pinf_batch_size', 0))}\n")
            f.write(f"pinf_lr: {float(getattr(args, 'pinf_lr', 0.0))}\n")
            f.write(f"pinf_hidden_dim: {int(getattr(args, 'pinf_hidden_dim', 0))}\n")
            f.write(f"pinf_transforms: {int(getattr(args, 'pinf_transforms', 0))}\n")
            f.write(f"pinf_min_std: {float(getattr(args, 'pinf_min_std', 0.0))}\n")
            f.write(f"pinf_recon_weight: {float(getattr(args, 'pinf_recon_weight', 1.0))}\n")
            f.write(f"pinf_early_patience: {int(getattr(args, 'pinf_early_patience', 0))}\n")
            f.write(f"pinf_pair_batch_size: {int(getattr(args, 'pinf_pair_batch_size', 0))}\n")
        if str(getattr(args, "theta_field_method", "")).strip().lower() in (
            "theta_flow_autoencoder",
            "x_flow_autoencoder",
        ):
            f.write(f"gn_ae_latent_dim: {getattr(args, 'gn_ae_latent_dim', None)}\n")
            f.write(f"gn_ae_epochs: {int(getattr(args, 'gn_ae_epochs', 0))}\n")
            f.write(f"gn_ae_batch_size: {int(getattr(args, 'gn_ae_batch_size', 0))}\n")
            f.write(f"gn_ae_lr: {float(getattr(args, 'gn_ae_lr', 0.0))}\n")
            f.write(f"gn_ae_hidden_dim: {int(getattr(args, 'gn_ae_hidden_dim', 0))}\n")
            f.write(f"gn_ae_depth: {int(getattr(args, 'gn_ae_depth', 0))}\n")
            f.write(f"gn_ae_early_patience: {int(getattr(args, 'gn_ae_early_patience', 0))}\n")
        if bool(getattr(args, "theta_flow_fourier_state", False)):
            f.write(f"theta_flow_fourier_k: {int(getattr(args, 'theta_flow_fourier_k', 0))}\n")
            f.write(
                "theta_flow_fourier_period_mult: "
                f"{float(getattr(args, 'theta_flow_fourier_period_mult', 0.0))}\n"
            )
            f.write(
                "theta_flow_fourier_include_linear: "
                f"{bool(getattr(args, 'theta_flow_fourier_include_linear', False))}\n"
            )
        f.write(f"subset_seed_offset: {args.subset_seed_offset}\n")
        f.write(f"permutation rng seed: {perm_seed}\n")
        f.write(f"dataset pool size: {n_pool}\n")
        f.write(f"reference run dir: {ref_dir}\n")
        f.write(f"meta seed: {meta.get('seed')}\n")
        for k, v in paths_out.items():
            f.write(f"{k}: {v}\n")
        f.write("\n# Per-n training-loss artifacts\n")
        for row in per_n_loss_rows:
            f.write(
                "n={n} status={status} run_dir={run_dir} src={src} dst={dst} note={note}\n".format(
                    n=row["n"],
                    status=row["status"],
                    run_dir=row["run_dir"],
                    src=row["src"],
                    dst=row["dst"],
                    note=row["note"],
                )
            )


class CachedConvergenceBundle(TypedDict):
    """Arrays loaded from ``h_decoding_convergence_results.npz`` for visualization-only mode."""

    n: np.ndarray
    corr_h: np.ndarray
    corr_clf: np.ndarray
    wall_s: np.ndarray
    h_cols: np.ndarray
    clf_cols: np.ndarray
    n_ref: int
    perm_seed: int
    base_seed: int
    meta_seed: int
    edges: np.ndarray
    centers: np.ndarray
    gt_n_mc: int
    gt_seed: int
    gt_symmetrize: bool
    out_npz: str
    # Optional: LLR scatter (newer full runs)
    llr_gt_mean_one_sided_mc: NotRequired[np.ndarray]
    llr_binned_columns: NotRequired[np.ndarray]
    corr_llr_binned_vs_gt_mc: NotRequired[np.ndarray]
    binned_gaussian_h_binned_columns: NotRequired[np.ndarray]
    binned_gaussian_corr_h_binned_vs_gt_mc: NotRequired[np.ndarray]
    binned_gaussian_variance_floor: NotRequired[float]
    binned_gaussian_label: NotRequired[str]


def _load_cached_convergence_results(output_dir: str) -> CachedConvergenceBundle:
    """Load metrics and matrices from a prior full run."""
    out_npz = os.path.join(output_dir, "h_decoding_convergence_results.npz")
    if not os.path.isfile(out_npz):
        raise FileNotFoundError(
            f"Visualization-only mode requires prior results; missing file:\n  {out_npz}\n"
            "Run once without --visualization-only to generate h_decoding_convergence_results.npz."
        )
    z = np.load(out_npz, allow_pickle=True)
    required = (
        "n",
        "corr_h_binned_vs_gt_mc",
        "corr_clf_vs_ref",
        "wall_seconds",
        "h_binned_columns",
        "clf_acc_columns",
        "n_ref",
        "theta_bin_edges",
    )
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{out_npz} missing keys: {missing}")

    h_cols = np.asarray(z["h_binned_columns"], dtype=np.float64)
    clf_cols = np.asarray(z["clf_acc_columns"], dtype=np.float64)
    if h_cols.ndim != 3 or clf_cols.ndim != 3:
        raise ValueError("h_binned_columns / clf_acc_columns must be 3D (columns, bins, bins).")
    if h_cols.shape != clf_cols.shape:
        raise ValueError("h_binned_columns and clf_acc_columns shape mismatch.")

    edges = np.asarray(z["theta_bin_edges"], dtype=np.float64).ravel()
    n_bins_file = int(h_cols.shape[1])
    if edges.size != n_bins_file + 1:
        raise ValueError(
            f"theta_bin_edges length {edges.size} inconsistent with matrix dim {n_bins_file} (expected n_bins+1 edges)."
        )

    if "theta_bin_centers" in z.files:
        centers = np.asarray(z["theta_bin_centers"], dtype=np.float64).ravel()
    else:
        centers = bin_centers_from_edges(edges)

    gt_n_mc = int(np.asarray(z["gt_hellinger_n_mc"]).reshape(-1)[0]) if "gt_hellinger_n_mc" in z.files else 0
    gt_seed = int(np.asarray(z["gt_hellinger_seed"]).reshape(-1)[0]) if "gt_hellinger_seed" in z.files else 0
    sym_raw = z["gt_hellinger_symmetrize"] if "gt_hellinger_symmetrize" in z.files else np.int32(0)
    gt_symmetrize = bool(int(np.asarray(sym_raw).reshape(-1)[0]))

    base_bundle: dict[str, Any] = {
        "n": np.asarray(z["n"], dtype=np.int64).ravel(),
        "corr_h": np.asarray(z["corr_h_binned_vs_gt_mc"], dtype=np.float64).ravel(),
        "corr_clf": np.asarray(z["corr_clf_vs_ref"], dtype=np.float64).ravel(),
        "wall_s": np.asarray(z["wall_seconds"], dtype=np.float64).ravel(),
        "h_cols": h_cols,
        "clf_cols": clf_cols,
        "n_ref": int(np.asarray(z["n_ref"]).reshape(-1)[0]),
        "perm_seed": int(np.asarray(z["perm_seed"]).reshape(-1)[0]) if "perm_seed" in z.files else 0,
        "base_seed": int(np.asarray(z["convergence_base_seed"]).reshape(-1)[0]) if "convergence_base_seed" in z.files else 0,
        "meta_seed": int(np.asarray(z["dataset_meta_seed"]).reshape(-1)[0]) if "dataset_meta_seed" in z.files else 0,
        "edges": np.asarray(z["theta_bin_edges"], dtype=np.float64),
        "centers": np.asarray(centers, dtype=np.float64).ravel(),
        "gt_n_mc": gt_n_mc,
        "gt_seed": gt_seed,
        "gt_symmetrize": gt_symmetrize,
        "out_npz": os.path.abspath(out_npz),
    }
    if "gt_mean_llr_one_sided_mc" in z.files:
        base_bundle["llr_gt_mean_one_sided_mc"] = np.asarray(z["gt_mean_llr_one_sided_mc"], dtype=np.float64)
    if "llr_binned_columns" in z.files:
        base_bundle["llr_binned_columns"] = np.asarray(z["llr_binned_columns"], dtype=np.float64)
    if "corr_llr_binned_vs_gt_mc" in z.files:
        base_bundle["corr_llr_binned_vs_gt_mc"] = np.asarray(z["corr_llr_binned_vs_gt_mc"], dtype=np.float64).ravel()
    if "binned_gaussian_h_binned_columns" in z.files:
        base_bundle["binned_gaussian_h_binned_columns"] = np.asarray(
            z["binned_gaussian_h_binned_columns"],
            dtype=np.float64,
        )
    if "binned_gaussian_corr_h_binned_vs_gt_mc" in z.files:
        base_bundle["binned_gaussian_corr_h_binned_vs_gt_mc"] = np.asarray(
            z["binned_gaussian_corr_h_binned_vs_gt_mc"],
            dtype=np.float64,
        ).ravel()
    if "binned_gaussian_variance_floor" in z.files:
        base_bundle["binned_gaussian_variance_floor"] = float(
            np.asarray(z["binned_gaussian_variance_floor"]).reshape(-1)[0]
        )
    if "binned_gaussian_label" in z.files:
        raw_label = z["binned_gaussian_label"]
        try:
            base_bundle["binned_gaussian_label"] = str(np.asarray(raw_label).reshape(-1)[0])
        except Exception:
            base_bundle["binned_gaussian_label"] = str(raw_label)
    return cast(CachedConvergenceBundle, base_bundle)


def _validate_cached_matches_cli(args: argparse.Namespace, cached: CachedConvergenceBundle, ns: list[int]) -> None:
    n_arr = np.asarray(cached["n"], dtype=np.int64).ravel()
    if n_arr.size != len(ns) or not np.array_equal(n_arr, np.asarray(ns, dtype=np.int64)):
        raise ValueError(
            f"--n-list {ns} does not match cached results n={n_arr.tolist()}. "
            "Use the same --n-list as the run that produced h_decoding_convergence_results.npz."
        )
    if int(cached["n_ref"]) != int(args.n_ref):
        raise ValueError(
            f"Cached n_ref={cached['n_ref']} does not match --n-ref={int(args.n_ref)}."
        )
    n_bins_h = int(cached["h_cols"].shape[1])
    if n_bins_h != int(args.num_theta_bins):
        raise ValueError(
            f"Cached matrices imply num_theta_bins={n_bins_h} but CLI has --num-theta-bins={args.num_theta_bins}."
        )
    n_cols = int(cached["h_cols"].shape[0])
    if n_cols != len(ns) + 1:
        raise ValueError(
            f"Expected h_binned_columns to have {len(ns) + 1} columns (n sweep + n_ref); got {n_cols}."
        )


def _render_convergence_figures_and_summary(
    *,
    args: argparse.Namespace,
    meta: dict,
    perm_seed: int,
    n_pool: int,
    ref_dir: str,
    ns: list[int],
    n_bins: int,
    theta_centers: np.ndarray,
    gt_n_mc: int,
    gt_seed: int,
    gt_symmetrize_effective: bool,
    corr_h: np.ndarray,
    corr_clf: np.ndarray,
    wall_s: np.ndarray,
    h_cols: np.ndarray,
    clf_cols: np.ndarray,
    out_npz: str,
    per_n_loss_rows: list[dict[str, str]],
    err_msg: list[str],
    visualization_only: bool,
    llr_cols: np.ndarray | None = None,
    corr_llr: np.ndarray | None = None,
    binned_gaussian_h_cols: np.ndarray | None = None,
    binned_gaussian_corr_h: np.ndarray | None = None,
    binned_gaussian_label: str | None = None,
    binned_gaussian_variance_floor: float | None = None,
) -> None:
    """Write CSV, figures, manifest, training-loss panel, summary, and print artifact paths."""
    csv_path = os.path.join(args.output_dir, "h_decoding_convergence_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "corr_h_binned_vs_gt_mc",
                "corr_clf_vs_ref",
                "corr_llr_binned_vs_gt_mc",
                "wall_seconds",
            ]
        )
        for i, n in enumerate(ns):
            r_llr: float | str
            if (
                corr_llr is not None
                and int(np.asarray(corr_llr, dtype=np.float64).ravel().size) > i
            ):
                r_llr = float(np.asarray(corr_llr, dtype=np.float64).ravel()[i])
            else:
                r_llr = ""
            w.writerow([n, corr_h[i], corr_clf[i], r_llr, wall_s[i]])

    fig_path = os.path.join(args.output_dir, "h_decoding_convergence.png")
    fig, ax = plt.subplots(1, 1, figsize=_H_DECODING_CURVE_FIGSIZE_IN)
    _populate_convergence_curve_ax(
        ax,
        list(ns),
        corr_h,
        corr_clf,
        binned_gaussian_corr_h=binned_gaussian_corr_h,
    )
    fig.tight_layout()
    conv_svg = _save_figure_png_svg(fig, fig_path, dpi=160)
    plt.close(fig)

    matrix_panel_path = os.path.join(args.output_dir, "h_decoding_matrices_panel.png")
    col_labels = [f"n={n}" for n in ns] + [f"Approx GT, n_ref={int(args.n_ref)}"]
    extra_h_rows: list[tuple[str, list[np.ndarray]]] = []
    if binned_gaussian_h_cols is not None:
        bg_label = binned_gaussian_label or r"$\sqrt{H^2}$, binned Gaussian"
        bg_arr = np.asarray(binned_gaussian_h_cols, dtype=np.float64)
        if bg_arr.shape != np.asarray(h_cols, dtype=np.float64).shape:
            raise ValueError(
                f"binned_gaussian_h_cols shape {bg_arr.shape} must match h_cols shape {np.asarray(h_cols).shape}."
            )
        extra_h_rows.append((bg_label, [np.asarray(bg_arr[j], dtype=np.float64) for j in range(bg_arr.shape[0])]))
    _render_matrix_panel(
        h_mats=list(h_cols),
        clf_mats=list(clf_cols),
        col_labels=col_labels,
        out_path=matrix_panel_path,
        n_bins=n_bins,
        theta_centers=theta_centers,
        extra_h_rows=extra_h_rows,
    )

    loss_dir = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_dir, exist_ok=True)
    manifest_path = os.path.join(loss_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write("# n\tstatus\trun_dir\tsrc_loss_npz\tdst_loss_npz\tnote\n")
        for row in per_n_loss_rows:
            mf.write(
                "{n}\t{status}\t{run_dir}\t{src}\t{dst}\t{note}\n".format(
                    n=row["n"],
                    status=row["status"],
                    run_dir=row["run_dir"],
                    src=row["src"],
                    dst=row["dst"],
                    note=row["note"],
                )
            )

    combined_path = os.path.join(args.output_dir, "h_decoding_convergence_combined.png")
    diagnostic_png_paths: list[str | None] = []
    for n in ns:
        diagnostic_png = os.path.join(
            args.output_dir,
            "sweep_runs",
            f"n_{int(n):06d}",
            "diagnostics",
            "theta_flow_single_x_posterior_hist.png",
        )
        diagnostic_png_paths.append(diagnostic_png if os.path.isfile(diagnostic_png) else None)
    llr_est: list[np.ndarray] | None = None
    llr_gt: np.ndarray | None = None
    corr_llr_a: np.ndarray | None = None
    if (
        llr_cols is not None
        and int(llr_cols.shape[0]) == len(ns) + 1
        and corr_llr is not None
        and int(np.asarray(corr_llr, dtype=np.float64).ravel().size) == len(ns)
    ):
        llr_gt = np.asarray(llr_cols[-1], dtype=np.float64)
        llr_est = [np.asarray(llr_cols[j], dtype=np.float64) for j in range(len(ns))]
        corr_llr_a = np.asarray(corr_llr, dtype=np.float64).ravel()
    combined_svg = _save_combined_convergence_figure(
        h_mats=list(h_cols),
        clf_mats=list(clf_cols),
        col_labels=col_labels,
        n_bins=n_bins,
        theta_centers=theta_centers,
        ns=list(ns),
        corr_h=corr_h,
        corr_clf=corr_clf,
        loss_dir=loss_dir,
        diagnostic_png_paths=diagnostic_png_paths,
        out_png_path=combined_path,
        dpi=160,
        llr_gt=llr_gt,
        llr_est_mats=llr_est,
        corr_llr=corr_llr_a,
        binned_gaussian_corr_h=binned_gaussian_corr_h,
        extra_h_rows=extra_h_rows,
    )

    loss_panel_png = os.path.join(args.output_dir, "h_decoding_training_losses_panel.png")
    loss_panel_svg = _render_training_losses_panel(
        ns=list(ns),
        loss_dir=loss_dir,
        out_png_path=loss_panel_png,
        dpi=160,
    )

    gn_pretrain_panel_png = os.path.join(args.output_dir, "h_decoding_gn_pretrain_losses_panel.png")
    gn_pretrain_panel_svg = ""
    if _loss_dir_has_gn_pretrain(list(ns), loss_dir):
        gn_pretrain_panel_svg = _render_gn_pretrain_losses_panel(
            ns=list(ns),
            loss_dir=loss_dir,
            out_png_path=gn_pretrain_panel_png,
            dpi=160,
        )

    summary_path = os.path.join(args.output_dir, "h_decoding_convergence_summary.txt")
    paths_out = {
        "results_npz": out_npz,
        "results_csv": csv_path,
        "figure": fig_path,
        "figure_svg": conv_svg,
        "matrix_panel": matrix_panel_path,
        "matrix_panel_svg": str(Path(matrix_panel_path).with_suffix(".svg")),
        "combined_figure": combined_path,
        "combined_figure_svg": combined_svg,
        "embedded_fixed_x_diagnostic_pngs": "; ".join(
            f"n={int(n)}:{p}" for n, p in zip(ns, diagnostic_png_paths) if p is not None
        ),
        "training_losses_panel": loss_panel_png,
        "training_losses_panel_svg": loss_panel_svg,
        "reference_npz": os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        "training_losses_dir": loss_dir,
        "training_losses_manifest": manifest_path,
    }
    if gn_pretrain_panel_svg:
        paths_out["gn_pretrain_losses_panel"] = gn_pretrain_panel_png
        paths_out["gn_pretrain_losses_panel_svg"] = gn_pretrain_panel_svg
    if visualization_only:
        paths_out["mode"] = "visualization-only (figures/csv/summary regenerated from cached NPZ)"
    if binned_gaussian_h_cols is not None:
        paths_out["binned_gaussian_label"] = binned_gaussian_label or ""
        if binned_gaussian_variance_floor is not None:
            paths_out["binned_gaussian_variance_floor"] = str(float(binned_gaussian_variance_floor))
    if err_msg:
        paths_out["errors"] = "; ".join(err_msg[:20])
    _write_summary(summary_path, args, meta, perm_seed, n_pool, ref_dir, paths_out, per_n_loss_rows)
    with open(summary_path, "a", encoding="utf-8") as sf:
        sf.write("\n# Correlation targets (Pearson r off-diagonal; matrix_corr_offdiag_pearson)\n")
        sf.write(
            "# corr_h_binned_vs_gt_mc: Pearson r, off-diagonal binned sqrt(H_sym) vs sqrt(generative GT H^2).\n"
        )
        sf.write(
            "# corr_clf_vs_ref: Pearson r, off-diagonal pairwise decoding vs n_ref subset decoding matrix (same bin edges).\n"
            "#   NaN off-diagonals in the estimated decoding matrix are filled with the mean of its finite off-diagonals before correlation.\n"
        )
        sf.write(
            "# h_binned_columns last column: GT sqrt(H^2) (not DSM/flow); hellinger_gt_sq_mc key stores sqrt(H^2).\n"
        )
        sf.write(
            "# corr_llr_binned_vs_gt_mc: off-diagonal Pearson r, binned model \\Delta L vs "
            "one-sided generative mean LLR E_x[log p(x|θ_j)-log p(x|θ_i)] (MC; GT uses llr_binned_columns last column).\n"
        )
        sf.write(
            f"gt_hellinger_n_mc: {int(gt_n_mc)}  # MC samples per bin row; floor(n_ref / num_theta_bins)\n"
        )
        sf.write(
            f"gt_hellinger_n_ref_budget: {int(args.n_ref)}  # reference training subset size (--n-ref)\n"
        )
        sf.write(
            f"gt_hellinger_n_bins_times_n_mc: {int(n_bins * gt_n_mc)}  # n_bins * n_mc (may be < n_ref)\n"
        )
        sf.write(f"gt_hellinger_seed: {int(gt_seed)}\n")
        sf.write(f"gt_hellinger_symmetrize: {bool(gt_symmetrize_effective)}\n")
        if binned_gaussian_h_cols is not None:
            sf.write("\n# Binned Gaussian row\n")
            sf.write(f"binned_gaussian_label: {binned_gaussian_label or ''}\n")
            if binned_gaussian_variance_floor is not None:
                sf.write(f"binned_gaussian_variance_floor: {float(binned_gaussian_variance_floor)}\n")
            sf.write(
                "binned_gaussian_h_binned_columns: no-flow binned Gaussian sqrt(H^2) columns; "
                "last column reuses MC GT sqrt(H^2).\n"
            )
            if binned_gaussian_corr_h is not None:
                sf.write(
                    "binned_gaussian_corr_h_binned_vs_gt_mc: Pearson r, off-diagonal no-flow "
                    "binned Gaussian sqrt(H^2) vs sqrt(generative GT H^2).\n"
                )

    tag = "[convergence] Saved (visualization-only):" if visualization_only else "[convergence] Saved:"
    print(tag)
    print(f"  - {out_npz}")
    print(f"  - {csv_path}")
    print(f"  - {fig_path}")
    print(f"  - {conv_svg}")
    print(f"  - {matrix_panel_path}")
    print(f"  - {paths_out['matrix_panel_svg']}")
    print(f"  - {combined_path}")
    print(f"  - {combined_svg}")
    print(f"  - {loss_panel_png}")
    print(f"  - {loss_panel_svg}")
    if gn_pretrain_panel_svg:
        print(f"  - {gn_pretrain_panel_png}")
        print(f"  - {gn_pretrain_panel_svg}")
    print(f"  - {loss_dir}/ (per-n training loss .npz + manifest.txt)")
    print(f"  - {summary_path}")
