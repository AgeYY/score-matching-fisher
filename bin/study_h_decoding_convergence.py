#!/usr/bin/env python3
"""Convergence of binned H and pairwise decoding vs references.

**Binned H:** off-diagonal Pearson r vs a **generative ground-truth** Hellinger matrix
estimated by Monte Carlo from the known toy likelihood (see ``fisher/hellinger_gt.py`` and
``report/notes/hellinger_idea.tex``). The MC routine returns squared Hellinger **H^2**; this
script compares and visualizes **elementwise square roots** ``sqrt(H^2)`` (GT) and
``sqrt(H^sym)`` (learned binned symmetric ``h_sym``), both clipped to ``[0, 1]`` before the
square root. With ``n_bins`` = ``--num-theta-bins``, the MC count per
bin row is ``n_mc = n_ref // n_bins`` (integer floor); ``n_bins * n_mc`` may be less than ``n_ref``.
Nested subset training for each ``n`` in ``--n-list`` uses up to ``n`` samples; the ``n_ref`` column does not train a model.

**Pairwise decoding:** off-diagonal Pearson r vs the decoding matrix from the ``--n-ref``
subset (same bin edges as GT), unchanged.

**Dataset:** the loaded NPZ must match ``--dataset-family`` (default ``randamp_gaussian_sqrtd``:
random-amplitude Gaussian bumps plus ``sqrt(x_dim)`` observation-noise scaling; see ``make_dataset.py``).
Regenerate the NPZ if the family does not match.

For each ``n`` in ``--n-list``, the H matrix is computed from trained models for the selected
field method. Supported methods are ``--theta-field-method theta_flow`` (theta-space flow ODE
log-likelihood: Bayes ratios train/evaluate prior + posterior flows; with
``--theta-flow-posterior-only-likelihood``, only the posterior flow is trained/evaluated),
``--theta-field-method theta_path_integral``
(same training as theta_flow but H from velocity-to-score plus trapezoid integral along sorted ``theta``),
``--theta-field-method x_flow`` (conditional x-space FM likelihood; no prior model),
``--theta-field-method theta-flow-autoencoder`` and ``--theta-field-method x-flow-autoencoder``
(plain autoencoder preprocessing followed by the corresponding flow method in encoded latent space),
``--theta-field-method x-flow-pca`` (theta-binned train-mean PCA preprocessing followed by
``x_flow`` in projected space),
``--theta-field-method ctsm_v`` (pair-conditioned CTSM-v time-score integration; no prior model),
``--theta-field-method nf`` (conditional normalizing flow log p(theta|x) with an NF prior
for posterior-minus-prior log-ratio construction), ``--theta-field-method gaussian-network``
(MLP conditional Gaussian log p(x|theta), with mean and precision Cholesky outputs), and
``--theta-field-method gaussian-network-diagonal`` (same but diagonal precision Cholesky), and
``--theta-field-method gaussian-network-low-rank`` (latent covariance Cholesky mapped through
learned low-rank projection plus diagonal residual covariance). The staged autoencoder variants
``gaussian-network-autoencoder`` and ``gaussian-network-diagonal-autoencoder`` first encode
observations with a plain reconstruction autoencoder, then fit the corresponding Gaussian
likelihood in latent space. ``gaussian-network-diagonal-binned-pca`` projects observations onto
PCA components fit from theta-binned train-set means, then fits a diagonal Gaussian likelihood
in that low-dimensional PCA space. ``--theta-field-method gaussian-x-flow`` trains a full covariance
Cholesky Gaussian via analytic flow-matching velocity on an affine noise bridge; ``gaussian-x-flow-diagonal``
uses the same objective with a diagonal covariance / Cholesky (see ``fisher/gaussian_x_flow.py``).
``--theta-field-method linear-x-flow`` trains ``v(x,theta)=A x+b_phi(theta)`` and evaluates the
induced Gaussian with theta-dependent mean and shared covariance.
``--theta-field-method linear-x-flow-diagonal-theta`` trains ``v(x,theta)=diag(a_phi(theta)) x+b_phi(theta)``
and evaluates a diagonal Gaussian with theta-dependent mean and diagonal covariance.
``--theta-field-method linear-x-flow-diagonal-theta-spline`` is the same diagonal flow, but ``a`` and ``b``
are linear maps of fixed scalar-theta B-spline features (default ``K=5``).
``--theta-field-method linear-x-flow-schedule`` uses the same time-independent velocity network
and likelihood, but trains on a scheduled affine probability path such as cosine.
``--theta-field-method linear-x-flow-diagonal-t`` trains ``v(x,t,theta)=diag(a(t))x+b(t,theta)``
on a scheduled affine probability path and evaluates its quadrature Gaussian likelihood.
``--theta-field-method linear-x-flow-nonlinear-pca`` first trains the full linear x-flow,
fits PCA on residuals around the induced linear-flow mean, then adds a low-dimensional nonlinear
correction and evaluates ODE log likelihood.
``--theta-field-method nf-reduction`` learns an invertible x-space flow to ``(z, epsilon)`` and
computes H from a conditional flow likelihood ``log p(z|theta)``.
Flow methods use ``--flow-arch``: ``mlp``, ``film`` (FiLM with raw-theta embeddings), or
``film_fourier`` for ``theta_flow`` / ``theta_path_integral`` / ``x_flow``.
``film_fourier`` uses FiLM conditioning with Fourier theta features
(``--flow-theta-fourier-*`` for ``theta_flow`` / ``theta_path_integral`` and ``--flow-x-theta-fourier-*`` for ``x_flow``).
The **reference column** (``n_ref``) does **not** run learned H
training: the
matrix-panel top row shows **MC generative** ``sqrt(H^2)`` (same as the H correlation
target), while the bottom row still shows pairwise decoding on the ``n_ref`` data subset.

**Visualization-only:** Pass ``--visualization-only`` to reload ``h_decoding_convergence_results.npz``
from ``--output-dir`` and regenerate figures/CSV/summary without retraining (requires a prior full run
and ``training_losses/n_*.npz`` for the loss panel). If
``sweep_runs/n_<max(n_list)>/h_matrix_results*.npz`` exists but the fixed-$x$ diagnostic was never
written (older runs), the script may **backfill**
``sweep_runs/.../diagnostics/theta_flow_single_x_posterior_hist.png`` so the combined figure can embed it.

**NPZ semantics:** Arrays ``h_binned_columns``, ``h_binned_ref``, and the key ``hellinger_gt_sq_mc``
hold **square-root** matrices for this study (legacy key name ``hellinger_gt_sq_mc``; values are
``sqrt(H^2)``, not ``H^2``). LLR diagnostics (newer runs) add ``gt_mean_llr_one_sided_mc``,
``llr_binned_columns``, and ``corr_llr_binned_vs_gt_mc`` (binned model ``\\Delta L`` vs one-sided
generative mean log-likelihood ratios; see ``fisher/hellinger_gt.py``).
"""

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

import visualize_h_matrix_binned as vhb
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.hellinger_gt import (
    bin_centers_from_edges,
    estimate_hellinger_sq_one_sided_mc,
    estimate_mean_llr_one_sided_mc,
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
    ConditionalPCANonlinearLinearXFlowMLP,
    ConditionalDiagonalLinearXFlowFiLMLP,
    ConditionalDiagonalLinearXFlowMLP,
    ConditionalLinearXFlowMLP,
    ConditionalLowRankLinearXFlowMLP,
    ConditionalRandomBasisLowRankLinearXFlowMLP,
    ConditionalScalarLinearXFlowMLP,
    ConditionalThetaDiagonalLinearXFlowMLP,
    ConditionalThetaDiagonalSplineLinearXFlowMLP,
    ConditionalTimeDiagonalLinearXFlowMLP,
    compute_pca_nonlinear_linear_x_flow_c_matrix,
    compute_time_diagonal_linear_x_flow_c_matrix,
    compute_linear_x_flow_analytic_hellinger_matrix,
    estimate_binned_gaussian_shared_diagonal_covariance,
    fit_residual_pca_basis_from_linear_mean,
    compute_linear_x_flow_c_matrix,
    train_linear_x_flow,
    train_pca_nonlinear_linear_x_flow,
    train_linear_x_flow_schedule,
    train_time_diagonal_linear_x_flow_schedule,
)
from fisher.linear_theta_flow import (
    ConditionalLinearThetaFlowMixtureMLP,
    compute_linear_theta_flow_c_matrix,
    train_linear_theta_flow,
)
from fisher.contrastive_llr import (
    ContrastiveAdditiveIndependentScorer,
    ContrastiveGaussianNetworkScorer,
    ContrastiveIndependentDotProductScorer,
    ContrastiveIndependentGaussianScorer,
    ContrastiveLLRMLP,
    ContrastiveNormalizedDotBiasScorer,
    ContrastiveNormalizedDotScorer,
    compute_contrastive_c_matrix,
    compute_contrastive_soft_c_matrix,
    contrastive_soft_metadata_without_training,
    h_directed_from_delta_l as compute_h_directed_contrastive,
    normalize_theta_encoding as normalize_contrastive_theta_encoding,
    theta_dim_for_encoding as contrastive_theta_dim_for_encoding,
    train_bidir_contrastive_soft_llr,
    train_contrastive_llr,
    train_contrastive_soft_llr,
)
from fisher.gmm_z_decode import GMMZDecodeModel, compute_gmm_z_decode_c_matrix, train_gmm_z_decode
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
    validate_gmm_z_decode_args,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, train score models for each n in --n-list, then compare "
            "sqrt(binned H_sym) to sqrt(MC generative H^2) and pairwise decoding to the n_ref-subset decoding matrix. "
            "The n_ref matrix-panel column uses MC GT sqrt(H^2) for the top row (no n_ref model training). "
            "Also writes h_decoding_convergence_combined.{png,svg} (matrix panel + correlation curves + "
            "off-diagonal est-vs-GT H scatter + training-loss panel in one figure) and h_decoding_training_losses_panel.{png,svg} "
            "(standalone training-loss panel, one column per n). Runs that save Gaussian-network pretrain curves "
            "(e.g. contrastive-soft-gaussian-net) also write h_decoding_gn_pretrain_losses_panel.{png,svg}."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from make_dataset.py (must match --dataset-family).",
    )
    p.add_argument(
        "--dataset-family",
        type=str,
        default="randamp_gaussian_sqrtd",
        choices=[
            "cosine_gaussian",
            "cosine_gaussian_const_noise",
            "cosine_gaussian_sqrtd",
            "cosine_gaussian_sqrtd_rand_tune",
            "cosine_gaussian_sqrtd_rand_tune_additive",
            "randamp_gaussian",
            "randamp_gaussian_sqrtd",
            "randamp_gaussian2d_sqrtd",
            "gridcos_gaussian2d_sqrtd_rand_tune_additive",
            "cosine_gmm",
            "cos_sin_piecewise",
            "linear_piecewise",
        ],
        help=(
            "Expected generative family stored in the NPZ meta; must match make_dataset.py "
            "when the archive was created. Default: randamp_gaussian_sqrtd (random-amplitude "
            "Gaussian bumps + sqrt(x_dim) observation-noise scaling)."
        ),
    )
    p.add_argument(
        "--n-ref",
        type=int,
        default=5000,
        help=(
            "Reference subset size (default 5000). Requires len(dataset) >= n_ref and max(n-list) <= n_ref. "
            "GT Hellinger MC uses n_mc = n_ref // num_theta_bins samples per bin row (floor division)."
        ),
    )
    p.add_argument(
        "--n-list",
        type=str,
        default="80,200,400,600",
        help=(
            "Comma-separated nested subset sizes n to compare to the reference (default: "
            "80,160,240,320,400 — min 80, max 400, four equal steps of 80). Each n must be <= --n-ref."
        ),
    )
    p.add_argument(
        "--subset-seed-offset",
        type=int,
        default=0,
        help=(
            "Added to the convergence base seed for the global permutation (default 0). "
            "The base seed is --run-seed when set, otherwise the dataset meta seed from the NPZ."
        ),
    )
    p.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help=(
            "If set, use this as the base RNG seed for this study (global permutation uses "
            "run_seed + --subset-seed-offset; shared Fisher training uses this via merged args.seed; "
            "default clf / GT Hellinger MC seeds use this when their dedicated flags are -1). "
            "Omit to keep the dataset meta seed from the NPZ (same behavior as before)."
        ),
    )
    p.add_argument(
        "--num-theta-bins",
        type=int,
        default=10,
        help=(
            "Number of equal-width theta bins (default 10). GT Hellinger MC uses "
            "n_mc = n_ref // num_theta_bins (floor) samples per bin row."
        ),
    )
    p.add_argument(
        "--theta-binning-mode",
        type=str,
        default="theta1",
        choices=["theta1", "theta2_grid"],
        help=(
            "Binning convention for H/decoding matrices. Default theta1 keeps legacy scalar/first-coordinate "
            "bins. theta2_grid requires theta_dim=2 and flattens a 2D theta grid into one matrix axis."
        ),
    )
    p.add_argument(
        "--num-theta-bins-y",
        type=int,
        default=0,
        help=(
            "theta2_grid only: number of equal-width bins for theta_2. "
            "If <=0, reuse --num-theta-bins."
        ),
    )
    p.add_argument(
        "--theta-flow-onehot-state",
        action="store_true",
        help=(
            "theta_flow only: replace scalar theta state with one-hot bin vectors built from "
            "--num-theta-bins fixed edges on the n_ref permutation prefix. "
            "Flow ODE state becomes R^num_theta_bins."
        ),
    )
    p.add_argument(
        "--theta-flow-fourier-state",
        action="store_true",
        help=(
            "theta_flow only: replace scalar theta state with Fourier features built from "
            "theta range on the n_ref permutation prefix. Base period is "
            "theta_flow_fourier_period_mult * (theta_max - theta_min), then harmonics k=1..K."
        ),
    )
    p.add_argument(
        "--theta-flow-fourier-k",
        type=int,
        default=4,
        help=(
            "Number of Fourier harmonics K for --theta-flow-fourier-state. "
            "State dim = 2*K (+1 when --theta-flow-fourier-include-linear)."
        ),
    )
    p.add_argument(
        "--theta-flow-fourier-period-mult",
        type=float,
        default=2.0,
        help=(
            "Base-period multiplier for Fourier theta state: "
            "period = multiplier * (theta_max - theta_min) from n_ref subset."
        ),
    )
    p.add_argument(
        "--theta-flow-fourier-include-linear",
        action="store_true",
        help="Include centered linear theta term in Fourier theta state.",
    )
    p.add_argument(
        "--theta-flow-segmented",
        action="store_true",
        help=(
            "theta_flow only: estimate H using equal-width theta segments "
            "(orchestration in visualize_h_matrix_binned)."
        ),
    )
    p.add_argument(
        "--clf-min-class-count",
        type=int,
        default=5,
        help="Minimum samples per bin class for pairwise classifiers (default 5).",
    )
    p.add_argument(
        "--clf-random-state",
        type=int,
        default=-1,
        help="Random seed for LogisticRegression; -1 uses dataset seed from NPZ meta.",
    )
    p.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep per-n training output directories under output-dir/sweep_runs/ (default: temp dirs).",
    )
    p.add_argument(
        "--visualization-only",
        action="store_true",
        help=(
            "Skip dataset sweep, GT Hellinger MC, and per-n training. Load "
            "h_decoding_convergence_results.npz from --output-dir and regenerate figures, CSV, "
            "and summary only. Requires a prior full run in that directory; "
            "training_losses/n_*.npz must exist for the loss panel."
        ),
    )
    p.add_argument(
        "--gt-hellinger-seed",
        type=int,
        default=-1,
        help="RNG seed for GT Hellinger MC; -1 uses dataset meta seed.",
    )
    p.add_argument(
        "--gt-hellinger-symmetrize",
        action="store_true",
        help="If set, symmetrize GT H^2 matrix as (H^2 + H^2.T) / 2 after one-sided MC estimation.",
    )
    p.add_argument("--nf-epochs", type=int, default=2000, help="NF method only: training epochs.")
    p.add_argument("--nf-batch-size", type=int, default=256, help="NF method only: training batch size.")
    p.add_argument("--nf-lr", type=float, default=1e-3, help="NF method only: learning rate.")
    p.add_argument("--nf-hidden-dim", type=int, default=128, help="NF method only: encoder hidden size.")
    p.add_argument("--nf-context-dim", type=int, default=32, help="NF method only: context size.")
    p.add_argument("--nf-transforms", type=int, default=5, help="NF method only: spline transform count.")
    p.add_argument(
        "--nf-pair-batch-size",
        type=int,
        default=65536,
        help="NF method only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument("--nf-early-patience", type=int, default=300, help="NF method only: early-stop patience.")
    p.add_argument("--nf-early-min-delta", type=float, default=1e-4, help="NF method only: early-stop min delta.")
    p.add_argument(
        "--nf-early-ema-alpha",
        type=float,
        default=0.05,
        help="NF method only: EMA alpha for validation monitor.",
    )
    p.add_argument(
        "--nf-prior-epochs",
        type=int,
        default=None,
        help="NF method only: optional prior-NF epochs override (default: --nf-epochs).",
    )
    p.add_argument(
        "--nf-prior-batch-size",
        type=int,
        default=None,
        help="NF method only: optional prior-NF batch size override (default: --nf-batch-size).",
    )
    p.add_argument(
        "--nf-prior-lr",
        type=float,
        default=None,
        help="NF method only: optional prior-NF learning rate override (default: --nf-lr).",
    )
    p.add_argument(
        "--nf-prior-hidden-dim",
        type=int,
        default=None,
        help="NF method only: optional prior-NF hidden size override (default: --nf-hidden-dim).",
    )
    p.add_argument(
        "--nf-prior-transforms",
        type=int,
        default=None,
        help="NF method only: optional prior-NF transform count override (default: --nf-transforms).",
    )
    p.add_argument(
        "--nf-prior-early-patience",
        type=int,
        default=None,
        help="NF method only: optional prior-NF early-stop patience override (default: --nf-early-patience).",
    )
    p.add_argument(
        "--nf-prior-early-min-delta",
        type=float,
        default=None,
        help="NF method only: optional prior-NF early-stop min delta override (default: --nf-early-min-delta).",
    )
    p.add_argument(
        "--nf-prior-early-ema-alpha",
        type=float,
        default=None,
        help="NF method only: optional prior-NF EMA alpha override (default: --nf-early-ema-alpha).",
    )
    p.add_argument(
        "--nfr-latent-dim",
        type=int,
        default=2,
        help="nf-reduction only: z dimension r; must satisfy 1 <= r < x_dim.",
    )
    p.add_argument("--nfr-epochs", type=int, default=2000, help="nf-reduction only: training epochs.")
    p.add_argument("--nfr-batch-size", type=int, default=256, help="nf-reduction only: training batch size.")
    p.add_argument("--nfr-lr", type=float, default=1e-3, help="nf-reduction only: learning rate.")
    p.add_argument("--nfr-hidden-dim", type=int, default=128, help="nf-reduction only: NSF hidden width.")
    p.add_argument("--nfr-context-dim", type=int, default=32, help="nf-reduction only: theta context size for z-flow.")
    p.add_argument(
        "--nfr-transforms",
        type=int,
        default=5,
        help="nf-reduction only: NSF transform count for representation and conditional z flows.",
    )
    p.add_argument(
        "--nfr-pair-batch-size",
        type=int,
        default=65536,
        help="nf-reduction only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--nfr-early-patience",
        type=int,
        default=300,
        help="nf-reduction only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--nfr-early-min-delta",
        type=float,
        default=1e-4,
        help="nf-reduction only: early-stop min delta.",
    )
    p.add_argument(
        "--nfr-early-ema-alpha",
        type=float,
        default=0.05,
        help="nf-reduction only: EMA alpha for validation NLL monitor.",
    )
    p.add_argument("--pinf-latent-dim", type=int, default=2, help="pi-nf only: z dimension r; must satisfy 1 <= r < x_dim.")
    p.add_argument("--pinf-epochs", type=int, default=2000, help="pi-nf only: training epochs.")
    p.add_argument("--pinf-batch-size", type=int, default=256, help="pi-nf only: training batch size.")
    p.add_argument("--pinf-lr", type=float, default=1e-3, help="pi-nf only: learning rate.")
    p.add_argument("--pinf-hidden-dim", type=int, default=128, help="pi-nf only: NSF and Gaussian-prior MLP hidden width.")
    p.add_argument("--pinf-transforms", type=int, default=5, help="pi-nf only: representation NSF transform count.")
    p.add_argument("--pinf-min-std", type=float, default=1e-3, help="pi-nf only: softplus floor on diagonal z std.")
    p.add_argument("--pinf-weight-decay", type=float, default=0.0, help="pi-nf only: AdamW weight decay.")
    p.add_argument("--pinf-recon-weight", type=float, default=1.0, help="pi-nf only: sampled-residual reconstruction MSE weight.")
    p.add_argument("--pinf-pair-batch-size", type=int, default=65536, help="pi-nf only: approximate pair budget per C-matrix block.")
    p.add_argument("--pinf-early-patience", type=int, default=300, help="pi-nf only: early-stop patience; 0 disables.")
    p.add_argument("--pinf-early-min-delta", type=float, default=1e-4, help="pi-nf only: early-stop min delta.")
    p.add_argument("--pinf-early-ema-alpha", type=float, default=0.05, help="pi-nf only: EMA alpha for validation NLL monitor.")
    p.add_argument("--gn-epochs", type=int, default=4000, help="gaussian-network method only: training epochs.")
    p.add_argument("--gn-batch-size", type=int, default=256, help="gaussian-network method only: training batch size.")
    p.add_argument("--gn-lr", type=float, default=1e-3, help="gaussian-network method only: learning rate.")
    p.add_argument("--gn-hidden-dim", type=int, default=128, help="gaussian-network method only: MLP hidden width.")
    p.add_argument("--gn-depth", type=int, default=3, help="gaussian-network method only: MLP depth.")
    p.add_argument("--gn-weight-decay", type=float, default=0.0, help="gaussian-network method only: AdamW weight decay.")
    p.add_argument(
        "--gn-diag-floor",
        type=float,
        default=1e-4,
        help="gaussian-network method only: positive floor added to Cholesky precision diagonal.",
    )
    p.add_argument(
        "--gn-early-patience",
        type=int,
        default=300,
        help="gaussian-network method only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--gn-early-min-delta",
        type=float,
        default=1e-4,
        help="gaussian-network method only: early-stop min delta.",
    )
    p.add_argument(
        "--gn-early-ema-alpha",
        type=float,
        default=0.05,
        help="gaussian-network method only: EMA alpha for validation monitor.",
    )
    p.add_argument(
        "--gn-max-grad-norm",
        type=float,
        default=10.0,
        help="gaussian-network method only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--gn-pair-batch-size",
        type=int,
        default=65536,
        help="gaussian-network method only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--gn-pca-dim",
        type=int,
        default=2,
        help="gaussian-network-diagonal-binned-pca only: PCA dimension M fit from theta-binned train means.",
    )
    p.add_argument(
        "--gn-pca-num-bins",
        type=int,
        default=None,
        help="gaussian-network-diagonal-binned-pca only: theta bins K for PCA means (default: --num-theta-bins).",
    )
    p.add_argument(
        "--gn-low-rank-dim",
        type=int,
        default=4,
        help="gaussian-network-low-rank only: latent covariance rank r.",
    )
    p.add_argument(
        "--gn-psi-floor",
        type=float,
        default=1e-6,
        help="gaussian-network-low-rank only: positive floor added to learned residual variances Psi.",
    )
    p.add_argument(
        "--gn-ae-latent-dim",
        type=int,
        default=None,
        help="gaussian-network-autoencoder only: encoded observation dimension (default: min(8, x_dim)).",
    )
    p.add_argument(
        "--gn-ae-epochs",
        type=int,
        default=1000,
        help="gaussian-network-autoencoder only: autoencoder training epochs.",
    )
    p.add_argument(
        "--gn-ae-batch-size",
        type=int,
        default=256,
        help="gaussian-network-autoencoder only: autoencoder training batch size.",
    )
    p.add_argument(
        "--gn-ae-lr",
        type=float,
        default=1e-3,
        help="gaussian-network-autoencoder only: autoencoder learning rate.",
    )
    p.add_argument(
        "--gn-ae-hidden-dim",
        type=int,
        default=128,
        help="gaussian-network-autoencoder only: autoencoder hidden width.",
    )
    p.add_argument(
        "--gn-ae-depth",
        type=int,
        default=2,
        help="gaussian-network-autoencoder only: autoencoder encoder/decoder depth.",
    )
    p.add_argument(
        "--gn-ae-weight-decay",
        type=float,
        default=0.0,
        help="gaussian-network-autoencoder only: autoencoder AdamW weight decay.",
    )
    p.add_argument(
        "--gn-ae-early-patience",
        type=int,
        default=200,
        help="gaussian-network-autoencoder only: autoencoder early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--gn-ae-early-min-delta",
        type=float,
        default=1e-4,
        help="gaussian-network-autoencoder only: autoencoder early-stop min delta.",
    )
    p.add_argument(
        "--gn-ae-early-ema-alpha",
        type=float,
        default=0.05,
        help="gaussian-network-autoencoder only: EMA alpha for autoencoder validation monitor.",
    )
    p.add_argument(
        "--flow-pca-dim",
        type=int,
        default=2,
        help="x-flow-pca only: PCA dimension M fit from theta-binned train means.",
    )
    p.add_argument(
        "--flow-pca-num-bins",
        type=int,
        default=None,
        help="x-flow-pca only: theta bins K for PCA means (default: --num-theta-bins).",
    )
    p.add_argument("--gxf-epochs", type=int, default=2000, help="gaussian-x-flow only: training epochs.")
    p.add_argument("--gxf-batch-size", type=int, default=256, help="gaussian-x-flow only: training batch size.")
    p.add_argument("--gxf-lr", type=float, default=1e-4, help="gaussian-x-flow only: learning rate.")
    p.add_argument("--gxf-hidden-dim", type=int, default=128, help="gaussian-x-flow only: MLP hidden width.")
    p.add_argument("--gxf-depth", type=int, default=3, help="gaussian-x-flow only: MLP depth.")
    p.add_argument(
        "--gxf-weight-decay", type=float, default=0.0, help="gaussian-x-flow only: AdamW weight decay."
    )
    p.add_argument(
        "--gxf-diag-floor",
        type=float,
        default=1e-4,
        help="gaussian-x-flow only: softplus floor on covariance Cholesky diagonal entries.",
    )
    p.add_argument(
        "--gxf-cov-jitter",
        type=float,
        default=1e-4,
        help="gaussian-x-flow only: diagonal jitter added to C_t in the analytic velocity (Cholesky).",
    )
    p.add_argument(
        "--gxf-t-eps",
        type=float,
        default=0.05,
        help="gaussian-x-flow only: bridge time is sampled in [t_eps, 1-t_eps] (open interval).",
    )
    p.add_argument(
        "--gxf-path-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cos", "straight"],
        help="gaussian-x-flow only: affine path a(t), b(t) for the noise bridge (linear or cosine).",
    )
    p.add_argument(
        "--gxf-early-patience",
        type=int,
        default=300,
        help="gaussian-x-flow only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--gxf-early-min-delta",
        type=float,
        default=1e-4,
        help="gaussian-x-flow only: early-stop min delta on smoothed validation FM loss.",
    )
    p.add_argument(
        "--gxf-early-ema-alpha",
        type=float,
        default=0.05,
        help="gaussian-x-flow only: EMA alpha for validation FM loss monitor.",
    )
    p.add_argument(
        "--gxf-weight-ema-decay",
        type=float,
        default=0.9,
        help="gaussian-x-flow only: model-weight EMA decay; <=0 disables weight EMA.",
    )
    p.add_argument(
        "--gxf-max-grad-norm",
        type=float,
        default=10.0,
        help="gaussian-x-flow only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--gxf-pair-batch-size",
        type=int,
        default=65536,
        help="gaussian-x-flow only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument("--lxf-epochs", type=int, default=50000, help="linear-x-flow only: training epochs.")
    p.add_argument("--lxf-batch-size", type=int, default=1024, help="linear-x-flow only: training batch size.")
    p.add_argument("--lxf-lr", type=float, default=1e-4, help="linear-x-flow only: learning rate.")
    p.add_argument("--lxf-hidden-dim", type=int, default=128, help="linear-x-flow only: b_phi MLP hidden width.")
    p.add_argument("--lxf-depth", type=int, default=3, help="linear-x-flow only: b_phi MLP depth.")
    p.add_argument(
        "--lxf-b-net",
        type=str,
        default="mlp",
        choices=["mlp", "film"],
        help="linear-x-flow-diagonal only: offset b_phi parameterization (plain MLP vs FiLM trunk).",
    )
    p.add_argument(
        "--lxf-spline-k",
        type=int,
        default=5,
        help="linear-x-flow-diagonal-theta-spline only: number of B-spline basis functions K (cubic, clamped).",
    )
    p.add_argument("--lxf-low-rank-dim", type=int, default=4, help="linear-x-flow-low-rank only: rank r.")
    p.add_argument("--lxf-randb-lambda-a", type=float, default=1e-4, help="linear-x-flow-low-rank-randb only: L2 penalty on diagonal a.")
    p.add_argument("--lxf-randb-lambda-s", type=float, default=1e-4, help="linear-x-flow-low-rank-randb only: L2 penalty on symmetric S.")
    p.add_argument("--lxf-weight-decay", type=float, default=0.0, help="linear-x-flow only: AdamW weight decay.")
    p.add_argument(
        "--lxf-t-eps",
        type=float,
        default=0.05,
        help="linear-x-flow only: bridge time is sampled uniformly in [t_eps, 1-t_eps].",
    )
    p.add_argument(
        "--lxf-solve-jitter",
        type=float,
        default=1e-6,
        help="linear-x-flow only: jitter for solving A mu=(exp(A)-I)b and Cholesky log likelihood.",
    )
    p.add_argument(
        "--lxf-early-patience",
        type=int,
        default=1000,
        help="linear-x-flow only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--lxf-early-min-delta",
        type=float,
        default=1e-4,
        help="linear-x-flow only: early-stop min delta on smoothed validation FM loss.",
    )
    p.add_argument(
        "--lxf-early-ema-alpha",
        type=float,
        default=0.05,
        help="linear-x-flow only: EMA alpha for validation FM loss monitor.",
    )
    p.add_argument(
        "--lxf-weight-ema-decay",
        type=float,
        default=0.9,
        help="linear-x-flow only: model-weight EMA decay; <=0 disables weight EMA.",
    )
    p.add_argument("--lxf-restore-best", action="store_true", default=True)
    p.add_argument("--no-lxf-restore-best", action="store_false", dest="lxf_restore_best")
    p.add_argument(
        "--lxf-max-grad-norm",
        type=float,
        default=10.0,
        help="linear-x-flow only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--lxf-pair-batch-size",
        type=int,
        default=65536,
        help="linear-x-flow only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--lxf-save-c-matrix",
        action="store_true",
        help=(
            "linear-x-flow Gaussian endpoint methods only: also save the legacy C and DeltaL matrices. "
            "By default h_sym is computed analytically from Gaussian Hellinger and these expensive "
            "likelihood-ratio diagnostics are omitted."
        ),
    )
    p.add_argument("--lxf-nlpca-dim", type=int, default=8, help="linear-x-flow-nonlinear-pca only: residual PCA dimension k.")
    p.add_argument("--lxf-nlpca-epochs", type=int, default=0, help="linear-x-flow-nonlinear-pca only: second-stage epochs; 0 uses --lxf-epochs.")
    p.add_argument("--lxf-nlpca-lr", type=float, default=0.0, help="linear-x-flow-nonlinear-pca only: second-stage learning rate; 0 uses --lxf-lr.")
    p.add_argument("--lxf-nlpca-hidden-dim", type=int, default=128, help="linear-x-flow-nonlinear-pca only: h_phi hidden width.")
    p.add_argument("--lxf-nlpca-depth", type=int, default=3, help="linear-x-flow-nonlinear-pca only: h_phi MLP depth.")
    p.add_argument("--lxf-nlpca-lambda-h", type=float, default=0.0, help="linear-x-flow-nonlinear-pca only: L2 penalty on h_phi output.")
    p.add_argument("--lxf-nlpca-freeze-linear", action="store_true", help="linear-x-flow-nonlinear-pca only: freeze A and b_phi during second-stage training.")
    p.add_argument("--lxf-nlpca-ode-steps", type=int, default=32, help="linear-x-flow-nonlinear-pca only: fixed Euler steps for ODE likelihood.")
    p.add_argument(
        "--lxfs-path-schedule",
        type=str,
        default="cosine",
        choices=["linear", "straight", "cosine", "cos"],
        help="linear-x-flow-schedule only: affine path a(t), b(t) for FM training.",
    )
    p.add_argument("--lxfs-epochs", type=int, default=2000, help="linear-x-flow-schedule only: training epochs.")
    p.add_argument("--lxfs-batch-size", type=int, default=1024, help="linear-x-flow-schedule only: training batch size.")
    p.add_argument("--lxfs-lr", type=float, default=1e-4, help="linear-x-flow-schedule only: learning rate.")
    p.add_argument("--lxfs-hidden-dim", type=int, default=128, help="linear-x-flow-schedule only: b_phi MLP hidden width.")
    p.add_argument("--lxfs-depth", type=int, default=3, help="linear-x-flow-schedule only: b_phi MLP depth.")
    p.add_argument("--lxfs-weight-decay", type=float, default=0.0, help="linear-x-flow-schedule only: AdamW weight decay.")
    p.add_argument(
        "--lxfs-t-eps",
        type=float,
        default=0.05,
        help="linear-x-flow-schedule only: bridge time is sampled in [t_eps, 1-t_eps].",
    )
    p.add_argument(
        "--lxfs-solve-jitter",
        type=float,
        default=1e-6,
        help="linear-x-flow-schedule only: jitter for solving A mu=(exp(A)-I)b and Cholesky log likelihood.",
    )
    p.add_argument(
        "--lxfs-early-patience",
        type=int,
        default=300,
        help="linear-x-flow-schedule only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--lxfs-early-min-delta",
        type=float,
        default=1e-4,
        help="linear-x-flow-schedule only: early-stop min delta on smoothed validation FM loss.",
    )
    p.add_argument(
        "--lxfs-early-ema-alpha",
        type=float,
        default=0.05,
        help="linear-x-flow-schedule only: EMA alpha for validation FM loss monitor.",
    )
    p.add_argument(
        "--lxfs-weight-ema-decay",
        type=float,
        default=0.9,
        help="linear-x-flow-schedule only: model-weight EMA decay; <=0 disables weight EMA.",
    )
    p.add_argument(
        "--lxfs-max-grad-norm",
        type=float,
        default=10.0,
        help="linear-x-flow-schedule only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--lxfs-pair-batch-size",
        type=int,
        default=65536,
        help="linear-x-flow-schedule only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--lxfs-quadrature-steps",
        type=int,
        default=64,
        help="linear-x-flow-diagonal-t only: fixed time grid size for endpoint Gaussian quadrature.",
    )
    p.add_argument("--ltf-num-components", type=int, default=3, help="linear-theta-flow only: Gaussian mixture components.")
    p.add_argument("--ltf-epochs", type=int, default=2000, help="linear-theta-flow only: training epochs.")
    p.add_argument("--ltf-batch-size", type=int, default=1024, help="linear-theta-flow only: training batch size.")
    p.add_argument("--ltf-lr", type=float, default=1e-4, help="linear-theta-flow only: learning rate.")
    p.add_argument("--ltf-hidden-dim", type=int, default=128, help="linear-theta-flow only: MLP hidden width.")
    p.add_argument("--ltf-depth", type=int, default=3, help="linear-theta-flow only: MLP depth.")
    p.add_argument("--ltf-weight-decay", type=float, default=0.0, help="linear-theta-flow only: AdamW weight decay.")
    p.add_argument(
        "--ltf-t-eps",
        type=float,
        default=0.05,
        help="linear-theta-flow only: bridge time is sampled uniformly in [t_eps, 1-t_eps].",
    )
    p.add_argument(
        "--ltf-solve-jitter",
        type=float,
        default=1e-6,
        help="linear-theta-flow only: jitter for solving A mu=(exp(A)-I)b.",
    )
    p.add_argument(
        "--ltf-early-patience",
        type=int,
        default=300,
        help="linear-theta-flow only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--ltf-early-min-delta",
        type=float,
        default=1e-4,
        help="linear-theta-flow only: early-stop min delta on smoothed validation FM loss.",
    )
    p.add_argument(
        "--ltf-early-ema-alpha",
        type=float,
        default=0.05,
        help="linear-theta-flow only: EMA alpha for validation FM loss monitor.",
    )
    p.add_argument(
        "--ltf-weight-ema-decay",
        type=float,
        default=0.9,
        help="linear-theta-flow only: model-weight EMA decay; <=0 disables weight EMA.",
    )
    p.add_argument(
        "--ltf-max-grad-norm",
        type=float,
        default=10.0,
        help="linear-theta-flow only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--ltf-pair-batch-size",
        type=int,
        default=65536,
        help="linear-theta-flow only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument("--contrastive-epochs", type=int, default=2000, help="contrastive method only: training epochs.")
    p.add_argument(
        "--contrastive-batch-size",
        type=int,
        default=256,
        help="contrastive method only: minibatch size for row-wise shuffled-theta cross entropy.",
    )
    p.add_argument("--contrastive-lr", type=float, default=1e-3, help="contrastive method only: learning rate.")
    p.add_argument(
        "--contrastive-hidden-dim",
        type=int,
        default=128,
        help="contrastive method only: MLP hidden width.",
    )
    p.add_argument("--contrastive-depth", type=int, default=3, help="contrastive method only: MLP depth.")
    p.add_argument(
        "--contrastive-weight-decay",
        type=float,
        default=0.0,
        help="contrastive method only: AdamW weight decay.",
    )
    p.add_argument(
        "--contrastive-early-patience",
        type=int,
        default=300,
        help="contrastive method only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--contrastive-early-min-delta",
        type=float,
        default=1e-4,
        help="contrastive method only: early-stop min delta.",
    )
    p.add_argument(
        "--contrastive-early-ema-alpha",
        type=float,
        default=0.05,
        help="contrastive method only: EMA alpha for validation monitor.",
    )
    p.add_argument(
        "--contrastive-max-grad-norm",
        type=float,
        default=10.0,
        help="contrastive method only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--contrastive-pair-batch-size",
        type=int,
        default=65536,
        help="contrastive method only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--contrastive-theta-encoding",
        type=str,
        default="one_hot_bin",
        choices=["one_hot_bin", "integer_bin"],
        help=(
            "contrastive method only: theta-bin code passed to scalar S(x, theta_code). "
            "one_hot_bin uses K-dimensional one-hot bins; integer_bin uses one normalized scalar in [-1,1]."
        ),
    )
    p.add_argument(
        "--contrastive-soft-bandwidth",
        type=float,
        default=1.0,
        help=(
            "contrastive-soft only: Gaussian theta-kernel bandwidth in raw theta units "
            "(default 1.0); <=0 uses auto kth-neighbor bandwidth (--contrastive-soft-bandwidth-k)."
        ),
    )
    p.add_argument(
        "--contrastive-soft-score-arch",
        type=str,
        default="normalized_dot",
        choices=[
            "normalized_dot",
            "norm_dot",
            "dot",
            "additive_independent",
            "additive",
            "additive_independent_feature",
            "independent",
            "independent_gaussian",
            "gaussian",
            "independent_dot_product",
            "independent_dot",
            "dot_independent",
            "mlp",
        ],
        help=(
            "contrastive-soft only: scalar score architecture. normalized_dot uses "
            "S(x,theta)=alpha normalize(g(x))^T normalize(a(theta)); additive_independent uses "
            "D^{-1} sum_d h_d(x_d)^T a(theta); independent_gaussian uses a diagonal Gaussian "
            "score; independent_dot_product uses alpha/sqrt(D) sum_d h(x_d,e_d)^T a(theta); "
            "mlp uses the old joint MLP scorer."
        ),
    )
    p.add_argument(
        "--contrastive-soft-dot-dim",
        type=int,
        default=16,
        help=(
            "contrastive-soft normalized_dot/additive_independent only: shared feature dimension "
            "for dot-product features."
        ),
    )
    p.add_argument(
        "--contrastive-soft-coordinate-embed-dim",
        type=int,
        default=16,
        help="contrastive-soft independent_dot_product only: learned coordinate embedding dimension.",
    )
    p.add_argument(
        "--contrastive-soft-gaussian-logvar-min",
        type=float,
        default=-8.0,
        help="contrastive-soft independent_gaussian only: minimum clipped log variance.",
    )
    p.add_argument(
        "--contrastive-soft-gaussian-logvar-max",
        type=float,
        default=5.0,
        help="contrastive-soft independent_gaussian only: maximum clipped log variance.",
    )
    p.add_argument(
        "--contrastive-soft-bandwidth-start",
        type=float,
        default=0.0,
        help=(
            "contrastive-soft only: start bandwidth in raw theta units for linear annealing; "
            "set both start and end > 0 to enable."
        ),
    )
    p.add_argument(
        "--contrastive-soft-bandwidth-end",
        type=float,
        default=0.0,
        help=(
            "contrastive-soft only: final bandwidth in raw theta units for linear annealing; "
            "set both start and end > 0 to enable."
        ),
    )
    p.add_argument(
        "--contrastive-soft-bandwidth-k",
        type=int,
        default=5,
        help="contrastive-soft only: kth nearest theta neighbor used for auto bandwidth.",
    )
    p.add_argument(
        "--contrastive-soft-periodic",
        action="store_true",
        help="contrastive-soft only: use circular theta distance in the soft target kernel.",
    )
    p.add_argument(
        "--contrastive-soft-period",
        type=float,
        default=2.0 * np.pi,
        help="contrastive-soft only: period for circular theta distance when --contrastive-soft-periodic is set.",
    )
    add_estimation_arguments(p)
    p.set_defaults(
        output_dir=str(Path(DATA_DIR) / "h_decoding_convergence"),
        theta_field_method="theta_flow",
        flow_arch="mlp",
        score_early_ema_alpha=0.05,
        prior_early_ema_alpha=0.05,
        score_early_ema_warmup_epochs=0,
        prior_early_ema_warmup_epochs=0,
    )
    return p


def _parse_n_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("--n-list must contain at least one integer.")
    return [int(x) for x in parts]


def _normalize_gaussian_network_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "gaussian-network": "gaussian_network",
        "gaussian_network": "gaussian_network",
        "gaussian-network-diagonal": "gaussian_network_diagonal",
        "gaussian_network_diagonal": "gaussian_network_diagonal",
        "gaussian-network-low-rank": "gaussian_network_low_rank",
        "gaussian_network_low_rank": "gaussian_network_low_rank",
        "gaussian-network-diagonal-binned-pca": "gaussian_network_diagonal_binned_pca",
        "gaussian_network_diagonal_binned_pca": "gaussian_network_diagonal_binned_pca",
        "gaussian-network-autoencoder": "gaussian_network_autoencoder",
        "gaussian_network_autoencoder": "gaussian_network_autoencoder",
        "gaussian-network-diagonal-autoencoder": "gaussian_network_diagonal_autoencoder",
        "gaussian_network_diagonal_autoencoder": "gaussian_network_diagonal_autoencoder",
        "gaussian-network-diagonal-antoencoder": "gaussian_network_diagonal_autoencoder",
        "gaussian_network_diagonal_antoencoder": "gaussian_network_diagonal_autoencoder",
    }
    return aliases.get(key)


def _normalize_gaussian_x_flow_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "gaussian-x-flow": "gaussian_x_flow",
        "gaussian_x_flow": "gaussian_x_flow",
        "gaussian-x-flow-diagonal": "gaussian_x_flow_diagonal",
        "gaussian_x_flow_diagonal": "gaussian_x_flow_diagonal",
    }
    return aliases.get(key)


def _normalize_linear_x_flow_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "linear-x-flow": "linear_x_flow",
        "linear_x_flow": "linear_x_flow",
        "linear-x-flow-scalar": "linear_x_flow_scalar",
        "linear_x_flow_scalar": "linear_x_flow_scalar",
        "linear-x-flow-diagonal": "linear_x_flow_diagonal",
        "linear_x_flow_diagonal": "linear_x_flow_diagonal",
        "bin-gaussian-linear-x-flow-diagonal": "bin_gaussian_linear_x_flow_diagonal",
        "bin_gaussian_linear_x_flow_diagonal": "bin_gaussian_linear_x_flow_diagonal",
        "bin-lxf-diagonal": "bin_gaussian_linear_x_flow_diagonal",
        "bin_lxf_diagonal": "bin_gaussian_linear_x_flow_diagonal",
        "linear-x-flow-diagonal-theta": "linear_x_flow_diagonal_theta",
        "linear_x_flow_diagonal_theta": "linear_x_flow_diagonal_theta",
        "linear-x-flow-diagonal-theta-spline": "linear_x_flow_diagonal_theta_spline",
        "linear_x_flow_diagonal_theta_spline": "linear_x_flow_diagonal_theta_spline",
        "linear-x-flow-low-rank": "linear_x_flow_low_rank",
        "linear_x_flow_low_rank": "linear_x_flow_low_rank",
        "linear-x-flow-low-rank-randb": "linear_x_flow_low_rank_randb",
        "linear_x_flow_low_rank_randb": "linear_x_flow_low_rank_randb",
        "linear-x-flow-nonlinear-pca": "linear_x_flow_nonlinear_pca",
        "linear_x_flow_nonlinear_pca": "linear_x_flow_nonlinear_pca",
        "linear-x-flow-schedule": "linear_x_flow_schedule",
        "linear_x_flow_schedule": "linear_x_flow_schedule",
        "linear-x-flow-diagonal-t": "linear_x_flow_diagonal_t",
        "linear_x_flow_diagonal_t": "linear_x_flow_diagonal_t",
    }
    return aliases.get(key)


def _normalize_linear_theta_flow_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "linear-theta-flow": "linear_theta_flow",
        "linear_theta_flow": "linear_theta_flow",
    }
    return aliases.get(key)


def _normalize_nf_reduction_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "nf-reduction": "nf_reduction",
        "nf_reduction": "nf_reduction",
    }
    return aliases.get(key)


def _normalize_gmm_z_decode_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "gmm-z-decode": "gmm_z_decode",
        "gmm_z_decode": "gmm_z_decode",
    }
    return aliases.get(key)


def _normalize_pi_nf_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "pi-nf": "pi_nf",
        "pi_nf": "pi_nf",
    }
    return aliases.get(key)


def _normalize_contrastive_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "contrastive": "contrastive",
        "contrasive": "contrastive",
        "shuffled-contrastive": "contrastive",
        "shuffled_contrastive": "contrastive",
        "contrastive-soft": "contrastive_soft",
        "contrastive_soft": "contrastive_soft",
        "contrasive-soft": "contrastive_soft",
        "contrasive_soft": "contrastive_soft",
        "bidir-contrastive-soft": "bidir_contrastive_soft",
        "bidir_contrastive_soft": "bidir_contrastive_soft",
        "bidirectional-contrastive-soft": "bidir_contrastive_soft",
        "bidirectional_contrastive_soft": "bidir_contrastive_soft",
        "bidir-contrasive-soft": "bidir_contrastive_soft",
        "bidir_contrasive_soft": "bidir_contrastive_soft",
        "contrastive-soft-gaussian-net": "contrastive_soft_gaussian_net",
        "contrastive_soft_gaussian_net": "contrastive_soft_gaussian_net",
        "contrasive-soft-gaussian-net": "contrastive_soft_gaussian_net",
        "contrasive_soft_gaussian_net": "contrastive_soft_gaussian_net",
        "contrastive-soft-gaussian-net-no-finetune": "contrastive_soft_gaussian_net_no_finetune",
        "contrastive_soft_gaussian_net_no_finetune": "contrastive_soft_gaussian_net_no_finetune",
        "contrasive-soft-gaussian-net-no-finetune": "contrastive_soft_gaussian_net_no_finetune",
        "contrasive_soft_gaussian_net_no_finetune": "contrastive_soft_gaussian_net_no_finetune",
    }
    return aliases.get(key)


def _normalize_flow_autoencoder_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "theta-flow-autoencoder": "theta_flow_autoencoder",
        "theta_flow_autoencoder": "theta_flow_autoencoder",
        "x-flow-autoencoder": "x_flow_autoencoder",
        "x_flow_autoencoder": "x_flow_autoencoder",
    }
    return aliases.get(key)


def _normalize_flow_pca_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    aliases = {
        "x-flow-pca": "x_flow_pca",
        "x_flow_pca": "x_flow_pca",
    }
    return aliases.get(key)


def _base_flow_method_for_autoencoder(tfm: str) -> str:
    method = str(tfm).strip().lower()
    if method == "theta_flow_autoencoder":
        return "theta_flow"
    if method == "x_flow_autoencoder":
        return "x_flow"
    raise ValueError(f"Unsupported flow autoencoder method: {tfm!r}.")


def _validate_autoencoder_cli(args: argparse.Namespace) -> None:
    ae_latent_dim = getattr(args, "gn_ae_latent_dim", None)
    if ae_latent_dim is not None and int(ae_latent_dim) < 1:
        raise ValueError("--gn-ae-latent-dim must be >= 1.")
    if int(getattr(args, "gn_ae_epochs", 0)) < 1:
        raise ValueError("--gn-ae-epochs must be >= 1.")
    if int(getattr(args, "gn_ae_batch_size", 0)) < 1:
        raise ValueError("--gn-ae-batch-size must be >= 1.")
    if float(getattr(args, "gn_ae_lr", 0.0)) <= 0.0:
        raise ValueError("--gn-ae-lr must be > 0.")
    if int(getattr(args, "gn_ae_hidden_dim", 0)) < 1:
        raise ValueError("--gn-ae-hidden-dim must be >= 1.")
    if int(getattr(args, "gn_ae_depth", 0)) < 1:
        raise ValueError("--gn-ae-depth must be >= 1.")
    if float(getattr(args, "gn_ae_weight_decay", 0.0)) < 0.0:
        raise ValueError("--gn-ae-weight-decay must be >= 0.")
    if int(getattr(args, "gn_ae_early_patience", -1)) < 0:
        raise ValueError("--gn-ae-early-patience must be >= 0.")
    if float(getattr(args, "gn_ae_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--gn-ae-early-min-delta must be >= 0.")
    ae_alpha = float(getattr(args, "gn_ae_early_ema_alpha", 0.0))
    if not np.isfinite(ae_alpha) or ae_alpha <= 0.0 or ae_alpha > 1.0:
        raise ValueError("--gn-ae-early-ema-alpha must be in (0, 1].")


def _validate_flow_pca_cli(args: argparse.Namespace) -> None:
    if int(getattr(args, "flow_pca_dim", 0)) < 1:
        raise ValueError("--flow-pca-dim must be >= 1.")
    pca_bins = getattr(args, "flow_pca_num_bins", None)
    if pca_bins is not None and int(pca_bins) < 2:
        raise ValueError("--flow-pca-num-bins must be >= 2.")
    base_args = argparse.Namespace(**vars(args).copy())
    setattr(base_args, "theta_field_method", "x_flow")
    validate_estimation_args(base_args)


def _validate_gxf_cli(args: argparse.Namespace) -> None:
    if int(getattr(args, "gxf_epochs", 0)) < 1:
        raise ValueError("--gxf-epochs must be >= 1.")
    if int(getattr(args, "gxf_batch_size", 0)) < 1:
        raise ValueError("--gxf-batch-size must be >= 1.")
    if float(getattr(args, "gxf_lr", 0.0)) <= 0.0:
        raise ValueError("--gxf-lr must be > 0.")
    if int(getattr(args, "gxf_hidden_dim", 0)) < 1:
        raise ValueError("--gxf-hidden-dim must be >= 1.")
    if int(getattr(args, "gxf_depth", 0)) < 1:
        raise ValueError("--gxf-depth must be >= 1.")
    if float(getattr(args, "gxf_weight_decay", 0.0)) < 0.0:
        raise ValueError("--gxf-weight-decay must be >= 0.")
    if float(getattr(args, "gxf_diag_floor", 0.0)) <= 0.0:
        raise ValueError("--gxf-diag-floor must be > 0.")
    if float(getattr(args, "gxf_cov_jitter", 0.0)) <= 0.0:
        raise ValueError("--gxf-cov-jitter must be > 0.")
    te = float(getattr(args, "gxf_t_eps", 0.05))
    if not (0.0 < te < 0.5):
        raise ValueError("--gxf-t-eps must be in (0, 0.5).")
    if int(getattr(args, "gxf_early_patience", -1)) < 0:
        raise ValueError("--gxf-early-patience must be >= 0.")
    if float(getattr(args, "gxf_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--gxf-early-min-delta must be >= 0.")
    gxfa = float(getattr(args, "gxf_early_ema_alpha", 0.0))
    if not np.isfinite(gxfa) or gxfa <= 0.0 or gxfa > 1.0:
        raise ValueError("--gxf-early-ema-alpha must be in (0, 1].")
    if int(getattr(args, "gxf_pair_batch_size", 0)) < 1:
        raise ValueError("--gxf-pair-batch-size must be >= 1.")
    _mg = float(getattr(args, "gxf_max_grad_norm", 10.0))
    if not np.isfinite(_mg):
        raise ValueError("--gxf-max-grad-norm must be finite.")
    _wema = float(getattr(args, "gxf_weight_ema_decay", 0.9))
    if not np.isfinite(_wema) or _wema >= 1.0:
        raise ValueError("--gxf-weight-ema-decay must be finite and < 1.")
    path_schedule_from_name(str(getattr(args, "gxf_path_schedule", "linear")))


def _validate_lxf_cli(args: argparse.Namespace) -> None:
    method = str(getattr(args, "theta_field_method", "linear_x_flow")).strip().lower()
    scheduled = method in ("linear_x_flow_schedule", "linear_x_flow_diagonal_t")
    prefix = "lxfs" if scheduled else "lxf"
    label = "--lxfs" if scheduled else "--lxf"
    if int(getattr(args, f"{prefix}_epochs", 0)) < 1:
        raise ValueError(f"{label}-epochs must be >= 1.")
    if int(getattr(args, f"{prefix}_batch_size", 0)) < 1:
        raise ValueError(f"{label}-batch-size must be >= 1.")
    if float(getattr(args, f"{prefix}_lr", 0.0)) <= 0.0:
        raise ValueError(f"{label}-lr must be > 0.")
    if int(getattr(args, f"{prefix}_hidden_dim", 0)) < 1:
        raise ValueError(f"{label}-hidden-dim must be >= 1.")
    if int(getattr(args, f"{prefix}_depth", 0)) < 1:
        raise ValueError(f"{label}-depth must be >= 1.")
    b_net_kind = str(getattr(args, "lxf_b_net", "mlp")).strip().lower()
    if b_net_kind not in ("mlp", "film"):
        raise ValueError("--lxf-b-net must be one of: mlp, film.")
    if method not in ("linear_x_flow_diagonal", "bin_gaussian_linear_x_flow_diagonal") and b_net_kind != "mlp":
        raise ValueError("--lxf-b-net=film is only supported for linear_x_flow_diagonal and bin_gaussian_linear_x_flow_diagonal.")
    if method == "bin_gaussian_linear_x_flow_diagonal":
        vf = float(getattr(args, "flow_theta_reg_variance_floor", getattr(args, "flow_x_reg_variance_floor", 1e-6)))
        if not np.isfinite(vf) or vf <= 0.0:
            raise ValueError("--flow-theta-reg-variance-floor must be finite and > 0.")
    if method in ("linear_x_flow_low_rank", "linear_x_flow_low_rank_randb") and int(getattr(args, "lxf_low_rank_dim", 0)) < 1:
        raise ValueError("--lxf-low-rank-dim must be >= 1.")
    if method == "linear_x_flow_nonlinear_pca":
        if int(getattr(args, "lxf_nlpca_dim", 0)) < 1:
            raise ValueError("--lxf-nlpca-dim must be >= 1.")
        if int(getattr(args, "lxf_nlpca_epochs", 0)) < 0:
            raise ValueError("--lxf-nlpca-epochs must be >= 0.")
        if float(getattr(args, "lxf_nlpca_lr", 0.0)) < 0.0:
            raise ValueError("--lxf-nlpca-lr must be >= 0.")
        if int(getattr(args, "lxf_nlpca_hidden_dim", 0)) < 1:
            raise ValueError("--lxf-nlpca-hidden-dim must be >= 1.")
        if int(getattr(args, "lxf_nlpca_depth", 0)) < 1:
            raise ValueError("--lxf-nlpca-depth must be >= 1.")
        if float(getattr(args, "lxf_nlpca_lambda_h", 0.0)) < 0.0:
            raise ValueError("--lxf-nlpca-lambda-h must be >= 0.")
        if int(getattr(args, "lxf_nlpca_ode_steps", 0)) < 1:
            raise ValueError("--lxf-nlpca-ode-steps must be >= 1.")
    if method == "linear_x_flow_low_rank_randb":
        if float(getattr(args, "lxf_randb_lambda_a", 0.0)) < 0.0:
            raise ValueError("--lxf-randb-lambda-a must be >= 0.")
        if float(getattr(args, "lxf_randb_lambda_s", 0.0)) < 0.0:
            raise ValueError("--lxf-randb-lambda-s must be >= 0.")
    if method == "linear_x_flow_diagonal_theta_spline":
        sk = int(getattr(args, "lxf_spline_k", 5))
        if sk < 4:
            raise ValueError("--lxf-spline-k must be >= 4 for cubic B-splines (degree 3).")
        if sk > 64:
            raise ValueError("--lxf-spline-k must be <= 64.")
    if float(getattr(args, f"{prefix}_weight_decay", 0.0)) < 0.0:
        raise ValueError(f"{label}-weight-decay must be >= 0.")
    te = float(getattr(args, f"{prefix}_t_eps", 1e-3))
    if scheduled:
        if not (0.0 < te < 0.5):
            raise ValueError("--lxfs-t-eps must be in (0, 0.5).")
        path_schedule_from_name(str(getattr(args, "lxfs_path_schedule", "cosine")))
        if method == "linear_x_flow_diagonal_t" and int(getattr(args, "lxfs_quadrature_steps", 0)) < 2:
            raise ValueError("--lxfs-quadrature-steps must be >= 2.")
    elif not (0.0 < te < 0.5):
        raise ValueError("--lxf-t-eps must be in (0, 0.5).")
    sj = float(getattr(args, f"{prefix}_solve_jitter", 1e-6))
    if not np.isfinite(sj) or sj <= 0.0:
        raise ValueError(f"{label}-solve-jitter must be finite and > 0.")
    if int(getattr(args, f"{prefix}_early_patience", -1)) < 0:
        raise ValueError(f"{label}-early-patience must be >= 0.")
    if float(getattr(args, f"{prefix}_early_min_delta", 0.0)) < 0.0:
        raise ValueError(f"{label}-early-min-delta must be >= 0.")
    alpha = float(getattr(args, f"{prefix}_early_ema_alpha", 0.0))
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"{label}-early-ema-alpha must be in (0, 1].")
    if int(getattr(args, f"{prefix}_pair_batch_size", 0)) < 1:
        raise ValueError(f"{label}-pair-batch-size must be >= 1.")
    max_grad = float(getattr(args, f"{prefix}_max_grad_norm", 10.0))
    if not np.isfinite(max_grad):
        raise ValueError(f"{label}-max-grad-norm must be finite.")
    wema = float(getattr(args, f"{prefix}_weight_ema_decay", 0.9))
    if not np.isfinite(wema) or wema >= 1.0:
        raise ValueError(f"{label}-weight-ema-decay must be finite and < 1.")


def _validate_ltf_cli(args: argparse.Namespace) -> None:
    if int(getattr(args, "ltf_num_components", 0)) < 1:
        raise ValueError("--ltf-num-components must be >= 1.")
    if int(getattr(args, "ltf_epochs", 0)) < 1:
        raise ValueError("--ltf-epochs must be >= 1.")
    if int(getattr(args, "ltf_batch_size", 0)) < 1:
        raise ValueError("--ltf-batch-size must be >= 1.")
    if float(getattr(args, "ltf_lr", 0.0)) <= 0.0:
        raise ValueError("--ltf-lr must be > 0.")
    if int(getattr(args, "ltf_hidden_dim", 0)) < 1:
        raise ValueError("--ltf-hidden-dim must be >= 1.")
    if int(getattr(args, "ltf_depth", 0)) < 1:
        raise ValueError("--ltf-depth must be >= 1.")
    if float(getattr(args, "ltf_weight_decay", 0.0)) < 0.0:
        raise ValueError("--ltf-weight-decay must be >= 0.")
    te = float(getattr(args, "ltf_t_eps", 0.05))
    if not (0.0 < te < 0.5):
        raise ValueError("--ltf-t-eps must be in (0, 0.5).")
    sj = float(getattr(args, "ltf_solve_jitter", 1e-6))
    if not np.isfinite(sj) or sj <= 0.0:
        raise ValueError("--ltf-solve-jitter must be finite and > 0.")
    if int(getattr(args, "ltf_early_patience", -1)) < 0:
        raise ValueError("--ltf-early-patience must be >= 0.")
    if float(getattr(args, "ltf_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--ltf-early-min-delta must be >= 0.")
    alpha = float(getattr(args, "ltf_early_ema_alpha", 0.0))
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--ltf-early-ema-alpha must be in (0, 1].")
    if int(getattr(args, "ltf_pair_batch_size", 0)) < 1:
        raise ValueError("--ltf-pair-batch-size must be >= 1.")
    max_grad = float(getattr(args, "ltf_max_grad_norm", 10.0))
    if not np.isfinite(max_grad):
        raise ValueError("--ltf-max-grad-norm must be finite.")
    wema = float(getattr(args, "ltf_weight_ema_decay", 0.9))
    if not np.isfinite(wema) or wema >= 1.0:
        raise ValueError("--ltf-weight-ema-decay must be finite and < 1.")


def _validate_nfr_cli(args: argparse.Namespace) -> None:
    require_zuko_for_nf()
    if int(getattr(args, "nfr_latent_dim", 0)) < 1:
        raise ValueError("--nfr-latent-dim must be >= 1.")
    if int(getattr(args, "nfr_epochs", 0)) < 1:
        raise ValueError("--nfr-epochs must be >= 1.")
    if int(getattr(args, "nfr_batch_size", 0)) < 1:
        raise ValueError("--nfr-batch-size must be >= 1.")
    if float(getattr(args, "nfr_lr", 0.0)) <= 0.0:
        raise ValueError("--nfr-lr must be > 0.")
    if int(getattr(args, "nfr_hidden_dim", 0)) < 1:
        raise ValueError("--nfr-hidden-dim must be >= 1.")
    if int(getattr(args, "nfr_context_dim", 0)) < 1:
        raise ValueError("--nfr-context-dim must be >= 1.")
    if int(getattr(args, "nfr_transforms", 0)) < 1:
        raise ValueError("--nfr-transforms must be >= 1.")
    if int(getattr(args, "nfr_pair_batch_size", 0)) < 1:
        raise ValueError("--nfr-pair-batch-size must be >= 1.")
    if int(getattr(args, "nfr_early_patience", -1)) < 0:
        raise ValueError("--nfr-early-patience must be >= 0.")
    if float(getattr(args, "nfr_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--nfr-early-min-delta must be >= 0.")
    alpha = float(getattr(args, "nfr_early_ema_alpha", 0.0))
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--nfr-early-ema-alpha must be in (0, 1].")


def _validate_pinf_cli(args: argparse.Namespace) -> None:
    require_zuko_for_nf()
    if int(getattr(args, "pinf_latent_dim", 0)) < 1:
        raise ValueError("--pinf-latent-dim must be >= 1.")
    if int(getattr(args, "pinf_epochs", 0)) < 1:
        raise ValueError("--pinf-epochs must be >= 1.")
    if int(getattr(args, "pinf_batch_size", 0)) < 1:
        raise ValueError("--pinf-batch-size must be >= 1.")
    if float(getattr(args, "pinf_lr", 0.0)) <= 0.0:
        raise ValueError("--pinf-lr must be > 0.")
    if int(getattr(args, "pinf_hidden_dim", 0)) < 1:
        raise ValueError("--pinf-hidden-dim must be >= 1.")
    if int(getattr(args, "pinf_transforms", 0)) < 1:
        raise ValueError("--pinf-transforms must be >= 1.")
    if float(getattr(args, "pinf_min_std", 0.0)) <= 0.0:
        raise ValueError("--pinf-min-std must be > 0.")
    if float(getattr(args, "pinf_weight_decay", 0.0)) < 0.0:
        raise ValueError("--pinf-weight-decay must be >= 0.")
    rw = float(getattr(args, "pinf_recon_weight", 1.0))
    if not np.isfinite(rw) or rw < 0.0:
        raise ValueError("--pinf-recon-weight must be finite and >= 0.")
    if int(getattr(args, "pinf_pair_batch_size", 0)) < 1:
        raise ValueError("--pinf-pair-batch-size must be >= 1.")
    if int(getattr(args, "pinf_early_patience", -1)) < 0:
        raise ValueError("--pinf-early-patience must be >= 0.")
    if float(getattr(args, "pinf_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--pinf-early-min-delta must be >= 0.")
    alpha = float(getattr(args, "pinf_early_ema_alpha", 0.0))
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--pinf-early-ema-alpha must be in (0, 1].")


def _validate_contrastive_soft_gaussian_net_no_finetune_cli(args: argparse.Namespace) -> None:
    """GN MLE pretrain only; same bandwidth metadata checks as soft contrastive (no Adam fine-tuning)."""
    if int(getattr(args, "gn_epochs", 0)) < 1:
        raise ValueError("--gn-epochs must be >= 1.")
    if int(getattr(args, "gn_batch_size", 0)) < 1:
        raise ValueError("--gn-batch-size must be >= 1.")
    if float(getattr(args, "gn_lr", 0.0)) <= 0.0:
        raise ValueError("--gn-lr must be > 0.")
    if int(getattr(args, "gn_hidden_dim", 0)) < 1:
        raise ValueError("--gn-hidden-dim must be >= 1.")
    if int(getattr(args, "gn_depth", 0)) < 1:
        raise ValueError("--gn-depth must be >= 1.")
    if float(getattr(args, "gn_weight_decay", 0.0)) < 0.0:
        raise ValueError("--gn-weight-decay must be >= 0.")
    if float(getattr(args, "gn_diag_floor", 0.0)) <= 0.0:
        raise ValueError("--gn-diag-floor must be > 0.")
    if int(getattr(args, "gn_early_patience", -1)) < 0:
        raise ValueError("--gn-early-patience must be >= 0.")
    if float(getattr(args, "gn_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--gn-early-min-delta must be >= 0.")
    alpha = float(getattr(args, "gn_early_ema_alpha", 0.0))
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--gn-early-ema-alpha must be in (0, 1].")
    if int(getattr(args, "gn_pair_batch_size", 0)) < 1:
        raise ValueError("--gn-pair-batch-size must be >= 1.")
    if int(getattr(args, "contrastive_pair_batch_size", 0)) < 1:
        raise ValueError("--contrastive-pair-batch-size must be >= 1.")
    bw = float(getattr(args, "contrastive_soft_bandwidth", 1.0))
    if not np.isfinite(bw):
        raise ValueError("--contrastive-soft-bandwidth must be finite.")
    bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
    bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
    if not np.isfinite(bw_start) or not np.isfinite(bw_end):
        raise ValueError("--contrastive-soft-bandwidth-start/end must be finite.")
    if (bw_start > 0.0) != (bw_end > 0.0):
        raise ValueError(
            "--contrastive-soft-bandwidth-start and --contrastive-soft-bandwidth-end must both be > 0 "
            "to enable annealing."
        )
    if int(getattr(args, "contrastive_soft_bandwidth_k", 0)) < 1:
        raise ValueError("--contrastive-soft-bandwidth-k must be >= 1.")
    period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("--contrastive-soft-period must be finite and > 0.")


def _validate_contrastive_cli(args: argparse.Namespace) -> None:
    tfm = str(getattr(args, "theta_field_method", "")).strip().lower()
    if tfm == "contrastive_soft_gaussian_net_no_finetune":
        _validate_contrastive_soft_gaussian_net_no_finetune_cli(args)
        return
    if int(getattr(args, "contrastive_epochs", 0)) < 1:
        raise ValueError("--contrastive-epochs must be >= 1.")
    if int(getattr(args, "contrastive_batch_size", 0)) < 2:
        raise ValueError("--contrastive-batch-size must be >= 2.")
    if float(getattr(args, "contrastive_lr", 0.0)) <= 0.0:
        raise ValueError("--contrastive-lr must be > 0.")
    if int(getattr(args, "contrastive_hidden_dim", 0)) < 1:
        raise ValueError("--contrastive-hidden-dim must be >= 1.")
    if int(getattr(args, "contrastive_depth", 0)) < 1:
        raise ValueError("--contrastive-depth must be >= 1.")
    if float(getattr(args, "contrastive_weight_decay", 0.0)) < 0.0:
        raise ValueError("--contrastive-weight-decay must be >= 0.")
    if int(getattr(args, "contrastive_early_patience", -1)) < 0:
        raise ValueError("--contrastive-early-patience must be >= 0.")
    if float(getattr(args, "contrastive_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--contrastive-early-min-delta must be >= 0.")
    alpha = float(getattr(args, "contrastive_early_ema_alpha", 0.0))
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("--contrastive-early-ema-alpha must be in (0, 1].")
    max_grad = float(getattr(args, "contrastive_max_grad_norm", 10.0))
    if not np.isfinite(max_grad) or max_grad < 0.0:
        raise ValueError("--contrastive-max-grad-norm must be finite and >= 0.")
    if int(getattr(args, "contrastive_pair_batch_size", 0)) < 1:
        raise ValueError("--contrastive-pair-batch-size must be >= 1.")
    enc = normalize_contrastive_theta_encoding(str(getattr(args, "contrastive_theta_encoding", "one_hot_bin")))
    setattr(args, "contrastive_theta_encoding", enc)
    soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
    soft_arch_aliases = {
        "normalized_dot": "normalized_dot",
        "norm_dot": "normalized_dot",
        "dot": "normalized_dot",
        "additive": "additive_independent",
        "additive_independent": "additive_independent",
        "additive_independent_feature": "additive_independent",
        "independent": "additive_independent",
        "gaussian": "independent_gaussian",
        "independent_gaussian": "independent_gaussian",
        "independent_dot": "independent_dot_product",
        "independent_dot_product": "independent_dot_product",
        "dot_independent": "independent_dot_product",
        "mlp": "mlp",
    }
    if soft_arch not in soft_arch_aliases:
        raise ValueError(
            "--contrastive-soft-score-arch must be one of "
            "{'normalized_dot','additive_independent','independent_gaussian','independent_dot_product','mlp'}."
        )
    setattr(args, "contrastive_soft_score_arch", soft_arch_aliases[soft_arch])
    if int(getattr(args, "contrastive_soft_dot_dim", 0)) < 1:
        raise ValueError("--contrastive-soft-dot-dim must be >= 1.")
    if int(getattr(args, "contrastive_soft_coordinate_embed_dim", 0)) < 1:
        raise ValueError("--contrastive-soft-coordinate-embed-dim must be >= 1.")
    logvar_min = float(getattr(args, "contrastive_soft_gaussian_logvar_min", -8.0))
    logvar_max = float(getattr(args, "contrastive_soft_gaussian_logvar_max", 5.0))
    if not np.isfinite(logvar_min) or not np.isfinite(logvar_max) or logvar_min >= logvar_max:
        raise ValueError("--contrastive-soft-gaussian-logvar-min/max must be finite with min < max.")
    bw = float(getattr(args, "contrastive_soft_bandwidth", 1.0))
    if not np.isfinite(bw):
        raise ValueError("--contrastive-soft-bandwidth must be finite.")
    bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
    bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
    if not np.isfinite(bw_start) or not np.isfinite(bw_end):
        raise ValueError("--contrastive-soft-bandwidth-start/end must be finite.")
    if (bw_start > 0.0) != (bw_end > 0.0):
        raise ValueError("--contrastive-soft-bandwidth-start and --contrastive-soft-bandwidth-end must both be > 0 to enable annealing.")
    if int(getattr(args, "contrastive_soft_bandwidth_k", 0)) < 1:
        raise ValueError("--contrastive-soft-bandwidth-k must be >= 1.")
    period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("--contrastive-soft-period must be finite and > 0.")


def theta_segment_ids_equal_width(
    theta: np.ndarray,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Delegate to ``visualize_h_matrix_binned.theta_segment_ids_equal_width``."""
    return vhb.theta_segment_ids_equal_width(theta, n_segments)


def _validate_cli(args: argparse.Namespace) -> None:
    tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
    nfr_norm = _normalize_nf_reduction_method(tfm)
    gzd_norm = _normalize_gmm_z_decode_method(tfm)
    pinf_norm = _normalize_pi_nf_method(tfm)
    gxf_norm = _normalize_gaussian_x_flow_method(tfm)
    lxf_norm = _normalize_linear_x_flow_method(tfm)
    ltf_norm = _normalize_linear_theta_flow_method(tfm)
    contrastive_norm = _normalize_contrastive_method(tfm)
    gn_norm = None if (
        gxf_norm is not None
        or lxf_norm is not None
        or ltf_norm is not None
        or nfr_norm is not None
        or gzd_norm is not None
        or pinf_norm is not None
        or contrastive_norm is not None
    ) else _normalize_gaussian_network_method(tfm)
    flow_ae_norm = _normalize_flow_autoencoder_method(tfm)
    flow_pca_norm = _normalize_flow_pca_method(tfm)
    if contrastive_norm is not None:
        setattr(args, "theta_field_method", contrastive_norm)
        _validate_contrastive_cli(args)
        tfm = contrastive_norm
    elif pinf_norm is not None:
        setattr(args, "theta_field_method", pinf_norm)
        _validate_pinf_cli(args)
        tfm = pinf_norm
    elif gzd_norm is not None:
        setattr(args, "theta_field_method", gzd_norm)
        validate_gmm_z_decode_args(args)
        tfm = gzd_norm
    elif nfr_norm is not None:
        setattr(args, "theta_field_method", nfr_norm)
        _validate_nfr_cli(args)
        tfm = nfr_norm
    elif gxf_norm is not None:
        setattr(args, "theta_field_method", gxf_norm)
        _validate_gxf_cli(args)
        tfm = gxf_norm
    elif lxf_norm is not None:
        setattr(args, "theta_field_method", lxf_norm)
        _validate_lxf_cli(args)
        tfm = lxf_norm
    elif ltf_norm is not None:
        setattr(args, "theta_field_method", ltf_norm)
        _validate_ltf_cli(args)
        tfm = ltf_norm
    elif gn_norm is not None:
        gn_method = gn_norm
        setattr(args, "theta_field_method", gn_method)
        if int(getattr(args, "gn_epochs", 0)) < 1:
            raise ValueError("--gn-epochs must be >= 1.")
        if int(getattr(args, "gn_batch_size", 0)) < 1:
            raise ValueError("--gn-batch-size must be >= 1.")
        if float(getattr(args, "gn_lr", 0.0)) <= 0.0:
            raise ValueError("--gn-lr must be > 0.")
        if int(getattr(args, "gn_hidden_dim", 0)) < 1:
            raise ValueError("--gn-hidden-dim must be >= 1.")
        if int(getattr(args, "gn_depth", 0)) < 1:
            raise ValueError("--gn-depth must be >= 1.")
        if float(getattr(args, "gn_weight_decay", 0.0)) < 0.0:
            raise ValueError("--gn-weight-decay must be >= 0.")
        if float(getattr(args, "gn_diag_floor", 0.0)) <= 0.0:
            raise ValueError("--gn-diag-floor must be > 0.")
        if int(getattr(args, "gn_early_patience", -1)) < 0:
            raise ValueError("--gn-early-patience must be >= 0.")
        if float(getattr(args, "gn_early_min_delta", 0.0)) < 0.0:
            raise ValueError("--gn-early-min-delta must be >= 0.")
        alpha = float(getattr(args, "gn_early_ema_alpha", 0.0))
        if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError("--gn-early-ema-alpha must be in (0, 1].")
        if int(getattr(args, "gn_pair_batch_size", 0)) < 1:
            raise ValueError("--gn-pair-batch-size must be >= 1.")
        if int(getattr(args, "gn_pca_dim", 0)) < 1:
            raise ValueError("--gn-pca-dim must be >= 1.")
        pca_bins = getattr(args, "gn_pca_num_bins", None)
        if pca_bins is not None and int(pca_bins) < 2:
            raise ValueError("--gn-pca-num-bins must be >= 2.")
        if int(getattr(args, "gn_low_rank_dim", 0)) < 1:
            raise ValueError("--gn-low-rank-dim must be >= 1.")
        if float(getattr(args, "gn_psi_floor", 0.0)) <= 0.0:
            raise ValueError("--gn-psi-floor must be > 0.")
        if gn_method in ("gaussian_network_autoencoder", "gaussian_network_diagonal_autoencoder"):
            _validate_autoencoder_cli(args)
        tfm = gn_method
    elif flow_ae_norm is not None:
        setattr(args, "theta_field_method", flow_ae_norm)
        _validate_autoencoder_cli(args)
        base_args = argparse.Namespace(**vars(args).copy())
        setattr(base_args, "theta_field_method", _base_flow_method_for_autoencoder(flow_ae_norm))
        validate_estimation_args(base_args)
        tfm = flow_ae_norm
    elif flow_pca_norm is not None:
        setattr(args, "theta_field_method", flow_pca_norm)
        _validate_flow_pca_cli(args)
        tfm = flow_pca_norm
    if tfm == "nf":
        setattr(args, "theta_field_method", "nf")
        require_zuko_for_nf()
        if int(getattr(args, "nf_epochs", 0)) < 1:
            raise ValueError("--nf-epochs must be >= 1.")
        if int(getattr(args, "nf_batch_size", 0)) < 1:
            raise ValueError("--nf-batch-size must be >= 1.")
        if float(getattr(args, "nf_lr", 0.0)) <= 0.0:
            raise ValueError("--nf-lr must be > 0.")
        if int(getattr(args, "nf_hidden_dim", 0)) < 1:
            raise ValueError("--nf-hidden-dim must be >= 1.")
        if int(getattr(args, "nf_context_dim", 0)) < 1:
            raise ValueError("--nf-context-dim must be >= 1.")
        if int(getattr(args, "nf_transforms", 0)) < 1:
            raise ValueError("--nf-transforms must be >= 1.")
        if int(getattr(args, "nf_pair_batch_size", 0)) < 1:
            raise ValueError("--nf-pair-batch-size must be >= 1.")
        if int(getattr(args, "nf_early_patience", -1)) < 0:
            raise ValueError("--nf-early-patience must be >= 0.")
        alpha = float(getattr(args, "nf_early_ema_alpha", 0.0))
        if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
            raise ValueError("--nf-early-ema-alpha must be in (0, 1].")
        _pe = getattr(args, "nf_prior_epochs", None)
        if _pe is not None and int(_pe) < 1:
            raise ValueError("--nf-prior-epochs must be >= 1.")
        _pb = getattr(args, "nf_prior_batch_size", None)
        if _pb is not None and int(_pb) < 1:
            raise ValueError("--nf-prior-batch-size must be >= 1.")
        _plr = getattr(args, "nf_prior_lr", None)
        if _plr is not None and float(_plr) <= 0.0:
            raise ValueError("--nf-prior-lr must be > 0.")
        _ph = getattr(args, "nf_prior_hidden_dim", None)
        if _ph is not None and int(_ph) < 1:
            raise ValueError("--nf-prior-hidden-dim must be >= 1.")
        _pt = getattr(args, "nf_prior_transforms", None)
        if _pt is not None and int(_pt) < 1:
            raise ValueError("--nf-prior-transforms must be >= 1.")
        _pp = getattr(args, "nf_prior_early_patience", None)
        if _pp is not None and int(_pp) < 0:
            raise ValueError("--nf-prior-early-patience must be >= 0.")
        _pm = getattr(args, "nf_prior_early_min_delta", None)
        if _pm is not None and float(_pm) < 0.0:
            raise ValueError("--nf-prior-early-min-delta must be >= 0.")
        _pa = getattr(args, "nf_prior_early_ema_alpha", None)
        if _pa is not None:
            pa = float(_pa)
            if (not np.isfinite(pa)) or pa <= 0.0 or pa > 1.0:
                raise ValueError("--nf-prior-early-ema-alpha must be in (0, 1].")
    elif tfm not in (
        "gaussian_network",
        "gaussian_network_diagonal",
        "gaussian_network_diagonal_binned_pca",
        "gaussian_network_low_rank",
        "gaussian_network_autoencoder",
        "gaussian_network_diagonal_autoencoder",
        "theta_flow_autoencoder",
        "x_flow_autoencoder",
        "x_flow_pca",
        "gaussian_x_flow",
        "gaussian_x_flow_diagonal",
        "linear_x_flow",
        "linear_x_flow_scalar",
        "linear_x_flow_diagonal",
        "bin_gaussian_linear_x_flow_diagonal",
        "linear_x_flow_diagonal_theta",
        "linear_x_flow_diagonal_theta_spline",
        "linear_x_flow_diagonal_t",
        "linear_x_flow_low_rank",
        "linear_x_flow_low_rank_randb",
        "linear_x_flow_nonlinear_pca",
        "linear_x_flow_schedule",
        "linear_theta_flow",
        "nf_reduction",
        "gmm_z_decode",
        "pi_nf",
        "contrastive",
        "contrastive_soft",
        "bidir_contrastive_soft",
        "contrastive_soft_gaussian_net",
        "contrastive_soft_gaussian_net_no_finetune",
    ):
        validate_estimation_args(args)
    if int(args.num_theta_bins) < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    theta_binning_mode = str(getattr(args, "theta_binning_mode", "theta1")).strip().lower()
    if theta_binning_mode not in ("theta1", "theta2_grid"):
        raise ValueError("--theta-binning-mode must be one of: theta1, theta2_grid.")
    if int(getattr(args, "num_theta_bins_y", 0)) < 0:
        raise ValueError("--num-theta-bins-y must be >= 0.")
    if int(args.clf_min_class_count) < 1:
        raise ValueError("--clf-min-class-count must be >= 1.")
    n_ref = int(args.n_ref)
    n_bins_cli = int(args.num_theta_bins)
    if theta_binning_mode == "theta2_grid":
        n_bins_y = int(getattr(args, "num_theta_bins_y", 0)) or n_bins_cli
        n_bins_cli = n_bins_cli * n_bins_y
    if n_ref < 2:
        raise ValueError("--n-ref must be >= 2.")
    if n_ref // n_bins_cli < 1:
        raise ValueError(
            "GT Hellinger requires n_mc = n_ref // total_theta_bins >= 1 "
            f"(got n_ref={n_ref} total_theta_bins={n_bins_cli})."
        )
    use_onehot = bool(getattr(args, "theta_flow_onehot_state", False))
    use_fourier = bool(getattr(args, "theta_flow_fourier_state", False))
    use_segmented = bool(getattr(args, "theta_flow_segmented", False))
    if use_onehot and use_fourier:
        raise ValueError("Use only one theta-flow state override: one-hot or Fourier, not both.")
    if use_segmented and use_onehot:
        raise ValueError("Use only one theta-flow mode: segmented or one-hot, not both.")
    if use_segmented and use_fourier:
        raise ValueError("Use only one theta-flow mode: segmented or Fourier, not both.")
    if use_onehot:
        tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
        arch = str(getattr(args, "flow_arch", "mlp")).strip().lower()
        if tfm not in ("theta_flow", "theta_flow_autoencoder"):
            raise ValueError(
                "--theta-flow-onehot-state requires --theta-field-method theta_flow or theta-flow-autoencoder "
                f"(got {getattr(args, 'theta_field_method', None)!r})."
            )
        if arch != "mlp":
            raise ValueError(
                "--theta-flow-onehot-state currently supports --flow-arch mlp only "
                f"(got {getattr(args, 'flow_arch', None)!r})."
            )
    if use_fourier:
        tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
        arch = str(getattr(args, "flow_arch", "mlp")).strip().lower()
        if tfm not in ("theta_flow", "theta_flow_autoencoder"):
            raise ValueError(
                "--theta-flow-fourier-state requires --theta-field-method theta_flow or theta-flow-autoencoder "
                f"(got {getattr(args, 'theta_field_method', None)!r})."
            )
        if arch != "mlp":
            raise ValueError(
                "--theta-flow-fourier-state currently supports --flow-arch mlp only "
                f"(got {getattr(args, 'flow_arch', None)!r})."
            )
        if int(getattr(args, "theta_flow_fourier_k", 0)) < 1:
            raise ValueError("--theta-flow-fourier-k must be >= 1.")
        period_mult = float(getattr(args, "theta_flow_fourier_period_mult", 0.0))
        if not np.isfinite(period_mult) or period_mult <= 0.0:
            raise ValueError("--theta-flow-fourier-period-mult must be a finite positive number.")
    if use_segmented:
        tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
        if tfm not in ("theta_flow", "theta_flow_autoencoder"):
            raise ValueError(
                "--theta-flow-segmented requires --theta-field-method theta_flow or theta-flow-autoencoder "
                f"(got {getattr(args, 'theta_field_method', None)!r})."
            )
    _parse_n_list(args.n_list)  # syntax check only; pool size checked in main


def _build_theta_fourier_state(
    theta_scalar: np.ndarray,
    *,
    theta_ref: np.ndarray,
    k: int,
    period_mult: float,
    include_linear: bool,
) -> tuple[np.ndarray, float, float, float]:
    """Build deterministic Fourier theta-state vectors from scalar theta."""
    theta_all = np.asarray(theta_scalar, dtype=np.float64).reshape(-1)
    theta_ref_vec = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    if theta_all.size < 1 or theta_ref_vec.size < 1:
        raise ValueError("Fourier theta state requires non-empty theta arrays.")
    ref_min = float(np.min(theta_ref_vec))
    ref_max = float(np.max(theta_ref_vec))
    ref_range = float(ref_max - ref_min)
    range_safe = max(ref_range, 1e-12)
    period = float(period_mult) * range_safe
    w0 = 2.0 * np.pi / period
    theta_center = 0.5 * (ref_min + ref_max)
    theta_shift = theta_all - theta_center
    cols: list[np.ndarray] = []
    if include_linear:
        cols.append((theta_shift / range_safe).reshape(-1, 1))
    for kk in range(1, int(k) + 1):
        phase = (float(kk) * w0) * theta_shift
        cols.append(np.sin(phase).reshape(-1, 1))
        cols.append(np.cos(phase).reshape(-1, 1))
    out = np.concatenate(cols, axis=1).astype(np.float64, copy=False)
    return out, ref_range, period, theta_center


def prepare_theta_binning_for_convergence(
    theta_raw_all: np.ndarray,
    perm: np.ndarray,
    n_ref: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Equal-width bins on θ₁ when ``theta_all`` has shape ``(N, 2)``; else scalar θ as before.

    Bin membership is defined by the first coordinate only; full θ rows stay in the bundle for
    training. Generative GT MC pairs each θ₁ bin center with an independent θ₂ ~ Uniform
    (see :func:`fisher.hellinger_gt.estimate_hellinger_sq_one_sided_mc`).
    """
    th = np.asarray(theta_raw_all, dtype=np.float64)
    if th.ndim == 1:
        th = th.reshape(-1, 1)
    elif th.ndim != 2:
        raise ValueError(
            "Convergence binning expects theta_all as 1D, (N, 1), or (N, 2); "
            f"got shape {th.shape}."
        )
    d = int(th.shape[1])
    if d > 2:
        raise ValueError(
            "Convergence binning supports theta_dim <= 2; " f"got theta_all shape={th.shape}."
        )
    if d == 2:
        theta_scalar_all = np.asarray(th[:, 0], dtype=np.float64).reshape(-1)
        print(
            "[convergence] theta_dim=2: binning on theta[:, 0] (theta_1); "
            "full (theta_1, theta_2) retained for model training.",
            flush=True,
        )
    else:
        theta_scalar_all = np.asarray(th[:, 0], dtype=np.float64).reshape(-1)
    n_ref_i = int(n_ref)
    n_bins_i = int(n_bins)
    theta_ref = np.asarray(theta_scalar_all[perm[:n_ref_i]], dtype=np.float64).reshape(-1)
    edges, edge_lo, edge_hi = vhb.theta_bin_edges(theta_ref, n_bins_i)
    bin_idx_all = vhb.theta_to_bin_index(theta_scalar_all, edges, n_bins_i)
    return theta_scalar_all, theta_ref, edges, float(edge_lo), float(edge_hi), bin_idx_all


class Theta2GridBinning(NamedTuple):
    theta_scalar_all: np.ndarray
    theta_ref: np.ndarray
    edges0: np.ndarray
    edges1: np.ndarray
    centers: np.ndarray
    bin_idx_all: np.ndarray
    grid_shape: tuple[int, int]
    edge_lo0: float
    edge_hi0: float
    edge_lo1: float
    edge_hi1: float


def theta2_grid_centers_from_edges(edges0: np.ndarray, edges1: np.ndarray) -> np.ndarray:
    """Flattened 2D bin centers in row-major order: ``flat = i * n_y + j``."""
    c0 = bin_centers_from_edges(np.asarray(edges0, dtype=np.float64))
    c1 = bin_centers_from_edges(np.asarray(edges1, dtype=np.float64))
    grid0, grid1 = np.meshgrid(c0, c1, indexing="ij")
    return np.stack([grid0.reshape(-1), grid1.reshape(-1)], axis=1).astype(np.float64, copy=False)


def prepare_theta2_grid_binning_for_convergence(
    theta_raw_all: np.ndarray,
    perm: np.ndarray,
    n_ref: int,
    n_bins_x: int,
    n_bins_y: int,
) -> Theta2GridBinning:
    """Equal-width bins on both theta coordinates, flattened to one matrix axis."""
    th = np.asarray(theta_raw_all, dtype=np.float64)
    if th.ndim != 2 or int(th.shape[1]) != 2:
        raise ValueError(
            "theta2_grid binning requires theta_all with shape (N, 2); "
            f"got shape={th.shape}."
        )
    nx = int(n_bins_x)
    ny = int(n_bins_y)
    if nx < 1 or ny < 1:
        raise ValueError("theta2_grid requires positive bin counts in both dimensions.")
    n_ref_i = int(n_ref)
    theta_ref_2d = np.asarray(th[np.asarray(perm, dtype=np.int64)[:n_ref_i]], dtype=np.float64)
    edges0, lo0, hi0 = vhb.theta_bin_edges(theta_ref_2d[:, 0], nx)
    edges1, lo1, hi1 = vhb.theta_bin_edges(theta_ref_2d[:, 1], ny)
    ix = vhb.theta_to_bin_index(th[:, 0], edges0, nx)
    iy = vhb.theta_to_bin_index(th[:, 1], edges1, ny)
    flat = (ix * ny + iy).astype(np.int64, copy=False)
    centers = theta2_grid_centers_from_edges(edges0, edges1)
    print(
        "[convergence] theta_dim=2: binning on a flattened theta_1 x theta_2 grid "
        f"({nx} x {ny} = {nx * ny} bins); full theta retained for model training.",
        flush=True,
    )
    return Theta2GridBinning(
        theta_scalar_all=flat.astype(np.float64, copy=False),
        theta_ref=theta_ref_2d,
        edges0=np.asarray(edges0, dtype=np.float64),
        edges1=np.asarray(edges1, dtype=np.float64),
        centers=centers,
        bin_idx_all=flat,
        grid_shape=(nx, ny),
        edge_lo0=float(lo0),
        edge_hi0=float(hi0),
        edge_lo1=float(lo1),
        edge_hi1=float(hi1),
    )


class SweepSubset(NamedTuple):
    bundle: SharedDatasetBundle
    bin_all: np.ndarray
    bin_train: np.ndarray
    bin_validation: np.ndarray


def _subset_bundle(
    bundle: SharedDatasetBundle,
    perm: np.ndarray,
    n: int,
    meta: dict,
    *,
    bin_idx_all: np.ndarray,
    theta_state_all: np.ndarray | None = None,
) -> SweepSubset:
    """First n indices in perm order (nested subsets). Train/validation split matches make_dataset."""
    n = int(n)
    sub_perm = perm[:n]
    theta_src_all = bundle.theta_all if theta_state_all is None else theta_state_all
    theta_all = np.asarray(theta_src_all[sub_perm], dtype=np.float64)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)
    elif theta_all.ndim != 2:
        raise ValueError("theta_state_all must be 1D or 2D.")
    x_all = np.asarray(bundle.x_all[sub_perm], dtype=np.float64)
    bin_all = np.asarray(bin_idx_all[sub_perm], dtype=np.int64).reshape(-1)
    if bin_all.shape[0] != n:
        raise ValueError("bin_idx_all subset length mismatch.")
    tf = float(meta["train_frac"])
    if tf >= 1.0:
        n_train = n
    else:
        n_train = int(tf * n)
        n_train = min(max(n_train, 1), n - 1)
    theta_train = theta_all[:n_train]
    x_train = x_all[:n_train]
    theta_validation = theta_all[n_train:]
    x_validation = x_all[n_train:]
    bin_train = bin_all[:n_train]
    bin_validation = bin_all[n_train:]
    train_idx = np.arange(n_train, dtype=np.int64)
    validation_idx = np.arange(n_train, n, dtype=np.int64)
    return SweepSubset(
        bundle=SharedDatasetBundle(
            meta=bundle.meta,
            theta_all=theta_all,
            x_all=x_all,
            train_idx=train_idx,
            validation_idx=validation_idx,
            theta_train=theta_train,
            x_train=x_train,
            theta_validation=theta_validation,
            x_validation=x_validation,
        ),
        bin_all=bin_all,
        bin_train=bin_train,
        bin_validation=bin_validation,
    )


def _make_full_args(args: argparse.Namespace, meta: dict) -> SimpleNamespace:
    full_args = merge_meta_into_args(meta, args)
    rs = getattr(args, "run_seed", None)
    if rs is not None:
        setattr(full_args, "seed", int(rs))
    setattr(full_args, "compute_h_matrix", True)
    setattr(full_args, "h_restore_original_order", True)
    setattr(full_args, "skip_shared_fisher_gt_compare", True)
    # Save delta_l_matrix in h_matrix_results*.npz for LLR binned vs generative mean LLR scatter.
    setattr(full_args, "h_save_intermediates", True)
    validate_estimation_args(full_args)
    return full_args


def _run_ctx_for_bundle(
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    full_args: SimpleNamespace,
    n_bins: int,
) -> vhb.RunContext:
    run_seed = int(getattr(full_args, "seed", meta["seed"]))
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    dev = require_device(str(full_args.device))
    dataset = build_dataset_from_meta(meta)
    cfg = vhb.BinnedVizConfig(args=args, dataset_npz=str(args.dataset_npz), n_bins=n_bins, h_only=False)
    return vhb.RunContext(
        args=args,
        config=cfg,
        bundle=bundle,
        meta=meta,
        full_args=full_args,
        dataset=dataset,
        rng=rng,
        device=dev,
    )


def _validate_theta_used_matches_bundle(theta_chk: np.ndarray, theta_used_npz: np.ndarray, *, err_suffix: str) -> None:
    """Align dimension-wise theta from bundle vs ``h_matrix_results*.npz`` (scalar or `(N, d)`)."""
    tc = np.asarray(theta_chk, dtype=np.float64)
    tu = np.asarray(theta_used_npz, dtype=np.float64)
    if tc.ndim == 1:
        tc = tc.reshape(-1, 1)
    if tu.ndim == 1:
        tu = tu.reshape(-1, 1)
    if tc.shape != tu.shape:
        raise ValueError(
            f"theta/H shape mismatch: theta_chk={tc.shape} theta_used={tu.shape} ({err_suffix})"
        )
    if not np.allclose(tc, tu, rtol=0.0, atol=1e-5):
        raise ValueError(
            "theta_used from H-matrix npz does not match expected dataset rows " + f"({err_suffix})."
        )


def _rewrite_npz_fields(path: str, **updates: Any) -> None:
    if not os.path.exists(path):
        return
    with np.load(path, allow_pickle=True) as z:
        payload = {name: z[name] for name in z.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def _train_autoencoder_and_encode_bundle(
    *,
    args: argparse.Namespace,
    bundle: SharedDatasetBundle,
    device: torch.device,
) -> tuple[SharedDatasetBundle, dict[str, Any], int]:
    x_train = np.asarray(bundle.x_train, dtype=np.float64)
    x_val = np.asarray(bundle.x_validation, dtype=np.float64)
    x_all = np.asarray(bundle.x_all, dtype=np.float64)
    if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
        raise ValueError("Autoencoder preprocessing expects x arrays to be 2D.")
    default_latent_dim = min(8, int(x_all.shape[1]))
    ae_latent_dim = int(getattr(args, "gn_ae_latent_dim", default_latent_dim) or default_latent_dim)
    if ae_latent_dim > int(x_all.shape[1]):
        raise ValueError(f"--gn-ae-latent-dim must be <= x_dim={int(x_all.shape[1])}; got {ae_latent_dim}.")
    ae_model = ObservationAutoencoder(
        x_dim=int(x_all.shape[1]),
        latent_dim=ae_latent_dim,
        hidden_dim=int(getattr(args, "gn_ae_hidden_dim", 128)),
        depth=int(getattr(args, "gn_ae_depth", 2)),
    ).to(device)
    ae_train_out = train_observation_autoencoder(
        model=ae_model,
        x_train=x_train,
        x_val=x_val,
        device=device,
        epochs=int(getattr(args, "gn_ae_epochs", 1000)),
        batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
        lr=float(getattr(args, "gn_ae_lr", 1e-3)),
        weight_decay=float(getattr(args, "gn_ae_weight_decay", 0.0)),
        patience=int(getattr(args, "gn_ae_early_patience", 200)),
        min_delta=float(getattr(args, "gn_ae_early_min_delta", 1e-4)),
        ema_alpha=float(getattr(args, "gn_ae_early_ema_alpha", 0.05)),
        log_every=max(1, int(getattr(args, "log_every", 50))),
        restore_best=True,
    )
    z_train = encode_observations(
        model=ae_model,
        x=x_train,
        device=device,
        batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
    )
    z_val = encode_observations(
        model=ae_model,
        x=x_val,
        device=device,
        batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
    )
    z_all = encode_observations(
        model=ae_model,
        x=x_all,
        device=device,
        batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
    )
    encoded_bundle = SharedDatasetBundle(
        meta=bundle.meta,
        theta_all=bundle.theta_all,
        x_all=z_all,
        train_idx=bundle.train_idx,
        validation_idx=bundle.validation_idx,
        theta_train=bundle.theta_train,
        x_train=z_train,
        theta_validation=bundle.theta_validation,
        x_validation=z_val,
    )
    return encoded_bundle, ae_train_out, ae_latent_dim


def _metrics_fixed_edges(
    loaded: vhb.LoadedHMatrix,
    subset: SweepSubset,
    n_bins: int,
    clf_min_class_count: int,
    clf_random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if loaded.h_sym.shape[0] != subset.bin_all.shape[0]:
        raise ValueError(
            f"h_sym rows {loaded.h_sym.shape[0]} do not match subset bins length {subset.bin_all.shape[0]}."
        )
    h_binned, _ = vhb.average_matrix_by_bins(loaded.h_sym, subset.bin_all, n_bins)
    clf_acc, _, _, _ = vhb.pairwise_bin_logistic_accuracy_train_val(
        subset.bundle.x_train,
        subset.bin_train,
        subset.bundle.x_all,
        subset.bin_all,
        n_bins,
        min_class_count=int(clf_min_class_count),
        random_state=int(clf_random_state),
    )
    return h_binned, clf_acc


def _binned_gaussian_hellinger_sq(
    subset: SweepSubset,
    n_bins: int,
    *,
    variance_floor: float = 1e-6,
) -> np.ndarray:
    r"""Binned-Gaussian ``H^2`` estimate using per-bin means and shared diagonal variance.

    This no-flow diagnostic fits ``p(x | bin(theta)=b) = N(mu_b, diag(global_var))``
    from the subset full pool, then computes the closed-form shared-covariance
    Gaussian Hellinger distance.
    """
    x_all = np.asarray(subset.bundle.x_all, dtype=np.float64)
    bin_all = np.asarray(subset.bin_all, dtype=np.int64).reshape(-1)
    nb = int(n_bins)
    vf = float(variance_floor)
    if x_all.ndim != 2:
        raise ValueError("x_all must be 2D.")
    if x_all.shape[0] != bin_all.shape[0]:
        raise ValueError("x_all and bin_all must have the same number of rows.")
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    if not np.isfinite(vf) or vf <= 0.0:
        raise ValueError("variance_floor must be a finite positive number.")

    x_dim = int(x_all.shape[1])
    means = np.zeros((nb, x_dim), dtype=np.float64)
    counts = np.bincount(np.clip(bin_all, 0, nb - 1), minlength=nb).astype(np.int64)
    for b in range(nb):
        idx = np.flatnonzero(bin_all == b)
        if idx.size > 0:
            means[b] = np.mean(x_all[idx], axis=0)

    nonempty = counts > 0
    nonempty_idx = np.flatnonzero(nonempty)
    out = np.full((nb, nb), np.nan, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    if nonempty_idx.size == 0:
        return out
    for b in np.flatnonzero(~nonempty):
        nearest = int(nonempty_idx[np.argmin(np.abs(nonempty_idx - int(b)))])
        means[int(b)] = means[nearest]

    train_means = means[np.clip(bin_all, 0, nb - 1)]
    global_var = np.maximum(np.mean((x_all - train_means) ** 2, axis=0), vf)
    inv_var = 1.0 / global_var
    for i in range(nb):
        for j in range(i + 1, nb):
            diff = means[i] - means[j]
            maha2 = float(np.sum(diff * diff * inv_var))
            if not np.isfinite(maha2):
                continue
            h2_ij = 1.0 - float(np.exp(-0.125 * max(0.0, maha2)))
            h2_ij = float(np.clip(h2_ij, 0.0, 1.0))
            out[i, j] = h2_ij
            out[j, i] = h2_ij
    return out


def _save_empty_no_training_losses(path: str, *, method_name: str, **metadata: object) -> None:
    empty = np.asarray([], dtype=np.float64)
    payload: dict[str, object] = {
        "theta_field_method": np.asarray([method_name], dtype=object),
        "prior_enable": np.bool_(False),
        "score_train_losses": empty,
        "score_val_losses": empty,
        "score_val_monitor_losses": empty,
        "score_best_epoch": np.int64(0),
        "score_stopped_epoch": np.int64(0),
        "score_stopped_early": np.bool_(False),
        "score_best_val_smooth": np.float64(float("nan")),
        "score_grad_norm_mean": np.float64(float("nan")),
        "score_grad_norm_max": np.float64(float("nan")),
        "score_param_norm_final": np.float64(float("nan")),
        "score_n_clipped_steps": np.int64(0),
        "score_n_total_steps": np.int64(0),
        "score_lr_last": np.float64(float("nan")),
        "score_final_eval_weights": np.asarray(["analytic"], dtype=object),
        "ae_train_losses": empty,
        "ae_val_losses": empty,
        "ae_val_monitor_losses": empty,
        "ae_best_epoch": np.int64(0),
        "ae_stopped_epoch": np.int64(0),
        "ae_stopped_early": np.bool_(False),
        "ae_latent_dim": np.int64(0),
        "score_likelihood_finetune_train_losses": empty,
        "score_likelihood_finetune_val_losses": empty,
        "score_likelihood_finetune_val_monitor_losses": empty,
        "prior_train_losses": empty,
        "prior_val_losses": empty,
        "prior_val_monitor_losses": empty,
        "prior_likelihood_finetune_train_losses": empty,
        "prior_likelihood_finetune_val_losses": empty,
        "prior_likelihood_finetune_val_monitor_losses": empty,
    }
    payload.update(metadata)
    np.savez_compressed(path, **payload)


def _fit_binned_mean_pca_projection(
    *,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    bin_train: np.ndarray,
    x_val: np.ndarray,
    x_all: np.ndarray,
    n_bins: int,
    pca_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | int]]:
    """Fit PCA from theta-binned train means and project train/val/all observations."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    th_tr = np.asarray(theta_train, dtype=np.float64)
    x_va = np.asarray(x_val, dtype=np.float64)
    x_full = np.asarray(x_all, dtype=np.float64)
    bins = np.asarray(bin_train, dtype=np.int64).reshape(-1)
    nb = int(n_bins)
    m = int(pca_dim)
    if x_tr.ndim != 2 or x_va.ndim != 2 or x_full.ndim != 2:
        raise ValueError("binned PCA expects x_train, x_val, and x_all to be 2D.")
    if x_tr.shape[0] != bins.shape[0]:
        raise ValueError("binned PCA bin_train length must match x_train rows.")
    if th_tr.shape[0] != x_tr.shape[0]:
        raise ValueError("binned PCA theta_train length must match x_train rows.")
    if x_tr.shape[1] != x_va.shape[1] or x_tr.shape[1] != x_full.shape[1]:
        raise ValueError("binned PCA x dimension mismatch.")
    if nb < 2:
        raise ValueError("--gn-pca-num-bins must be >= 2.")
    if m < 1:
        raise ValueError("--gn-pca-dim must be >= 1.")
    if m > int(x_tr.shape[1]):
        raise ValueError(f"--gn-pca-dim must be <= x_dim={int(x_tr.shape[1])}; got {m}.")

    counts = np.bincount(np.clip(bins, 0, nb - 1), minlength=nb).astype(np.int64)
    nonempty = counts > 0
    nonempty_idx = np.flatnonzero(nonempty)
    if nonempty_idx.size < 2:
        raise ValueError(
            "binned PCA projection requires at least two non-empty theta bins in the train split."
        )
    max_rank = min(int(x_tr.shape[1]), int(nonempty_idx.size) - 1)
    if m > max_rank:
        raise ValueError(
            f"--gn-pca-dim={m} exceeds available binned-mean PCA rank {max_rank} "
            f"(non_empty_bins={int(nonempty_idx.size)}, x_dim={int(x_tr.shape[1])})."
        )

    means = np.full((nb, int(x_tr.shape[1])), np.nan, dtype=np.float64)
    theta_centers = np.full(nb, np.nan, dtype=np.float64)
    th_flat = th_tr.reshape(th_tr.shape[0], -1)[:, 0]
    for b in nonempty_idx:
        mask = bins == int(b)
        means[int(b)] = np.mean(x_tr[mask], axis=0, dtype=np.float64)
        theta_centers[int(b)] = float(np.mean(th_flat[mask], dtype=np.float64))
    fit_means = means[nonempty]
    pca_mean = np.mean(fit_means, axis=0, dtype=np.float64)
    centered = fit_means - pca_mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:m].T.astype(np.float64, copy=False)

    z_train = (x_tr - pca_mean) @ components
    z_val = (x_va - pca_mean) @ components
    z_all = (x_full - pca_mean) @ components
    meta: dict[str, np.ndarray | int] = {
        "pca_mean": pca_mean.astype(np.float64, copy=False),
        "pca_components": components,
        "pca_singular_values": np.asarray(singular_values[:m], dtype=np.float64),
        "pca_bin_counts": counts,
        "pca_theta_bin_centers": theta_centers,
        "pca_binned_train_means": means,
        "pca_nonempty_bins": nonempty_idx.astype(np.int64, copy=False),
    }
    return z_train, z_val, z_all, meta


def _pairwise_clf_from_bundle(
    *,
    args: argparse.Namespace,
    meta: dict,
    subset: SweepSubset,
    output_dir: str,
    n_bins: int,
    clf_min_class_count: int,
    clf_random_state: int,
) -> np.ndarray:
    """Pairwise bin decoding: train on NPZ train rows, accuracy on NPZ full pool."""
    clf_acc, _, _, _ = vhb.pairwise_bin_logistic_accuracy_train_val(
        subset.bundle.x_train,
        subset.bin_train,
        subset.bundle.x_all,
        subset.bin_all,
        n_bins,
        min_class_count=int(clf_min_class_count),
        random_state=int(clf_random_state),
    )
    return clf_acc


def _estimate_one(
    *,
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    output_dir: str,
    n_bins: int,
    bin_train: np.ndarray | None = None,
    bin_validation: np.ndarray | None = None,
    bin_all: np.ndarray | None = None,
) -> tuple[vhb.LoadedHMatrix, np.ndarray, torch.device]:
    """Train (unless h-only), load H, return loaded H, x_aligned, and device."""
    tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
    contrastive_norm = _normalize_contrastive_method(tfm)
    if contrastive_norm in ("contrastive_soft_gaussian_net", "contrastive_soft_gaussian_net_no_finetune"):
        method_name = contrastive_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("contrastive-soft-gaussian-net expects theta arrays to be 1D or 2D.")
        if int(theta_train.shape[1]) != 1 or int(theta_all.shape[1]) != 1:
            raise ValueError("contrastive-soft-gaussian-net v1 requires scalar theta.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("contrastive-soft-gaussian-net expects x arrays to be 2D.")
        if theta_train.shape[0] < 2 or theta_val.shape[0] < 2:
            raise ValueError("contrastive-soft-gaussian-net requires at least two train and two validation rows.")

        x_mean_pre = np.mean(x_train, axis=0, dtype=np.float64)
        x_std_pre = np.maximum(np.std(x_train, axis=0, dtype=np.float64), 1e-6)
        theta_mean_pre = np.mean(theta_train, axis=0, dtype=np.float64)
        theta_std_pre = np.maximum(np.std(theta_train, axis=0, dtype=np.float64), 1e-6)
        x_train_n = (x_train - x_mean_pre) / x_std_pre
        x_val_n = (x_val - x_mean_pre) / x_std_pre
        theta_train_n = (theta_train - theta_mean_pre) / theta_std_pre
        theta_val_n = (theta_val - theta_mean_pre) / theta_std_pre

        gaussian_model = ConditionalDiagonalGaussianPrecisionMLP(
            theta_dim=1,
            x_dim=int(x_all.shape[1]),
            hidden_dim=int(getattr(args, "gn_hidden_dim", 128)),
            depth=int(getattr(args, "gn_depth", 3)),
            diag_floor=float(getattr(args, "gn_diag_floor", 1e-4)),
        ).to(dev)
        gn_train_out = train_gaussian_network(
            model=gaussian_model,
            theta_train=theta_train_n,
            x_train=x_train_n,
            theta_val=theta_val_n,
            x_val=x_val_n,
            device=dev,
            epochs=int(getattr(args, "gn_epochs", 4000)),
            batch_size=int(getattr(args, "gn_batch_size", 256)),
            lr=float(getattr(args, "gn_lr", 1e-3)),
            weight_decay=float(getattr(args, "gn_weight_decay", 0.0)),
            patience=int(getattr(args, "gn_early_patience", 300)),
            min_delta=float(getattr(args, "gn_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "gn_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "gn_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        model = ContrastiveGaussianNetworkScorer(gaussian_model).to(dev)
        bw_arg = float(getattr(args, "contrastive_soft_bandwidth", 1.0))
        bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
        bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
        bw_k = int(getattr(args, "contrastive_soft_bandwidth_k", 5))
        periodic = bool(getattr(args, "contrastive_soft_periodic", False))
        period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
        if contrastive_norm == "contrastive_soft_gaussian_net_no_finetune":
            train_out = contrastive_soft_metadata_without_training(
                theta_train=theta_train,
                x_train=x_train,
                bandwidth=bw_arg,
                bandwidth_start=bw_start,
                bandwidth_end=bw_end,
                bandwidth_k=bw_k,
                periodic=periodic,
                period=period,
            )
        else:
            train_out = train_contrastive_soft_llr(
                model=model,
                theta_train=theta_train,
                x_train=x_train,
                theta_val=theta_val,
                x_val=x_val,
                device=dev,
                epochs=int(getattr(args, "contrastive_epochs", 2000)),
                batch_size=int(getattr(args, "contrastive_batch_size", 256)),
                lr=float(getattr(args, "contrastive_lr", 1e-3)),
                bandwidth=bw_arg,
                bandwidth_start=bw_start,
                bandwidth_end=bw_end,
                bandwidth_k=bw_k,
                periodic=periodic,
                period=period,
                weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
                patience=int(getattr(args, "contrastive_early_patience", 300)),
                min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
                ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
                max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
                log_every=max(1, int(getattr(args, "log_every", 50))),
                restore_best=True,
            )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        theta_mean = np.asarray(train_out["theta_mean"], dtype=np.float64)
        theta_std = np.asarray(train_out["theta_std"], dtype=np.float64)
        c_matrix = compute_contrastive_soft_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
            pair_batch_size=int(getattr(args, "contrastive_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_contrastive(delta_l))
        theta_used = theta_all.reshape(-1)

        h_eval_name = (
            "contrastive_soft_gaussian_net_no_finetune_log_p_x_given_theta"
            if contrastive_norm == "contrastive_soft_gaussian_net_no_finetune"
            else "contrastive_soft_gaussian_net_log_p_x_given_theta"
        )
        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray([h_eval_name], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            gn_hidden_dim=np.int64(getattr(args, "gn_hidden_dim", 128)),
            gn_depth=np.int64(getattr(args, "gn_depth", 3)),
            gn_diag_floor=np.float64(getattr(args, "gn_diag_floor", 1e-4)),
            contrastive_effective_batch_size=np.int64(train_out.get("effective_batch_size", 0)),
            contrastive_soft_bandwidth=np.float64(train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_normalized=np.float64(train_out["bandwidth_normalized"]),
            contrastive_soft_bandwidth_auto=np.bool_(train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_start_normalized=np.float64(train_out["bandwidth_start_normalized"]),
            contrastive_soft_bandwidth_end_normalized=np.float64(train_out["bandwidth_end_normalized"]),
            contrastive_soft_bandwidth_schedule=np.asarray(train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            contrastive_soft_bandwidth_k=np.int64(bw_k),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            gn_pretrain_train_losses=np.asarray(gn_train_out["train_losses"], dtype=np.float64),
            gn_pretrain_val_losses=np.asarray(gn_train_out["val_losses"], dtype=np.float64),
            gn_pretrain_val_monitor_losses=np.asarray(gn_train_out["val_monitor_losses"], dtype=np.float64),
            gn_pretrain_best_epoch=np.int64(gn_train_out["best_epoch"]),
            gn_pretrain_stopped_epoch=np.int64(gn_train_out["stopped_epoch"]),
            gn_pretrain_stopped_early=np.bool_(gn_train_out["stopped_early"]),
            gn_pretrain_best_val_smooth=np.float64(gn_train_out["best_val_loss"]),
            gn_pretrain_grad_norm_mean=np.float64(gn_train_out.get("grad_norm_mean", float("nan"))),
            gn_pretrain_grad_norm_max=np.float64(gn_train_out.get("grad_norm_max", float("nan"))),
            gn_pretrain_param_norm_final=np.float64(gn_train_out.get("param_norm_final", float("nan"))),
            contrastive_batch_size=np.int64(int(getattr(args, "contrastive_batch_size", 256))),
            contrastive_effective_batch_size=np.int64(train_out.get("effective_batch_size", 0)),
            contrastive_soft_bandwidth=np.float64(train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_auto=np.bool_(train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_schedule=np.asarray(train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded = SimpleNamespace(h_sym=np.asarray(h_sym, dtype=np.float64), theta_used=np.asarray(theta_used, dtype=np.float64))
        return loaded, np.asarray(x_all, dtype=np.float64), dev

    if contrastive_norm == "bidir_contrastive_soft":
        method_name = contrastive_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("bidir-contrastive-soft expects theta arrays to be 1D or 2D.")
        if int(theta_train.shape[1]) != 1 or int(theta_all.shape[1]) != 1:
            raise ValueError("bidir-contrastive-soft v1 requires scalar theta.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("bidir-contrastive-soft expects x arrays to be 2D.")
        if theta_train.shape[0] < 2 or theta_val.shape[0] < 2:
            raise ValueError("bidir-contrastive-soft requires at least two train and two validation rows.")

        hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
        depth = int(getattr(args, "contrastive_depth", 3))
        dot_dim = int(getattr(args, "contrastive_soft_dot_dim", 64))
        soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
        if soft_arch == "mlp":
            model = ContrastiveLLRMLP(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        else:
            model = ContrastiveNormalizedDotBiasScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                feature_dim=dot_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        bw_arg = float(getattr(args, "contrastive_soft_bandwidth", 1.0))
        bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
        bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
        bw_k = int(getattr(args, "contrastive_soft_bandwidth_k", 5))
        periodic = bool(getattr(args, "contrastive_soft_periodic", False))
        period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
        train_out = train_bidir_contrastive_soft_llr(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=int(getattr(args, "contrastive_epochs", 2000)),
            batch_size=int(getattr(args, "contrastive_batch_size", 256)),
            lr=float(getattr(args, "contrastive_lr", 1e-3)),
            bandwidth=bw_arg,
            bandwidth_start=bw_start,
            bandwidth_end=bw_end,
            bandwidth_k=bw_k,
            periodic=periodic,
            period=period,
            weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
            patience=int(getattr(args, "contrastive_early_patience", 300)),
            min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        theta_mean = np.asarray(train_out["theta_mean"], dtype=np.float64)
        theta_std = np.asarray(train_out["theta_std"], dtype=np.float64)
        if hasattr(model, "rho") and hasattr(model, "alpha"):
            contrastive_soft_logit_rho = float(model.rho.detach().cpu().item())
            contrastive_soft_logit_alpha = float(model.alpha.detach().cpu().item())
        else:
            contrastive_soft_logit_rho = float("nan")
            contrastive_soft_logit_alpha = float("nan")
        bidir_score_tag = "mlp" if soft_arch == "mlp" else "normalized_dot_bias"
        c_matrix = compute_contrastive_soft_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
            pair_batch_size=int(getattr(args, "contrastive_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_contrastive(delta_l))
        theta_used = theta_all.reshape(-1)

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray(["bidir_contrastive_soft_llr_score"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            contrastive_hidden_dim=np.int64(hidden_dim),
            contrastive_depth=np.int64(depth),
            contrastive_effective_batch_size=np.int64(train_out.get("effective_batch_size", 0)),
            contrastive_soft_score_arch=np.asarray([bidir_score_tag], dtype=object),
            contrastive_soft_dot_dim=np.int64(dot_dim),
            contrastive_soft_logit_rho=np.float64(contrastive_soft_logit_rho),
            contrastive_soft_logit_alpha=np.float64(contrastive_soft_logit_alpha),
            contrastive_soft_bandwidth=np.float64(train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_normalized=np.float64(train_out["bandwidth_normalized"]),
            contrastive_soft_bandwidth_auto=np.bool_(train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_start_normalized=np.float64(train_out["bandwidth_start_normalized"]),
            contrastive_soft_bandwidth_end_normalized=np.float64(train_out["bandwidth_end_normalized"]),
            contrastive_soft_bandwidth_schedule=np.asarray(train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            contrastive_soft_bandwidth_k=np.int64(bw_k),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_train_row_losses=np.asarray(train_out["train_row_losses"], dtype=np.float64),
            score_train_col_losses=np.asarray(train_out["train_col_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_row_losses=np.asarray(train_out["val_row_losses"], dtype=np.float64),
            score_val_col_losses=np.asarray(train_out["val_col_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(float("nan")),
            score_grad_norm_max=np.float64(float("nan")),
            score_param_norm_final=np.float64(float("nan")),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            contrastive_batch_size=np.int64(int(getattr(args, "contrastive_batch_size", 256))),
            contrastive_effective_batch_size=np.int64(train_out.get("effective_batch_size", 0)),
            contrastive_soft_score_arch=np.asarray([bidir_score_tag], dtype=object),
            contrastive_soft_dot_dim=np.int64(dot_dim),
            contrastive_soft_logit_rho=np.float64(contrastive_soft_logit_rho),
            contrastive_soft_logit_alpha=np.float64(contrastive_soft_logit_alpha),
            contrastive_soft_bandwidth=np.float64(train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_auto=np.bool_(train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_schedule=np.asarray(train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded_bidir = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_bidir, np.asarray(x_all, dtype=np.float64), dev

    if contrastive_norm == "contrastive_soft":
        method_name = contrastive_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("contrastive-soft expects theta arrays to be 1D or 2D.")
        if int(theta_train.shape[1]) != 1 or int(theta_all.shape[1]) != 1:
            raise ValueError("contrastive-soft v1 requires scalar theta.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("contrastive-soft expects x arrays to be 2D.")
        if theta_train.shape[0] < 2 or theta_val.shape[0] < 2:
            raise ValueError("contrastive-soft requires at least two train and two validation rows.")

        hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
        depth = int(getattr(args, "contrastive_depth", 3))
        soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
        dot_dim = int(getattr(args, "contrastive_soft_dot_dim", 64))
        coord_embed_dim = int(getattr(args, "contrastive_soft_coordinate_embed_dim", 16))
        gaussian_logvar_min = float(getattr(args, "contrastive_soft_gaussian_logvar_min", -8.0))
        gaussian_logvar_max = float(getattr(args, "contrastive_soft_gaussian_logvar_max", 5.0))
        if soft_arch == "normalized_dot":
            model = ContrastiveNormalizedDotScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                feature_dim=dot_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        elif soft_arch == "additive_independent":
            model = ContrastiveAdditiveIndependentScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                feature_dim=dot_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        elif soft_arch == "independent_gaussian":
            model = ContrastiveIndependentGaussianScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                hidden_dim=hidden_dim,
                depth=depth,
                logvar_min=gaussian_logvar_min,
                logvar_max=gaussian_logvar_max,
            ).to(dev)
        elif soft_arch == "independent_dot_product":
            model = ContrastiveIndependentDotProductScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                feature_dim=dot_dim,
                coord_embed_dim=coord_embed_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        elif soft_arch == "mlp":
            model = ContrastiveLLRMLP(
                x_dim=int(x_all.shape[1]),
                theta_dim=1,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        else:
            raise ValueError(
                "--contrastive-soft-score-arch must be one of "
                "{'normalized_dot','additive_independent','independent_gaussian','independent_dot_product','mlp'}."
            )
        bw_arg = float(getattr(args, "contrastive_soft_bandwidth", 1.0))
        bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
        bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
        bw_k = int(getattr(args, "contrastive_soft_bandwidth_k", 5))
        periodic = bool(getattr(args, "contrastive_soft_periodic", False))
        period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
        train_out = train_contrastive_soft_llr(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=int(getattr(args, "contrastive_epochs", 2000)),
            batch_size=int(getattr(args, "contrastive_batch_size", 256)),
            lr=float(getattr(args, "contrastive_lr", 1e-3)),
            bandwidth=bw_arg,
            bandwidth_start=bw_start,
            bandwidth_end=bw_end,
            bandwidth_k=bw_k,
            periodic=periodic,
            period=period,
            weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
            patience=int(getattr(args, "contrastive_early_patience", 300)),
            min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        theta_mean = np.asarray(train_out["theta_mean"], dtype=np.float64)
        theta_std = np.asarray(train_out["theta_std"], dtype=np.float64)
        if hasattr(model, "rho") and hasattr(model, "alpha"):
            contrastive_soft_logit_rho = float(model.rho.detach().cpu().item())
            contrastive_soft_logit_alpha = float(model.alpha.detach().cpu().item())
        else:
            contrastive_soft_logit_rho = float("nan")
            contrastive_soft_logit_alpha = float("nan")
        c_matrix = compute_contrastive_soft_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
            pair_batch_size=int(getattr(args, "contrastive_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_contrastive(delta_l))
        theta_used = theta_all.reshape(-1)

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray(["contrastive_soft_llr_score"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            contrastive_hidden_dim=np.int64(hidden_dim),
            contrastive_depth=np.int64(depth),
            contrastive_effective_batch_size=np.int64(train_out.get("effective_batch_size", 0)),
            contrastive_soft_score_arch=np.asarray([soft_arch], dtype=object),
            contrastive_soft_dot_dim=np.int64(dot_dim),
            contrastive_soft_coordinate_embed_dim=np.int64(coord_embed_dim),
            contrastive_soft_gaussian_logvar_min=np.float64(gaussian_logvar_min),
            contrastive_soft_gaussian_logvar_max=np.float64(gaussian_logvar_max),
            contrastive_soft_logit_rho=np.float64(contrastive_soft_logit_rho),
            contrastive_soft_logit_alpha=np.float64(contrastive_soft_logit_alpha),
            contrastive_soft_bandwidth=np.float64(train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_normalized=np.float64(train_out["bandwidth_normalized"]),
            contrastive_soft_bandwidth_auto=np.bool_(train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_start_normalized=np.float64(train_out["bandwidth_start_normalized"]),
            contrastive_soft_bandwidth_end_normalized=np.float64(train_out["bandwidth_end_normalized"]),
            contrastive_soft_bandwidth_schedule=np.asarray(train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            contrastive_soft_bandwidth_k=np.int64(bw_k),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(float("nan")),
            score_grad_norm_max=np.float64(float("nan")),
            score_param_norm_final=np.float64(float("nan")),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            contrastive_batch_size=np.int64(int(getattr(args, "contrastive_batch_size", 256))),
            contrastive_effective_batch_size=np.int64(train_out.get("effective_batch_size", 0)),
            contrastive_soft_score_arch=np.asarray([soft_arch], dtype=object),
            contrastive_soft_dot_dim=np.int64(dot_dim),
            contrastive_soft_coordinate_embed_dim=np.int64(coord_embed_dim),
            contrastive_soft_gaussian_logvar_min=np.float64(gaussian_logvar_min),
            contrastive_soft_gaussian_logvar_max=np.float64(gaussian_logvar_max),
            contrastive_soft_logit_rho=np.float64(contrastive_soft_logit_rho),
            contrastive_soft_logit_alpha=np.float64(contrastive_soft_logit_alpha),
            contrastive_soft_bandwidth=np.float64(train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_auto=np.bool_(train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_schedule=np.asarray(train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded_contrastive_soft = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_contrastive_soft, np.asarray(x_all, dtype=np.float64), dev

    if contrastive_norm is not None:
        method_name = contrastive_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("contrastive expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("contrastive expects x arrays to be 2D.")
        if theta_train.shape[0] < 2 or theta_val.shape[0] < 2:
            raise ValueError("contrastive requires at least two train and two validation rows.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError("contrastive theta dimension mismatch.")
        if bin_train is None or bin_validation is None or bin_all is None:
            raise ValueError("contrastive requires theta bin labels from the convergence subset.")

        hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
        depth = int(getattr(args, "contrastive_depth", 3))
        theta_encoding = normalize_contrastive_theta_encoding(
            str(getattr(args, "contrastive_theta_encoding", "one_hot_bin"))
        )
        model = ContrastiveLLRMLP(
            x_dim=int(x_all.shape[1]),
            theta_dim=contrastive_theta_dim_for_encoding(int(n_bins), theta_encoding),
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(dev)
        train_out = train_contrastive_llr(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            bin_train=np.asarray(bin_train, dtype=np.int64),
            bin_val=np.asarray(bin_validation, dtype=np.int64),
            n_bins=int(n_bins),
            theta_encoding=theta_encoding,
            device=dev,
            epochs=int(getattr(args, "contrastive_epochs", 2000)),
            batch_size=int(getattr(args, "contrastive_batch_size", 256)),
            lr=float(getattr(args, "contrastive_lr", 1e-3)),
            weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
            patience=int(getattr(args, "contrastive_early_patience", 300)),
            min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        theta_mean = np.asarray(train_out["theta_mean"], dtype=np.float64)
        theta_std = np.asarray(train_out["theta_std"], dtype=np.float64)
        c_matrix = compute_contrastive_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            bin_all=np.asarray(bin_all, dtype=np.int64),
            n_bins=int(n_bins),
            theta_encoding=theta_encoding,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            pair_batch_size=int(getattr(args, "contrastive_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_contrastive(delta_l))
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray(["contrastive_llr_score"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            contrastive_hidden_dim=np.int64(hidden_dim),
            contrastive_depth=np.int64(depth),
            contrastive_num_theta_bins=np.int64(int(n_bins)),
            contrastive_theta_encoding=np.asarray([theta_encoding], dtype=object),
            contrastive_unique_bin_batches=np.bool_(True),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(float("nan")),
            score_grad_norm_max=np.float64(float("nan")),
            score_param_norm_final=np.float64(float("nan")),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            contrastive_batch_size=np.int64(int(getattr(args, "contrastive_batch_size", 256))),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded_contrastive = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_contrastive, np.asarray(x_all, dtype=np.float64), dev

    gxf_norm = _normalize_gaussian_x_flow_method(tfm)
    if gxf_norm is not None:
        method_name = gxf_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError(f"{method_name} expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError(f"{method_name} expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError(f"{method_name} requires non-empty train and validation splits.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError(f"{method_name} theta dimension mismatch.")

        sched_name = str(getattr(args, "gxf_path_schedule", "linear")).strip().lower()
        schedule = path_schedule_from_name(sched_name)
        gxf_diag_cov = method_name == "gaussian_x_flow_diagonal"
        if gxf_diag_cov:
            model = ConditionalDiagonalGaussianCovarianceFMMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=int(x_all.shape[1]),
                hidden_dim=int(getattr(args, "gxf_hidden_dim", 128)),
                depth=int(getattr(args, "gxf_depth", 3)),
                diag_floor=float(getattr(args, "gxf_diag_floor", 1e-4)),
            ).to(dev)
        else:
            model = ConditionalGaussianCovarianceFMMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=int(x_all.shape[1]),
                hidden_dim=int(getattr(args, "gxf_hidden_dim", 128)),
                depth=int(getattr(args, "gxf_depth", 3)),
                diag_floor=float(getattr(args, "gxf_diag_floor", 1e-4)),
            ).to(dev)
        h_eval_name = (
            "gaussian_x_flow_diagonal_log_p_x_given_theta"
            if gxf_diag_cov
            else "gaussian_x_flow_log_p_x_given_theta"
        )
        train_out = train_gaussian_x_flow(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            schedule=schedule,
            epochs=int(getattr(args, "gxf_epochs", 2000)),
            batch_size=int(getattr(args, "gxf_batch_size", 256)),
            lr=float(getattr(args, "gxf_lr", 1e-3)),
            weight_decay=float(getattr(args, "gxf_weight_decay", 0.0)),
            t_eps=float(getattr(args, "gxf_t_eps", 1e-3)),
            cov_jitter=float(getattr(args, "gxf_cov_jitter", 1e-4)),
            patience=int(getattr(args, "gxf_early_patience", 300)),
            min_delta=float(getattr(args, "gxf_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "gxf_early_ema_alpha", 0.05)),
            weight_ema_decay=float(getattr(args, "gxf_weight_ema_decay", 0.9)),
            max_grad_norm=float(getattr(args, "gxf_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        c_matrix = compute_gaussian_x_flow_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            pair_batch_size=int(getattr(args, "gxf_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray([h_eval_name], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            gxf_path_schedule=np.asarray([sched_name], dtype=object),
            gxf_t_eps=np.float64(float(getattr(args, "gxf_t_eps", 1e-3))),
            gxf_cov_jitter=np.float64(float(getattr(args, "gxf_cov_jitter", 1e-4))),
            gxf_hidden_dim=np.int64(int(getattr(args, "gxf_hidden_dim", 128))),
            gxf_depth=np.int64(int(getattr(args, "gxf_depth", 3))),
            gxf_diag_floor=np.float64(float(getattr(args, "gxf_diag_floor", 1e-4))),
            gxf_diagonal_covariance=np.bool_(gxf_diag_cov),
            gxf_weight_ema_decay=np.float64(train_out.get("weight_ema_decay", float(getattr(args, "gxf_weight_ema_decay", 0.9)))),
            gxf_weight_ema_enabled=np.bool_(train_out.get("weight_ema_enabled", False)),
            gxf_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            gxf_x_mean=np.asarray(x_mean, dtype=np.float64),
            gxf_x_std=np.asarray(x_std, dtype=np.float64),
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(float("nan")),
            score_grad_norm_max=np.float64(float("nan")),
            score_param_norm_final=np.float64(float("nan")),
            score_n_clipped_steps=np.int64(0),
            score_n_total_steps=np.int64(0),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            score_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            ae_train_losses=empty,
            ae_val_losses=empty,
            ae_val_monitor_losses=empty,
            ae_best_epoch=np.int64(0),
            ae_stopped_epoch=np.int64(0),
            ae_stopped_early=np.bool_(False),
            ae_latent_dim=np.int64(0),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
            gxf_path_schedule=np.asarray([sched_name], dtype=object),
            gxf_fm_train=np.bool_(True),
            gxf_diagonal_covariance=np.bool_(gxf_diag_cov),
            gxf_weight_ema_decay=np.float64(train_out.get("weight_ema_decay", float(getattr(args, "gxf_weight_ema_decay", 0.9)))),
            gxf_weight_ema_enabled=np.bool_(train_out.get("weight_ema_enabled", False)),
            gxf_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
        )
        loaded_gxf = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_gxf, np.asarray(x_all, dtype=np.float64), dev

    lxf_norm = _normalize_linear_x_flow_method(tfm)
    if lxf_norm is not None:
        method_name = lxf_norm
        scheduled_lxf = method_name in ("linear_x_flow_schedule", "linear_x_flow_diagonal_t")
        lxf_prefix = "lxfs" if scheduled_lxf else "lxf"
        sched_name = str(getattr(args, "lxfs_path_schedule", "cosine")).strip().lower() if scheduled_lxf else ""
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError(f"{method_name} expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError(f"{method_name} expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError(f"{method_name} requires non-empty train and validation splits.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError(f"{method_name} theta dimension mismatch.")

        binned_lxf_cov_meta: dict[str, Any] | None = None
        if method_name == "bin_gaussian_linear_x_flow_diagonal":
            if bin_train is None:
                raise ValueError("bin_gaussian_linear_x_flow_diagonal requires bin_train from the sweep subset.")
            binned_lxf_cov_meta = estimate_binned_gaussian_shared_diagonal_covariance(
                x_train=x_train,
                bin_train=np.asarray(bin_train, dtype=np.int64),
                n_bins=n_bins,
                variance_floor=float(
                    getattr(args, "flow_theta_reg_variance_floor", getattr(args, "flow_x_reg_variance_floor", 1e-6))
                ),
            )

        x_dim_lxf = int(x_all.shape[1])
        lxf_rank = int(getattr(args, "lxf_low_rank_dim", 4))
        if method_name == "linear_x_flow_scalar":
            drift_type = "scalar"
            model = ConditionalScalarLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
            ).to(dev)
        elif method_name in ("linear_x_flow_diagonal", "bin_gaussian_linear_x_flow_diagonal"):
            drift_type = "diagonal"
            hidden_dim_lxf = int(getattr(args, f"{lxf_prefix}_hidden_dim", 128))
            depth_lxf = int(getattr(args, f"{lxf_prefix}_depth", 3))
            b_net_kind = str(getattr(args, "lxf_b_net", "mlp")).strip().lower()
            if b_net_kind == "film":
                model = ConditionalDiagonalLinearXFlowFiLMLP(
                    theta_dim=int(theta_all.shape[1]),
                    x_dim=x_dim_lxf,
                    hidden_dim=hidden_dim_lxf,
                    depth=depth_lxf,
                ).to(dev)
            else:
                model = ConditionalDiagonalLinearXFlowMLP(
                    theta_dim=int(theta_all.shape[1]),
                    x_dim=x_dim_lxf,
                    hidden_dim=hidden_dim_lxf,
                    depth=depth_lxf,
                ).to(dev)
            if binned_lxf_cov_meta is not None:
                with torch.no_grad():
                    model.a.copy_(torch.as_tensor(binned_lxf_cov_meta["a"], dtype=model.a.dtype, device=dev))
                model.a.requires_grad_(False)
        elif method_name == "linear_x_flow_diagonal_theta":
            drift_type = "diagonal_theta"
            model = ConditionalThetaDiagonalLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
            ).to(dev)
        elif method_name == "linear_x_flow_diagonal_theta_spline":
            if int(theta_all.shape[1]) != 1:
                raise ValueError(
                    f"{method_name} requires scalar theta (theta.shape[1]==1); got theta_dim={int(theta_all.shape[1])}."
                )
            drift_type = "diagonal_theta_spline"
            tcol = np.asarray(theta_train[:, 0], dtype=np.float64).reshape(-1)
            theta_min = float(np.min(tcol))
            theta_max = float(np.max(tcol))
            spline_k = int(getattr(args, "lxf_spline_k", 5))
            model = ConditionalThetaDiagonalSplineLinearXFlowMLP(
                theta_dim=1,
                x_dim=x_dim_lxf,
                theta_min=theta_min,
                theta_max=theta_max,
                num_basis=spline_k,
                spline_degree=3,
            ).to(dev)
        elif method_name == "linear_x_flow_diagonal_t":
            drift_type = "diagonal_time"
            model = ConditionalTimeDiagonalLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
                quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)),
            ).to(dev)
        elif method_name == "linear_x_flow_low_rank":
            if lxf_rank > x_dim_lxf:
                raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "low_rank"
            model = ConditionalLowRankLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                rank=lxf_rank,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
            ).to(dev)
        elif method_name == "linear_x_flow_low_rank_randb":
            if lxf_rank > x_dim_lxf:
                raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "low_rank_randb"
            model = ConditionalRandomBasisLowRankLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                rank=lxf_rank,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
                lambda_a=float(getattr(args, "lxf_randb_lambda_a", 1e-4)),
                lambda_s=float(getattr(args, "lxf_randb_lambda_s", 1e-4)),
            ).to(dev)
        elif method_name == "linear_x_flow_nonlinear_pca":
            drift_type = "nonlinear_pca_full_symmetric"
            model = ConditionalLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
            ).to(dev)
        else:
            drift_type = "full_symmetric"
            model = ConditionalLinearXFlowMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=x_dim_lxf,
                hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)),
                depth=int(getattr(args, f"{lxf_prefix}_depth", 3)),
            ).to(dev)
        train_kwargs = dict(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=int(getattr(args, f"{lxf_prefix}_epochs", 2000)),
            batch_size=int(getattr(args, f"{lxf_prefix}_batch_size", 1024)),
            lr=float(getattr(args, f"{lxf_prefix}_lr", 1e-3)),
            weight_decay=float(getattr(args, f"{lxf_prefix}_weight_decay", 0.0)),
            t_eps=float(getattr(args, f"{lxf_prefix}_t_eps", 0.05)),
            patience=int(getattr(args, f"{lxf_prefix}_early_patience", 1000)),
            min_delta=float(getattr(args, f"{lxf_prefix}_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, f"{lxf_prefix}_early_ema_alpha", 0.05)),
            weight_ema_decay=float(getattr(args, f"{lxf_prefix}_weight_ema_decay", 0.9)),
            max_grad_norm=float(getattr(args, f"{lxf_prefix}_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=bool(getattr(args, "lxf_restore_best", True)),
        )
        if method_name == "linear_x_flow_diagonal_t":
            train_out = train_time_diagonal_linear_x_flow_schedule(
                **train_kwargs,
                schedule=path_schedule_from_name(sched_name),
            )
        elif scheduled_lxf:
            train_out = train_linear_x_flow_schedule(
                **train_kwargs,
                schedule=path_schedule_from_name(sched_name),
            )
        else:
            train_out = train_linear_x_flow(**train_kwargs)
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        nonlinear_train_out: dict[str, Any] | None = None
        pca_basis = np.asarray([], dtype=np.float32)
        analytic_lxf_h = method_name != "linear_x_flow_nonlinear_pca"
        save_lxf_c_matrix = bool(getattr(args, "lxf_save_c_matrix", False))
        c_matrix: np.ndarray | None
        delta_l: np.ndarray | None
        endpoint_mu = np.asarray([], dtype=np.float64)
        endpoint_cov_or_diag = np.asarray([], dtype=np.float64)
        endpoint_is_diag = False
        if method_name == "linear_x_flow_nonlinear_pca":
            x_train_norm = (x_train - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
            pca_basis = fit_residual_pca_basis_from_linear_mean(
                linear_model=cast(ConditionalLinearXFlowMLP, model),
                theta_train=theta_train,
                x_train_norm=x_train_norm,
                pca_dim=int(getattr(args, "lxf_nlpca_dim", 8)),
                device=dev,
                solve_jitter=float(getattr(args, "lxf_solve_jitter", 1e-6)),
            )
            model = ConditionalPCANonlinearLinearXFlowMLP(
                linear_model=cast(ConditionalLinearXFlowMLP, model),
                pca_basis=pca_basis,
                hidden_dim=int(getattr(args, "lxf_nlpca_hidden_dim", 128)),
                depth=int(getattr(args, "lxf_nlpca_depth", 3)),
            ).to(dev)
            nlpca_epochs = int(getattr(args, "lxf_nlpca_epochs", 0))
            if nlpca_epochs <= 0:
                nlpca_epochs = int(getattr(args, "lxf_epochs", 2000))
            nlpca_lr = float(getattr(args, "lxf_nlpca_lr", 0.0))
            if nlpca_lr <= 0.0:
                nlpca_lr = float(getattr(args, "lxf_lr", 1e-3))
            nonlinear_train_out = train_pca_nonlinear_linear_x_flow(
                model=cast(ConditionalPCANonlinearLinearXFlowMLP, model),
                theta_train=theta_train,
                x_train=x_train,
                theta_val=theta_val,
                x_val=x_val,
                device=dev,
                x_mean=x_mean,
                x_std=x_std,
                epochs=nlpca_epochs,
                batch_size=int(getattr(args, "lxf_batch_size", 1024)),
                lr=nlpca_lr,
                weight_decay=float(getattr(args, "lxf_weight_decay", 0.0)),
                t_eps=float(getattr(args, "lxf_t_eps", 0.05)),
                lambda_h=float(getattr(args, "lxf_nlpca_lambda_h", 0.0)),
                freeze_linear=bool(getattr(args, "lxf_nlpca_freeze_linear", False)),
                patience=int(getattr(args, "lxf_early_patience", 1000)),
                min_delta=float(getattr(args, "lxf_early_min_delta", 1e-4)),
                ema_alpha=float(getattr(args, "lxf_early_ema_alpha", 0.05)),
                weight_ema_decay=float(getattr(args, "lxf_weight_ema_decay", 0.9)),
                max_grad_norm=float(getattr(args, "lxf_max_grad_norm", 10.0)),
                solve_jitter=float(getattr(args, "lxf_solve_jitter", 1e-6)),
                log_every=max(1, int(getattr(args, "log_every", 50))),
                restore_best=True,
            )
            c_matrix = compute_pca_nonlinear_linear_x_flow_c_matrix(
                model=cast(ConditionalPCANonlinearLinearXFlowMLP, model),
                theta_all=theta_all,
                x_all=x_all,
                device=dev,
                x_mean=x_mean,
                x_std=x_std,
                solve_jitter=float(getattr(args, "lxf_solve_jitter", 1e-6)),
                ode_steps=int(getattr(args, "lxf_nlpca_ode_steps", 32)),
                pair_batch_size=int(getattr(args, "lxf_pair_batch_size", 65536)),
            )
            delta_l = compute_delta_l_nf(c_matrix)
            h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        elif method_name == "linear_x_flow_diagonal_t":
            h_sym, endpoint_mu, endpoint_cov_or_diag, endpoint_is_diag = compute_linear_x_flow_analytic_hellinger_matrix(
                model=model,
                theta_all=theta_all,
                device=dev,
                solve_jitter=float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6)),
                quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)),
            )
            c_matrix = None
            delta_l = None
            if save_lxf_c_matrix:
                c_matrix = compute_time_diagonal_linear_x_flow_c_matrix(
                    model=cast(ConditionalTimeDiagonalLinearXFlowMLP, model),
                    theta_all=theta_all,
                    x_all=x_all,
                    device=dev,
                    x_mean=x_mean,
                    x_std=x_std,
                    solve_jitter=float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6)),
                    quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)),
                    pair_batch_size=int(getattr(args, f"{lxf_prefix}_pair_batch_size", 65536)),
                )
                delta_l = compute_delta_l_nf(c_matrix)
        else:
            h_sym, endpoint_mu, endpoint_cov_or_diag, endpoint_is_diag = compute_linear_x_flow_analytic_hellinger_matrix(
                model=model,
                theta_all=theta_all,
                device=dev,
                solve_jitter=float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6)),
            )
            c_matrix = None
            delta_l = None
            if save_lxf_c_matrix:
                c_matrix = compute_linear_x_flow_c_matrix(
                    model=model,
                    theta_all=theta_all,
                    x_all=x_all,
                    device=dev,
                    x_mean=x_mean,
                    x_std=x_std,
                    solve_jitter=float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6)),
                    pair_batch_size=int(getattr(args, f"{lxf_prefix}_pair_batch_size", 65536)),
                )
                delta_l = compute_delta_l_nf(c_matrix)
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        lxf_b_net_saved = (
            str(getattr(args, "lxf_b_net", "mlp")).strip().lower()
            if method_name in ("linear_x_flow_diagonal", "bin_gaussian_linear_x_flow_diagonal")
            else "mlp"
        )
        binned_lxf_artifacts: dict[str, Any] = {}
        if binned_lxf_cov_meta is not None:
            binned_lxf_artifacts = {
                "lxf_a_fixed": np.bool_(True),
                "lxf_a_source": np.asarray(["binned_gaussian_shared_diagonal_covariance"], dtype=object),
                "lxf_a": np.asarray(binned_lxf_cov_meta["a"], dtype=np.float64),
                "lxf_shared_variance": np.asarray(binned_lxf_cov_meta["shared_variance"], dtype=np.float64),
                "lxf_bin_counts": np.asarray(binned_lxf_cov_meta["bin_counts"], dtype=np.int64),
                "lxf_normalized_bin_means": np.asarray(binned_lxf_cov_meta["normalized_bin_means"], dtype=np.float64),
                "lxf_variance_floor": np.float64(float(binned_lxf_cov_meta["variance_floor"])),
            }

        h_payload: dict[str, Any] = dict(
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray(
                [
                    f"{method_name}_log_p_x_given_theta"
                    if not analytic_lxf_h
                    else f"{method_name}_analytic_gaussian_hellinger"
                ],
                dtype=object,
            ),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            lxf_analytic_gaussian_hellinger=np.bool_(analytic_lxf_h),
            lxf_save_c_matrix=np.bool_(save_lxf_c_matrix),
            lxf_endpoint_mu=np.asarray(endpoint_mu, dtype=np.float64),
            lxf_endpoint_covariance_or_variance_diag=np.asarray(endpoint_cov_or_diag, dtype=np.float64),
            lxf_endpoint_is_diagonal=np.bool_(endpoint_is_diag),
            lxf_fm_train=np.bool_(True),
            lxf_t_eps=np.float64(float(getattr(args, f"{lxf_prefix}_t_eps", 0.05))),
            lxf_solve_jitter=np.float64(float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6))),
            lxf_hidden_dim=np.int64(int(getattr(args, f"{lxf_prefix}_hidden_dim", 128))),
            lxf_depth=np.int64(int(getattr(args, f"{lxf_prefix}_depth", 3))),
            lxf_b_net=np.asarray([lxf_b_net_saved], dtype=object),
            lxf_drift_type=np.asarray([drift_type], dtype=object),
            lxf_low_rank_dim=np.int64(lxf_rank if method_name in ("linear_x_flow_low_rank", "linear_x_flow_low_rank_randb") else 0),
            lxf_randb_lambda_a=np.float64(float(getattr(args, "lxf_randb_lambda_a", 1e-4)) if method_name == "linear_x_flow_low_rank_randb" else 0.0),
            lxf_randb_lambda_s=np.float64(float(getattr(args, "lxf_randb_lambda_s", 1e-4)) if method_name == "linear_x_flow_low_rank_randb" else 0.0),
            lxf_spline_k=np.int64(int(getattr(args, "lxf_spline_k", 5)) if method_name == "linear_x_flow_diagonal_theta_spline" else 0),
            lxf_nlpca_dim=np.int64(int(getattr(args, "lxf_nlpca_dim", 8)) if method_name == "linear_x_flow_nonlinear_pca" else 0),
            lxf_nlpca_lambda_h=np.float64(float(getattr(args, "lxf_nlpca_lambda_h", 0.0)) if method_name == "linear_x_flow_nonlinear_pca" else 0.0),
            lxf_nlpca_freeze_linear=np.bool_(bool(getattr(args, "lxf_nlpca_freeze_linear", False)) if method_name == "linear_x_flow_nonlinear_pca" else False),
            lxf_nlpca_ode_steps=np.int64(int(getattr(args, "lxf_nlpca_ode_steps", 32)) if method_name == "linear_x_flow_nonlinear_pca" else 0),
            lxf_nlpca_pca_basis=np.asarray(pca_basis, dtype=np.float32),
            lxf_weight_ema_decay=np.float64(train_out.get("weight_ema_decay", float(getattr(args, f"{lxf_prefix}_weight_ema_decay", 0.9)))),
            lxf_weight_ema_enabled=np.bool_(train_out.get("weight_ema_enabled", False)),
            lxf_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            lxf_restore_best=np.bool_(bool(getattr(args, "lxf_restore_best", True))),
            lxfs_path_schedule=np.asarray([sched_name], dtype=object),
            lxfs_scheduled_train=np.bool_(scheduled_lxf),
            lxfs_quadrature_steps=np.int64(int(getattr(args, "lxfs_quadrature_steps", 64)) if method_name == "linear_x_flow_diagonal_t" else 0),
            lxf_x_mean=np.asarray(x_mean, dtype=np.float64),
            lxf_x_std=np.asarray(x_std, dtype=np.float64),
            **binned_lxf_artifacts,
        )
        if c_matrix is not None:
            h_payload["c_matrix"] = np.asarray(c_matrix, dtype=np.float64)
        if delta_l is not None:
            h_payload["delta_l_matrix"] = np.asarray(delta_l, dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            **h_payload,
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(float("nan")),
            score_grad_norm_max=np.float64(float("nan")),
            score_param_norm_final=np.float64(float("nan")),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            score_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            ae_train_losses=empty,
            ae_val_losses=empty,
            ae_val_monitor_losses=empty,
            ae_best_epoch=np.int64(0),
            ae_stopped_epoch=np.int64(0),
            ae_stopped_early=np.bool_(False),
            ae_latent_dim=np.int64(0),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
            lxf_fm_train=np.bool_(True),
            lxf_b_net=np.asarray([lxf_b_net_saved], dtype=object),
            lxf_drift_type=np.asarray([drift_type], dtype=object),
            lxf_low_rank_dim=np.int64(lxf_rank if method_name in ("linear_x_flow_low_rank", "linear_x_flow_low_rank_randb") else 0),
            lxf_randb_lambda_a=np.float64(float(getattr(args, "lxf_randb_lambda_a", 1e-4)) if method_name == "linear_x_flow_low_rank_randb" else 0.0),
            lxf_randb_lambda_s=np.float64(float(getattr(args, "lxf_randb_lambda_s", 1e-4)) if method_name == "linear_x_flow_low_rank_randb" else 0.0),
            lxf_spline_k=np.int64(int(getattr(args, "lxf_spline_k", 5)) if method_name == "linear_x_flow_diagonal_theta_spline" else 0),
            lxf_nlpca_train_losses=np.asarray(nonlinear_train_out["train_losses"], dtype=np.float64) if nonlinear_train_out is not None else empty,
            lxf_nlpca_val_losses=np.asarray(nonlinear_train_out["val_losses"], dtype=np.float64) if nonlinear_train_out is not None else empty,
            lxf_nlpca_val_monitor_losses=np.asarray(nonlinear_train_out["val_monitor_losses"], dtype=np.float64) if nonlinear_train_out is not None else empty,
            lxf_nlpca_best_epoch=np.int64(nonlinear_train_out["best_epoch"] if nonlinear_train_out is not None else 0),
            lxf_nlpca_stopped_epoch=np.int64(nonlinear_train_out["stopped_epoch"] if nonlinear_train_out is not None else 0),
            lxf_nlpca_stopped_early=np.bool_(nonlinear_train_out["stopped_early"] if nonlinear_train_out is not None else False),
            lxf_nlpca_best_val_smooth=np.float64(nonlinear_train_out["best_val_loss"] if nonlinear_train_out is not None else float("nan")),
            lxf_nlpca_dim=np.int64(int(getattr(args, "lxf_nlpca_dim", 8)) if method_name == "linear_x_flow_nonlinear_pca" else 0),
            lxf_nlpca_lambda_h=np.float64(float(getattr(args, "lxf_nlpca_lambda_h", 0.0)) if method_name == "linear_x_flow_nonlinear_pca" else 0.0),
            lxf_nlpca_freeze_linear=np.bool_(bool(getattr(args, "lxf_nlpca_freeze_linear", False)) if method_name == "linear_x_flow_nonlinear_pca" else False),
            lxf_nlpca_ode_steps=np.int64(int(getattr(args, "lxf_nlpca_ode_steps", 32)) if method_name == "linear_x_flow_nonlinear_pca" else 0),
            lxf_weight_ema_decay=np.float64(train_out.get("weight_ema_decay", float(getattr(args, f"{lxf_prefix}_weight_ema_decay", 0.9)))),
            lxf_weight_ema_enabled=np.bool_(train_out.get("weight_ema_enabled", False)),
            lxf_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            lxf_restore_best=np.bool_(bool(getattr(args, "lxf_restore_best", True))),
            lxfs_path_schedule=np.asarray([sched_name], dtype=object),
            lxfs_scheduled_train=np.bool_(scheduled_lxf),
            lxfs_quadrature_steps=np.int64(int(getattr(args, "lxfs_quadrature_steps", 64)) if method_name == "linear_x_flow_diagonal_t" else 0),
            **binned_lxf_artifacts,
        )
        loaded_lxf = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_lxf, np.asarray(x_all, dtype=np.float64), dev

    ltf_norm = _normalize_linear_theta_flow_method(tfm)
    if ltf_norm is not None:
        method_name = ltf_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError(f"{method_name} expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError(f"{method_name} expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError(f"{method_name} requires non-empty train and validation splits.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError(f"{method_name} theta dimension mismatch.")

        model = ConditionalLinearThetaFlowMixtureMLP(
            theta_dim=int(theta_all.shape[1]),
            x_dim=int(x_all.shape[1]),
            num_components=int(getattr(args, "ltf_num_components", 3)),
            hidden_dim=int(getattr(args, "ltf_hidden_dim", 128)),
            depth=int(getattr(args, "ltf_depth", 3)),
        ).to(dev)
        train_out = train_linear_theta_flow(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=int(getattr(args, "ltf_epochs", 2000)),
            batch_size=int(getattr(args, "ltf_batch_size", 1024)),
            lr=float(getattr(args, "ltf_lr", 1e-3)),
            weight_decay=float(getattr(args, "ltf_weight_decay", 0.0)),
            t_eps=float(getattr(args, "ltf_t_eps", 0.05)),
            patience=int(getattr(args, "ltf_early_patience", 300)),
            min_delta=float(getattr(args, "ltf_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "ltf_early_ema_alpha", 0.05)),
            weight_ema_decay=float(getattr(args, "ltf_weight_ema_decay", 0.9)),
            max_grad_norm=float(getattr(args, "ltf_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        theta_mean = np.asarray(train_out["theta_mean"], dtype=np.float64)
        theta_std = np.asarray(train_out["theta_std"], dtype=np.float64)
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        c_matrix = compute_linear_theta_flow_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            theta_mean=theta_mean,
            theta_std=theta_std,
            x_mean=x_mean,
            x_std=x_std,
            solve_jitter=float(getattr(args, "ltf_solve_jitter", 1e-6)),
            pair_batch_size=int(getattr(args, "ltf_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            theta_flow_log_post_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray([f"{method_name}_log_p_theta_given_x"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            ltf_num_components=np.int64(int(getattr(args, "ltf_num_components", 3))),
            ltf_t_eps=np.float64(float(getattr(args, "ltf_t_eps", 0.05))),
            ltf_solve_jitter=np.float64(float(getattr(args, "ltf_solve_jitter", 1e-6))),
            ltf_hidden_dim=np.int64(int(getattr(args, "ltf_hidden_dim", 128))),
            ltf_depth=np.int64(int(getattr(args, "ltf_depth", 3))),
            ltf_weight_ema_decay=np.float64(train_out.get("weight_ema_decay", float(getattr(args, "ltf_weight_ema_decay", 0.9)))),
            ltf_weight_ema_enabled=np.bool_(train_out.get("weight_ema_enabled", False)),
            ltf_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            ltf_theta_mean=np.asarray(theta_mean, dtype=np.float64),
            ltf_theta_std=np.asarray(theta_std, dtype=np.float64),
            ltf_x_mean=np.asarray(x_mean, dtype=np.float64),
            ltf_x_std=np.asarray(x_std, dtype=np.float64),
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([method_name], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(float("nan")),
            score_grad_norm_max=np.float64(float("nan")),
            score_param_norm_final=np.float64(float("nan")),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            score_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
            ae_train_losses=empty,
            ae_val_losses=empty,
            ae_val_monitor_losses=empty,
            ae_best_epoch=np.int64(0),
            ae_stopped_epoch=np.int64(0),
            ae_stopped_early=np.bool_(False),
            ae_latent_dim=np.int64(0),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
            ltf_fm_train=np.bool_(True),
            ltf_num_components=np.int64(int(getattr(args, "ltf_num_components", 3))),
            ltf_weight_ema_decay=np.float64(train_out.get("weight_ema_decay", float(getattr(args, "ltf_weight_ema_decay", 0.9)))),
            ltf_weight_ema_enabled=np.bool_(train_out.get("weight_ema_enabled", False)),
            ltf_final_eval_weights=np.asarray([str(train_out.get("final_eval_weights", "raw"))], dtype=object),
        )
        loaded_ltf = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_ltf, np.asarray(x_all, dtype=np.float64), dev

    gn_norm = _normalize_gaussian_network_method(tfm)
    if gn_norm is not None:
        gn_method = gn_norm
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError(f"{gn_method} expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError(f"{gn_method} expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError(f"{gn_method} requires non-empty train and validation splits.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError(f"{gn_method} theta dimension mismatch.")

        ae_train_out: dict[str, Any] | None = None
        ae_latent_dim = 0
        gn_x_train = x_train
        gn_x_val = x_val
        gn_x_all = x_all
        h_eval_scalar_name = f"{gn_method}_log_p_x_given_theta"
        pca_meta: dict[str, np.ndarray | int] | None = None
        gn_pca_num_bins = 0
        gn_pca_dim = 0
        if gn_method in ("gaussian_network_autoencoder", "gaussian_network_diagonal_autoencoder"):
            default_latent_dim = min(8, int(x_all.shape[1]))
            ae_latent_dim = int(getattr(args, "gn_ae_latent_dim", default_latent_dim) or default_latent_dim)
            if ae_latent_dim > int(x_all.shape[1]):
                raise ValueError(f"--gn-ae-latent-dim must be <= x_dim={int(x_all.shape[1])}; got {ae_latent_dim}.")
            ae_model = ObservationAutoencoder(
                x_dim=int(x_all.shape[1]),
                latent_dim=ae_latent_dim,
                hidden_dim=int(getattr(args, "gn_ae_hidden_dim", 128)),
                depth=int(getattr(args, "gn_ae_depth", 2)),
            ).to(dev)
            ae_train_out = train_observation_autoencoder(
                model=ae_model,
                x_train=x_train,
                x_val=x_val,
                device=dev,
                epochs=int(getattr(args, "gn_ae_epochs", 1000)),
                batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
                lr=float(getattr(args, "gn_ae_lr", 1e-3)),
                weight_decay=float(getattr(args, "gn_ae_weight_decay", 0.0)),
                patience=int(getattr(args, "gn_ae_early_patience", 200)),
                min_delta=float(getattr(args, "gn_ae_early_min_delta", 1e-4)),
                ema_alpha=float(getattr(args, "gn_ae_early_ema_alpha", 0.05)),
                log_every=max(1, int(getattr(args, "log_every", 50))),
                restore_best=True,
            )
            gn_x_train = encode_observations(
                model=ae_model,
                x=x_train,
                device=dev,
                batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
            )
            gn_x_val = encode_observations(
                model=ae_model,
                x=x_val,
                device=dev,
                batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
            )
            gn_x_all = encode_observations(
                model=ae_model,
                x=x_all,
                device=dev,
                batch_size=int(getattr(args, "gn_ae_batch_size", 256)),
            )
            h_eval_scalar_name = f"{gn_method}_log_p_z_given_theta"
        elif gn_method == "gaussian_network_diagonal_binned_pca":
            if bin_train is None:
                raise ValueError("gaussian-network-diagonal-binned-pca requires bin_train from the convergence sweep.")
            if int(theta_train.shape[1]) != 1:
                raise ValueError("gaussian-network-diagonal-binned-pca v1 requires scalar theta.")
            gn_pca_num_bins = int(getattr(args, "gn_pca_num_bins", None) or n_bins)
            gn_pca_dim = int(getattr(args, "gn_pca_dim", 2))
            if gn_pca_num_bins == int(n_bins):
                pca_bin_train = np.asarray(bin_train, dtype=np.int64)
            else:
                pca_edges, _, _ = vhb.theta_bin_edges(theta_train.reshape(-1), gn_pca_num_bins)
                pca_bin_train = vhb.theta_to_bin_index(theta_train.reshape(-1), pca_edges, gn_pca_num_bins)
            gn_x_train, gn_x_val, gn_x_all, pca_meta = _fit_binned_mean_pca_projection(
                x_train=x_train,
                theta_train=theta_train,
                bin_train=pca_bin_train,
                x_val=x_val,
                x_all=x_all,
                n_bins=gn_pca_num_bins,
                pca_dim=gn_pca_dim,
            )
            h_eval_scalar_name = f"{gn_method}_log_p_z_given_theta"

        if gn_method == "gaussian_network_low_rank":
            rank = int(getattr(args, "gn_low_rank_dim", 4))
            if rank > int(gn_x_all.shape[1]):
                raise ValueError(f"--gn-low-rank-dim must be <= x_dim={int(gn_x_all.shape[1])}; got {rank}.")
            model = ConditionalLowRankGaussianCovarianceMLP(
                theta_dim=int(theta_all.shape[1]),
                x_dim=int(gn_x_all.shape[1]),
                rank=rank,
                hidden_dim=int(getattr(args, "gn_hidden_dim", 128)),
                depth=int(getattr(args, "gn_depth", 3)),
                diag_floor=float(getattr(args, "gn_diag_floor", 1e-4)),
                psi_floor=float(getattr(args, "gn_psi_floor", 1e-6)),
            ).to(dev)
        else:
            model_cls = (
                ConditionalDiagonalGaussianPrecisionMLP
                if gn_method in (
                    "gaussian_network_diagonal",
                    "gaussian_network_diagonal_autoencoder",
                    "gaussian_network_diagonal_binned_pca",
                )
                else ConditionalGaussianPrecisionMLP
            )
            model = model_cls(
                theta_dim=int(theta_all.shape[1]),
                x_dim=int(gn_x_all.shape[1]),
                hidden_dim=int(getattr(args, "gn_hidden_dim", 128)),
                depth=int(getattr(args, "gn_depth", 3)),
                diag_floor=float(getattr(args, "gn_diag_floor", 1e-4)),
            ).to(dev)
        train_out = train_gaussian_network(
            model=model,
            theta_train=theta_train,
            x_train=gn_x_train,
            theta_val=theta_val,
            x_val=gn_x_val,
            device=dev,
            epochs=int(getattr(args, "gn_epochs", 4000)),
            batch_size=int(getattr(args, "gn_batch_size", 256)),
            lr=float(getattr(args, "gn_lr", 1e-3)),
            weight_decay=float(getattr(args, "gn_weight_decay", 0.0)),
            patience=int(getattr(args, "gn_early_patience", 300)),
            min_delta=float(getattr(args, "gn_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "gn_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "gn_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        c_matrix = compute_gaussian_network_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=gn_x_all,
            device=dev,
            pair_batch_size=int(getattr(args, "gn_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray([gn_method], dtype=object),
            h_eval_scalar_name=np.asarray([h_eval_scalar_name], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            gn_hidden_dim=np.int64(getattr(args, "gn_hidden_dim", 128)),
            gn_depth=np.int64(getattr(args, "gn_depth", 3)),
            gn_diag_floor=np.float64(getattr(args, "gn_diag_floor", 1e-4)),
            gn_low_rank_dim=np.int64(getattr(args, "gn_low_rank_dim", 0)),
            gn_psi_floor=np.float64(getattr(args, "gn_psi_floor", np.nan)),
            gn_autoencoder_enabled=np.bool_(ae_train_out is not None),
            gn_ae_latent_dim=np.int64(ae_latent_dim),
            gn_ae_reconstruction_val_loss=np.float64(
                ae_train_out["best_val_loss"] if ae_train_out is not None else np.nan
            ),
            gn_binned_pca_enabled=np.bool_(pca_meta is not None),
            gn_pca_dim=np.int64(gn_pca_dim),
            gn_pca_num_bins=np.int64(gn_pca_num_bins),
            gn_pca_mean=np.asarray(
                pca_meta["pca_mean"] if pca_meta is not None else [],
                dtype=np.float64,
            ),
            gn_pca_components=np.asarray(
                pca_meta["pca_components"] if pca_meta is not None else np.zeros((0, 0)),
                dtype=np.float64,
            ),
            gn_pca_singular_values=np.asarray(
                pca_meta["pca_singular_values"] if pca_meta is not None else [],
                dtype=np.float64,
            ),
            gn_pca_bin_counts=np.asarray(
                pca_meta["pca_bin_counts"] if pca_meta is not None else [],
                dtype=np.int64,
            ),
            gn_pca_theta_bin_centers=np.asarray(
                pca_meta["pca_theta_bin_centers"] if pca_meta is not None else [],
                dtype=np.float64,
            ),
            gn_pca_binned_train_means=np.asarray(
                pca_meta["pca_binned_train_means"] if pca_meta is not None else np.zeros((0, 0)),
                dtype=np.float64,
            ),
            gn_pca_nonempty_bins=np.asarray(
                pca_meta["pca_nonempty_bins"] if pca_meta is not None else [],
                dtype=np.int64,
            ),
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray([gn_method], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_loss"]),
            score_grad_norm_mean=np.float64(train_out.get("grad_norm_mean", float("nan"))),
            score_grad_norm_max=np.float64(train_out.get("grad_norm_max", float("nan"))),
            score_param_norm_final=np.float64(train_out.get("param_norm_final", float("nan"))),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            ae_train_losses=np.asarray(
                ae_train_out["train_losses"] if ae_train_out is not None else [],
                dtype=np.float64,
            ),
            ae_val_losses=np.asarray(
                ae_train_out["val_losses"] if ae_train_out is not None else [],
                dtype=np.float64,
            ),
            ae_val_monitor_losses=np.asarray(
                ae_train_out["val_monitor_losses"] if ae_train_out is not None else [],
                dtype=np.float64,
            ),
            ae_best_epoch=np.int64(ae_train_out["best_epoch"] if ae_train_out is not None else 0),
            ae_stopped_epoch=np.int64(ae_train_out["stopped_epoch"] if ae_train_out is not None else 0),
            ae_stopped_early=np.bool_(ae_train_out["stopped_early"] if ae_train_out is not None else False),
            ae_latent_dim=np.int64(ae_latent_dim),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded_gn = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_gn, np.asarray(x_all, dtype=np.float64), dev

    if tfm == "gmm_z_decode":
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("gmm-z-decode expects theta arrays to be 1D or 2D.")
        if int(theta_all.shape[1]) != 1:
            raise ValueError("gmm-z-decode v1 requires scalar theta.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("gmm-z-decode expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError("gmm-z-decode requires non-empty train and validation splits.")

        latent_dim = int(getattr(args, "gzd_latent_dim", 2))
        components = int(getattr(args, "gzd_components", 5))
        hidden_dim = int(getattr(args, "gzd_hidden_dim", 128))
        depth = int(getattr(args, "gzd_depth", 2))
        model = GMMZDecodeModel(
            x_dim=int(x_all.shape[1]),
            latent_dim=latent_dim,
            components=components,
            hidden_dim=hidden_dim,
            depth=depth,
            min_std=float(getattr(args, "gzd_min_std", 1e-3)),
        ).to(dev)
        train_out = train_gmm_z_decode(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=int(getattr(args, "gzd_epochs", 2000)),
            batch_size=int(getattr(args, "gzd_batch_size", 256)),
            lr=float(getattr(args, "gzd_lr", 1e-3)),
            weight_decay=float(getattr(args, "gzd_weight_decay", 0.0)),
            patience=int(getattr(args, "gzd_early_patience", 300)),
            min_delta=float(getattr(args, "gzd_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "gzd_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "gzd_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        c_matrix, z_all = compute_gmm_z_decode_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=np.asarray(train_out["x_mean"], dtype=np.float64),
            x_std=np.asarray(train_out["x_std"], dtype=np.float64),
            theta_mean=np.asarray(train_out["theta_mean"], dtype=np.float64),
            theta_std=np.asarray(train_out["theta_std"], dtype=np.float64),
            pair_batch_size=int(getattr(args, "gzd_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        theta_used = theta_all.reshape(-1)

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray(["gmm_z_decode"], dtype=object),
            h_eval_scalar_name=np.asarray(["gmm_z_decode_log_q_theta_given_z_uniform_prior"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray(["gmm_z_decode"], dtype=object),
            gzd_latent_dim=np.int64(latent_dim),
            gzd_components=np.int64(components),
            gzd_hidden_dim=np.int64(hidden_dim),
            gzd_depth=np.int64(depth),
            gzd_x_mean=np.asarray(train_out["x_mean"], dtype=np.float64),
            gzd_x_std=np.asarray(train_out["x_std"], dtype=np.float64),
            gzd_theta_mean=np.asarray(train_out["theta_mean"], dtype=np.float64),
            gzd_theta_std=np.asarray(train_out["theta_std"], dtype=np.float64),
            gzd_z_all=np.asarray(z_all, dtype=np.float64),
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray(["gmm_z_decode"], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_ema_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_ema"]),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            score_n_clipped_steps=np.int64(train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=np.int64(train_out.get("n_total_steps", 0)),
            gzd_latent_dim=np.int64(latent_dim),
            gzd_components=np.int64(components),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded, np.asarray(x_all, dtype=np.float64), dev

    if tfm == "pi_nf":
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("pi-nf expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("pi-nf expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError("pi-nf requires non-empty train and validation splits.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError("pi-nf theta dimension mismatch.")
        latent_dim = int(getattr(args, "pinf_latent_dim", 2))
        if latent_dim >= int(x_all.shape[1]):
            raise ValueError(f"--pinf-latent-dim must be < x_dim={int(x_all.shape[1])}; got {latent_dim}.")

        pinf_hidden_dim = int(getattr(args, "pinf_hidden_dim", 128))
        pinf_transforms = int(getattr(args, "pinf_transforms", 5))
        model = PiNFModel(
            theta_dim=int(theta_all.shape[1]),
            x_dim=int(x_all.shape[1]),
            latent_dim=latent_dim,
            hidden_dim=pinf_hidden_dim,
            transforms=pinf_transforms,
            min_std=float(getattr(args, "pinf_min_std", 1e-3)),
        ).to(dev)
        train_out = train_pi_nf(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=int(getattr(args, "pinf_epochs", 2000)),
            batch_size=int(getattr(args, "pinf_batch_size", 256)),
            lr=float(getattr(args, "pinf_lr", 1e-3)),
            weight_decay=float(getattr(args, "pinf_weight_decay", 0.0)),
            recon_weight=float(getattr(args, "pinf_recon_weight", 1.0)),
            patience=int(getattr(args, "pinf_early_patience", 300)),
            min_delta=float(getattr(args, "pinf_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "pinf_early_ema_alpha", 0.05)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        theta_mean = np.asarray(train_out["theta_mean"], dtype=np.float64)
        theta_std = np.asarray(train_out["theta_std"], dtype=np.float64)
        c_matrix, z_all, r_all = compute_pi_nf_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
            pair_batch_size=int(getattr(args, "pinf_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()
        diag = pi_nf_diagnostics(z_all=z_all, r_all=r_all, theta_all=theta_all)

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray(["pi_nf"], dtype=object),
            h_eval_scalar_name=np.asarray(["pi_nf_log_p_z_given_theta"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray(["pi_nf"], dtype=object),
            pinf_latent_dim=np.int64(latent_dim),
            pinf_residual_dim=np.int64(int(x_all.shape[1]) - latent_dim),
            pinf_hidden_dim=np.int64(pinf_hidden_dim),
            pinf_transforms=np.int64(pinf_transforms),
            pinf_recon_weight=np.float64(float(getattr(args, "pinf_recon_weight", 1.0))),
            pinf_x_mean=np.asarray(x_mean, dtype=np.float64),
            pinf_x_std=np.asarray(x_std, dtype=np.float64),
            pinf_theta_mean=np.asarray(theta_mean, dtype=np.float64),
            pinf_theta_std=np.asarray(theta_std, dtype=np.float64),
            pinf_z_all=np.asarray(z_all, dtype=np.float64),
            pinf_r_all=np.asarray(r_all, dtype=np.float64),
            pinf_z_to_theta_r2=np.float64(diag["pinf_z_to_theta_r2"]),
            pinf_r_to_theta_r2=np.float64(diag["pinf_r_to_theta_r2"]),
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray(["pi_nf"], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_ema_losses"], dtype=np.float64),
            score_nll_train_losses=np.asarray(train_out["train_nll_losses"], dtype=np.float64),
            score_recon_train_losses=np.asarray(train_out["train_recon_losses"], dtype=np.float64),
            score_total_train_losses=np.asarray(train_out["train_total_losses"], dtype=np.float64),
            score_nll_val_losses=np.asarray(train_out["val_nll_losses"], dtype=np.float64),
            score_recon_val_losses=np.asarray(train_out["val_recon_losses"], dtype=np.float64),
            score_total_val_losses=np.asarray(train_out["val_total_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_ema"]),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            pinf_latent_dim=np.int64(latent_dim),
            pinf_residual_dim=np.int64(int(x_all.shape[1]) - latent_dim),
            pinf_recon_weight=np.float64(float(getattr(args, "pinf_recon_weight", 1.0))),
            pinf_z_to_theta_r2=np.float64(diag["pinf_z_to_theta_r2"]),
            pinf_r_to_theta_r2=np.float64(diag["pinf_r_to_theta_r2"]),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded_pinf = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_pinf, np.asarray(x_all, dtype=np.float64), dev

    if tfm == "nf_reduction":
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if theta_train.ndim != 2 or theta_val.ndim != 2 or theta_all.ndim != 2:
            raise ValueError("nf-reduction expects theta arrays to be 1D or 2D.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("nf-reduction expects x arrays to be 2D.")
        if theta_train.shape[0] < 1 or theta_val.shape[0] < 1:
            raise ValueError("nf-reduction requires non-empty train and validation splits.")
        if theta_train.shape[1] != theta_all.shape[1]:
            raise ValueError("nf-reduction theta dimension mismatch.")
        latent_dim = int(getattr(args, "nfr_latent_dim", 2))
        if latent_dim >= int(x_all.shape[1]):
            raise ValueError(f"--nfr-latent-dim must be < x_dim={int(x_all.shape[1])}; got {latent_dim}.")

        nfr_epochs = int(getattr(args, "nfr_epochs", 2000))
        nfr_batch_size = int(getattr(args, "nfr_batch_size", 256))
        nfr_lr = float(getattr(args, "nfr_lr", 1e-3))
        nfr_hidden_dim = int(getattr(args, "nfr_hidden_dim", 128))
        nfr_context_dim = int(getattr(args, "nfr_context_dim", 32))
        nfr_transforms = int(getattr(args, "nfr_transforms", 5))
        nfr_early_patience = int(getattr(args, "nfr_early_patience", 300))
        nfr_early_min_delta = float(getattr(args, "nfr_early_min_delta", 1e-4))
        nfr_early_ema_alpha = float(getattr(args, "nfr_early_ema_alpha", 0.05))

        model = NFReductionModel(
            theta_dim=int(theta_all.shape[1]),
            x_dim=int(x_all.shape[1]),
            latent_dim=latent_dim,
            hidden_dim=nfr_hidden_dim,
            transforms=nfr_transforms,
            context_dim=nfr_context_dim,
        ).to(dev)
        train_out = train_nf_reduction(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=nfr_epochs,
            batch_size=nfr_batch_size,
            lr=nfr_lr,
            patience=nfr_early_patience,
            min_delta=nfr_early_min_delta,
            ema_alpha=nfr_early_ema_alpha,
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
        )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        c_matrix, z_all = compute_nf_reduction_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            pair_batch_size=int(getattr(args, "nfr_pair_batch_size", 65536)),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray(["nf_reduction"], dtype=object),
            h_eval_scalar_name=np.asarray(["nf_reduction_log_p_z_given_theta"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray(["nf_reduction"], dtype=object),
            nfr_latent_dim=np.int64(latent_dim),
            nfr_residual_dim=np.int64(int(x_all.shape[1]) - latent_dim),
            nfr_hidden_dim=np.int64(nfr_hidden_dim),
            nfr_context_dim=np.int64(nfr_context_dim),
            nfr_transforms=np.int64(nfr_transforms),
            nfr_x_mean=np.asarray(x_mean, dtype=np.float64),
            nfr_x_std=np.asarray(x_std, dtype=np.float64),
            nfr_z_all=np.asarray(z_all, dtype=np.float64),
        )
        empty = np.asarray([], dtype=np.float64)
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray(["nf_reduction"], dtype=object),
            prior_enable=np.bool_(False),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_ema_losses"], dtype=np.float64),
            score_best_epoch=np.int64(train_out["best_epoch"]),
            score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
            score_stopped_early=np.bool_(train_out["stopped_early"]),
            score_best_val_smooth=np.float64(train_out["best_val_ema"]),
            score_lr_last=np.float64(train_out.get("lr_last", float("nan"))),
            nfr_latent_dim=np.int64(latent_dim),
            nfr_residual_dim=np.int64(int(x_all.shape[1]) - latent_dim),
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
        )
        loaded_nfr = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_used, dtype=np.float64),
        )
        return loaded_nfr, np.asarray(x_all, dtype=np.float64), dev

    if tfm == "nf":
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64).reshape(-1)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if x_train.ndim != 2 or x_all.ndim != 2:
            raise ValueError("NF method expects x arrays to be 2D.")
        if theta_train.size < 1 or theta_val.size < 1:
            raise ValueError("NF method requires non-empty train and validation splits.")
        nf_epochs = int(getattr(args, "nf_epochs", 2000))
        nf_batch_size = int(getattr(args, "nf_batch_size", 256))
        nf_lr = float(getattr(args, "nf_lr", 1e-3))
        nf_hidden_dim = int(getattr(args, "nf_hidden_dim", 128))
        nf_context_dim = int(getattr(args, "nf_context_dim", 32))
        nf_transforms = int(getattr(args, "nf_transforms", 5))
        nf_early_patience = int(getattr(args, "nf_early_patience", 300))
        nf_early_min_delta = float(getattr(args, "nf_early_min_delta", 1e-4))
        nf_early_ema_alpha = float(getattr(args, "nf_early_ema_alpha", 0.05))
        nf_prior_epochs = int(getattr(args, "nf_prior_epochs", nf_epochs) or nf_epochs)
        nf_prior_batch_size = int(getattr(args, "nf_prior_batch_size", nf_batch_size) or nf_batch_size)
        nf_prior_lr = float(getattr(args, "nf_prior_lr", nf_lr) or nf_lr)
        nf_prior_hidden_dim = int(getattr(args, "nf_prior_hidden_dim", nf_hidden_dim) or nf_hidden_dim)
        nf_prior_transforms = int(getattr(args, "nf_prior_transforms", nf_transforms) or nf_transforms)
        nf_prior_early_patience = int(getattr(args, "nf_prior_early_patience", nf_early_patience) or nf_early_patience)
        nf_prior_early_min_delta = float(
            getattr(args, "nf_prior_early_min_delta", nf_early_min_delta) or nf_early_min_delta
        )
        nf_prior_early_ema_alpha = float(
            getattr(args, "nf_prior_early_ema_alpha", nf_early_ema_alpha) or nf_early_ema_alpha
        )

        model = ConditionalThetaNF(
            x_dim=int(x_all.shape[1]),
            context_dim=nf_context_dim,
            hidden_dim=nf_hidden_dim,
            transforms=nf_transforms,
        ).to(dev)
        train_out = train_conditional_nf(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=dev,
            epochs=nf_epochs,
            batch_size=nf_batch_size,
            lr=nf_lr,
            patience=nf_early_patience,
            min_delta=nf_early_min_delta,
            ema_alpha=nf_early_ema_alpha,
        )
        prior_model = PriorThetaNF(
            hidden_dim=nf_prior_hidden_dim,
            transforms=nf_prior_transforms,
        ).to(dev)
        prior_out = train_prior_nf(
            model=prior_model,
            theta_train=theta_train,
            theta_val=theta_val,
            device=dev,
            epochs=nf_prior_epochs,
            batch_size=nf_prior_batch_size,
            lr=nf_prior_lr,
            patience=nf_prior_early_patience,
            min_delta=nf_prior_early_min_delta,
            ema_alpha=nf_prior_early_ema_alpha,
        )
        c_matrix = compute_c_matrix_nf(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            pair_batch_size=int(getattr(args, "nf_pair_batch_size", 65536)),
        )
        log_p_theta_prior = compute_log_p_theta_prior_nf(
            model=prior_model,
            theta_all=theta_all,
            device=dev,
        )
        r_matrix = compute_ratio_matrix_posterior_minus_prior(
            c_matrix_post=c_matrix,
            log_p_theta_prior=log_p_theta_prior,
        )
        delta_l = compute_delta_l_nf(r_matrix)
        h_sym = symmetrize_nf(compute_h_directed_nf(delta_l))

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_all, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            c_matrix_ratio=np.asarray(r_matrix, dtype=np.float64),
            log_p_theta_prior=np.asarray(log_p_theta_prior, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            h_field_method=np.asarray(["nf"], dtype=object),
            h_eval_scalar_name=np.asarray(["nf_log_ratio_post_minus_prior"], dtype=object),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
        )
        np.savez_compressed(
            os.path.join(output_dir, "score_prior_training_losses.npz"),
            theta_field_method=np.asarray(["nf"], dtype=object),
            prior_enable=np.bool_(True),
            score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
            score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out["val_ema_losses"], dtype=np.float64),
            prior_train_losses=np.asarray(prior_out["train_losses"], dtype=np.float64),
            prior_val_losses=np.asarray(prior_out["val_losses"], dtype=np.float64),
            prior_val_monitor_losses=np.asarray(prior_out["val_ema_losses"], dtype=np.float64),
        )
        loaded_nf = SimpleNamespace(
            h_sym=np.asarray(h_sym, dtype=np.float64),
            theta_used=np.asarray(theta_all, dtype=np.float64),
        )
        return loaded_nf, np.asarray(x_all, dtype=np.float64), dev

    flow_ae_norm = _normalize_flow_autoencoder_method(tfm)
    if flow_ae_norm is not None:
        base_method = _base_flow_method_for_autoencoder(flow_ae_norm)
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        encoded_bundle, ae_train_out, ae_latent_dim = _train_autoencoder_and_encode_bundle(
            args=args,
            bundle=bundle,
            device=dev,
        )
        if flow_ae_norm == "theta_flow_autoencoder" and ae_latent_dim < 2:
            raise ValueError("theta-flow-autoencoder requires --gn-ae-latent-dim >= 2 for theta-flow conditioning.")
        d = vars(args).copy()
        d.setdefault("h_matrix_npz", None)
        d.setdefault("h_only", False)
        args2 = argparse.Namespace(**d)
        args2.theta_field_method = base_method
        args2.output_dir = output_dir
        full_args = _make_full_args(args2, meta)
        setattr(full_args, "theta_field_method", base_method)
        setattr(full_args, "x_dim", int(ae_latent_dim))
        ctx = _run_ctx_for_bundle(args2, meta, encoded_bundle, full_args, n_bins)
        vhb.run_h_estimation_if_needed(ctx)

        h_path = os.path.join(output_dir, _h_matrix_results_npz_basename(dataset_family=str(meta.get("dataset_family", ""))))
        if not os.path.exists(h_path):
            h_path = os.path.join(output_dir, "h_matrix_results_theta_cov.npz")
        if base_method == "theta_flow":
            eval_name = "theta_flow_autoencoder_log_ratio_theta_given_z"
        else:
            eval_name = "x_flow_autoencoder_log_p_z_given_theta"
        _rewrite_npz_fields(
            h_path,
            h_field_method=np.asarray([flow_ae_norm], dtype=object),
            h_eval_scalar_name=np.asarray([eval_name], dtype=object),
            autoencoder_enabled=np.bool_(True),
            ae_latent_dim=np.int64(ae_latent_dim),
            ae_reconstruction_val_loss=np.float64(ae_train_out["best_val_loss"]),
        )
        loss_path = os.path.join(output_dir, "score_prior_training_losses.npz")
        _rewrite_npz_fields(
            loss_path,
            theta_field_method=np.asarray([flow_ae_norm], dtype=object),
            ae_train_losses=np.asarray(ae_train_out["train_losses"], dtype=np.float64),
            ae_val_losses=np.asarray(ae_train_out["val_losses"], dtype=np.float64),
            ae_val_monitor_losses=np.asarray(ae_train_out["val_monitor_losses"], dtype=np.float64),
            ae_best_epoch=np.int64(ae_train_out["best_epoch"]),
            ae_stopped_epoch=np.int64(ae_train_out["stopped_epoch"]),
            ae_stopped_early=np.bool_(ae_train_out["stopped_early"]),
            ae_latent_dim=np.int64(ae_latent_dim),
        )
        loaded = vhb.load_h_matrix(ctx)
        theta_chk = vhb.theta_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
        _validate_theta_used_matches_bundle(
            theta_chk,
            loaded.theta_used,
            err_suffix=f"h_field_method={flow_ae_norm!r}",
        )
        return loaded, np.asarray(bundle.x_all, dtype=np.float64), dev

    flow_pca_norm = _normalize_flow_pca_method(tfm)
    if flow_pca_norm is not None:
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
        theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
        theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
        x_train = np.asarray(bundle.x_train, dtype=np.float64)
        x_val = np.asarray(bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(bundle.x_all, dtype=np.float64)
        if theta_train.ndim == 1:
            theta_train = theta_train.reshape(-1, 1)
        if theta_val.ndim == 1:
            theta_val = theta_val.reshape(-1, 1)
        if theta_all.ndim == 1:
            theta_all = theta_all.reshape(-1, 1)
        if int(theta_train.shape[1]) != 1 or int(theta_all.shape[1]) != 1:
            raise ValueError("x-flow-pca v1 requires scalar theta.")
        if bin_train is None:
            raise ValueError("x-flow-pca requires bin_train from the convergence sweep.")
        flow_pca_num_bins = int(getattr(args, "flow_pca_num_bins", None) or n_bins)
        flow_pca_dim = int(getattr(args, "flow_pca_dim", 2))
        if flow_pca_num_bins == int(n_bins):
            pca_bin_train = np.asarray(bin_train, dtype=np.int64)
        else:
            pca_edges, _, _ = vhb.theta_bin_edges(theta_train.reshape(-1), flow_pca_num_bins)
            pca_bin_train = vhb.theta_to_bin_index(theta_train.reshape(-1), pca_edges, flow_pca_num_bins)
        z_train, z_val, z_all, pca_meta = _fit_binned_mean_pca_projection(
            x_train=x_train,
            theta_train=theta_train,
            bin_train=pca_bin_train,
            x_val=x_val,
            x_all=x_all,
            n_bins=flow_pca_num_bins,
            pca_dim=flow_pca_dim,
        )
        encoded_bundle = SharedDatasetBundle(
            meta=bundle.meta,
            theta_all=theta_all,
            x_all=z_all,
            train_idx=bundle.train_idx,
            validation_idx=bundle.validation_idx,
            theta_train=theta_train,
            x_train=z_train,
            theta_validation=theta_val,
            x_validation=z_val,
        )
        d = vars(args).copy()
        d.setdefault("h_matrix_npz", None)
        d.setdefault("h_only", False)
        args2 = argparse.Namespace(**d)
        args2.theta_field_method = "x_flow"
        args2.output_dir = output_dir
        full_args = _make_full_args(args2, meta)
        setattr(full_args, "theta_field_method", "x_flow")
        setattr(full_args, "x_dim", int(flow_pca_dim))
        ctx = _run_ctx_for_bundle(args2, meta, encoded_bundle, full_args, n_bins)
        vhb.run_h_estimation_if_needed(ctx)

        h_path = os.path.join(output_dir, _h_matrix_results_npz_basename(dataset_family=str(meta.get("dataset_family", ""))))
        if not os.path.exists(h_path):
            h_path = os.path.join(output_dir, "h_matrix_results_theta_cov.npz")
        _rewrite_npz_fields(
            h_path,
            h_field_method=np.asarray([flow_pca_norm], dtype=object),
            h_eval_scalar_name=np.asarray(["x_flow_pca_log_p_z_given_theta"], dtype=object),
            pca_enabled=np.bool_(True),
            flow_pca_dim=np.int64(flow_pca_dim),
            flow_pca_num_bins=np.int64(flow_pca_num_bins),
            flow_pca_mean=np.asarray(pca_meta["pca_mean"], dtype=np.float64),
            flow_pca_components=np.asarray(pca_meta["pca_components"], dtype=np.float64),
            flow_pca_singular_values=np.asarray(pca_meta["pca_singular_values"], dtype=np.float64),
            flow_pca_bin_counts=np.asarray(pca_meta["pca_bin_counts"], dtype=np.int64),
            flow_pca_theta_bin_centers=np.asarray(pca_meta["pca_theta_bin_centers"], dtype=np.float64),
            flow_pca_binned_train_means=np.asarray(pca_meta["pca_binned_train_means"], dtype=np.float64),
            flow_pca_nonempty_bins=np.asarray(pca_meta["pca_nonempty_bins"], dtype=np.int64),
        )
        loss_path = os.path.join(output_dir, "score_prior_training_losses.npz")
        _rewrite_npz_fields(
            loss_path,
            theta_field_method=np.asarray([flow_pca_norm], dtype=object),
            pca_enabled=np.bool_(True),
            flow_pca_dim=np.int64(flow_pca_dim),
            flow_pca_num_bins=np.int64(flow_pca_num_bins),
        )
        loaded = vhb.load_h_matrix(ctx)
        theta_chk = vhb.theta_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
        _validate_theta_used_matches_bundle(
            theta_chk,
            loaded.theta_used,
            err_suffix="x-flow-pca",
        )
        return loaded, np.asarray(bundle.x_all, dtype=np.float64), dev

    d = vars(args).copy()
    d.setdefault("h_matrix_npz", None)
    d.setdefault("h_only", False)
    args2 = argparse.Namespace(**d)
    args2.output_dir = output_dir
    full_args = _make_full_args(args2, meta)
    ctx = _run_ctx_for_bundle(args2, meta, bundle, full_args, n_bins)
    vhb.run_h_estimation_if_needed(ctx)
    loaded = vhb.load_h_matrix(ctx)
    theta_chk = vhb.theta_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    _validate_theta_used_matches_bundle(
        theta_chk,
        loaded.theta_used,
        err_suffix=f"h_field_method={str(getattr(args, 'theta_field_method', ''))!r}",
    )
    x_aligned = vhb.x_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    if x_aligned.shape[0] != theta_chk.shape[0]:
        raise ValueError(
            f"x/H row mismatch: x_aligned={x_aligned.shape[0]} theta_used_rows={theta_chk.shape[0]}"
        )
    return loaded, x_aligned, ctx.device


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
    if method == "gmm_z_decode":
        return c, r"GMM-z-decode log $q(\theta|z)$"
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
    if method == "gaussian_x_flow":
        return c, r"Gaussian X-flow log $p(x|\theta)$"
    if method == "gaussian_x_flow_diagonal":
        return c, r"Gaussian X-flow (diagonal cov.) log $p(x|\theta)$"
    if method == "linear_x_flow":
        return c, r"Linear X-flow log $p(x|\theta)$"
    if method == "linear_x_flow_scalar":
        return c, r"Linear X-flow scalar $A$ log $p(x|\theta)$"
    if method == "linear_x_flow_diagonal":
        return c, r"Linear X-flow diagonal $A$ log $p(x|\theta)$"
    if method == "linear_x_flow_diagonal_theta":
        return c, r"Linear X-flow diagonal $A(\theta)$ log $p(x|\theta)$"
    if method == "linear_x_flow_diagonal_theta_spline":
        return c, r"Linear X-flow diagonal $A(\theta)$ spline log $p(x|\theta)$"
    if method == "linear_x_flow_low_rank":
        return c, r"Linear X-flow low-rank $A$ log $p(x|\theta)$"
    if method == "linear_x_flow_low_rank_randb":
        return c, r"Linear X-flow random-basis low-rank $A$ log $p(x|\theta)$"
    if method == "linear_x_flow_schedule":
        return c, r"Linear X-flow schedule log $p(x|\theta)$"
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
        elif tfm == "ctsm_v":
            post_lab = "pair-conditioned CTSM-v"
        elif tfm == "nf":
            post_lab = "normalizing-flow posterior"
        elif tfm == "nf_reduction":
            post_lab = "NF-reduction z likelihood"
        elif tfm == "gmm_z_decode":
            post_lab = "GMM-z-decode posterior NLL"
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
        elif tfm == "linear_x_flow_scalar":
            post_lab = "linear-x-flow scalar FM likelihood"
        elif tfm == "linear_x_flow_diagonal":
            post_lab = "linear-x-flow diagonal FM likelihood"
        elif tfm == "linear_x_flow_diagonal_theta":
            post_lab = "linear-x-flow diagonal-theta FM likelihood"
        elif tfm == "linear_x_flow_diagonal_theta_spline":
            post_lab = "linear-x-flow diagonal-theta-spline FM likelihood"
        elif tfm == "linear_x_flow_diagonal_t":
            post_lab = "linear-x-flow diagonal-t FM likelihood"
        elif tfm == "linear_x_flow_low_rank":
            post_lab = "linear-x-flow low-rank FM likelihood"
        elif tfm == "linear_x_flow_low_rank_randb":
            post_lab = "linear-x-flow random-basis low-rank FM likelihood"
        elif tfm == "linear_x_flow_schedule":
            post_lab = "linear-x-flow schedule FM likelihood"
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
        elif tfm == "ctsm_v":
            post_lab = "pair-conditioned CTSM-v"
        elif tfm == "nf":
            post_lab = "normalizing-flow posterior"
        elif tfm == "nf_reduction":
            post_lab = "NF-reduction z likelihood"
        elif tfm == "gmm_z_decode":
            post_lab = "GMM-z-decode posterior NLL"
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
        elif tfm == "linear_x_flow_scalar":
            post_lab = "linear-x-flow scalar FM likelihood"
        elif tfm == "linear_x_flow_diagonal":
            post_lab = "linear-x-flow diagonal FM likelihood"
        elif tfm == "linear_x_flow_diagonal_theta":
            post_lab = "linear-x-flow diagonal-theta FM likelihood"
        elif tfm == "linear_x_flow_diagonal_theta_spline":
            post_lab = "linear-x-flow diagonal-theta-spline FM likelihood"
        elif tfm == "linear_x_flow_diagonal_t":
            post_lab = "linear-x-flow diagonal-t FM likelihood"
        elif tfm == "linear_x_flow_low_rank":
            post_lab = "linear-x-flow low-rank FM likelihood"
        elif tfm == "linear_x_flow_low_rank_randb":
            post_lab = "linear-x-flow random-basis low-rank FM likelihood"
        elif tfm == "linear_x_flow_schedule":
            post_lab = "linear-x-flow schedule FM likelihood"
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
            "linear_x_flow",
            "linear_x_flow_scalar",
            "linear_x_flow_diagonal",
            "linear_x_flow_diagonal_theta",
            "linear_x_flow_diagonal_theta_spline",
            "linear_x_flow_diagonal_t",
            "linear_x_flow_low_rank",
            "linear_x_flow_low_rank_randb",
            "linear_x_flow_nonlinear_pca",
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
            f.write(f"lxf_spline_k: {int(getattr(args, 'lxf_spline_k', 0))}\n")
            f.write(f"lxf_nlpca_dim: {int(getattr(args, 'lxf_nlpca_dim', 0))}\n")
            f.write(f"lxf_nlpca_epochs: {int(getattr(args, 'lxf_nlpca_epochs', 0))}\n")
            f.write(f"lxf_nlpca_lr: {float(getattr(args, 'lxf_nlpca_lr', 0.0))}\n")
            f.write(f"lxf_nlpca_hidden_dim: {int(getattr(args, 'lxf_nlpca_hidden_dim', 0))}\n")
            f.write(f"lxf_nlpca_depth: {int(getattr(args, 'lxf_nlpca_depth', 0))}\n")
            f.write(f"lxf_nlpca_lambda_h: {float(getattr(args, 'lxf_nlpca_lambda_h', 0.0))}\n")
            f.write(f"lxf_nlpca_freeze_linear: {bool(getattr(args, 'lxf_nlpca_freeze_linear', False))}\n")
            f.write(f"lxf_nlpca_ode_steps: {int(getattr(args, 'lxf_nlpca_ode_steps', 0))}\n")
        if _tfm_sum == "linear_x_flow_schedule":
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
        if _tfm_sum == "linear_x_flow_diagonal_t":
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
        if _tfm_sum == "gmm_z_decode":
            f.write(f"gzd_latent_dim: {int(getattr(args, 'gzd_latent_dim', 0))}\n")
            f.write(f"gzd_components: {int(getattr(args, 'gzd_components', 0))}\n")
            f.write(f"gzd_epochs: {int(getattr(args, 'gzd_epochs', 0))}\n")
            f.write(f"gzd_batch_size: {int(getattr(args, 'gzd_batch_size', 0))}\n")
            f.write(f"gzd_lr: {float(getattr(args, 'gzd_lr', 0.0))}\n")
            f.write(f"gzd_hidden_dim: {int(getattr(args, 'gzd_hidden_dim', 0))}\n")
            f.write(f"gzd_depth: {int(getattr(args, 'gzd_depth', 0))}\n")
            f.write(f"gzd_early_patience: {int(getattr(args, 'gzd_early_patience', 0))}\n")
            f.write(f"gzd_pair_batch_size: {int(getattr(args, 'gzd_pair_batch_size', 0))}\n")
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
            "# corr_h_binned_vs_gt_mc: Pearson r, off-diagonal binned sqrt(H_sym) vs sqrt(generative GT H^2) (MC one-sided Hellinger).\n"
        )
        sf.write(
            "# corr_clf_vs_ref: Pearson r, off-diagonal pairwise decoding vs n_ref subset decoding matrix (same bin edges).\n"
        )
        sf.write(
            "# h_binned_columns last column: MC GT sqrt(H^2) (not DSM/flow); hellinger_gt_sq_mc key stores sqrt(H^2).\n"
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


def main(argv: list[str] | None = None) -> None:
    # When stdout is redirected (e.g. nohup), default block buffering delays run.log updates.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    p = build_parser()
    args = p.parse_args(argv)
    args.output_dir = os.path.abspath(str(args.output_dir))
    args.dataset_npz = os.path.abspath(str(args.dataset_npz))
    _validate_cli(args)
    ns = _parse_n_list(args.n_list)

    os.makedirs(args.output_dir, exist_ok=True)
    bundle = load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
    meta_family = str(meta.get("dataset_family", ""))
    if meta_family != str(args.dataset_family):
        raise ValueError(
            f"NPZ meta dataset_family={meta_family!r} does not match --dataset-family={str(args.dataset_family)!r}. "
            "Regenerate with matching make_dataset.py --dataset-family, or pass --dataset-family to match the NPZ."
        )
    n_pool = int(bundle.theta_all.shape[0])
    need = max(int(args.n_ref), max(ns))
    if n_pool < need:
        raise ValueError(
            f"Dataset has n_total={n_pool} but need at least max(n_ref, max(n_list))={need}. "
            "Regenerate with make_dataset.py --n-total >= that value."
        )
    for n in ns:
        if n < 1:
            raise ValueError(f"Each n in --n-list must be >= 1; got {n}.")
        if n > n_pool:
            raise ValueError(f"Each n in --n-list must be <= n_total={n_pool}; got n={n}.")
    if max(ns) > int(args.n_ref):
        raise ValueError(
            f"Require max(n-list) <= n-ref for nested subsets; got max(n_list)={max(ns)} n_ref={args.n_ref}."
        )

    if bool(getattr(args, "visualization_only", False)):
        cached = _load_cached_convergence_results(args.output_dir)
        _validate_cached_matches_cli(args, cached, ns)
        if int(cached["meta_seed"]) != 0 and int(meta["seed"]) != int(cached["meta_seed"]):
            raise ValueError(
                f"Dataset NPZ meta seed={meta['seed']} does not match cached results dataset_meta_seed={cached['meta_seed']}. "
                "Use the same --dataset-npz as the run that produced the cached results."
            )
        ref_dir_v = os.path.join(args.output_dir, "reference")
        n_bins_v = int(args.num_theta_bins)
        per_n_loss_rows_viz: list[dict[str, str]] = []
        for n in ns:
            loss_npz = os.path.join(args.output_dir, "training_losses", f"n_{int(n):06d}.npz")
            if not os.path.isfile(loss_npz):
                raise FileNotFoundError(
                    f"Visualization-only mode requires per-n training loss NPZ (for loss panel):\n  {loss_npz}\n"
                    "Run a full study first so training_losses/n_*.npz exists, or copy artifacts from a prior run."
                )
            loss_abs = os.path.abspath(loss_npz)
            per_n_loss_rows_viz.append(
                {
                    "n": str(n),
                    "status": "cached",
                    "run_dir": "",
                    "src": loss_abs,
                    "dst": loss_abs,
                    "note": "visualization-only (no re-training)",
                }
            )
        print(
            "[convergence] --visualization-only: skipping GT MC, reference sweep, and per-n training; "
            "regenerating figures from cached NPZ.",
            flush=True,
        )
        _backfill_fixed_x_posterior_diagnostic_if_missing(
            output_dir=args.output_dir,
            bundle=bundle,
            meta=meta,
            ns=ns,
            perm_seed=int(cached["perm_seed"]),
            n_pool=n_pool,
        )
        _llr_c = cached.get("llr_binned_columns")
        _corr_llr_c = cached.get("corr_llr_binned_vs_gt_mc")
        _bg_h_c = cached.get("binned_gaussian_h_binned_columns")
        _bg_corr_c = cached.get("binned_gaussian_corr_h_binned_vs_gt_mc")
        _bg_label_c = cached.get("binned_gaussian_label")
        _bg_vf_c = cached.get("binned_gaussian_variance_floor")
        _render_convergence_figures_and_summary(
            args=args,
            meta=meta,
            perm_seed=int(cached["perm_seed"]),
            n_pool=n_pool,
            ref_dir=ref_dir_v,
            ns=ns,
            n_bins=n_bins_v,
            theta_centers=cached["centers"],
            gt_n_mc=int(cached["gt_n_mc"]),
            gt_seed=int(cached["gt_seed"]),
            gt_symmetrize_effective=bool(cached["gt_symmetrize"]),
            corr_h=cached["corr_h"],
            corr_clf=cached["corr_clf"],
            wall_s=cached["wall_s"],
            h_cols=cached["h_cols"],
            clf_cols=cached["clf_cols"],
            out_npz=cached["out_npz"],
            per_n_loss_rows=per_n_loss_rows_viz,
            err_msg=[],
            visualization_only=True,
            llr_cols=None if _llr_c is None else np.asarray(_llr_c, dtype=np.float64),
            corr_llr=None if _corr_llr_c is None else np.asarray(_corr_llr_c, dtype=np.float64).ravel(),
            binned_gaussian_h_cols=None if _bg_h_c is None else np.asarray(_bg_h_c, dtype=np.float64),
            binned_gaussian_corr_h=None if _bg_corr_c is None else np.asarray(_bg_corr_c, dtype=np.float64).ravel(),
            binned_gaussian_label=None if _bg_label_c is None else str(_bg_label_c),
            binned_gaussian_variance_floor=None if _bg_vf_c is None else float(_bg_vf_c),
        )
        return

    n_bins = int(args.num_theta_bins)
    base_seed = int(args.run_seed) if args.run_seed is not None else int(meta["seed"])
    perm_seed = base_seed + int(args.subset_seed_offset)
    rng_perm = np.random.default_rng(perm_seed)
    perm = rng_perm.permutation(n_pool)
    if args.run_seed is not None:
        print(
            f"[convergence] --run-seed={int(args.run_seed)} "
            f"(dataset meta seed={int(meta['seed'])}; perm_seed={perm_seed})",
            flush=True,
        )

    theta_raw_all = np.asarray(bundle.theta_all, dtype=np.float64)
    theta_scalar_all, theta_ref, edges, edge_lo, edge_hi, bin_idx_all = prepare_theta_binning_for_convergence(
        theta_raw_all,
        perm,
        int(args.n_ref),
        n_bins,
    )
    theta_state_all: np.ndarray | None = None
    if bool(getattr(args, "theta_flow_onehot_state", False)):
        theta_state_all = np.eye(n_bins, dtype=np.float64)[bin_idx_all]
        print(
            f"[convergence] theta_flow one-hot state enabled: theta -> one_hot(bin(theta), K={n_bins})",
            flush=True,
        )
    elif bool(getattr(args, "theta_flow_fourier_state", False)):
        theta_state_all, theta_fourier_ref_range, theta_fourier_period, theta_fourier_center = _build_theta_fourier_state(
            theta_scalar_all,
            theta_ref=theta_ref,
            k=int(args.theta_flow_fourier_k),
            period_mult=float(args.theta_flow_fourier_period_mult),
            include_linear=bool(args.theta_flow_fourier_include_linear),
        )
        print(
            "[convergence] theta_flow Fourier state enabled: "
            f"dim={theta_state_all.shape[1]} K={int(args.theta_flow_fourier_k)} "
            f"period={theta_fourier_period:.6g} "
            f"(mult={float(args.theta_flow_fourier_period_mult):.3g}, ref_range={theta_fourier_ref_range:.6g}, "
            f"center={theta_fourier_center:.6g}, include_linear={bool(args.theta_flow_fourier_include_linear)})",
            flush=True,
        )

    clf_rs = base_seed if int(args.clf_random_state) < 0 else int(args.clf_random_state)

    # Generative GT uses fisher.shared_fisher_est.generative_x_dim_from_meta when meta has PR embedding,
    # so MC likelihood matches the low-d toy model behind embedded NPZs (not embedded x_dim).
    dataset_for_gt = build_dataset_from_meta(meta)
    gt_seed = base_seed if int(args.gt_hellinger_seed) < 0 else int(args.gt_hellinger_seed)
    if hasattr(dataset_for_gt, "rng"):
        dataset_for_gt.rng = np.random.default_rng(gt_seed)
    centers = bin_centers_from_edges(edges)
    gt_n_mc = int(args.n_ref) // n_bins
    t_gt0 = time.time()
    h_gt_mc = estimate_hellinger_sq_one_sided_mc(
        dataset_for_gt,
        centers,
        n_mc=gt_n_mc,
        symmetrize=bool(args.gt_hellinger_symmetrize),
    )
    h_gt_sqrt = _sqrt_h_like(h_gt_mc)
    print(
        f"[convergence] GT Hellinger (MC likelihood) n_bins={n_bins} n_mc={gt_n_mc} "
        f"(n_bins*n_mc={n_bins * gt_n_mc} <= n_ref={int(args.n_ref)}) wall time: {time.time() - t_gt0:.1f}s "
        f"(H track uses sqrt(H^2) vs sqrt(binned h_sym))",
        flush=True,
    )
    t_llr0 = time.time()
    llr_gt_mc = estimate_mean_llr_one_sided_mc(
        dataset_for_gt,
        centers,
        n_mc=gt_n_mc,
    )
    print(
        f"[convergence] GT one-sided mean LLR (MC likelihood) n_bins={n_bins} n_mc={gt_n_mc} "
        f"wall time: {time.time() - t_llr0:.1f}s (LLR track: E_x[log p(x|θ_j)-log p(x|θ_i)] vs binned ΔL).",
        flush=True,
    )

    ref_dir = os.path.join(args.output_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
    if tfm == "theta_flow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')})",
            flush=True,
        )
        if bool(getattr(args, "theta_flow_posterior_only_likelihood", False)):
            print(
                "[convergence] theta_flow mode uses ODE log-likelihood on the conditional theta-flow only "
                "(log p(theta|x) via compute_likelihood; prior flow skipped; no theta-axis score integral).",
                flush=True,
            )
        else:
            print(
                "[convergence] theta_flow mode uses ODE log-likelihood on theta-space flows "
                "(log p(theta|x) - log p(theta) via compute_likelihood; no theta-axis score integral).",
                flush=True,
            )
    elif tfm == "theta_flow_autoencoder":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')}; plain AE preprocessing)",
            flush=True,
        )
        print(
            "[convergence] theta_flow_autoencoder mode trains x->z autoencoder first, "
            "then uses ODE log-likelihood on theta-space flows conditioned on z "
            "(log p(theta|z) - log p(theta), or posterior-only if requested).",
            flush=True,
        )
    elif tfm == "theta_path_integral":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')})",
            flush=True,
        )
        print(
            "[convergence] theta_path_integral mode uses score-from-velocity conversion "
            "(path.velocity_to_epsilon then s=-eps/sigma_t) and trapezoid integral along sorted theta.",
            flush=True,
        )
    elif tfm == "x_flow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')}; conditional x-flow only)",
            flush=True,
        )
        print(
            "[convergence] x_flow mode uses ODE likelihood on x-space flow log p(x|theta) "
            "(no prior model).",
            flush=True,
        )
    elif tfm == "x_flow_autoencoder":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')}; conditional latent z-flow only)",
            flush=True,
        )
        print(
            "[convergence] x_flow_autoencoder mode trains x->z autoencoder first, "
            "uses ODE likelihood log p(z|theta), then DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "x_flow_pca":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')}; conditional binned-PCA z-flow only)",
            flush=True,
        )
        print(
            "[convergence] x_flow_pca mode fits PCA from theta-binned train means, projects x to z, "
            "uses ODE likelihood log p(z|theta), then DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "linear_theta_flow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(K={int(getattr(args, 'ltf_num_components', 3))}; time-independent mixture velocity)",
            flush=True,
        )
        print(
            "[convergence] linear_theta_flow mode trains pure FM on v_k(theta,x)=A_k theta+b_k(x), "
            "uses analytic Gaussian-mixture log p(theta|x), then DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "ctsm_v":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(ctsm_arch={getattr(args, 'ctsm_arch', 'film')}; pair-conditioned bridge model)",
            flush=True,
        )
        print(
            "[convergence] ctsm_v mode uses pair-conditioned CTSM-v to integrate per-pair "
            "log-ratio fields over t (no prior model).",
            flush=True,
        )
    elif tfm == "nf":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            "(conditional normalizing flow posterior + learned prior)",
            flush=True,
        )
        print(
            "[convergence] nf mode builds C_post[i,j]=log p(theta_j|x_i), learns log p(theta_j), "
            "forms R=C_post-log p(theta_j), then DeltaL=R-diag(R), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "contrastive":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(hidden_dim={int(getattr(args, 'contrastive_hidden_dim', 128))}; "
            f"depth={int(getattr(args, 'contrastive_depth', 3))}; "
            f"theta_encoding={str(getattr(args, 'contrastive_theta_encoding', 'one_hot_bin'))}; "
            f"theta bins={int(args.num_theta_bins)}; identity x embedding)",
            flush=True,
        )
        print(
            "[convergence] contrastive mode trains scalar S(x,encode(bin(theta))) with row-wise shuffled-batch cross entropy, "
            "then uses C[i,j]=S(x_i,theta_j), DeltaL=C-diag(C), and one-sided H^2 from exp(DeltaL/2).",
            flush=True,
        )
    elif tfm == "contrastive_soft":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(score_arch={str(getattr(args, 'contrastive_soft_score_arch', 'normalized_dot'))}; "
            f"dot_dim={int(getattr(args, 'contrastive_soft_dot_dim', 64))}; "
            f"coord_embed_dim={int(getattr(args, 'contrastive_soft_coordinate_embed_dim', 16))}; "
            f"gaussian_logvar=[{float(getattr(args, 'contrastive_soft_gaussian_logvar_min', -8.0)):g},"
            f"{float(getattr(args, 'contrastive_soft_gaussian_logvar_max', 5.0)):g}]; "
            f"hidden_dim={int(getattr(args, 'contrastive_hidden_dim', 128))}; "
            f"depth={int(getattr(args, 'contrastive_depth', 3))}; "
            f"bandwidth={float(getattr(args, 'contrastive_soft_bandwidth', 1.0)):g}; "
            f"bandwidth_start={float(getattr(args, 'contrastive_soft_bandwidth_start', 0.0)):g}; "
            f"bandwidth_end={float(getattr(args, 'contrastive_soft_bandwidth_end', 0.0)):g}; "
            f"bandwidth_k={int(getattr(args, 'contrastive_soft_bandwidth_k', 5))}; "
            f"periodic={bool(getattr(args, 'contrastive_soft_periodic', False))}; identity x embedding)",
            flush=True,
        )
        print(
            "[convergence] contrastive_soft mode trains scalar S(x,theta) with Gaussian-kernel soft positives "
            "over shuffled minibatch theta candidates, then uses C[i,j]=S(x_i,theta_j), DeltaL=C-diag(C), "
            "and one-sided H^2 from exp(DeltaL/2).",
            flush=True,
        )
    elif tfm == "bidir_contrastive_soft":
        _bidir_sa = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot"))
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(score_arch={_bidir_sa}; dot_dim={int(getattr(args, 'contrastive_soft_dot_dim', 64))}; "
            f"hidden_dim={int(getattr(args, 'contrastive_hidden_dim', 128))}; "
            f"depth={int(getattr(args, 'contrastive_depth', 3))}; "
            f"bandwidth={float(getattr(args, 'contrastive_soft_bandwidth', 1.0)):g}; "
            f"bandwidth_start={float(getattr(args, 'contrastive_soft_bandwidth_start', 0.0)):g}; "
            f"bandwidth_end={float(getattr(args, 'contrastive_soft_bandwidth_end', 0.0)):g}; "
            f"bandwidth_k={int(getattr(args, 'contrastive_soft_bandwidth_k', 5))}; "
            f"periodic={bool(getattr(args, 'contrastive_soft_periodic', False))})",
            flush=True,
        )
        if _bidir_sa == "mlp":
            print(
                "[convergence] bidir_contrastive_soft mode trains joint MLP S(x,theta) "
                "with 0.5 row soft CE + 0.5 column soft CE over Gaussian-kernel theta targets.",
                flush=True,
            )
        else:
            print(
                "[convergence] bidir_contrastive_soft mode trains S(x,theta)=alpha cos(g(x),a(theta))+b(theta) "
                "with 0.5 row soft CE + 0.5 column soft CE over Gaussian-kernel theta targets.",
                flush=True,
            )
    elif tfm == "contrastive_soft_gaussian_net":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(gn_hidden_dim={int(getattr(args, 'gn_hidden_dim', 128))}; "
            f"gn_depth={int(getattr(args, 'gn_depth', 3))}; "
            f"gn_diag_floor={float(getattr(args, 'gn_diag_floor', 1e-4)):g}; "
            f"contrastive_hidden_dim={int(getattr(args, 'contrastive_hidden_dim', 128))}; "
            f"bandwidth={float(getattr(args, 'contrastive_soft_bandwidth', 1.0)):g}; "
            f"bandwidth_start={float(getattr(args, 'contrastive_soft_bandwidth_start', 0.0)):g}; "
            f"bandwidth_end={float(getattr(args, 'contrastive_soft_bandwidth_end', 0.0)):g}; "
            f"bandwidth_k={int(getattr(args, 'contrastive_soft_bandwidth_k', 5))}; "
            f"periodic={bool(getattr(args, 'contrastive_soft_periodic', False))}; identity x embedding)",
            flush=True,
        )
        print(
            "[convergence] contrastive_soft_gaussian_net mode first trains a diagonal Gaussian "
            "log p(x|theta) by MLE, then fine-tunes the same scalar Gaussian score with "
            "Gaussian-kernel soft contrastive positives over shuffled minibatch theta candidates.",
            flush=True,
        )
    elif tfm == "contrastive_soft_gaussian_net_no_finetune":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(gn_hidden_dim={int(getattr(args, 'gn_hidden_dim', 128))}; "
            f"gn_depth={int(getattr(args, 'gn_depth', 3))}; "
            f"gn_diag_floor={float(getattr(args, 'gn_diag_floor', 1e-4)):g}; "
            f"bandwidth={float(getattr(args, 'contrastive_soft_bandwidth', 1.0)):g}; "
            f"bandwidth_start={float(getattr(args, 'contrastive_soft_bandwidth_start', 0.0)):g}; "
            f"bandwidth_end={float(getattr(args, 'contrastive_soft_bandwidth_end', 0.0)):g}; "
            f"bandwidth_k={int(getattr(args, 'contrastive_soft_bandwidth_k', 5))}; "
            f"periodic={bool(getattr(args, 'contrastive_soft_periodic', False))})",
            flush=True,
        )
        print(
            "[convergence] contrastive_soft_gaussian_net_no_finetune mode trains a diagonal Gaussian "
            "log p(x|theta) by MLE only, then evaluates C[i,j]=log p(x_i|theta_j) without soft-contrastive fine-tuning.",
            flush=True,
        )
    elif tfm == "nf_reduction":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(latent_dim={int(getattr(args, 'nfr_latent_dim', 2))}; invertible x->(z,epsilon) reduction)",
            flush=True,
        )
        print(
            "[convergence] nf_reduction mode trains an invertible x-space NSF and conditional z-flow, "
            "then uses C[i,j]=log p(z_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "pi_nf":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(latent_dim={int(getattr(args, 'pinf_latent_dim', 2))}; diagonal Gaussian p(z|theta); "
            f"recon_weight={float(getattr(args, 'pinf_recon_weight', 1.0)):g})",
            flush=True,
        )
        print(
            "[convergence] pi_nf mode trains an invertible x->(z,r) NSF with p(z|theta) diagonal Gaussian and r~N(0,I), "
            "then uses C[i,j]=log p(z_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "gmm_z_decode":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(latent_dim={int(getattr(args, 'gzd_latent_dim', 2))}; components={int(getattr(args, 'gzd_components', 5))})",
            flush=True,
        )
        print(
            "[convergence] gmm_z_decode mode trains z=E(x) and q(theta|z), "
            "then uses C[i,j]=log q(theta_j|z_i), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "gaussian_x_flow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(path_schedule={str(getattr(args, 'gxf_path_schedule', 'linear'))}; full covariance Cholesky FM)",
            flush=True,
        )
        print(
            "[convergence] gaussian_x_flow mode trains μ(θ) and full covariance Cholesky L(θ) with analytic FM velocity, "
            "then uses C[i,j]=log p(x_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "gaussian_x_flow_diagonal":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(path_schedule={str(getattr(args, 'gxf_path_schedule', 'linear'))}; diagonal covariance FM)",
            flush=True,
        )
        print(
            "[convergence] gaussian_x_flow_diagonal mode trains μ(θ) and diagonal covariance Cholesky L(θ) with analytic FM velocity, "
            "then uses C[i,j]=log p(x_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "linear_x_flow_diagonal_theta":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            "(theta-conditioned diagonal drift a_phi(theta) and offset b_phi(theta); straight-bridge FM)",
            flush=True,
        )
        print(
            "[convergence] linear_x_flow_diagonal_theta mode trains "
            "v(x,theta)=diag(a_phi(theta)) x + b_phi(theta), "
            "then uses diagonal Gaussian p(x|theta) with Sigma_ii=exp(2 a_i(theta)) and "
            "theta-dependent mean for C[i,j]=log p(x_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "linear_x_flow_diagonal_theta_spline":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(K={int(getattr(args, 'lxf_spline_k', 5))}; cubic B-spline features of scalar theta; straight-bridge FM)",
            flush=True,
        )
        print(
            "[convergence] linear_x_flow_diagonal_theta_spline mode trains "
            "v(x,theta)=diag(a(theta)) x + b(theta) with a,b linear in fixed B-spline(phi(theta)), "
            "then uses diagonal Gaussian p(x|theta) with Sigma_ii=exp(2 a_i(theta)) and "
            "theta-dependent mean for C[i,j]=log p(x_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "linear_x_flow_diagonal_t":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(path_schedule={str(getattr(args, 'lxfs_path_schedule', 'cosine'))}; "
            f"quadrature_steps={int(getattr(args, 'lxfs_quadrature_steps', 64))}; "
            "time-dependent diagonal A(t) and b(t,theta) FM)",
            flush=True,
        )
        print(
            "[convergence] linear_x_flow_diagonal_t mode trains "
            "v(x,t,theta)=diag(a(t)) x + b(t,theta) on a scheduled affine bridge, "
            "then uses quadrature for the induced diagonal Gaussian p(x|theta) with shared covariance.",
            flush=True,
        )
    elif tfm in (
        "linear_x_flow",
        "linear_x_flow_scalar",
        "linear_x_flow_diagonal",
        "linear_x_flow_low_rank",
        "linear_x_flow_low_rank_randb",
        "linear_x_flow_nonlinear_pca",
        "linear_x_flow_diagonal_t",
    ):
        if tfm == "linear_x_flow_scalar":
            drift_desc = "scalar A=aI"
        elif tfm == "linear_x_flow_diagonal":
            _bn = str(getattr(args, "lxf_b_net", "mlp")).strip().lower()
            drift_desc = f"diagonal A; b_phi={'FiLM trunk' if _bn == 'film' else 'MLP'}"
        elif tfm == "linear_x_flow_diagonal_t":
            drift_desc = (
                f"time-dependent diagonal A(t) "
                f"(path_schedule={str(getattr(args, 'lxfs_path_schedule', 'cosine'))}; "
                f"quadrature_steps={int(getattr(args, 'lxfs_quadrature_steps', 64))})"
            )
        elif tfm == "linear_x_flow_low_rank":
            drift_desc = f"low-rank A (rank={int(getattr(args, 'lxf_low_rank_dim', 4))})"
        elif tfm == "linear_x_flow_low_rank_randb":
            drift_desc = (
                f"random-basis low-rank A (rank={int(getattr(args, 'lxf_low_rank_dim', 4))}; "
                f"lambda_a={float(getattr(args, 'lxf_randb_lambda_a', 1e-4)):g}; "
                f"lambda_s={float(getattr(args, 'lxf_randb_lambda_s', 1e-4)):g})"
            )
        elif tfm == "linear_x_flow_nonlinear_pca":
            drift_desc = (
                f"full symmetric A plus nonlinear residual PCA correction "
                f"(k={int(getattr(args, 'lxf_nlpca_dim', 8))}; "
                f"lambda_h={float(getattr(args, 'lxf_nlpca_lambda_h', 0.0)):g}; "
                f"freeze_linear={bool(getattr(args, 'lxf_nlpca_freeze_linear', False))})"
            )
        else:
            drift_desc = "full symmetric A"
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"({drift_desc}; shared linear A x plus theta-MLP offset FM)",
            flush=True,
        )
        print(
            (
                f"[convergence] {tfm} mode first trains v(x,theta)=A x + b_phi(theta), "
                "fits residual PCA around the induced linear-flow mean, retrains with "
                "U h_phi(U^T(x_t - t mu_linear(theta)), t, theta), then uses ODE log p(x|theta) "
                "for C[i,j]=log p(x_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2)."
                if tfm == "linear_x_flow_nonlinear_pca"
                else f"[convergence] {tfm} mode trains v(x,theta)=A x + b_phi(theta), "
                "then uses the induced Gaussian with theta-dependent mean and shared covariance for "
                "C[i,j]=log p(x_i|theta_j), DeltaL=C-diag(C), and H via 1-sech(DeltaL/2)."
            ),
            flush=True,
        )
    elif tfm == "linear_x_flow_schedule":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(path_schedule={str(getattr(args, 'lxfs_path_schedule', 'cosine'))}; "
            "time-independent A x plus theta-MLP offset FM)",
            flush=True,
        )
        print(
            "[convergence] linear_x_flow_schedule mode trains v(x,theta)=A x + b_phi(theta) "
            "on a scheduled affine bridge with no time input to the network, then uses the induced "
            "Gaussian with theta-dependent mean and shared covariance for C[i,j]=log p(x_i|theta_j), "
            "DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm == "linear_theta_flow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(K={int(getattr(args, 'ltf_num_components', 3))}; time-independent mixture velocity)",
            flush=True,
        )
        print(
            "[convergence] linear_theta_flow mode trains pure FM on v_k(theta,x)=A_k theta+b_k(x), "
            "uses analytic Gaussian-mixture log p(theta|x), then DeltaL=C-diag(C), and H via 1-sech(DeltaL/2).",
            flush=True,
        )
    elif tfm in (
        "gaussian_network",
        "gaussian_network_diagonal",
        "gaussian_network_diagonal_binned_pca",
        "gaussian_network_low_rank",
        "gaussian_network_autoencoder",
        "gaussian_network_diagonal_autoencoder",
    ):
        if tfm == "gaussian_network_diagonal":
            desc = " diagonal precision"
            detail = "predicts mean and diagonal precision Cholesky factor L(theta)"
        elif tfm == "gaussian_network_diagonal_binned_pca":
            desc = " diagonal binned-PCA precision"
            detail = (
                "fits PCA from theta-binned train means, projects x to z, then predicts latent mean "
                "and diagonal precision Cholesky factor L(theta)"
            )
        elif tfm == "gaussian_network_low_rank":
            desc = " low-rank covariance"
            detail = "predicts high-dimensional mean and latent covariance Cholesky factor L(theta), with learned A and Psi"
        elif tfm == "gaussian_network_autoencoder":
            desc = " autoencoder"
            detail = "trains a plain x autoencoder first, then predicts latent mean and precision Cholesky factor L(theta)"
        elif tfm == "gaussian_network_diagonal_autoencoder":
            desc = " diagonal autoencoder"
            detail = "trains a plain x autoencoder first, then predicts latent mean and diagonal precision Cholesky factor L(theta)"
        else:
            desc = ""
            detail = "predicts mean and precision Cholesky factor L(theta)"
        c_detail = (
            "uses C[i,j]=log p(z_i|theta_j), then DeltaL=C-diag(C), and H via 1-sech(DeltaL/2)."
            if "autoencoder" in tfm or "binned_pca" in tfm
            else "uses C[i,j]=log p(x_i|theta_j), then DeltaL=C-diag(C), and H via 1-sech(DeltaL/2)."
        )
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(MLP conditional Gaussian{desc} likelihood)",
            flush=True,
        )
        print(
            f"[convergence] {tfm} mode {detail}, "
            f"{c_detail}",
            flush=True,
        )
    else:
        raise ValueError(
            f"Unsupported --theta-field-method={tfm!r}; use "
            "theta_flow, theta-flow-autoencoder, theta_path_integral, x_flow, x-flow-autoencoder, x-flow-pca, "
            "ctsm_v, nf, contrastive, nf-reduction, gaussian-x-flow, gaussian-x-flow-diagonal, "
            "linear-x-flow, linear-x-flow-scalar, linear-x-flow-diagonal, linear-x-flow-diagonal-theta, "
            "linear-x-flow-diagonal-theta-spline, "
            "linear-x-flow-low-rank, linear-x-flow-low-rank-randb, linear-x-flow-nonlinear-pca, "
            "linear-x-flow-schedule, linear-x-flow-diagonal-t, linear-theta-flow, gaussian-network, "
            "gaussian-network-diagonal, gaussian-network-diagonal-binned-pca, gaussian-network-low-rank, gaussian-network-autoencoder, "
            "or gaussian-network-diagonal-autoencoder."
        )
    print(
        "[convergence] n_ref reference: no learned H training at n_ref; matrix-panel top row = MC GT sqrt(H^2); "
        "pairwise decoding from n_ref subset only.",
        flush=True,
    )
    print(f"[convergence] reference dir (decoding-only artifacts) n={args.n_ref} -> {ref_dir}", flush=True)
    print(f"[convergence] n_list={ns}", flush=True)
    t0 = time.time()
    subset_ref = _subset_bundle(
        bundle,
        perm,
        int(args.n_ref),
        meta,
        bin_idx_all=bin_idx_all,
        theta_state_all=theta_state_all,
    )
    h_ref = np.asarray(h_gt_sqrt, dtype=np.float64)
    clf_ref = _pairwise_clf_from_bundle(
        args=args,
        meta=meta,
        subset=subset_ref,
        output_dir=ref_dir,
        n_bins=n_bins,
        clf_min_class_count=int(args.clf_min_class_count),
        clf_random_state=clf_rs,
    )
    print(f"[convergence] reference (GT + decoding) wall time: {time.time() - t0:.1f}s")

    loss_dir = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        h_binned_ref=h_ref,
        clf_acc_ref=clf_ref,
        # Legacy key name: stores sqrt(H^2) from MC (not raw H^2); see module docstring.
        hellinger_gt_sq_mc=h_gt_sqrt,
        gt_mean_llr_one_sided_mc=np.asarray(llr_gt_mc, dtype=np.float64),
        h_binned_ref_is_gt_mc=np.int32(1),
        theta_bin_centers=centers,
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_n_ref_budget=np.int64(args.n_ref),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        theta_bin_edges=edges,
        edge_lo=np.float64(edge_lo),
        edge_hi=np.float64(edge_hi),
        perm_seed=np.int64(perm_seed),
        convergence_base_seed=np.int64(base_seed),
        dataset_meta_seed=np.int64(meta["seed"]),
        n_ref=np.int64(args.n_ref),
    )

    corr_h = np.full(len(ns), np.nan, dtype=np.float64)
    corr_clf = np.full(len(ns), np.nan, dtype=np.float64)
    corr_llr = np.full(len(ns), np.nan, dtype=np.float64)
    wall_s = np.full(len(ns), np.nan, dtype=np.float64)
    err_msg: list[str] = []
    h_sweep: list[np.ndarray] = []
    clf_sweep: list[np.ndarray] = []
    llr_sweep: list[np.ndarray] = []
    per_n_loss_rows: list[dict[str, str]] = []
    ds_fam = str(meta.get("dataset_family", "cosine_gaussian"))

    sweep_root = os.path.join(args.output_dir, "sweep_runs")
    if bool(args.keep_intermediate):
        os.makedirs(sweep_root, exist_ok=True)
    print(
        "[convergence] per-n training sweep is enabled; collecting training_losses from each run artifact.",
        flush=True,
    )

    for k, n in enumerate(ns):
        run_dir: str
        tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
        if bool(args.keep_intermediate):
            run_dir = os.path.join(sweep_root, f"n_{n:06d}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            tmp_ctx = tempfile.TemporaryDirectory(
                prefix=f"h_conv_n{n}_",
                dir=args.output_dir,
                ignore_cleanup_errors=True,
            )
            run_dir = tmp_ctx.name

        try:
            t1 = time.time()
            subset_n = _subset_bundle(
                bundle,
                perm,
                int(n),
                meta,
                bin_idx_all=bin_idx_all,
                theta_state_all=theta_state_all,
            )
            loaded_n, x_aligned, _ = _estimate_one(
                args=args,
                meta=meta,
                bundle=subset_n.bundle,
                output_dir=run_dir,
                n_bins=n_bins,
                bin_train=subset_n.bin_train,
                bin_validation=subset_n.bin_validation,
                bin_all=subset_n.bin_all,
            )
            per_diag = os.path.join(
                args.output_dir,
                "sweep_runs",
                f"n_{int(n):06d}",
                "diagnostics",
            )
            _write_fixed_x_posterior_diagnostic(
                run_dir=run_dir,
                persistent_diagnostics_dir=per_diag,
                meta=meta,
                perm_seed=int(perm_seed),
                n_subset=int(n),
                x_aligned=x_aligned,
            )
            h_n, clf_n = _metrics_fixed_edges(
                loaded_n,
                subset_n,
                n_bins,
                int(args.clf_min_class_count),
                clf_rs,
            )
            h_n_sqrt = _sqrt_h_like(h_n)
            corr_h[k] = vhb.matrix_corr_offdiag_pearson(h_n_sqrt, h_gt_sqrt)
            corr_clf[k] = vhb.matrix_corr_offdiag_pearson(clf_n, clf_ref)
            wall_s[k] = time.time() - t1
            h_sweep.append(np.asarray(h_n_sqrt, dtype=np.float64))
            clf_sweep.append(np.asarray(clf_n, dtype=np.float64))
            try:
                delta_l_in = _load_delta_l_from_run_dir(run_dir, dataset_family=ds_fam)
                llr_n = _metrics_delta_l_binned(delta_l_in, subset_n, n_bins)
                corr_llr[k] = vhb.matrix_corr_offdiag_pearson(llr_n, np.asarray(llr_gt_mc, dtype=np.float64))
            except KeyError as exc:
                llr_n = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
                print(f"[convergence] n={n}  LLR diagnostic unavailable: {exc}", flush=True)
            llr_sweep.append(np.asarray(llr_n, dtype=np.float64))
            print(
                f"[convergence] n={n}  corr_h={corr_h[k]:.4f}  corr_clf={corr_clf[k]:.4f}  "
                f"corr_llr={corr_llr[k]:.4f}  wall={wall_s[k]:.1f}s",
                flush=True,
            )
            run_loss_npz = os.path.join(run_dir, "score_prior_training_losses.npz")
            run_loss_npz_abs = os.path.abspath(run_loss_npz)
            if not os.path.isfile(run_loss_npz_abs):
                raise FileNotFoundError(
                    f"Expected per-n training loss artifact is missing: {run_loss_npz_abs}"
                )
            dst_loss_npz_abs = os.path.abspath(os.path.join(loss_dir, f"n_{n:06d}.npz"))
            shutil.copy2(run_loss_npz_abs, dst_loss_npz_abs)
            per_n_loss_rows.append(
                {
                    "n": str(n),
                    "status": "copied",
                    "run_dir": os.path.abspath(run_dir),
                    "src": run_loss_npz_abs,
                    "dst": dst_loss_npz_abs,
                    "note": "from per-n training sweep run",
                }
            )
            print(
                f"[convergence] n={n} training_loss copied -> {dst_loss_npz_abs}",
                flush=True,
            )
        except Exception as e:
            err_msg.append(f"n={n}: {e!r}")
            print(f"[convergence] ERROR n={n}: {e}")
            per_n_loss_rows.append(
                {
                    "n": str(n),
                    "status": "error",
                    "run_dir": os.path.abspath(run_dir),
                    "src": os.path.abspath(os.path.join(run_dir, "score_prior_training_losses.npz")),
                    "dst": os.path.abspath(os.path.join(loss_dir, f"n_{n:06d}.npz")),
                    "note": repr(e),
                }
            )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if err_msg:
        msg = "\n".join([f"  - {m}" for m in err_msg[:20]])
        raise RuntimeError(
            "Per-n sweep failed (including required training-loss collection).\n"
            f"{msg}"
        )
    if len(h_sweep) != len(ns) or len(clf_sweep) != len(ns) or len(llr_sweep) != len(ns):
        raise RuntimeError(
            "Missing binned matrices for some n (partial failures). "
            "Fix errors above or re-run with a smaller n-list."
        )
    h_cols = np.stack(h_sweep + [h_ref], axis=0)
    clf_cols = np.stack(clf_sweep + [clf_ref], axis=0)
    llr_ref = np.asarray(llr_gt_mc, dtype=np.float64)
    llr_cols = np.stack(llr_sweep + [llr_ref], axis=0)
    column_n = np.asarray(list(ns) + [int(args.n_ref)], dtype=np.int64)
    binned_gaussian_label = r"$\sqrt{H^2}$, binned Gaussian"
    binned_gaussian_variance_floor = float(
        getattr(args, "flow_theta_reg_variance_floor", getattr(args, "flow_x_reg_variance_floor", 1e-6))
    )
    binned_gaussian_h_sweep: list[np.ndarray] = []
    binned_gaussian_corr_h = np.full(len(ns), np.nan, dtype=np.float64)
    for k, n in enumerate(ns):
        subset_bg = _subset_bundle(
            bundle,
            perm,
            int(n),
            meta,
            bin_idx_all=bin_idx_all,
            theta_state_all=None,
        )
        bg_h2 = _binned_gaussian_hellinger_sq(
            subset_bg,
            n_bins,
            variance_floor=binned_gaussian_variance_floor,
        )
        bg_h_sqrt = _sqrt_h_like(bg_h2)
        binned_gaussian_h_sweep.append(np.asarray(bg_h_sqrt, dtype=np.float64))
        binned_gaussian_corr_h[k] = vhb.matrix_corr_offdiag_pearson(bg_h_sqrt, h_gt_sqrt)
    binned_gaussian_h_cols = np.stack(binned_gaussian_h_sweep + [h_ref], axis=0)

    out_npz = os.path.join(args.output_dir, "h_decoding_convergence_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.asarray(ns, dtype=np.int64),
        corr_h_binned_vs_gt_mc=corr_h,
        corr_clf_vs_ref=corr_clf,
        wall_seconds=wall_s,
        n_ref=np.int64(args.n_ref),
        perm_seed=np.int64(perm_seed),
        convergence_base_seed=np.int64(base_seed),
        dataset_meta_seed=np.int64(meta["seed"]),
        theta_bin_edges=edges,
        theta_bin_centers=centers,
        # Legacy key name: sqrt(H^2) from MC; see module docstring.
        hellinger_gt_sq_mc=h_gt_sqrt,
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_n_ref_budget=np.int64(args.n_ref),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        h_binned_ref_is_gt_mc=np.int32(1),
        h_binned_columns=h_cols,
        clf_acc_columns=clf_cols,
        column_n=column_n,
        gt_mean_llr_one_sided_mc=np.asarray(llr_gt_mc, dtype=np.float64),
        llr_binned_columns=llr_cols,
        corr_llr_binned_vs_gt_mc=corr_llr,
        binned_gaussian_h_binned_columns=binned_gaussian_h_cols,
        binned_gaussian_corr_h_binned_vs_gt_mc=binned_gaussian_corr_h,
        binned_gaussian_variance_floor=np.float64(binned_gaussian_variance_floor),
        binned_gaussian_label=np.asarray([binned_gaussian_label], dtype=object),
    )

    _render_convergence_figures_and_summary(
        args=args,
        meta=meta,
        perm_seed=int(perm_seed),
        n_pool=n_pool,
        ref_dir=ref_dir,
        ns=ns,
        n_bins=n_bins,
        theta_centers=centers,
        gt_n_mc=int(gt_n_mc),
        gt_seed=int(gt_seed),
        gt_symmetrize_effective=bool(args.gt_hellinger_symmetrize),
        corr_h=corr_h,
        corr_clf=corr_clf,
        wall_s=wall_s,
        h_cols=h_cols,
        clf_cols=clf_cols,
        out_npz=out_npz,
        per_n_loss_rows=per_n_loss_rows,
        err_msg=err_msg,
        visualization_only=False,
        llr_cols=llr_cols,
        corr_llr=corr_llr,
        binned_gaussian_h_cols=binned_gaussian_h_cols,
        binned_gaussian_corr_h=binned_gaussian_corr_h,
        binned_gaussian_label=binned_gaussian_label,
        binned_gaussian_variance_floor=binned_gaussian_variance_floor,
    )


if __name__ == "__main__":
    main()
