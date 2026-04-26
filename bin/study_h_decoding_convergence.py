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
log-likelihood Bayes ratios; prior + posterior theta-flows), ``--theta-field-method theta_flow_reg``
(same H evaluation as theta_flow, with binned Gaussian synthetic-pair regularization during posterior training),
``--theta-field-method theta_flow_pre_post`` (same H evaluation as theta_flow, with posterior
regularization-only pretraining followed by readout-only real-data fine-tuning),
``--theta-field-method theta_path_integral``
(same training as theta_flow but H from velocity-to-score plus trapezoid integral along sorted ``theta``),
``--theta-field-method x_flow`` (conditional x-space FM likelihood; no prior model),
``--theta-field-method x_flow_reg`` (same x-space FM likelihood with KNN Gaussian velocity-prior regularization),
``--theta-field-method ctsm_v`` (pair-conditioned CTSM-v time-score integration; no prior model), and
``--theta-field-method nf`` (conditional normalizing flow log p(theta|x) with an NF prior
for posterior-minus-prior log-ratio construction).
Flow methods use ``--flow-arch``: ``mlp``, ``film`` (FiLM with raw-theta embeddings), or
``film_fourier`` for ``theta_flow`` / ``theta_flow_reg`` / ``theta_flow_pre_post`` /
``theta_path_integral`` / ``x_flow`` / ``x_flow_reg``.
``film_fourier`` uses FiLM conditioning with Fourier theta features
(``--flow-theta-fourier-*`` for ``theta_flow`` / ``theta_flow_reg`` /
``theta_flow_pre_post`` / ``theta_path_integral`` and ``--flow-x-theta-fourier-*`` for ``x_flow`` / ``x_flow_reg``).
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
generative mean log-likelihood ratios; see ``fisher/hellinger_gt.py``). The optional
``binned_gaussian_h_binned_columns`` row is also stored as ``sqrt(H^2)``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
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
from fisher.dataset_visualization import pca_project
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.hellinger_gt import (
    bin_centers_from_edges,
    estimate_hellinger_sq_one_sided_mc,
    estimate_mean_llr_one_sided_mc,
)
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


def _x_pca_bundle(
    bundle: SharedDatasetBundle,
    *,
    n_components: int,
) -> SharedDatasetBundle:
    """Project all bundle ``x`` arrays onto the first ``k`` PCs of ``x_all`` (SVD, centered)."""
    if int(n_components) <= 0:
        return bundle
    x_all = np.asarray(bundle.x_all, dtype=np.float64)
    if x_all.ndim != 2:
        raise ValueError("PCA requires 2D x_all.")
    n, d = int(x_all.shape[0]), int(x_all.shape[1])
    k_req = min(int(n_components), n, d)
    if k_req < 1:
        raise ValueError("PCA: need positive n_components and valid x shape.")
    # If we cannot reduce below ambient dimension, keep raw x (avoids a pure rotation of full-rank data).
    if k_req >= d:
        print(
            f"[convergence] x PCA: requested/allowed k={k_req} but x_dim={d}; keeping raw x (no reduction).",
            flush=True,
        )
        return bundle
    k = k_req
    proj_all, x_mean, basis = pca_project(x_all, n_components=k)
    if int(n_components) > k:
        print(
            f"[convergence] x PCA: requested n_components={int(n_components)} but using k={k} (min of n, d, request).",
            flush=True,
        )

    def _t(x: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x, dtype=np.float64)
        if x0.ndim != 2 or int(x0.shape[1]) != d:
            raise ValueError("PCA transform requires x with same num_features as x_all.")
        return (x0 - x_mean) @ basis

    new_meta = dict(bundle.meta)
    new_meta["x_dim"] = int(k)
    new_meta["x_pca_n_components"] = int(k)
    new_meta["x_pca_fitted_n"] = int(n)
    new_meta["x_pca_from_dim"] = int(d)
    return replace(
        bundle,
        meta=new_meta,
        x_all=proj_all.astype(np.float64, copy=False),
        x_train=_t(bundle.x_train).astype(np.float64, copy=False),
        x_validation=_t(bundle.x_validation).astype(np.float64, copy=False),
    )


def _meta_for_gt_hellinger_mc(meta: dict[str, Any]) -> dict[str, Any]:
    """Return meta for MC GT Hellinger / one-sided mean LLR.

    For PR-autoencoder-embedded **cosine** families, the NPZ has ambient ``x_dim = h_dim`` (e.g. 100)
    but the generative toy is still the low-dimensional (pre-embed) process; ``x`` in the archive
    is a nonlinear lift. MC ground truth should use :func:`fisher.shared_fisher_est.build_dataset_from_meta`
    with ``x_dim = pr_autoencoder_z_dim`` so ``sample_x`` and ``log_p_x_given_theta`` are consistent
    with the original space.

    ``randamp_gaussian_sqrtd`` + PR is already handled inside ``build_dataset_from_meta`` (generative
    ``x_dim`` set from ``pr_autoencoder_z_dim``); this helper only applies to cosine* families.
    """
    if not bool(meta.get("pr_autoencoder_embedded", False)):
        return meta
    fam = str(meta.get("dataset_family", ""))
    if fam not in (
        "cosine_gaussian",
        "cosine_gaussian_const_noise",
        "cosine_gaussian_sqrtd",
        "cosine_gaussian_sqrtd_rand_tune",
    ):
        return meta
    h = int(meta.get("x_dim", 0))
    z = int(meta.get("pr_autoencoder_z_dim", h))
    if h <= z:
        return meta
    out = dict(meta)
    out["x_dim"] = z
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
            "(standalone training-loss panel, one column per n)."
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
            "randamp_gaussian",
            "randamp_gaussian_sqrtd",
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
        "--x-pca-dim",
        type=int,
        default=10,
        help=(
            "If >0, project x onto this many leading PCs of centered x_all (SVD) before the sweep, "
            "and update meta x_dim. Applies to x_all, x_train, and x_validation consistently. "
            "0 leaves x unchanged (raw ambient features)."
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
        "--prior-row-flow-x-reg-lambda",
        type=float,
        default=None,
        help=(
            "If set, add an auxiliary matrix-panel H row computed by a second x_flow_reg sweep "
            "with this KNN Gaussian velocity-prior regularization weight. The first H row remains "
            "the user-selected --theta-field-method."
        ),
    )
    p.add_argument(
        "--warm-start-flow-x-reg-source-lambda",
        type=float,
        default=None,
        help=(
            "If set, train an x_flow_reg source sweep with this lambda, save per-n checkpoints, "
            "then run the primary sweep initialized from the matching source checkpoint."
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
    add_estimation_arguments(p)
    p.set_defaults(
        output_dir=str(Path(DATA_DIR) / "h_decoding_convergence"),
        theta_field_method="theta_flow",
        flow_arch="mlp",
        flow_epochs=10000,
        flow_theta_pre_post_pretrain_epochs=10000,
        flow_theta_pre_post_finetune_epochs=10000,
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


def theta_segment_ids_equal_width(
    theta: np.ndarray,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Delegate to ``visualize_h_matrix_binned.theta_segment_ids_equal_width``."""
    return vhb.theta_segment_ids_equal_width(theta, n_segments)


def _validate_cli(args: argparse.Namespace) -> None:
    tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
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
    else:
        validate_estimation_args(args)
    if int(args.num_theta_bins) < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    if int(args.clf_min_class_count) < 1:
        raise ValueError("--clf-min-class-count must be >= 1.")
    if int(getattr(args, "x_pca_dim", 0)) < 0:
        raise ValueError("--x-pca-dim must be >= 0.")
    prior_row_lam = getattr(args, "prior_row_flow_x_reg_lambda", None)
    if prior_row_lam is not None:
        prior_row_lam_f = float(prior_row_lam)
        if not np.isfinite(prior_row_lam_f) or prior_row_lam_f <= 0.0:
            raise ValueError("--prior-row-flow-x-reg-lambda must be a finite positive number.")
    n_ref = int(args.n_ref)
    n_bins_cli = int(args.num_theta_bins)
    if n_ref < 2:
        raise ValueError("--n-ref must be >= 2.")
    if n_ref // n_bins_cli < 1:
        raise ValueError(
            "GT Hellinger requires n_mc = n_ref // num_theta_bins >= 1 "
            f"(got n_ref={n_ref} num_theta_bins={n_bins_cli})."
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
        if tfm not in ("theta_flow", "theta_flow_reg", "theta_flow_pre_post"):
            raise ValueError(
                "--theta-flow-onehot-state requires --theta-field-method theta_flow, "
                "theta_flow_reg, or theta_flow_pre_post "
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
        if tfm not in ("theta_flow", "theta_flow_reg", "theta_flow_pre_post"):
            raise ValueError(
                "--theta-flow-fourier-state requires --theta-field-method theta_flow, "
                "theta_flow_reg, or theta_flow_pre_post "
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
        if tfm not in ("theta_flow", "theta_flow_reg", "theta_flow_pre_post"):
            raise ValueError(
                "--theta-flow-segmented requires --theta-field-method theta_flow, "
                "theta_flow_reg, or theta_flow_pre_post "
                f"(got {getattr(args, 'theta_field_method', None)!r})."
            )
    warm_lam = getattr(args, "warm_start_flow_x_reg_source_lambda", None)
    if warm_lam is not None:
        tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
        if tfm != "x_flow_reg":
            raise ValueError("--warm-start-flow-x-reg-source-lambda requires --theta-field-method x_flow_reg.")
        if not np.isfinite(float(warm_lam)) or float(warm_lam) < 0.0:
            raise ValueError("--warm-start-flow-x-reg-source-lambda must be a finite non-negative number.")
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

    This is the no-flow diagnostic matching the binned Gaussian regularizer: fit
    ``p(x | bin(theta)=b) = N(mu_b, diag(global_var))`` from the subset full pool,
    then compute the closed-form shared-covariance Gaussian Hellinger distance.
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


class PerNSweepResult(NamedTuple):
    corr_h: np.ndarray
    corr_clf: np.ndarray
    corr_llr: np.ndarray
    wall_s: np.ndarray
    h_sweep: list[np.ndarray]
    clf_sweep: list[np.ndarray]
    llr_sweep: list[np.ndarray]
    per_n_loss_rows: list[dict[str, str]]


def _run_per_n_method_sweep(
    *,
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    perm: np.ndarray,
    ns: list[int],
    n_bins: int,
    bin_idx_all: np.ndarray,
    theta_state_all: np.ndarray | None,
    h_gt_sqrt: np.ndarray,
    clf_ref: np.ndarray,
    llr_gt_mc: np.ndarray,
    clf_random_state: int,
    run_root_name: str,
    loss_dir_name: str,
    sweep_label: str,
    loss_note: str,
    save_checkpoint_dir_name: str | None = None,
    init_checkpoint_dir_name: str | None = None,
) -> PerNSweepResult:
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

    sweep_root = os.path.join(args.output_dir, run_root_name)
    loss_dir = os.path.join(args.output_dir, loss_dir_name)
    save_checkpoint_dir = (
        os.path.join(args.output_dir, save_checkpoint_dir_name)
        if save_checkpoint_dir_name
        else None
    )
    init_checkpoint_dir = (
        os.path.join(args.output_dir, init_checkpoint_dir_name)
        if init_checkpoint_dir_name
        else None
    )
    os.makedirs(loss_dir, exist_ok=True)
    if save_checkpoint_dir is not None:
        os.makedirs(save_checkpoint_dir, exist_ok=True)
    if bool(args.keep_intermediate):
        os.makedirs(sweep_root, exist_ok=True)
    print(
        f"[convergence] per-n training sweep ({sweep_label}) is enabled; "
        f"collecting training_losses from each run artifact.",
        flush=True,
    )

    for k, n in enumerate(ns):
        run_dir: str
        tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
        if bool(args.keep_intermediate):
            run_dir = os.path.join(sweep_root, f"n_{n:06d}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"h_conv_n{n}_", dir=args.output_dir)
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
            run_args = argparse.Namespace(**vars(args))
            if save_checkpoint_dir is not None:
                setattr(
                    run_args,
                    "flow_x_save_checkpoint",
                    os.path.join(save_checkpoint_dir, f"n_{int(n):06d}.pt"),
                )
            if init_checkpoint_dir is not None:
                init_ckpt = os.path.join(init_checkpoint_dir, f"n_{int(n):06d}.pt")
                if not os.path.isfile(init_ckpt):
                    raise FileNotFoundError(f"Missing warm-start source checkpoint for n={n}: {init_ckpt}")
                setattr(run_args, "flow_x_init_checkpoint", init_ckpt)
            loaded_n, x_aligned, _ = _estimate_one(
                args=run_args,
                meta=meta,
                bundle=subset_n.bundle,
                output_dir=run_dir,
                n_bins=n_bins,
            )
            per_diag = os.path.join(
                args.output_dir,
                run_root_name,
                f"n_{int(n):06d}",
                "diagnostics",
            )
            _write_fixed_x_posterior_diagnostic(
                run_dir=run_dir,
                persistent_diagnostics_dir=per_diag,
                meta=meta,
                perm_seed=int(getattr(args, "_convergence_perm_seed", 0)),
                n_subset=int(n),
                x_aligned=x_aligned,
            )
            h_n, clf_n = _metrics_fixed_edges(
                loaded_n,
                subset_n,
                n_bins,
                int(args.clf_min_class_count),
                int(clf_random_state),
            )
            h_n_sqrt = _sqrt_h_like(h_n)
            corr_h[k] = vhb.matrix_corr_offdiag_pearson(h_n_sqrt, h_gt_sqrt)
            corr_clf[k] = vhb.matrix_corr_offdiag_pearson(clf_n, clf_ref)
            wall_s[k] = time.time() - t1
            h_sweep.append(np.asarray(h_n_sqrt, dtype=np.float64))
            clf_sweep.append(np.asarray(clf_n, dtype=np.float64))
            delta_l_in = _load_delta_l_from_run_dir(run_dir, dataset_family=ds_fam)
            llr_n = _metrics_delta_l_binned(delta_l_in, subset_n, n_bins)
            llr_sweep.append(np.asarray(llr_n, dtype=np.float64))
            corr_llr[k] = vhb.matrix_corr_offdiag_pearson(llr_n, np.asarray(llr_gt_mc, dtype=np.float64))
            print(
                f"[convergence] {sweep_label} n={n}  corr_h={corr_h[k]:.4f}  "
                f"corr_clf={corr_clf[k]:.4f}  corr_llr={corr_llr[k]:.4f}  wall={wall_s[k]:.1f}s",
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
                    "note": loss_note,
                }
            )
            print(
                f"[convergence] {sweep_label} n={n} training_loss copied -> {dst_loss_npz_abs}",
                flush=True,
            )
        except Exception as e:
            err_msg.append(f"n={n}: {e!r}")
            print(f"[convergence] ERROR {sweep_label} n={n}: {e}")
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
            f"Per-n sweep failed for {sweep_label} (including required training-loss collection).\n"
            f"{msg}"
        )
    if len(h_sweep) != len(ns) or len(clf_sweep) != len(ns) or len(llr_sweep) != len(ns):
        raise RuntimeError(
            f"Missing binned matrices for some n in {sweep_label} (partial failures). "
            "Fix errors above or re-run with a smaller n-list."
        )

    return PerNSweepResult(
        corr_h=corr_h,
        corr_clf=corr_clf,
        corr_llr=corr_llr,
        wall_s=wall_s,
        h_sweep=h_sweep,
        clf_sweep=clf_sweep,
        llr_sweep=llr_sweep,
        per_n_loss_rows=per_n_loss_rows,
    )


def _estimate_one(
    *,
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    output_dir: str,
    n_bins: int,
) -> tuple[vhb.LoadedHMatrix, np.ndarray, torch.device]:
    """Train (unless h-only), load H, return loaded H, x_aligned, and device."""
    tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
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
    if theta_chk.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(
            f"theta/H row mismatch: theta_chk={theta_chk.shape[0]} theta_used={loaded.theta_used.shape[0]}"
        )
    if not np.allclose(theta_chk, loaded.theta_used, rtol=0.0, atol=1e-5):
        raise ValueError(
            "theta_used from H-matrix npz does not match expected dataset rows for this h_field_method."
        )
    x_aligned = vhb.x_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    if x_aligned.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(
            f"x/H row mismatch: x_aligned={x_aligned.shape[0]} theta_used={loaded.theta_used.shape[0]}"
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


def _diagnostic_posterior_log_weights(
    *,
    hfm: str,
    c_row: np.ndarray,
    theta_flat: np.ndarray,
    log_p_theta_prior: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Return log weights for the fixed-x posterior diagnostic."""
    method = str(hfm).strip().lower()
    c = np.asarray(c_row, dtype=np.float64).reshape(-1)
    th = np.asarray(theta_flat, dtype=np.float64).reshape(-1)
    if method == "theta_flow":
        if log_p_theta_prior is not None:
            lp = np.asarray(log_p_theta_prior, dtype=np.float64).reshape(-1)
            if lp.size != c.size:
                raise ValueError(f"log_p_theta_prior length {lp.size} does not match c row length {c.size}.")
            return c + lp, "learned prior"
        return c + _log_std_normal_scalar(th), "standard-normal fallback"
    return c, "direct"


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


def _approx_gt_posterior_density(
    *,
    dataset: Any,
    x_fixed: np.ndarray,
    theta_dense: np.ndarray,
    theta_low: float,
    theta_high: float,
) -> np.ndarray:
    td = np.asarray(theta_dense, dtype=np.float64).reshape(-1)
    x1 = np.asarray(x_fixed, dtype=np.float64).reshape(1, -1)
    gen_d = int(getattr(dataset, "x_dim", int(x1.shape[1])))
    if int(x1.shape[1]) > gen_d:
        # e.g. PR-AE embedded x in R^h_dim while build_dataset_from_meta uses latent z_dim.
        x1 = x1[:, :gen_d]
    elif int(x1.shape[1]) < gen_d:
        raise ValueError(
            f"x has fewer dims than generative x_dim: got {x1.shape[1]} need {gen_d}."
        )
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
    log_p_theta_prior: np.ndarray | None,
    th_flat: np.ndarray,
    xa: np.ndarray,
    th_grid: np.ndarray,
    mu: np.ndarray,
    dataset: Any,
    lo: float,
    hi: float,
) -> None:
    c_row = np.asarray(c[int(i_fix), :], dtype=np.float64).reshape(-1)
    logp, posterior_source = _diagnostic_posterior_log_weights(
        hfm=hfm,
        c_row=c_row,
        theta_flat=th_flat,
        log_p_theta_prior=log_p_theta_prior,
    )
    w = _stable_softmax_log(logp)
    order = np.argsort(th_flat, kind="mergesort")
    th_s = th_flat[order]
    w_s = w[order]
    q_model = _weighted_kde_density(th_s, w_s, th_grid)

    x_fixed = np.asarray(xa[int(i_fix)], dtype=np.float64).reshape(-1)
    _gen_d = int(getattr(dataset, "x_dim", int(x_fixed.size)))
    if int(x_fixed.size) > _gen_d:
        x_fixed = x_fixed[:_gen_d]
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
    ax_top.plot(th_grid, q_model, color="#1f77b4", lw=1.6, label="Model posterior (approx)")
    ax_top.plot(th_grid, q_gt, color="#d62728", lw=1.5, ls="--", label="GT posterior (approx)")
    ax_top.set_ylabel("density")
    ax_top.set_title(
        f"Fixed-$x$ posterior diagnostics  (row $i$={int(i_fix)},  method={hfm}, {posterior_source})",
        fontsize=9,
    )
    ax_top.legend(loc="upper right", fontsize=7)
    ax_top.annotate(
        f"x[0]={x0:.3g}  ||x||={xn:.3g}  theta_map={theta_map:.3g}",
        xy=(0.5, 1.12),
        xycoords="axes fraction",
        ha="center",
        fontsize=7,
    )

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
        if dd < int(x_fixed.size):
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

    - ``theta_flow`` / ``theta_flow_reg`` / ``theta_flow_pre_post`` H artifacts store
      ``C[i,j] = log p(θ_j|x_i) - log p(θ_j)``. If available, add the saved learned
      ``log p(θ_j)`` back for posterior mass on the training θ grid. Older artifacts
      without this vector fall back to a standard-normal prior approximation.
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
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 2.2), dpi=120, layout="tight")
        ax.text(
            0.5,
            0.5,
            "c_matrix not in H-matrix npz (need h_save_intermediates=True).",
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
    log_p_theta_prior: np.ndarray | None = None
    if "log_p_theta_prior" in z.files:
        lp = np.asarray(z["log_p_theta_prior"], dtype=np.float64).reshape(-1)
        if lp.size == n and np.all(np.isfinite(lp)):
            log_p_theta_prior = lp
        else:
            print(
                "[convergence] fixed-x diagnostic: ignoring bad log_p_theta_prior "
                f"shape={getattr(lp, 'shape', None)} finite={bool(np.all(np.isfinite(lp)))}",
                flush=True,
            )
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
            logp, posterior_source = _diagnostic_posterior_log_weights(
                hfm=hfm,
                c_row=c_row,
                theta_flat=th_flat,
                log_p_theta_prior=log_p_theta_prior,
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
            ax.plot(th_s, w_s, "o-", color="#1f77b4", ms=2, lw=0.8, label="softmax C row (weights on θ grid)")
            ax.set_ylabel("mass")
            ax.set_xlabel(r"$\theta$")
            ax.set_title(
                f"fixed $x$  i={int(i_fix)}  method={hfm}, {posterior_source}\n"
                f"(posterior overlay failed: {e!s})",
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
        log_p_theta_prior=log_p_theta_prior,
        th_flat=th_flat,
        xa=xa,
        th_grid=th_grid,
        mu=mu,
        dataset=dataset,
        lo=lo,
        hi=hi,
    )
    _plot_fixed_x_column(
        ax_top=ax01,
        ax_bot=ax11,
        i_fix=int(i_fix_b),
        hfm=hfm,
        c=c,
        log_p_theta_prior=log_p_theta_prior,
        th_flat=th_flat,
        xa=xa,
        th_grid=th_grid,
        mu=mu,
        dataset=dataset,
        lo=lo,
        hi=hi,
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
    """If ``h_decoding_convergence_combined`` expects a diagnostic PNG but only an H-matrix NPZ exists (e.g. old run), write it."""
    n_max = int(max(ns))
    diag_png = os.path.join(
        output_dir,
        "sweep_runs",
        f"n_{n_max:06d}",
        "diagnostics",
        "theta_flow_single_x_posterior_hist.png",
    )
    if os.path.isfile(diag_png):
        return
    run_dir = os.path.join(output_dir, "sweep_runs", f"n_{n_max:06d}")
    ds_fam = str(meta.get("dataset_family", ""))
    suffix = "_non_gauss" if ds_fam == "cosine_gmm" else "_theta_cov"
    h_path = os.path.join(run_dir, f"h_matrix_results{suffix}.npz")
    if not os.path.isfile(h_path):
        return
    rng = np.random.default_rng(int(perm_seed))
    perm = rng.permutation(int(n_pool))
    sub = perm[:n_max]
    x_aligned = np.asarray(bundle.x_all[sub], dtype=np.float64)
    per_diag = os.path.join(output_dir, "sweep_runs", f"n_{n_max:06d}", "diagnostics")
    _write_fixed_x_posterior_diagnostic(
        run_dir=run_dir,
        persistent_diagnostics_dir=per_diag,
        meta=meta,
        perm_seed=int(perm_seed),
        n_subset=int(n_max),
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
        "prior_train_losses": _arr("prior_train_losses"),
        "prior_val_losses": _arr("prior_val_losses"),
        "prior_val_monitor_losses": _arr("prior_val_monitor_losses"),
        "theta_pre_post_pretrain_train_losses": _arr("theta_pre_post_pretrain_train_losses"),
        "theta_pre_post_pretrain_reg_train_losses": _arr("theta_pre_post_pretrain_reg_train_losses"),
        "theta_pre_post_pretrain_val_losses": _arr("theta_pre_post_pretrain_val_losses"),
        "theta_pre_post_pretrain_reg_val_losses": _arr("theta_pre_post_pretrain_reg_val_losses"),
        "theta_pre_post_pretrain_val_monitor_losses": _arr("theta_pre_post_pretrain_val_monitor_losses"),
        "theta_pre_post_finetune_train_losses": _arr("theta_pre_post_finetune_train_losses"),
        "theta_pre_post_finetune_fm_train_losses": _arr("theta_pre_post_finetune_fm_train_losses"),
        "theta_pre_post_finetune_val_losses": _arr("theta_pre_post_finetune_val_losses"),
        "theta_pre_post_finetune_fm_val_losses": _arr("theta_pre_post_finetune_fm_val_losses"),
        "theta_pre_post_finetune_val_monitor_losses": _arr("theta_pre_post_finetune_val_monitor_losses"),
    }


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
    """Plot train / val / EMA monitor curves on one axis."""
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
    """Two rows (score, prior), one column per ``n``; save PNG + SVG."""
    n_cols = len(ns)
    if n_cols < 1:
        raise ValueError("n-list must be non-empty for training loss panel.")
    w = max(2.6 * n_cols, 6.0)
    h = 5.8
    fig, axes = plt.subplots(2, n_cols, figsize=(w, h), squeeze=False, sharex="col")

    row0_ylabel = "score / posterior loss"
    row1_ylabel = "prior loss"

    for j, n in enumerate(ns):
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        if not os.path.isfile(path):
            for r in (0, 1):
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
            for r in (0, 1):
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
        elif tfm == "theta_flow_reg":
            post_lab = "theta-flow-reg ODE Bayes-ratio"
        elif tfm == "theta_flow_pre_post":
            post_lab = "theta-flow pre/post ODE Bayes-ratio"
        elif tfm == "theta_path_integral":
            post_lab = "theta-path-integral score"
        elif tfm == "x_flow":
            post_lab = "x-flow direct likelihood"
        elif tfm == "x_flow_reg":
            post_lab = "x-flow-reg direct likelihood"
        elif tfm == "ctsm_v":
            post_lab = "pair-conditioned CTSM-v"
        elif tfm == "nf":
            post_lab = "normalizing-flow posterior"
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

    fig.suptitle(
        "Training loss vs epoch (top: posterior; bottom: prior). Columns: nested subset sizes n.",
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
    diagnostic_png_path: str | None,
    out_png_path: str,
    dpi: int = 160,
    binned_gaussian_corr_h: np.ndarray | None = None,
    llr_gt: np.ndarray | None = None,
    llr_est_mats: list[np.ndarray] | None = None,
    corr_llr: np.ndarray | None = None,
    extra_h_rows: list[tuple[str, list[np.ndarray]]] | None = None,
) -> str:
    """Single figure with matrix panel, correlation curves, H and LLR est-vs-GT scatters, losses, and optional diagnostic.

    PNG is raster as usual. SVG keeps the right-hand curve as vector paths (not a single
    embedded screenshot); heatmaps still use matplotlib's normal SVG image handling for ``imshow``.

    With LLR data: top row = matrix + curves; then H scatter row; then LLR scatter row (binned
    model ``ΔL`` vs generative one-sided mean LLR); then training losses; then diagnostic.
    If ``llr_gt``/``llr_est_mats``/``corr_llr`` are omitted, the LLR row is skipped (older runs).
    """
    crv_w, crv_h = _H_DECODING_CURVE_FIGSIZE_IN
    if crv_h <= 0:
        raise ValueError("_H_DECODING_CURVE_FIGSIZE_IN height must be > 0.")
    n_cols = len(h_mats)
    n_loss_cols = len(ns)
    n_matrix_rows = 2 + len(extra_h_rows or [])
    m_w, m_h = 2.8 * n_cols, 2.5 * n_matrix_rows
    l_w, l_h = max(2.6 * n_loss_cols, 6.0), 5.8
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

    gs_loss = gs0[loss_row, :].subgridspec(2, n_loss_cols)
    axes_loss = np.empty((2, n_loss_cols), dtype=object)
    row0_ylabel = "score / posterior loss"
    row1_ylabel = "prior loss"
    for j, n in enumerate(ns):
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        axes_loss[0, j] = fig.add_subplot(gs_loss[0, j])
        axes_loss[1, j] = fig.add_subplot(gs_loss[1, j])
        if not os.path.isfile(path):
            for r in (0, 1):
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
            for r in (0, 1):
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
        elif tfm == "theta_flow_reg":
            post_lab = "theta-flow-reg ODE Bayes-ratio"
        elif tfm == "theta_flow_pre_post":
            post_lab = "theta-flow pre/post ODE Bayes-ratio"
        elif tfm == "theta_path_integral":
            post_lab = "theta-path-integral score"
        elif tfm == "x_flow":
            post_lab = "x-flow direct likelihood"
        elif tfm == "x_flow_reg":
            post_lab = "x-flow-reg direct likelihood"
        elif tfm == "ctsm_v":
            post_lab = "pair-conditioned CTSM-v"
        elif tfm == "nf":
            post_lab = "normalizing-flow posterior"
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

    diag_row = 4 if use_llr else 3
    ax_diag = fig.add_subplot(gs0[diag_row, :])
    if diagnostic_png_path is None or not os.path.isfile(str(diagnostic_png_path)):
        ax_diag.text(
            0.5,
            0.5,
            "Fixed-x posterior+tuning diagnostic not found.\n"
            "Expected: sweep_runs/n_<max(n_list)>/diagnostics/theta_flow_single_x_posterior_hist.png",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax_diag.set_axis_off()
    else:
        try:
            img = plt.imread(str(diagnostic_png_path))
            ax_diag.imshow(img)
            ax_diag.set_title(
                f"Fixed-x posterior+tuning diagnostic ({os.path.basename(str(diagnostic_png_path))})",
                fontsize=10,
            )
            ax_diag.axis("off")
        except Exception as e:
            ax_diag.text(
                0.5,
                0.5,
                f"Failed to load diagnostic image:\n{e!s}",
                ha="center",
                va="center",
                fontsize=9,
                color="crimson",
            )
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
    """Draw H heatmap rows plus one decoding row on existing axes."""
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
    extra_rows = extra_h_rows or []
    h_rows = [(r"$\sqrt{H^2}$, user method", h_mats)] + list(extra_rows)
    n_h_rows = len(h_rows)
    n_rows = n_h_rows + 1
    n_cols = len(h_mats)
    if n_cols != len(clf_mats) or n_cols != len(col_labels):
        raise ValueError("h_mats, clf_mats, col_labels length mismatch.")
    for label, mats in extra_rows:
        if len(mats) != n_cols:
            raise ValueError(f"extra H row {label!r} has {len(mats)} columns; expected {n_cols}.")
    if axes.shape != (n_rows, n_cols):
        raise ValueError(f"axes must be shape ({n_rows}, {n_cols}); got {axes.shape}.")

    vmin_h, vmax_h = 0.0, 1.0
    vmin_c, vmax_c = _finite_min_max(clf_mats)
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12
    cmap = "viridis"

    for c in range(n_cols):
        for r, (row_label, row_mats) in enumerate(h_rows):
            ax0 = axes[r, c]
            im0 = ax0.imshow(
                row_mats[c],
                vmin=vmin_h,
                vmax=vmax_h,
                cmap=cmap,
                aspect="equal",
                origin="lower",
            )
            if r == 0:
                ax0.set_title(col_labels[c], fontsize=10)
            ax0.set_xticks(tick_pos)
            ax0.set_xticklabels(
                tick_labs,
                rotation=x_rot,
                ha="right" if x_rot else "center",
                fontsize=_matrix_tick_labelsize,
            )
            ax0.set_yticks(tick_pos)
            ax0.set_yticklabels(tick_labs, fontsize=_matrix_tick_labelsize)
            ax0.tick_params(axis="both", labelsize=_matrix_tick_labelsize)
            _matrix_axes_show_top_right_spines(ax0)
            if c == 0:
                ax0.set_ylabel(row_label, fontsize=11)
            _cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
            _cb0.ax.tick_params(labelsize=_matrix_colorbar_tick_labelsize)

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
    """H rows: primary plus optional auxiliaries; final row: pairwise decoding."""
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
    prior_row_h_binned_columns: NotRequired[np.ndarray]
    prior_row_label: NotRequired[str]
    prior_row_flow_x_reg_lambda: NotRequired[float]
    prior_row_corr_h_binned_vs_gt_mc: NotRequired[np.ndarray]
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
    if "prior_row_h_binned_columns" in z.files:
        base_bundle["prior_row_h_binned_columns"] = np.asarray(z["prior_row_h_binned_columns"], dtype=np.float64)
    if "prior_row_label" in z.files:
        raw_label = z["prior_row_label"]
        if np.asarray(raw_label).size > 0:
            base_bundle["prior_row_label"] = str(np.asarray(raw_label).reshape(-1)[0])
    if "prior_row_flow_x_reg_lambda" in z.files:
        base_bundle["prior_row_flow_x_reg_lambda"] = float(
            np.asarray(z["prior_row_flow_x_reg_lambda"]).reshape(-1)[0]
        )
    if "prior_row_corr_h_binned_vs_gt_mc" in z.files:
        base_bundle["prior_row_corr_h_binned_vs_gt_mc"] = np.asarray(
            z["prior_row_corr_h_binned_vs_gt_mc"],
            dtype=np.float64,
        ).ravel()
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
        if np.asarray(raw_label).size > 0:
            base_bundle["binned_gaussian_label"] = str(np.asarray(raw_label).reshape(-1)[0])
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
    prior_row_h_cols: np.ndarray | None = None,
    prior_row_label: str | None = None,
    prior_row_loss_rows: list[dict[str, str]] | None = None,
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
                "corr_binned_gaussian_h_binned_vs_gt_mc",
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
            r_bg: float | str
            if (
                binned_gaussian_corr_h is not None
                and int(np.asarray(binned_gaussian_corr_h, dtype=np.float64).ravel().size) > i
            ):
                r_bg = float(np.asarray(binned_gaussian_corr_h, dtype=np.float64).ravel()[i])
            else:
                r_bg = ""
            w.writerow([n, corr_h[i], corr_clf[i], r_llr, r_bg, wall_s[i]])

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
    if prior_row_h_cols is not None:
        prior_label = prior_row_label or r"$\sqrt{H^2}$, x-flow-reg prior"
        prior_arr = np.asarray(prior_row_h_cols, dtype=np.float64)
        if prior_arr.shape != np.asarray(h_cols, dtype=np.float64).shape:
            raise ValueError(
                f"prior_row_h_cols shape {prior_arr.shape} must match h_cols shape {np.asarray(h_cols).shape}."
            )
        extra_h_rows.append((prior_label, [np.asarray(prior_arr[j], dtype=np.float64) for j in range(prior_arr.shape[0])]))
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
    prior_row_manifest_path = ""
    if prior_row_loss_rows:
        prior_loss_dir = os.path.dirname(os.path.abspath(prior_row_loss_rows[0]["dst"]))
        os.makedirs(prior_loss_dir, exist_ok=True)
        prior_row_manifest_path = os.path.join(prior_loss_dir, "manifest.txt")
        with open(prior_row_manifest_path, "w", encoding="utf-8") as mf:
            mf.write("# n\tstatus\trun_dir\tsrc_loss_npz\tdst_loss_npz\tnote\n")
            for row in prior_row_loss_rows:
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
    diagnostic_png = os.path.join(
        args.output_dir,
        "sweep_runs",
        f"n_{int(max(ns)):06d}",
        "diagnostics",
        "theta_flow_single_x_posterior_hist.png",
    )
    diagnostic_png_use = diagnostic_png if os.path.isfile(diagnostic_png) else None
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
        binned_gaussian_corr_h=binned_gaussian_corr_h,
        loss_dir=loss_dir,
        diagnostic_png_path=diagnostic_png_use,
        out_png_path=combined_path,
        dpi=160,
        llr_gt=llr_gt,
        llr_est_mats=llr_est,
        corr_llr=corr_llr_a,
        extra_h_rows=extra_h_rows,
    )

    loss_panel_png = os.path.join(args.output_dir, "h_decoding_training_losses_panel.png")
    loss_panel_svg = _render_training_losses_panel(
        ns=list(ns),
        loss_dir=loss_dir,
        out_png_path=loss_panel_png,
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
        "embedded_fixed_x_diagnostic_png": diagnostic_png_use or "",
        "training_losses_panel": loss_panel_png,
        "training_losses_panel_svg": loss_panel_svg,
        "reference_npz": os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        "training_losses_dir": loss_dir,
        "training_losses_manifest": manifest_path,
    }
    if prior_row_h_cols is not None:
        paths_out["prior_row_label"] = prior_row_label or ""
        paths_out["prior_row_training_losses_manifest"] = prior_row_manifest_path
    if binned_gaussian_h_cols is not None:
        paths_out["binned_gaussian_label"] = binned_gaussian_label or ""
        if binned_gaussian_variance_floor is not None:
            paths_out["binned_gaussian_variance_floor"] = str(float(binned_gaussian_variance_floor))
    if visualization_only:
        paths_out["mode"] = "visualization-only (figures/csv/summary regenerated from cached NPZ)"
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
        if prior_row_h_cols is not None:
            sf.write("\n# Auxiliary prior row\n")
            sf.write(f"prior_row_label: {prior_row_label or ''}\n")
            sf.write(
                "prior_row_h_binned_columns: auxiliary H row columns, last column reuses MC GT sqrt(H^2).\n"
            )
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
    if int(getattr(args, "x_pca_dim", 0)) > 0:
        k = int(args.x_pca_dim)
        print(
            f"[convergence] projecting x to first {k} PCs (SVD, centered on x_all) for learned models; "
            "GT Hellinger / LLR MC still uses the generative toy from meta (see [convergence] GT line).",
            flush=True,
        )
        bundle = _x_pca_bundle(bundle, n_components=k)
        meta = bundle.meta
        print(
            f"[convergence] after x PCA: x_all.shape={tuple(bundle.x_all.shape)} "
            f"meta[x_dim]={int(meta.get('x_dim', -1))}",
            flush=True,
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
        _prior_row_h_c = cached.get("prior_row_h_binned_columns")
        _prior_row_label_c = cached.get("prior_row_label")
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
            prior_row_h_cols=None if _prior_row_h_c is None else np.asarray(_prior_row_h_c, dtype=np.float64),
            prior_row_label=None if _prior_row_label_c is None else str(_prior_row_label_c),
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
    if theta_raw_all.ndim == 2 and int(theta_raw_all.shape[1]) != 1:
        raise ValueError(
            "Convergence binning requires scalar theta in dataset bundle; "
            f"got theta_all shape={theta_raw_all.shape}."
        )
    theta_scalar_all = theta_raw_all.reshape(-1)
    theta_ref = np.asarray(theta_scalar_all[perm[: int(args.n_ref)]], dtype=np.float64).reshape(-1)
    edges, edge_lo, edge_hi = vhb.theta_bin_edges(theta_ref, n_bins)
    bin_idx_all = vhb.theta_to_bin_index(theta_scalar_all, edges, n_bins)
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

    meta_gt = _meta_for_gt_hellinger_mc(meta)
    if meta_gt is not meta:
        print(
            "[convergence] GT Hellinger / LLR MC: using generative "
            f"x_dim={int(meta_gt['x_dim'])} (PR latent z_dim) instead of ambient "
            f"x_dim={int(meta.get('x_dim', meta_gt['x_dim']))}.",
            flush=True,
        )
    dataset_for_gt = build_dataset_from_meta(meta_gt)
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
    if tfm in ("theta_flow", "theta_flow_reg", "theta_flow_pre_post"):
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')})",
            flush=True,
        )
        print(
            f"[convergence] {tfm} mode uses ODE log-likelihood on theta-space flows "
            "(log p(theta|x) - log p(theta) via compute_likelihood; no theta-axis score integral).",
            flush=True,
        )
        if tfm == "theta_flow_reg":
            print(
                "[convergence] theta_flow_reg adds binned Gaussian synthetic-pair posterior FM regularization "
                f"lambda={float(getattr(args, 'flow_theta_reg_lambda', 0.01)):.6g} "
                f"bins={int(getattr(args, 'flow_theta_reg_bin_n_bins', 10))} "
                f"var_floor={float(getattr(args, 'flow_theta_reg_variance_floor', 1e-6)):.6g}.",
                flush=True,
            )
        elif tfm == "theta_flow_pre_post":
            _ft_ep = int(getattr(args, "flow_theta_pre_post_finetune_epochs", 10000))
            _pre_synth = int(getattr(args, "flow_theta_pre_post_pretrain_synthetic_size", 0))
            _pre_synth_online = bool(
                getattr(args, "flow_theta_pre_post_pretrain_resample_synthetic_each_epoch", False)
            )
            _pre_patience = getattr(args, "flow_theta_pre_post_pretrain_early_patience", None)
            _pre_patience_eff = (
                int(getattr(args, "flow_early_patience", 1000))
                if _pre_patience is None
                else int(_pre_patience)
            )
            _pre_synth_msg = (
                (
                    f"online synthetic pretrain pool per_epoch_total={_pre_synth} "
                    "resampled each epoch and split by dataset train fraction"
                )
                if _pre_synth > 0 and _pre_synth_online
                else f"fixed synthetic pretrain pool total={_pre_synth} split by dataset train fraction"
                if _pre_synth > 0
                else "legacy pretrain over the per-n post-training split with per-batch synthetic x resampling"
            )
            if _ft_ep < 1:
                print(
                    "[convergence] theta_flow_pre_post: pretrains the posterior theta-flow on binned Gaussian "
                    "synthetic-pair regularization only (unweighted FM MSE); readout real-data fine-tuning is **skipped** "
                    f"(fine_epochs=0; flow_theta_reg_lambda_metadata={float(getattr(args, 'flow_theta_reg_lambda', 0.01)):.6g} "
                    f"(NPZ only, not applied to pretrain loss); "
                    f"bins={int(getattr(args, 'flow_theta_reg_bin_n_bins', 10))}; "
                    f"pretrain_patience={_pre_patience_eff}; {_pre_synth_msg}).",
                    flush=True,
                )
            else:
                print(
                    "[convergence] theta_flow_pre_post pretrains the posterior theta-flow on binned Gaussian "
                    "synthetic-pair regularization only (unweighted FM MSE), then freezes all but the readout for real-data FM "
                    f"fine-tuning (readout-only real-data FM MSE; flow_theta_reg_lambda_metadata={float(getattr(args, 'flow_theta_reg_lambda', 0.01)):.6g} "
                    f"NPZ-only; "
                    f"bins={int(getattr(args, 'flow_theta_reg_bin_n_bins', 10))}; "
                    f"fine_epochs={_ft_ep}; pretrain_patience={_pre_patience_eff}; {_pre_synth_msg}).",
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
    elif tfm in ("x_flow", "x_flow_reg"):
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_arch={getattr(args, 'flow_arch', 'mlp')}; conditional x-flow only)",
            flush=True,
        )
        if tfm == "x_flow_reg":
            _reg_method = str(getattr(args, "flow_x_reg_prior_method", "binned")).strip().lower()
            _reg_detail = (
                f"equal-width binned Gaussian prior (bins={int(getattr(args, 'flow_x_reg_bin_n_bins', 10))})"
                if _reg_method == "binned"
                else f"KNN Gaussian prior (k={int(getattr(args, 'flow_x_reg_knn_k', 64))})"
            )
            print(
                "[convergence] x_flow_reg mode uses ODE likelihood on x-space flow log p(x|theta) "
                f"after {_reg_detail} velocity-prior regularization "
                f"(lambda={float(getattr(args, 'flow_x_reg_lambda', 0.01)):g}).",
                flush=True,
            )
        else:
            print(
                "[convergence] x_flow mode uses ODE likelihood on x-space flow log p(x|theta) "
                "(no prior model).",
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
    else:
        raise ValueError(
            f"Unsupported --theta-field-method={tfm!r}; use "
            "theta_flow, theta_flow_reg, theta_flow_pre_post, theta_path_integral, x_flow, x_flow_reg, ctsm_v, or nf."
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

    setattr(args, "_convergence_perm_seed", int(perm_seed))
    warm_source_lambda = getattr(args, "warm_start_flow_x_reg_source_lambda", None)
    warm_source_checkpoint_dir: str | None = None
    primary_save_checkpoint_dir: str | None = None
    if warm_source_lambda is not None:
        source_lam_f = float(warm_source_lambda)
        safe_source_lam = f"{source_lam_f:g}".replace("-", "m").replace(".", "p")
        target_lam_f = float(getattr(args, "flow_x_reg_lambda", 0.0))
        safe_target_lam = f"{target_lam_f:g}".replace("-", "m").replace(".", "p")
        warm_source_checkpoint_dir = f"warm_source_lambda{safe_source_lam}_checkpoints"
        primary_save_checkpoint_dir = (
            f"warm_target_lambda{safe_target_lam}_from_lambda{safe_source_lam}_checkpoints"
        )
        source_args = argparse.Namespace(**vars(args))
        setattr(source_args, "theta_field_method", "x_flow_reg")
        setattr(source_args, "flow_x_reg_lambda", source_lam_f)
        setattr(source_args, "_convergence_perm_seed", int(perm_seed))
        print(
            "[convergence] warm-start source sweep enabled: "
            f"--theta-field-method=x_flow_reg --flow-x-reg-lambda={source_lam_f:g} "
            f"--flow-x-reg-prior-method={getattr(source_args, 'flow_x_reg_prior_method', 'binned')}; "
            f"checkpoints -> {os.path.join(args.output_dir, warm_source_checkpoint_dir)}",
            flush=True,
        )
        _run_per_n_method_sweep(
            args=source_args,
            meta=meta,
            bundle=bundle,
            perm=perm,
            ns=ns,
            n_bins=n_bins,
            bin_idx_all=bin_idx_all,
            theta_state_all=theta_state_all,
            h_gt_sqrt=h_gt_sqrt,
            clf_ref=clf_ref,
            llr_gt_mc=llr_gt_mc,
            clf_random_state=clf_rs,
            run_root_name=f"sweep_runs_warm_source_lambda{safe_source_lam}",
            loss_dir_name=f"training_losses_warm_source_lambda{safe_source_lam}",
            sweep_label=f"warm-source-lambda{source_lam_f:g}",
            loss_note=f"from warm-start source sweep lambda={source_lam_f:g}",
            save_checkpoint_dir_name=warm_source_checkpoint_dir,
        )
        print(
            "[convergence] primary sweep will warm-start each n from matching source checkpoint "
            f"and train target lambda={target_lam_f:g}.",
            flush=True,
        )
    primary_sweep = _run_per_n_method_sweep(
        args=args,
        meta=meta,
        bundle=bundle,
        perm=perm,
        ns=ns,
        n_bins=n_bins,
        bin_idx_all=bin_idx_all,
        theta_state_all=theta_state_all,
        h_gt_sqrt=h_gt_sqrt,
        clf_ref=clf_ref,
        llr_gt_mc=llr_gt_mc,
        clf_random_state=clf_rs,
        run_root_name="sweep_runs",
        loss_dir_name="training_losses",
        sweep_label="primary-warm-start" if warm_source_checkpoint_dir is not None else "primary",
        loss_note=(
            f"from warm-start target sweep initialized from {warm_source_checkpoint_dir}"
            if warm_source_checkpoint_dir is not None
            else "from per-n training sweep run"
        ),
        save_checkpoint_dir_name=primary_save_checkpoint_dir,
        init_checkpoint_dir_name=warm_source_checkpoint_dir,
    )
    corr_h = primary_sweep.corr_h
    corr_clf = primary_sweep.corr_clf
    corr_llr = primary_sweep.corr_llr
    wall_s = primary_sweep.wall_s
    h_sweep = primary_sweep.h_sweep
    clf_sweep = primary_sweep.clf_sweep
    llr_sweep = primary_sweep.llr_sweep
    per_n_loss_rows = primary_sweep.per_n_loss_rows
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

    prior_row_lambda = getattr(args, "prior_row_flow_x_reg_lambda", None)
    prior_row_sweep: PerNSweepResult | None = None
    prior_row_h_cols: np.ndarray | None = None
    prior_row_llr_cols: np.ndarray | None = None
    prior_row_label: str | None = None
    if prior_row_lambda is not None:
        prior_lam_f = float(prior_row_lambda)
        _prior_method_label = str(getattr(args, "flow_x_reg_prior_method", "binned")).strip().lower()
        prior_row_label = rf"$\sqrt{{H^2}}$, x-flow-reg {_prior_method_label} $\lambda$={prior_lam_f:g}"
        aux_args = argparse.Namespace(**vars(args))
        setattr(aux_args, "theta_field_method", "x_flow_reg")
        setattr(aux_args, "flow_x_reg_lambda", prior_lam_f)
        setattr(aux_args, "_convergence_perm_seed", int(perm_seed))
        safe_lam = f"{prior_lam_f:g}".replace("-", "m").replace(".", "p")
        aux_run_root = f"sweep_runs_prior_lambda{safe_lam}"
        aux_loss_dir = f"training_losses_prior_lambda{safe_lam}"
        print(
            "[convergence] auxiliary prior row enabled: "
            f"--theta-field-method=x_flow_reg --flow-x-reg-lambda={prior_lam_f:g} "
            f"--flow-x-reg-prior-method={getattr(aux_args, 'flow_x_reg_prior_method', 'binned')}; "
            f"runs -> {os.path.join(args.output_dir, aux_run_root)}",
            flush=True,
        )
        prior_row_sweep = _run_per_n_method_sweep(
            args=aux_args,
            meta=meta,
            bundle=bundle,
            perm=perm,
            ns=ns,
            n_bins=n_bins,
            bin_idx_all=bin_idx_all,
            theta_state_all=None,
            h_gt_sqrt=h_gt_sqrt,
            clf_ref=clf_ref,
            llr_gt_mc=llr_gt_mc,
            clf_random_state=clf_rs,
            run_root_name=aux_run_root,
            loss_dir_name=aux_loss_dir,
            sweep_label=f"prior-row-lambda{prior_lam_f:g}",
            loss_note=f"from auxiliary x_flow_reg prior-row sweep lambda={prior_lam_f:g}",
        )
        prior_row_h_cols = np.stack(prior_row_sweep.h_sweep + [h_ref], axis=0)
        prior_row_llr_cols = np.stack(prior_row_sweep.llr_sweep + [np.asarray(llr_gt_mc, dtype=np.float64)], axis=0)
    h_cols = np.stack(h_sweep + [h_ref], axis=0)
    clf_cols = np.stack(clf_sweep + [clf_ref], axis=0)
    llr_ref = np.asarray(llr_gt_mc, dtype=np.float64)
    llr_cols = np.stack(llr_sweep + [llr_ref], axis=0)
    column_n = np.asarray(list(ns) + [int(args.n_ref)], dtype=np.int64)

    out_npz = os.path.join(args.output_dir, "h_decoding_convergence_results.npz")
    result_payload: dict[str, Any] = {
        "n": np.asarray(ns, dtype=np.int64),
        "corr_h_binned_vs_gt_mc": corr_h,
        "corr_clf_vs_ref": corr_clf,
        "wall_seconds": wall_s,
        "n_ref": np.int64(args.n_ref),
        "perm_seed": np.int64(perm_seed),
        "convergence_base_seed": np.int64(base_seed),
        "dataset_meta_seed": np.int64(meta["seed"]),
        "theta_bin_edges": edges,
        "theta_bin_centers": centers,
        # Legacy key name: sqrt(H^2) from MC; see module docstring.
        "hellinger_gt_sq_mc": h_gt_sqrt,
        "gt_hellinger_n_mc": np.int64(gt_n_mc),
        "gt_hellinger_n_ref_budget": np.int64(args.n_ref),
        "gt_hellinger_seed": np.int64(gt_seed),
        "gt_hellinger_symmetrize": np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        "h_binned_ref_is_gt_mc": np.int32(1),
        "h_binned_columns": h_cols,
        "clf_acc_columns": clf_cols,
        "column_n": column_n,
        "gt_mean_llr_one_sided_mc": np.asarray(llr_gt_mc, dtype=np.float64),
        "llr_binned_columns": llr_cols,
        "corr_llr_binned_vs_gt_mc": corr_llr,
        "theta_field_method": np.asarray([tfm], dtype=object),
        "binned_gaussian_h_binned_columns": binned_gaussian_h_cols,
        "binned_gaussian_corr_h_binned_vs_gt_mc": binned_gaussian_corr_h,
        "binned_gaussian_variance_floor": np.float64(binned_gaussian_variance_floor),
        "binned_gaussian_label": np.asarray([binned_gaussian_label], dtype=object),
    }
    if warm_source_lambda is not None:
        result_payload.update(
            {
                "warm_start_flow_x_reg_source_lambda": np.float64(float(warm_source_lambda)),
                "warm_start_flow_x_reg_target_lambda": np.float64(
                    float(getattr(args, "flow_x_reg_lambda", 0.0))
                ),
                "warm_start_source_checkpoint_dir": np.asarray(
                    [os.path.abspath(os.path.join(args.output_dir, warm_source_checkpoint_dir or ""))],
                    dtype=object,
                ),
                "warm_start_target_checkpoint_dir": np.asarray(
                    [os.path.abspath(os.path.join(args.output_dir, primary_save_checkpoint_dir or ""))],
                    dtype=object,
                ),
            }
        )
    if prior_row_sweep is not None and prior_row_h_cols is not None:
        result_payload.update(
            {
                "prior_row_h_binned_columns": prior_row_h_cols,
                "prior_row_corr_h_binned_vs_gt_mc": prior_row_sweep.corr_h,
                "prior_row_wall_seconds": prior_row_sweep.wall_s,
                "prior_row_llr_binned_columns": prior_row_llr_cols,
                "prior_row_corr_llr_binned_vs_gt_mc": prior_row_sweep.corr_llr,
                "prior_row_flow_x_reg_lambda": np.float64(float(prior_row_lambda)),
                "prior_row_label": np.asarray([prior_row_label or ""], dtype=object),
            }
        )
    np.savez_compressed(out_npz, **result_payload)

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
        err_msg=[],
        visualization_only=False,
        llr_cols=llr_cols,
        corr_llr=corr_llr,
        binned_gaussian_h_cols=binned_gaussian_h_cols,
        binned_gaussian_corr_h=binned_gaussian_corr_h,
        binned_gaussian_label=binned_gaussian_label,
        binned_gaussian_variance_floor=binned_gaussian_variance_floor,
        prior_row_h_cols=prior_row_h_cols,
        prior_row_label=prior_row_label,
        prior_row_loss_rows=None if prior_row_sweep is None else prior_row_sweep.per_n_loss_rows,
    )


if __name__ == "__main__":
    main()
