#!/usr/bin/env python3
"""Per-method estimation helpers for H-decoding convergence."""

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
    ContrastiveGaussianNetworkScorer,
    ContrastiveIndependentDotProductScorer,
    ContrastiveIndependentGaussianScorer,
    ContrastiveLLRMLP,
    ContrastiveNormalizedDotBiasScorer,
    ContrastiveNormalizedDotScorer,
    compute_contrastive_c_matrix,
    compute_contrastive_soft_c_matrix,
    contrastive_soft_metadata_without_training,
    dot_scorer_augmented_theta_dim,
    h_directed_from_delta_l as compute_h_directed_contrastive,
    normalize_theta_encoding as normalize_contrastive_theta_encoding,
    theta_dim_for_encoding as contrastive_theta_dim_for_encoding,
    train_bidir_contrastive_soft_llr,
    train_contrastive_llr,
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
from fisher.h_decoding_convergence_cli import (
    _normalize_contrastive_flow_method,
    _normalize_contrastive_method,
    _normalize_flow_autoencoder_method,
    _normalize_flow_pca_method,
    _normalize_gaussian_network_method,
    _normalize_gaussian_x_flow_method,
    _normalize_linear_theta_flow_method,
    _normalize_linear_x_flow_method,
    _normalize_sir_wrapper_method,
)


# Isolated ``h_decoding_convergence.{png,svg}`` size; combined figure uses the same width/height
# ratio for the right-hand curve column (column height matches the matrix panel height).
_H_DECODING_CURVE_FIGSIZE_IN: tuple[float, float] = (3.5, 3.5)

def theta_segment_ids_equal_width(
    theta: np.ndarray,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Delegate to ``visualize_h_matrix_binned.theta_segment_ids_equal_width``."""
    return vhb.theta_segment_ids_equal_width(theta, n_segments)

def _build_theta_fourier_state(
    theta_all_in: np.ndarray,
    *,
    theta_ref: np.ndarray,
    k: int,
    period_mult: float,
    include_linear: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic Fourier theta-state vectors from scalar or multi-dimensional theta.

    Accepts ``theta`` as 1D ``(N,)``, ``(N, 1)``, or ``(N, d_theta)``. Reference slice ``theta_ref``
    must have the same width ``d_theta``. For each coordinate independently (dimension-major order):

    - Optional scaled linear term ``(theta - center) / max(range, 1e-12)`` when ``include_linear``.
    - Harmonics ``sin(k * omega * shift)``, ``cos(...)`` for ``k = 1..K`` with
      ``omega = 2*pi / (period_mult * max(range, 1e-12))``.

    Output width is ``d_theta * ((1 if include_linear else 0) + 2*K)``. For ``d_theta == 1`` this
    matches the legacy scalar construction exactly.

    Returns ``(state, ref_range_vec, period_vec, center_vec)`` each of shape ``(d_theta,)`` for the
    metadata vectors (train-reference ranges, periods, centers per coordinate).
    """
    th = np.asarray(theta_all_in, dtype=np.float64)
    if th.ndim == 1:
        th = th.reshape(-1, 1)
    elif th.ndim != 2:
        raise ValueError(f"_build_theta_fourier_state expects theta of shape (N,), (N,1), or (N,d); got {th.shape}")

    tr = np.asarray(theta_ref, dtype=np.float64)
    if tr.ndim == 1:
        tr = tr.reshape(-1, 1)
    elif tr.ndim != 2:
        raise ValueError(f"_build_theta_fourier_state expects theta_ref 1D or 2D; got {tr.shape}")

    if int(th.shape[1]) != int(tr.shape[1]):
        raise ValueError(f"theta width {th.shape[1]} != theta_ref width {tr.shape[1]}.")
    if th.shape[0] < 1 or tr.shape[0] < 1:
        raise ValueError("Fourier theta state requires non-empty theta arrays.")

    kk = int(k)
    if kk < 1:
        raise ValueError("Fourier theta state requires K >= 1 harmonic pairs.")
    pm = float(period_mult)

    d_theta = int(th.shape[1])
    cols: list[np.ndarray] = []
    ref_ranges: list[float] = []
    periods: list[float] = []
    centers: list[float] = []

    for j in range(d_theta):
        ref_min = float(np.min(tr[:, j]))
        ref_max = float(np.max(tr[:, j]))
        ref_range = max(ref_max - ref_min, 1e-12)
        period = pm * ref_range
        center = 0.5 * (ref_min + ref_max)
        shift = th[:, j] - center
        w0 = 2.0 * np.pi / period

        ref_ranges.append(ref_range)
        periods.append(period)
        centers.append(center)

        if include_linear:
            cols.append((shift / ref_range).reshape(-1, 1))
        for h in range(1, kk + 1):
            phase = float(h) * w0 * shift
            cols.append(np.sin(phase).reshape(-1, 1))
            cols.append(np.cos(phase).reshape(-1, 1))

    out = np.concatenate(cols, axis=1).astype(np.float64, copy=False)
    ref_arr = np.asarray(ref_ranges, dtype=np.float64)
    per_arr = np.asarray(periods, dtype=np.float64)
    cen_arr = np.asarray(centers, dtype=np.float64)
    return out, ref_arr, per_arr, cen_arr


def theta_phys_rows_and_ref_for_fourier(
    theta_raw_all: np.ndarray,
    perm: np.ndarray,
    n_ref: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape raw dataset theta to ``(N, d)`` and take the same n_ref prefix as binning (phys coords for Fourier)."""
    th = np.asarray(theta_raw_all, dtype=np.float64)
    if th.ndim == 1:
        th = th.reshape(-1, 1)
    elif th.ndim != 2:
        raise ValueError(
            "Fourier theta state requires theta_all as 1D, (N,1), or (N,d); " f"got shape {th.shape}."
        )
    nref = int(n_ref)
    if nref < 1:
        raise ValueError("Fourier reference slice requires n_ref >= 1.")
    pref = th[np.asarray(perm, dtype=np.int64)[:nref]]
    return th, pref


def format_theta_fourier_state_log_message(
    *,
    tag: str,
    state_dim: int,
    k: int,
    ref_range_vec: np.ndarray,
    period_vec: np.ndarray,
    center_vec: np.ndarray,
    period_mult: float,
    include_linear: bool,
) -> str:
    """One-line log for multi-coordinate Fourier state (scalars when ``d==1``)."""
    rr = np.asarray(ref_range_vec, dtype=np.float64).reshape(-1)
    pp = np.asarray(period_vec, dtype=np.float64).reshape(-1)
    cc = np.asarray(center_vec, dtype=np.float64).reshape(-1)

    def _fmt(name: str, v: np.ndarray) -> str:
        if v.size == 1:
            return f"{name}={float(v[0]):.6g}"
        inner = ", ".join(f"{float(x):.6g}" for x in v.tolist())
        return f"{name}=[{inner}]"

    return (
        f"{tag} theta_flow Fourier state enabled: "
        f"dim={int(state_dim)} K={int(k)} "
        f"{_fmt('period', pp)} "
        f"(mult={float(period_mult):.3g}, {_fmt('ref_range', rr)}, {_fmt('center', cc)}, "
        f"include_linear={bool(include_linear)})"
    )


def contrastive_soft_fourier_settings_from_theta_flow_args(args: Any) -> tuple[int, float, bool]:
    """Map canonical ``--theta-flow-fourier-*`` CLI to contrastive-soft Fourier hyperparameters.

    When ``theta_flow_fourier_state`` is false, returns ``(0, period_mult, False)`` so the dot-family
    theta branch uses scalar coordinates only (``k=0`` disables harmonics in ``contrastive_llr``).
    """
    if not bool(getattr(args, "theta_flow_fourier_state", False)):
        return 0, float(getattr(args, "theta_flow_fourier_period_mult", 2.0)), False
    return (
        int(getattr(args, "theta_flow_fourier_k", 4)),
        float(getattr(args, "theta_flow_fourier_period_mult", 2.0)),
        bool(getattr(args, "theta_flow_fourier_include_linear", False)),
    )


def _build_lxf_theta_fourier_features(
    theta: np.ndarray,
    *,
    theta_ref: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-dimension theta features [centered linear, sin/cos harmonics]."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    theta_ref_arr = np.asarray(theta_ref, dtype=np.float64)
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)
    if theta_ref_arr.ndim == 1:
        theta_ref_arr = theta_ref_arr.reshape(-1, 1)
    if theta_arr.ndim != 2 or theta_ref_arr.ndim != 2:
        raise ValueError("LXF Fourier theta features require 1D or 2D theta arrays.")
    if theta_arr.shape[1] != theta_ref_arr.shape[1]:
        raise ValueError("LXF Fourier theta feature dimension mismatch.")
    if theta_arr.shape[0] < 1 or theta_ref_arr.shape[0] < 1:
        raise ValueError("LXF Fourier theta features require non-empty theta arrays.")
    kk = int(k)
    if kk < 1:
        raise ValueError("--lxf-theta-fourier-k must be >= 1 for linear_x_flow *_p methods.")

    ref_min = np.min(theta_ref_arr, axis=0)
    ref_max = np.max(theta_ref_arr, axis=0)
    ref_range = np.maximum(ref_max - ref_min, 1e-12)
    center = 0.5 * (ref_min + ref_max)
    period = ref_range.copy()
    theta_shift = theta_arr - center.reshape(1, -1)
    theta_scaled = theta_shift / ref_range.reshape(1, -1)
    w0 = (2.0 * np.pi) / period

    cols: list[np.ndarray] = []
    for j in range(theta_arr.shape[1]):
        cols.append(theta_scaled[:, j : j + 1])
        for h in range(1, kk + 1):
            phase = float(h) * float(w0[j]) * theta_shift[:, j : j + 1]
            cols.append(np.sin(phase))
            cols.append(np.cos(phase))
    return np.concatenate(cols, axis=1).astype(np.float64, copy=False), center, ref_range, period


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


def prepare_categorical_binning_for_convergence(
    theta_raw_all: np.ndarray,
    num_categories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Use category IDs directly as H/decoding bins.

    Accepts current one-hot theta rows ``(N, K)`` and legacy scalar integer labels
    ``(N, 1)`` / ``(N,)``.
    """
    k = int(num_categories)
    if k < 2:
        raise ValueError("Categorical binning requires num_categories >= 2.")
    theta_arr = np.asarray(theta_raw_all, dtype=np.float64)
    if theta_arr.ndim == 2 and int(theta_arr.shape[1]) == k:
        row_sums = theta_arr.sum(axis=1)
        is_binary = np.all((np.abs(theta_arr) <= 1e-6) | (np.abs(theta_arr - 1.0) <= 1e-6), axis=1)
        if np.any(np.abs(row_sums - 1.0) > 1e-6) or not bool(np.all(is_binary)):
            raise ValueError("Categorical one-hot theta rows must contain one 1 and otherwise 0s.")
        labels = np.argmax(theta_arr, axis=1).astype(np.int64)
        label_desc = "one-hot labels"
    else:
        if theta_arr.ndim == 2 and int(theta_arr.shape[1]) != 1:
            raise ValueError(
                f"Categorical theta must have shape (N, 1) or one-hot shape (N, {k}); got {theta_arr.shape}."
            )
        flat = theta_arr.reshape(-1)
        labels = np.rint(flat).astype(np.int64)
        if np.any(np.abs(flat - labels.astype(np.float64)) > 1e-6):
            raise ValueError("Categorical theta labels must be integer-valued.")
        label_desc = "integer labels"
    if np.any((labels < 0) | (labels >= k)):
        raise ValueError(f"Categorical theta labels must be in [0, {k - 1}].")
    edges = np.arange(k + 1, dtype=np.float64) - 0.5
    print(
        f"[convergence] categorical theta: using {label_desc} as {k} category bins.",
        flush=True,
    )
    return labels.astype(np.float64), labels.astype(np.float64), edges, float(edges[0]), float(edges[-1]), labels


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


def _encode_x_contrastive_normalized_dot(
    model: ContrastiveNormalizedDotScorer,
    x_norm: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Apply encode_x in batches (x already standardized like contrastive training)."""
    model.eval()
    x_norm = np.asarray(x_norm, dtype=np.float64)
    if x_norm.ndim != 2:
        raise ValueError("x_norm must be 2D.")
    out: list[np.ndarray] = []
    n = int(x_norm.shape[0])
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            xt = torch.from_numpy(x_norm[i0:i1].astype(np.float32)).to(device)
            z = model.encode_x(xt).detach().cpu().numpy().astype(np.float64, copy=False)
            out.append(z)
    return np.concatenate(out, axis=0)


def _train_contrastive_soft_and_encode_bundle(
    *,
    args: argparse.Namespace,
    bundle: SharedDatasetBundle,
    device: torch.device,
    theta_binning_mode: str,
) -> tuple[SharedDatasetBundle, dict[str, Any], int]:
    """Train normalized-dot contrastive-soft encoder; replace bundle x with L2-normalized z."""
    if str(theta_binning_mode).strip().lower() == "theta2_grid":
        raise ValueError("contrastive_theta_flow / contrastive_x_flow v1 require theta1 (--theta-binning-mode theta1).")
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
        raise ValueError("contrastive+flow v1 requires scalar theta (shape (N,1)).")
    if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
        raise ValueError("contrastive+flow expects x arrays to be 2D.")
    if theta_train.shape[0] < 2 or theta_val.shape[0] < 2:
        raise ValueError("contrastive+flow requires at least two train and two validation rows.")
    embed_dim = int(getattr(args, "contrastive_soft_dot_dim", 10))
    if embed_dim < 1:
        raise ValueError("--contrastive-soft-dot-dim must be >= 1 for contrastive+flow.")
    hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
    depth = int(getattr(args, "contrastive_depth", 3))
    _fk, _pm, _inc = contrastive_soft_fourier_settings_from_theta_flow_args(args)
    theta_in_dim = dot_scorer_augmented_theta_dim(fourier_k=int(_fk), fourier_include_linear=bool(_inc))
    model = ContrastiveNormalizedDotScorer(
        x_dim=int(x_all.shape[1]),
        theta_dim=int(theta_in_dim),
        feature_dim=embed_dim,
        hidden_dim=hidden_dim,
        depth=depth,
    ).to(device)
    print(
        f"[contrastive+flow] contrastive-soft normalized_dot encoder: x_dim={int(x_all.shape[1])} "
        f"z_dim={embed_dim} theta_in_dim={int(theta_in_dim)}",
        flush=True,
    )
    train_out = train_contrastive_soft_llr(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        epochs=int(getattr(args, "contrastive_epochs", 2000)),
        batch_size=int(getattr(args, "contrastive_batch_size", 256)),
        lr=float(getattr(args, "contrastive_lr", 1e-3)),
        bandwidth_bins=int(getattr(args, "contrastive_soft_bandwidth_bins", 10)),
        bandwidth_start=float(getattr(args, "contrastive_soft_bandwidth_start", 0.0)),
        bandwidth_end=float(getattr(args, "contrastive_soft_bandwidth_end", 0.0)),
        periodic=bool(getattr(args, "contrastive_soft_periodic", False)),
        period=float(getattr(args, "contrastive_soft_period", 2.0 * np.pi)),
        weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
        patience=int(getattr(args, "contrastive_early_patience", 300)),
        min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
        ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
        max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
        log_every=max(1, int(getattr(args, "log_every", 50))),
        restore_best=True,
        contrastive_theta_fourier_k=int(_fk),
        contrastive_theta_fourier_period_mult=float(_pm),
        contrastive_theta_fourier_include_linear=bool(_inc),
    )
    x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
    x_std = np.asarray(train_out["x_std"], dtype=np.float64)
    x_train_n = (x_train - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    x_val_n = (x_val - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    x_all_n = (x_all - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    enc_bs = max(int(getattr(args, "contrastive_batch_size", 256)), 1)
    z_train = _encode_x_contrastive_normalized_dot(
        model, x_train_n, device=device, batch_size=enc_bs
    )
    z_val = _encode_x_contrastive_normalized_dot(
        model, x_val_n, device=device, batch_size=enc_bs
    )
    z_all = _encode_x_contrastive_normalized_dot(
        model, x_all_n, device=device, batch_size=enc_bs
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
    return encoded_bundle, train_out, embed_dim


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


def _fit_sir_projection(
    *,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    x_val: np.ndarray,
    x_all: np.ndarray,
    sir_dim: int,
    num_bins: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | int | float]]:
    """Fit SIR on train data and project train/val/all observations."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    th_tr = np.asarray(theta_train, dtype=np.float64)
    x_va = np.asarray(x_val, dtype=np.float64)
    x_full = np.asarray(x_all, dtype=np.float64)
    if th_tr.ndim == 1:
        th_tr = th_tr.reshape(-1, 1)
    if x_tr.ndim != 2 or x_va.ndim != 2 or x_full.ndim != 2:
        raise ValueError("SIR expects x_train, x_val, and x_all to be 2D.")
    if th_tr.ndim != 2:
        raise ValueError("SIR expects theta_train to be 1D or 2D.")
    if x_tr.shape[0] != th_tr.shape[0]:
        raise ValueError("SIR theta_train length must match x_train rows.")
    if x_tr.shape[1] != x_va.shape[1] or x_tr.shape[1] != x_full.shape[1]:
        raise ValueError("SIR x dimension mismatch.")
    m = int(sir_dim)
    nb = int(num_bins)
    lam = float(ridge)
    if m < 1:
        raise ValueError("--sir-dim must be >= 1.")
    if nb < 2:
        raise ValueError("--sir-num-bins must be >= 2.")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("--sir-ridge must be finite and > 0.")
    x_dim = int(x_tr.shape[1])
    if m > x_dim:
        raise ValueError(f"--sir-dim must be <= x_dim={x_dim}; got {m}.")

    theta_dim = int(th_tr.shape[1])
    per_dim_bins = np.zeros((int(th_tr.shape[0]), theta_dim), dtype=np.int64)
    theta_edges = np.zeros((theta_dim, nb + 1), dtype=np.float64)
    for j in range(theta_dim):
        col = np.asarray(th_tr[:, j], dtype=np.float64)
        if not np.all(np.isfinite(col)):
            raise ValueError("SIR theta_train contains non-finite values.")
        lo = float(np.min(col))
        hi = float(np.max(col))
        if hi <= lo:
            raise ValueError(f"SIR theta dimension {j} is constant; cannot form equal-width bins.")
        edges = np.linspace(lo, hi, nb + 1, dtype=np.float64)
        theta_edges[j] = edges
        idx = np.searchsorted(edges, col, side="right") - 1
        per_dim_bins[:, j] = np.clip(idx, 0, nb - 1).astype(np.int64, copy=False)

    multipliers = (nb ** np.arange(theta_dim, dtype=np.int64)).astype(np.int64, copy=False)
    joint_ids = np.sum(per_dim_bins * multipliers.reshape(1, -1), axis=1, dtype=np.int64)
    nonempty_ids, inverse, counts = np.unique(joint_ids, return_inverse=True, return_counts=True)
    n_slices = int(nonempty_ids.size)
    if n_slices < 2:
        raise ValueError("SIR projection requires at least two non-empty theta bins in the train split.")
    max_rank = min(x_dim, n_slices - 1)
    if m > max_rank:
        raise ValueError(
            f"--sir-dim={m} exceeds available SIR rank {max_rank} "
            f"(non_empty_bins={n_slices}, x_dim={x_dim})."
        )

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    centered = x_tr - x_mean
    cov = (centered.T @ centered) / max(1, int(x_tr.shape[0]) - 1)
    cov_reg = cov + lam * np.eye(x_dim, dtype=np.float64)
    slice_means = np.zeros((n_slices, x_dim), dtype=np.float64)
    for h in range(n_slices):
        slice_means[h] = np.mean(x_tr[inverse == h], axis=0, dtype=np.float64)
    probs = counts.astype(np.float64) / float(x_tr.shape[0])
    mean_diff = slice_means - x_mean.reshape(1, -1)
    between = (mean_diff.T * probs.reshape(1, -1)) @ mean_diff

    chol = np.linalg.cholesky(cov_reg)
    whitened = np.linalg.solve(chol, between)
    whitened = np.linalg.solve(chol, whitened.T).T
    whitened = 0.5 * (whitened + whitened.T)
    eigvals, eigvecs = np.linalg.eigh(whitened)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.asarray(eigvals[order], dtype=np.float64)
    eigvecs = np.asarray(eigvecs[:, order], dtype=np.float64)
    components = np.linalg.solve(chol.T, eigvecs[:, :m]).astype(np.float64, copy=False)

    z_train = centered @ components
    z_val = (x_va - x_mean) @ components
    z_all = (x_full - x_mean) @ components
    meta: dict[str, np.ndarray | int | float] = {
        "sir_dim": np.int64(m),
        "sir_num_bins": np.int64(nb),
        "sir_ridge": np.float64(lam),
        "sir_theta_dim": np.int64(theta_dim),
        "sir_x_mean": x_mean.astype(np.float64, copy=False),
        "sir_components": components,
        "sir_eigenvalues": np.asarray(eigvals[:m], dtype=np.float64),
        "sir_bin_counts": np.asarray(counts, dtype=np.int64),
        "sir_nonempty_bin_ids": np.asarray(nonempty_ids, dtype=np.int64),
        "sir_slice_means": slice_means,
        "sir_theta_edges": theta_edges,
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
    decode_x_train: np.ndarray | None = None,
    decode_x_all: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise bin decoding: train on NPZ train rows, accuracy on NPZ full pool.

    Optional ``decode_x_train`` / ``decode_x_all`` override observation features (e.g. native
    pre-PR ``x`` for GT decoding while flows still train on embedded ``x``).
    """
    x_tr = subset.bundle.x_train if decode_x_train is None else decode_x_train
    x_ev = subset.bundle.x_all if decode_x_all is None else decode_x_all
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_ev = np.asarray(x_ev, dtype=np.float64)
    if int(x_tr.shape[0]) != int(subset.bin_train.shape[0]):
        raise ValueError(
            f"decode_x_train rows {x_tr.shape[0]} != bin_train {subset.bin_train.shape[0]}."
        )
    if int(x_ev.shape[0]) != int(subset.bin_all.shape[0]):
        raise ValueError(f"decode_x_all rows {x_ev.shape[0]} != bin_all {subset.bin_all.shape[0]}.")
    clf_acc, _, _, _ = vhb.pairwise_bin_logistic_accuracy_train_val(
        x_tr,
        subset.bin_train,
        x_ev,
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
    def contrastive_bin_likelihood_h_payload(method_label: str, c_matrix: np.ndarray) -> dict[str, np.ndarray]:
        if bin_all is None:
            raise ValueError(
                f"{method_label} requires bin_all to construct bin-marginal contrastive H; "
                "pass theta bin labels aligned with the evaluation pool."
            )
        bin_h = lxf_bin_likelihood_hellinger(c_matrix, np.asarray(bin_all, dtype=np.int64), int(n_bins))
        return {
            "h_sym_bin_likelihood": np.asarray(bin_h["h_sym"], dtype=np.float64),
            "bin_log_likelihood": np.asarray(bin_h["bin_log_likelihood"], dtype=np.float64),
            "bin_log_likelihood_matrix": np.asarray(bin_h["bin_log_likelihood"], dtype=np.float64),
            "bin_delta_l_matrix": np.asarray(bin_h["bin_delta_l"], dtype=np.float64),
            "h_directed_bin": np.asarray(bin_h["h_directed_bin"], dtype=np.float64),
            "h_directed_bin_likelihood": np.asarray(bin_h["h_directed_bin"], dtype=np.float64),
            "h_binned_direct": np.asarray(bin_h["h_binned"], dtype=np.float64),
            "h_binned_bin_likelihood": np.asarray(bin_h["h_binned"], dtype=np.float64),
            "bin_counts": np.asarray(bin_h["bin_counts"], dtype=np.int64),
        }

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
        if bin_all is None:
            raise ValueError("contrastive-soft-gaussian-net requires theta bin labels from the convergence subset.")

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
        bw_bins = int(getattr(args, "contrastive_soft_bandwidth_bins", 10))
        bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
        bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
        periodic = bool(getattr(args, "contrastive_soft_periodic", False))
        period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
        _fk_gn, _pm_gn, _inc_gn = contrastive_soft_fourier_settings_from_theta_flow_args(args)
        if contrastive_norm == "contrastive_soft_gaussian_net_no_finetune":
            train_out = contrastive_soft_metadata_without_training(
                theta_train=theta_train,
                x_train=x_train,
                bandwidth_bins=bw_bins,
                bandwidth_start=bw_start,
                bandwidth_end=bw_end,
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
                bandwidth_bins=bw_bins,
                bandwidth_start=bw_start,
                bandwidth_end=bw_end,
                periodic=periodic,
                period=period,
                weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
                patience=int(getattr(args, "contrastive_early_patience", 300)),
                min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
                ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
                max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
                log_every=max(1, int(getattr(args, "log_every", 50))),
                restore_best=True,
                contrastive_theta_fourier_k=int(_fk_gn),
                contrastive_theta_fourier_period_mult=float(_pm_gn),
                contrastive_theta_fourier_include_linear=bool(_inc_gn),
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
            contrastive_theta_fourier_k=int(train_out.get("contrastive_theta_fourier_k", 0)),
            contrastive_theta_fourier_period_mult=float(train_out.get("contrastive_theta_fourier_period_mult", 2.0)),
            contrastive_theta_fourier_include_linear=bool(
                train_out.get("contrastive_theta_fourier_include_linear", False)
            ),
            theta_fourier_ref=(
                theta_train
                if int(train_out.get("contrastive_theta_fourier_k", 0)) > 0
                else None
            ),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        bin_h_payload = contrastive_bin_likelihood_h_payload(method_name, c_matrix)
        h_sym = bin_h_payload["h_sym_bin_likelihood"]
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
            **bin_h_payload,
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
            contrastive_soft_bandwidth_bins=np.int64(train_out["bandwidth_bins"]),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
            contrastive_theta_fourier_k=np.int64(train_out.get("contrastive_theta_fourier_k", 0)),
            contrastive_theta_fourier_period_mult=np.float64(train_out.get("contrastive_theta_fourier_period_mult", 2.0)),
            contrastive_theta_fourier_include_linear=np.bool_(train_out.get("contrastive_theta_fourier_include_linear", False)),
            theta_fourier_ref_min=np.float64(train_out.get("theta_fourier_ref_min", float("nan"))),
            theta_fourier_ref_max=np.float64(train_out.get("theta_fourier_ref_max", float("nan"))),
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
        if bin_all is None:
            raise ValueError("bidir-contrastive-soft requires theta bin labels from the convergence subset.")

        hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
        depth = int(getattr(args, "contrastive_depth", 3))
        dot_dim = int(getattr(args, "contrastive_soft_dot_dim", 10))
        soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
        _fk_bidir, _pm_bidir, _inc_bidir = contrastive_soft_fourier_settings_from_theta_flow_args(args)
        theta_in_dim = dot_scorer_augmented_theta_dim(
            fourier_k=int(_fk_bidir),
            fourier_include_linear=bool(_inc_bidir),
        )
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
                theta_dim=int(theta_in_dim),
                feature_dim=dot_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        bw_bins = int(getattr(args, "contrastive_soft_bandwidth_bins", 10))
        bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
        bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
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
            bandwidth_bins=bw_bins,
            bandwidth_start=bw_start,
            bandwidth_end=bw_end,
            periodic=periodic,
            period=period,
            weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
            patience=int(getattr(args, "contrastive_early_patience", 300)),
            min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
            contrastive_theta_fourier_k=int(_fk_bidir),
            contrastive_theta_fourier_period_mult=float(_pm_bidir),
            contrastive_theta_fourier_include_linear=bool(_inc_bidir),
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
            contrastive_theta_fourier_k=int(train_out.get("contrastive_theta_fourier_k", 0)),
            contrastive_theta_fourier_period_mult=float(train_out.get("contrastive_theta_fourier_period_mult", 2.0)),
            contrastive_theta_fourier_include_linear=bool(
                train_out.get("contrastive_theta_fourier_include_linear", False)
            ),
            theta_fourier_ref=(
                theta_train
                if int(train_out.get("contrastive_theta_fourier_k", 0)) > 0
                else None
            ),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        bin_h_payload = contrastive_bin_likelihood_h_payload(method_name, c_matrix)
        h_sym = bin_h_payload["h_sym_bin_likelihood"]
        theta_used = theta_all.reshape(-1)

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            **bin_h_payload,
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
            contrastive_soft_bandwidth_bins=np.int64(train_out["bandwidth_bins"]),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
            contrastive_theta_fourier_k=np.int64(train_out.get("contrastive_theta_fourier_k", 0)),
            contrastive_theta_fourier_period_mult=np.float64(train_out.get("contrastive_theta_fourier_period_mult", 2.0)),
            contrastive_theta_fourier_include_linear=np.bool_(train_out.get("contrastive_theta_fourier_include_linear", False)),
            theta_fourier_ref_min=np.float64(train_out.get("theta_fourier_ref_min", float("nan"))),
            theta_fourier_ref_max=np.float64(train_out.get("theta_fourier_ref_max", float("nan"))),
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
        d_theta = int(theta_train.shape[1])
        if int(theta_val.shape[1]) != d_theta or int(theta_all.shape[1]) != d_theta:
            raise ValueError("contrastive-soft requires matching theta width across train, validation, and all splits.")
        if x_train.ndim != 2 or x_val.ndim != 2 or x_all.ndim != 2:
            raise ValueError("contrastive-soft expects x arrays to be 2D.")
        if theta_train.shape[0] < 2 or theta_val.shape[0] < 2:
            raise ValueError("contrastive-soft requires at least two train and two validation rows.")
        if bin_all is None:
            raise ValueError("contrastive-soft requires theta bin labels from the convergence subset.")

        hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
        depth = int(getattr(args, "contrastive_depth", 3))
        soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
        dot_dim = int(getattr(args, "contrastive_soft_dot_dim", 10))
        coord_embed_dim = int(getattr(args, "contrastive_soft_coordinate_embed_dim", 16))
        gaussian_logvar_min = float(getattr(args, "contrastive_soft_gaussian_logvar_min", -8.0))
        gaussian_logvar_max = float(getattr(args, "contrastive_soft_gaussian_logvar_max", 5.0))
        _fk_soft, _pm_soft, _inc_soft = contrastive_soft_fourier_settings_from_theta_flow_args(args)
        # Bundled Fourier θ from twofig/convergence (--theta-flow-fourier-state): θ rows are already the
        # Fourier feature vector; do not append harmonics again inside contrastive-soft training.
        bundled_fourier_theta = bool(getattr(args, "theta_flow_fourier_state", False)) and int(d_theta) > 1
        if bundled_fourier_theta:
            train_fourier_k = 0
            theta_in_dim = int(d_theta)
        elif int(_fk_soft) > 0:
            train_fourier_k = int(_fk_soft)
            theta_in_dim = int(
                dot_scorer_augmented_theta_dim(fourier_k=int(_fk_soft), fourier_include_linear=bool(_inc_soft))
            )
        else:
            train_fourier_k = 0
            theta_in_dim = int(d_theta)
        if soft_arch == "normalized_dot":
            model = ContrastiveNormalizedDotScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=int(theta_in_dim),
                feature_dim=dot_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        elif soft_arch == "additive_independent":
            model = ContrastiveAdditiveIndependentScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=int(theta_in_dim),
                feature_dim=dot_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        elif soft_arch == "independent_gaussian":
            model = ContrastiveIndependentGaussianScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=int(theta_in_dim),
                hidden_dim=hidden_dim,
                depth=depth,
                logvar_min=gaussian_logvar_min,
                logvar_max=gaussian_logvar_max,
            ).to(dev)
        elif soft_arch == "independent_dot_product":
            model = ContrastiveIndependentDotProductScorer(
                x_dim=int(x_all.shape[1]),
                theta_dim=int(theta_in_dim),
                feature_dim=dot_dim,
                coord_embed_dim=coord_embed_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        elif soft_arch == "mlp":
            model = ContrastiveLLRMLP(
                x_dim=int(x_all.shape[1]),
                theta_dim=int(theta_in_dim),
                hidden_dim=hidden_dim,
                depth=depth,
            ).to(dev)
        else:
            raise ValueError(
                "--contrastive-soft-score-arch must be one of "
                "{'normalized_dot','additive_independent','independent_gaussian','independent_dot_product','mlp'}."
            )
        bw_bins = int(getattr(args, "contrastive_soft_bandwidth_bins", 10))
        bw_start = float(getattr(args, "contrastive_soft_bandwidth_start", 0.0))
        bw_end = float(getattr(args, "contrastive_soft_bandwidth_end", 0.0))
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
            bandwidth_bins=bw_bins,
            bandwidth_start=bw_start,
            bandwidth_end=bw_end,
            periodic=periodic,
            period=period,
            weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
            patience=int(getattr(args, "contrastive_early_patience", 300)),
            min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
            ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
            max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
            log_every=max(1, int(getattr(args, "log_every", 50))),
            restore_best=True,
            contrastive_theta_fourier_k=int(train_fourier_k),
            contrastive_theta_fourier_period_mult=float(_pm_soft),
            contrastive_theta_fourier_include_linear=bool(_inc_soft),
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
            contrastive_theta_fourier_k=int(train_out.get("contrastive_theta_fourier_k", 0)),
            contrastive_theta_fourier_period_mult=float(train_out.get("contrastive_theta_fourier_period_mult", 2.0)),
            contrastive_theta_fourier_include_linear=bool(
                train_out.get("contrastive_theta_fourier_include_linear", False)
            ),
            theta_fourier_ref=(
                theta_train
                if int(train_out.get("contrastive_theta_fourier_k", 0)) > 0
                else None
            ),
        )
        delta_l = compute_delta_l_nf(c_matrix)
        bin_h_payload = contrastive_bin_likelihood_h_payload(method_name, c_matrix)
        h_sym = bin_h_payload["h_sym_bin_likelihood"]
        theta_used = theta_all.reshape(-1) if d_theta == 1 else np.asarray(theta_all, dtype=np.float64)

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            **bin_h_payload,
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
            contrastive_soft_bandwidth_bins=np.int64(train_out["bandwidth_bins"]),
            contrastive_theta_dim=np.int64(d_theta),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=x_mean,
            contrastive_x_std=x_std,
            contrastive_theta_mean=theta_mean,
            contrastive_theta_std=theta_std,
            contrastive_theta_fourier_k=np.int64(train_out.get("contrastive_theta_fourier_k", 0)),
            contrastive_theta_fourier_period_mult=np.float64(train_out.get("contrastive_theta_fourier_period_mult", 2.0)),
            contrastive_theta_fourier_include_linear=np.bool_(train_out.get("contrastive_theta_fourier_include_linear", False)),
            theta_fourier_ref_min=np.float64(train_out.get("theta_fourier_ref_min", float("nan"))),
            theta_fourier_ref_max=np.float64(train_out.get("theta_fourier_ref_max", float("nan"))),
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
        bin_h_payload = contrastive_bin_likelihood_h_payload(method_name, c_matrix)
        h_sym = bin_h_payload["h_sym_bin_likelihood"]
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()

        np.savez_compressed(
            os.path.join(output_dir, "h_matrix_results_theta_cov.npz"),
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            c_matrix=np.asarray(c_matrix, dtype=np.float64),
            delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
            **bin_h_payload,
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

    sir_pub = _normalize_sir_wrapper_method(tfm)
    if sir_pub is not None:
        inner = _sir_inner_theta_field_method(sir_pub)
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
        sir_dim = int(getattr(args, "sir_dim", 5))
        z_train, z_val, z_all, sir_meta = _fit_sir_projection(
            x_train=x_train,
            theta_train=theta_train,
            x_val=x_val,
            x_all=x_all,
            sir_dim=sir_dim,
            num_bins=int(getattr(args, "sir_num_bins", 10)),
            ridge=float(getattr(args, "sir_ridge", 1e-6)),
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

        if inner == "linear_x_flow_low_rank_t":
            args_lxf = argparse.Namespace(**d)
            args_lxf.theta_field_method = "linear_x_flow_low_rank_t"
            args_lxf.output_dir = output_dir
            loaded, _, _ = _estimate_one(
                args=args_lxf,
                meta=meta,
                bundle=encoded_bundle,
                output_dir=output_dir,
                n_bins=n_bins,
                bin_train=bin_train,
                bin_validation=bin_validation,
                bin_all=bin_all,
            )
        else:
            args_flow = argparse.Namespace(**d)
            args_flow.theta_field_method = inner
            args_flow.output_dir = output_dir
            full_args = _make_full_args(args_flow, meta)
            setattr(full_args, "theta_field_method", inner)
            setattr(full_args, "x_dim", int(sir_dim))
            if sir_pub == "sir_xflow":
                if bin_all is None:
                    raise ValueError(
                        "sir_xflow bin-likelihood H requires bin_all (theta bin indices aligned with the "
                        "evaluation pool); pass bin_all from study_h_decoding_twofig / convergence."
                    )
                setattr(full_args, "sir_xflow_bin_likelihood_h", True)
                setattr(full_args, "convergence_bin_all", np.asarray(bin_all, dtype=np.int64).reshape(-1))
                setattr(full_args, "convergence_n_bins", int(n_bins))
            ctx = _run_ctx_for_bundle(args_flow, meta, encoded_bundle, full_args, n_bins)
            vhb.run_h_estimation_if_needed(ctx)
            loaded = vhb.load_h_matrix(ctx)

        if sir_pub == "sir_xflow_lrank_t":
            h_eval_name = "sir_xflow_lrank_t_log_p_z_given_theta"
        elif sir_pub == "sir_xflow":
            h_eval_name = "sir_xflow_bin_log_p_z_given_theta"
        elif sir_pub == "sir_thetaflow":
            h_eval_name = "sir_thetaflow_log_p_theta_given_z"
        else:
            h_eval_name = f"{sir_pub}_h_eval_scalar"

        h_path = os.path.join(output_dir, _h_matrix_results_npz_basename(dataset_family=str(meta.get("dataset_family", ""))))
        if not os.path.exists(h_path):
            h_path = os.path.join(output_dir, "h_matrix_results_theta_cov.npz")
        _rewrite_npz_fields(
            h_path,
            h_field_method=np.asarray([sir_pub], dtype=object),
            theta_field_method=np.asarray([sir_pub], dtype=object),
            h_eval_scalar_name=np.asarray([h_eval_name], dtype=object),
            sir_enabled=np.bool_(True),
            sir_dim=np.int64(sir_dim),
            sir_num_bins=np.int64(int(getattr(args, "sir_num_bins", 10))),
            sir_ridge=np.float64(float(getattr(args, "sir_ridge", 1e-6))),
            sir_theta_dim=np.int64(int(sir_meta["sir_theta_dim"])),
            sir_x_mean=np.asarray(sir_meta["sir_x_mean"], dtype=np.float64),
            sir_components=np.asarray(sir_meta["sir_components"], dtype=np.float64),
            sir_eigenvalues=np.asarray(sir_meta["sir_eigenvalues"], dtype=np.float64),
            sir_bin_counts=np.asarray(sir_meta["sir_bin_counts"], dtype=np.int64),
            sir_nonempty_bin_ids=np.asarray(sir_meta["sir_nonempty_bin_ids"], dtype=np.int64),
            sir_slice_means=np.asarray(sir_meta["sir_slice_means"], dtype=np.float64),
            sir_theta_edges=np.asarray(sir_meta["sir_theta_edges"], dtype=np.float64),
        )
        loss_path = os.path.join(output_dir, "score_prior_training_losses.npz")
        _rewrite_npz_fields(
            loss_path,
            theta_field_method=np.asarray([sir_pub], dtype=object),
            sir_enabled=np.bool_(True),
            sir_dim=np.int64(sir_dim),
            sir_num_bins=np.int64(int(getattr(args, "sir_num_bins", 10))),
            sir_ridge=np.float64(float(getattr(args, "sir_ridge", 1e-6))),
        )
        full_args_chk = merge_meta_into_args(meta, args)
        theta_chk = vhb.theta_for_h_matrix_alignment(bundle, full_args_chk)
        _validate_theta_used_matches_bundle(
            theta_chk,
            loaded.theta_used,
            err_suffix=str(sir_pub),
        )
        return loaded, np.asarray(bundle.x_all, dtype=np.float64), dev

    lxf_norm = _normalize_linear_x_flow_method(tfm)
    if lxf_norm is not None:
        method_name = lxf_norm
        scheduled_lxf = method_name in _TIME_LXF_METHODS
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

        theta_train_model = theta_train
        theta_val_model = theta_val
        theta_all_model = theta_all
        x_dim_lxf = int(x_all.shape[1])
        lxf_rank = int(getattr(args, "lxf_low_rank_dim", 3))
        sir_meta_lxf: dict[str, np.ndarray | int | float] | None = None
        fixed_sir_u: np.ndarray | None = None
        if method_name in (
            "xflow_sir_lrank",
            "xflow_sir_lrank_dia",
            "xflow_sir_lrank_dia_theta",
            "xflow_sir_lrank_scalar",
            "xflow_sir_lrank_scalar_theta",
            "xflow_sir_pure_lrank",
        ):
            _, _, _, sir_meta_lxf = _fit_sir_projection(
                x_train=x_train,
                theta_train=theta_train,
                x_val=x_val,
                x_all=x_all,
                sir_dim=lxf_rank,
                num_bins=int(getattr(args, "sir_num_bins", 10)),
                ridge=float(getattr(args, "sir_ridge", 1e-6)),
            )
            fixed_sir_u = np.asarray(sir_meta_lxf["sir_components"], dtype=np.float64)
        common = dict(theta_dim=int(theta_all.shape[1]), x_dim=x_dim_lxf, hidden_dim=int(getattr(args, f"{lxf_prefix}_hidden_dim", 128)), depth=int(getattr(args, f"{lxf_prefix}_depth", 3)))
        if method_name == "linear_x_flow_t":
            drift_type = "full_symmetric_time"
            model = ConditionalTimeLinearXFlowMLP(**common, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64))).to(dev)
        elif method_name == "linear_x_flow_scalar_t":
            drift_type = "scalar_time"
            model = ConditionalTimeScalarLinearXFlowMLP(**common, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64))).to(dev)
        elif method_name == "linear_x_flow_diagonal_t":
            drift_type = "diagonal_time"
            model = ConditionalTimeDiagonalLinearXFlowMLP(**common, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64))).to(dev)
        elif method_name == "linear_x_flow_diagonal_theta_t":
            drift_type = "diagonal_theta_time"
            model = ConditionalTimeThetaDiagonalLinearXFlowMLP(**common, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64))).to(dev)
        elif method_name == "linear_x_flow_low_rank_t":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "low_rank_correction_time"
            model = ConditionalTimeLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1))).to(dev)
        elif method_name == "xflow_sir_lrank":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            if fixed_sir_u is None:
                raise RuntimeError("xflow_sir_lrank expected a fitted SIR basis.")
            drift_type = "sir_low_rank_correction_time"
            model = ConditionalTimeLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1),), fixed_u=fixed_sir_u).to(dev)
        elif method_name == "xflow_sir_lrank_dia":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            if fixed_sir_u is None:
                raise RuntimeError("xflow_sir_lrank_dia expected a fitted SIR basis.")
            drift_type = "sir_low_rank_correction_diagonal_time"
            model = ConditionalTimeDiagonalLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1),), fixed_u=fixed_sir_u).to(dev)
        elif method_name == "xflow_sir_lrank_dia_theta":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            if fixed_sir_u is None:
                raise RuntimeError("xflow_sir_lrank_dia_theta expected a fitted SIR basis.")
            drift_type = "sir_low_rank_correction_diagonal_theta_time"
            model = ConditionalTimeThetaDiagonalLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1),), fixed_u=fixed_sir_u).to(dev)
        elif method_name == "xflow_sir_lrank_scalar":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            if fixed_sir_u is None:
                raise RuntimeError("xflow_sir_lrank_scalar expected a fitted SIR basis.")
            drift_type = "sir_low_rank_correction_scalar_time"
            model = ConditionalTimeScalarLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1),), fixed_u=fixed_sir_u).to(dev)
        elif method_name == "xflow_sir_lrank_scalar_theta":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            if fixed_sir_u is None:
                raise RuntimeError("xflow_sir_lrank_scalar_theta expected a fitted SIR basis.")
            drift_type = "sir_low_rank_correction_scalar_theta_time"
            model = ConditionalTimeThetaScalarLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1),), fixed_u=fixed_sir_u).to(dev)
        elif method_name == "xflow_sir_pure_lrank":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            if fixed_sir_u is None:
                raise RuntimeError("xflow_sir_pure_lrank expected a fitted SIR basis.")
            drift_type = "sir_pure_low_rank_time"
            model = ConditionalTimePureLowRankLinearXFlowMLP(
                **common,
                correction_rank=lxf_rank,
                quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)),
                divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(),
                hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1)),
                fixed_u=fixed_sir_u,
            ).to(dev)
        elif method_name == "linear_x_flow_pure_low_rank_t":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "pure_low_rank_time"
            model = ConditionalTimePureLowRankLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1))).to(dev)
        elif method_name == "linear_x_flow_pure_cond_low_rank_t":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "pure_cond_low_rank_time"
            model = ConditionalTimePureConditionalLowRankLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1))).to(dev)
        elif method_name == "linear_x_flow_lr_t_ts":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "low_rank_correction_time_theta_only_b"
            model = ConditionalTimeThetaOnlyBLowRankCorrectionLinearXFlowMLP(**common, correction_rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), divergence_estimator=str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower(), hutchinson_probes=int(getattr(args, "lxf_hutchinson_probes", 1))).to(dev)
        elif method_name == "linear_x_flow_low_rank_randb_t":
            if lxf_rank > x_dim_lxf: raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_dim_lxf}; got {lxf_rank}.")
            drift_type = "low_rank_randb_time"
            model = ConditionalTimeRandomBasisLowRankLinearXFlowMLP(**common, rank=lxf_rank, quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), lambda_a=float(getattr(args, "lxf_randb_lambda_a", 1e-4)), lambda_s=float(getattr(args, "lxf_randb_lambda_s", 1e-4))).to(dev)
        else:
            raise ValueError(f"Unsupported linear_x_flow method: {method_name!r}.")
        train_kwargs = dict(
            model=model,
            theta_train=theta_train_model,
            x_train=x_train,
            theta_val=theta_val_model,
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
        schedule = path_schedule_from_name(sched_name)
        warmup_epochs = int(getattr(args, "lxf_low_rank_t_warmup_epochs", 0))
        if method_name == "linear_x_flow_lr_t_ts":
            train_out = train_low_rank_t_theta_only_b_mean_regression_pretrain_then_freeze_b(**train_kwargs, schedule=schedule, warmup_epochs=warmup_epochs, log_name=method_name)
        elif method_name in ("linear_x_flow_low_rank_t", "xflow_sir_lrank", "xflow_sir_lrank_dia", "xflow_sir_lrank_dia_theta", "xflow_sir_lrank_scalar", "xflow_sir_lrank_scalar_theta") and warmup_epochs > 0:
            train_out = train_low_rank_t_warmup_then_full(**train_kwargs, schedule=schedule, warmup_epochs=warmup_epochs, log_name=method_name)
        else:
            train_out = train_time_linear_x_flow_schedule(
                **train_kwargs, schedule=schedule, log_name=method_name
            )
        x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
        x_std = np.asarray(train_out["x_std"], dtype=np.float64)
        correction_methods = {
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
        }
        analytic_lxf_h = (
            bool(getattr(args, "lxf_analytic_gaussian_hellinger", False))
            and method_name not in correction_methods
        )
        c_matrix = None; delta_l = None; lxf_bin_h = None
        endpoint_mu = np.asarray([], dtype=np.float64); endpoint_cov_or_diag = np.asarray([], dtype=np.float64); endpoint_is_diag = False
        if bin_all is None: raise ValueError("linear-x-flow bin-likelihood Hellinger requires theta bin labels from the convergence subset.")
        if method_name in correction_methods:
            c_matrix = compute_ode_time_linear_x_flow_c_matrix(model=model, theta_all=theta_all_model, x_all=x_all, device=dev, x_mean=x_mean, x_std=x_std, solve_jitter=float(getattr(args, "lxfs_solve_jitter", 1e-6)), quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), ode_steps=int(getattr(args, "lxf_nlpca_ode_steps", 32)), pair_batch_size=int(getattr(args, "lxfs_pair_batch_size", 65536)))
            delta_l = compute_delta_l_nf(c_matrix); lxf_bin_h = lxf_bin_likelihood_hellinger(c_matrix, np.asarray(bin_all, dtype=np.int64), int(n_bins)); h_sym = lxf_bin_h["h_sym"]
        else:
            endpoint_h_sym, endpoint_mu, endpoint_cov_or_diag, endpoint_is_diag = compute_linear_x_flow_analytic_hellinger_matrix(model=model, theta_all=theta_all_model, device=dev, solve_jitter=float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6)), quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)))
            if analytic_lxf_h: h_sym = endpoint_h_sym
            else:
                c_matrix = compute_time_linear_x_flow_c_matrix(model=model, theta_all=theta_all_model, x_all=x_all, device=dev, x_mean=x_mean, x_std=x_std, solve_jitter=float(getattr(args, f"{lxf_prefix}_solve_jitter", 1e-6)), quadrature_steps=int(getattr(args, "lxfs_quadrature_steps", 64)), pair_batch_size=int(getattr(args, f"{lxf_prefix}_pair_batch_size", 65536)))
                delta_l = compute_delta_l_nf(c_matrix); lxf_bin_h = lxf_bin_likelihood_hellinger(c_matrix, np.asarray(bin_all, dtype=np.int64), int(n_bins)); h_sym = lxf_bin_h["h_sym"]
        theta_used = theta_all.reshape(-1) if int(theta_all.shape[1]) == 1 else theta_all.copy()
        empty = np.asarray([], dtype=np.float64)
        lxfs_extra_npz: dict[str, object] = {}
        if method_name in (
            "xflow_sir_lrank",
            "xflow_sir_lrank_dia",
            "xflow_sir_lrank_dia_theta",
            "xflow_sir_lrank_scalar",
            "xflow_sir_lrank_scalar_theta",
            "xflow_sir_pure_lrank",
        ):
            if sir_meta_lxf is None:
                raise RuntimeError(f"{method_name} expected SIR metadata.")
            lxfs_extra_npz.update(
                {
                    "sir_enabled": np.bool_(True),
                    "sir_dim": np.int64(lxf_rank),
                    "sir_num_bins": np.int64(int(getattr(args, "sir_num_bins", 10))),
                    "sir_ridge": np.float64(float(getattr(args, "sir_ridge", 1e-6))),
                    "sir_theta_dim": np.int64(int(sir_meta_lxf["sir_theta_dim"])),
                    "sir_x_mean": np.asarray(sir_meta_lxf["sir_x_mean"], dtype=np.float64),
                    "sir_components": np.asarray(sir_meta_lxf["sir_components"], dtype=np.float64),
                    "sir_eigenvalues": np.asarray(sir_meta_lxf["sir_eigenvalues"], dtype=np.float64),
                    "sir_bin_counts": np.asarray(sir_meta_lxf["sir_bin_counts"], dtype=np.int64),
                    "sir_nonempty_bin_ids": np.asarray(sir_meta_lxf["sir_nonempty_bin_ids"], dtype=np.int64),
                    "sir_slice_means": np.asarray(sir_meta_lxf["sir_slice_means"], dtype=np.float64),
                    "sir_theta_edges": np.asarray(sir_meta_lxf["sir_theta_edges"], dtype=np.float64),
                    "lxf_low_rank_u_source": np.asarray(["raw_sir"], dtype=object),
                }
            )
        h_payload = dict(
            theta_used=np.asarray(theta_used, dtype=np.float64),
            h_sym=np.asarray(h_sym, dtype=np.float64),
            h_field_method=np.asarray([method_name], dtype=object),
            h_eval_scalar_name=np.asarray(
                [f"{method_name}_bin_log_p_x_given_theta" if not analytic_lxf_h else f"{method_name}_analytic_gaussian_hellinger"],
                dtype=object,
            ),
            sigma_eval=np.asarray([np.nan], dtype=np.float64),
            theta_field_method=np.asarray([method_name], dtype=object),
            lxf_drift_type=np.asarray([drift_type], dtype=object),
            lxf_endpoint_mu=np.asarray(endpoint_mu, dtype=np.float64),
            lxf_endpoint_covariance_or_variance_diag=np.asarray(endpoint_cov_or_diag, dtype=np.float64),
            lxf_endpoint_is_diagonal=np.bool_(endpoint_is_diag),
            lxfs_path_schedule=np.asarray([sched_name], dtype=object),
            lxfs_scheduled_train=np.bool_(True),
            lxfs_quadrature_steps=np.int64(int(getattr(args, "lxfs_quadrature_steps", 64))),
            **lxfs_extra_npz,
        )
        if c_matrix is not None:
            h_payload["c_matrix"] = np.asarray(c_matrix, dtype=np.float64)
        if delta_l is not None:
            h_payload["delta_l_matrix"] = np.asarray(delta_l, dtype=np.float64)
        if lxf_bin_h is not None:
            h_payload.update(
                bin_log_likelihood_matrix=np.asarray(lxf_bin_h["bin_log_likelihood"], dtype=np.float64),
                bin_delta_l_matrix=np.asarray(lxf_bin_h["bin_delta_l"], dtype=np.float64),
                h_directed_bin_likelihood=np.asarray(lxf_bin_h["h_directed_bin"], dtype=np.float64),
                h_binned_bin_likelihood=np.asarray(lxf_bin_h["h_binned"], dtype=np.float64),
                bin_counts=np.asarray(lxf_bin_h["bin_counts"], dtype=np.int64),
            )
        np.savez_compressed(os.path.join(output_dir, "h_matrix_results_theta_cov.npz"), **h_payload)
        loss_npz_extra: dict[str, object] = dict(lxfs_extra_npz)
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
            score_likelihood_finetune_train_losses=empty,
            score_likelihood_finetune_val_losses=empty,
            score_likelihood_finetune_val_monitor_losses=empty,
            prior_train_losses=empty,
            prior_val_losses=empty,
            prior_val_monitor_losses=empty,
            prior_likelihood_finetune_train_losses=empty,
            prior_likelihood_finetune_val_losses=empty,
            prior_likelihood_finetune_val_monitor_losses=empty,
            lxf_drift_type=np.asarray([drift_type], dtype=object),
            lxfs_path_schedule=np.asarray([sched_name], dtype=object),
            lxfs_scheduled_train=np.bool_(True),
            lxfs_quadrature_steps=np.int64(int(getattr(args, "lxfs_quadrature_steps", 64))),
            **loss_npz_extra,
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

    contrastive_flow_norm = _normalize_contrastive_flow_method(tfm)
    if contrastive_flow_norm is not None:
        base_method = _base_flow_method_for_contrastive_flow(contrastive_flow_norm)
        dev = require_device(str(getattr(args, "device", "cuda")))
        os.makedirs(output_dir, exist_ok=True)
        theta_binning_mode = str(getattr(args, "theta_binning_mode", "theta1")).strip().lower()
        encoded_bundle, ctr_train_out, latent_dim = _train_contrastive_soft_and_encode_bundle(
            args=args,
            bundle=bundle,
            device=dev,
            theta_binning_mode=theta_binning_mode,
        )
        if contrastive_flow_norm == "contrastive_theta_flow" and latent_dim < 2:
            raise ValueError(
                "contrastive-theta-flow requires --contrastive-soft-dot-dim >= 2 for theta-flow conditioning."
            )
        d = vars(args).copy()
        d.setdefault("h_matrix_npz", None)
        d.setdefault("h_only", False)
        args2 = argparse.Namespace(**d)
        args2.theta_field_method = base_method
        args2.output_dir = output_dir
        full_args = _make_full_args(args2, meta)
        setattr(full_args, "theta_field_method", base_method)
        setattr(full_args, "x_dim", int(latent_dim))
        ctx = _run_ctx_for_bundle(args2, meta, encoded_bundle, full_args, n_bins)
        vhb.run_h_estimation_if_needed(ctx)

        h_path = os.path.join(
            output_dir, _h_matrix_results_npz_basename(dataset_family=str(meta.get("dataset_family", "")))
        )
        if not os.path.exists(h_path):
            h_path = os.path.join(output_dir, "h_matrix_results_theta_cov.npz")
        if base_method == "theta_flow":
            eval_name = "contrastive_theta_flow_log_ratio_theta_given_z"
        else:
            eval_name = "contrastive_x_flow_log_p_z_given_theta"
        periodic = bool(getattr(args, "contrastive_soft_periodic", False))
        period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
        _rewrite_npz_fields(
            h_path,
            h_field_method=np.asarray([contrastive_flow_norm], dtype=object),
            h_eval_scalar_name=np.asarray([eval_name], dtype=object),
            contrastive_soft_encoder_arch=np.asarray(["normalized_dot"], dtype=object),
            contrastive_embed_dim=np.int64(latent_dim),
            contrastive_embed_best_val_smooth=np.float64(ctr_train_out["best_val_loss"]),
            contrastive_effective_batch_size=np.int64(ctr_train_out.get("effective_batch_size", 0)),
            contrastive_soft_bandwidth=np.float64(ctr_train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_normalized=np.float64(ctr_train_out["bandwidth_normalized"]),
            contrastive_soft_bandwidth_auto=np.bool_(ctr_train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(ctr_train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(ctr_train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(ctr_train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_start_normalized=np.float64(ctr_train_out["bandwidth_start_normalized"]),
            contrastive_soft_bandwidth_end_normalized=np.float64(ctr_train_out["bandwidth_end_normalized"]),
            contrastive_soft_bandwidth_schedule=np.asarray(ctr_train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                ctr_train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            contrastive_soft_bandwidth_bins=np.int64(ctr_train_out["bandwidth_bins"]),
            contrastive_soft_periodic=np.bool_(periodic),
            contrastive_soft_period=np.float64(period),
            contrastive_x_mean=np.asarray(ctr_train_out["x_mean"], dtype=np.float64),
            contrastive_x_std=np.asarray(ctr_train_out["x_std"], dtype=np.float64),
            contrastive_theta_mean=np.asarray(ctr_train_out["theta_mean"], dtype=np.float64),
            contrastive_theta_std=np.asarray(ctr_train_out["theta_std"], dtype=np.float64),
        )
        loss_path = os.path.join(output_dir, "score_prior_training_losses.npz")
        _rewrite_npz_fields(
            loss_path,
            theta_field_method=np.asarray([contrastive_flow_norm], dtype=object),
            contrastive_embed_train_losses=np.asarray(ctr_train_out["train_losses"], dtype=np.float64),
            contrastive_embed_val_losses=np.asarray(ctr_train_out["val_losses"], dtype=np.float64),
            contrastive_embed_val_monitor_losses=np.asarray(ctr_train_out["val_monitor_losses"], dtype=np.float64),
            contrastive_embed_best_epoch=np.int64(ctr_train_out["best_epoch"]),
            contrastive_embed_stopped_epoch=np.int64(ctr_train_out["stopped_epoch"]),
            contrastive_embed_stopped_early=np.bool_(ctr_train_out["stopped_early"]),
            contrastive_embed_dim=np.int64(latent_dim),
            contrastive_embed_n_clipped_steps=np.int64(ctr_train_out.get("n_clipped_steps", 0)),
            contrastive_embed_n_total_steps=np.int64(ctr_train_out.get("n_total_steps", 0)),
            contrastive_embed_lr_last=np.float64(ctr_train_out.get("lr_last", float("nan"))),
            contrastive_soft_bandwidth=np.float64(ctr_train_out["bandwidth_raw"]),
            contrastive_soft_bandwidth_auto=np.bool_(ctr_train_out["bandwidth_auto"]),
            contrastive_soft_bandwidth_anneal_enabled=np.bool_(ctr_train_out["bandwidth_anneal_enabled"]),
            contrastive_soft_bandwidth_start=np.float64(ctr_train_out["bandwidth_start_raw"]),
            contrastive_soft_bandwidth_end=np.float64(ctr_train_out["bandwidth_end_raw"]),
            contrastive_soft_bandwidth_schedule=np.asarray(ctr_train_out["bandwidth_raw_schedule"], dtype=np.float64),
            contrastive_soft_bandwidth_schedule_normalized=np.asarray(
                ctr_train_out["bandwidth_normalized_schedule"],
                dtype=np.float64,
            ),
            contrastive_batch_size=np.int64(int(getattr(args, "contrastive_batch_size", 256))),
            contrastive_effective_batch_size=np.int64(ctr_train_out.get("effective_batch_size", 0)),
        )
        loaded = vhb.load_h_matrix(ctx)
        theta_chk = vhb.theta_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
        _validate_theta_used_matches_bundle(
            theta_chk,
            loaded.theta_used,
            err_suffix=f"h_field_method={contrastive_flow_norm!r}",
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
