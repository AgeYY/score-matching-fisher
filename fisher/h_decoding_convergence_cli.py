#!/usr/bin/env python3
"""CLI construction and validation for H-decoding convergence."""

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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, train score models for each n in --n-list, then compare "
            "sqrt(binned H_sym) to sqrt(MC generative H^2) and pairwise decoding to the n_ref-subset decoding matrix. "
            "The n_ref matrix-panel column uses MC GT sqrt(H^2) for the top row (no n_ref model training). "
            "Also writes h_decoding_convergence_combined.{png,svg} (matrix panel + correlation curves + "
            "off-diagonal est-vs-GT H scatter + training-loss panel in one figure) and h_decoding_training_losses_panel.{png,svg} "
            "(standalone training-loss panel, one column per n). Runs that save Gaussian-network pretrain curves "
            "Gaussian-network methods also write h_decoding_gn_pretrain_losses_panel.{png,svg}."
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
            "random_mog_categorical",
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
            "Replace scalar / native θ coordinates with Fourier features built per coordinate from reference ranges "
            "on the n_ref permutation prefix (theta_flow / theta_flow_autoencoder / linear_x_flow_t / xflow_sir_lrank), "
            "or enable contrastive-soft dot-family Fourier augmentation (contrastive_soft). "
            "With d θ coordinates, flow state width is d*(2*K + optional linear channel). Base period per coordinate is "
            "theta_flow_fourier_period_mult * (theta_max - theta_min) on that coordinate's ref slice, "
            "then harmonics k=1..K."
        ),
    )
    p.add_argument(
        "--theta-flow-fourier-k",
        type=int,
        default=4,
        help=(
            "Number of Fourier harmonics K per θ coordinate when --theta-flow-fourier-state is set "
            "(flow state dim = d_theta*(2*K + optional linear); contrastive-soft dot branch uses the same K "
            "for sin/cos on raw θ after the z-scored θ channel when harmonics are applied inside training)."
        ),
    )
    p.add_argument(
        "--theta-flow-fourier-period-mult",
        type=float,
        default=2.0,
        help=(
            "Base-period multiplier for Fourier theta state (per coordinate): "
            "period_j = multiplier * (max(theta_j)-min(theta_j)) on the n_ref subset for coordinate j."
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
    p.add_argument(
        "--sir-dim",
        type=int,
        default=5,
        help="SIR wrapper methods (sir_xflow_lrank_t, sir_xflow, sir_thetaflow): projection dimension.",
    )
    p.add_argument(
        "--sir-num-bins",
        type=int,
        default=10,
        help="SIR wrapper methods: equal-width bins per theta dimension for SIR slices.",
    )
    p.add_argument(
        "--sir-ridge",
        type=float,
        default=1e-6,
        help="SIR wrapper methods: diagonal ridge added to x covariance before SIR whitening.",
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
        "--lxf-low-rank-dim",
        type=int,
        default=None,
        help=(
            "Default: automatic for xflow_sir_lrank variants (smallest SIR rank explaining >=90%% "
            "inverse-regression eigenvalue mass, plus one, capped to available SIR rank); "
            "3 for non-SIR low-rank linear_x_flow variants. "
            "linear_x_flow_low_rank_t / linear_x_flow_pure_low_rank_t / linear_x_flow_pure_cond_low_rank_t / "
            "linear_x_flow_lr_t_ts / xflow_sir_lrank / xflow_sir_lrank_dia / xflow_sir_lrank_dia_theta / xflow_sir_lrank_scalar / "
            "xflow_sir_lrank_scalar_theta / xflow_sir_pure_lrank: "
            "rank r of U in U h(U^T x) "
            "(xflow_sir_lrank variants: "
            "fixed raw SIR directions from the train split; xflow_sir_pure_lrank: fixed raw SIR U with pure U h(U^T x) only; "
            "pure_low_rank_t: fixed orthonormal U; "
            "pure_cond_low_rank_t: U(theta,t) from MLP; pure_low_rank_t velocity is U h(U^T x) only). "
            "linear_x_flow_low_rank_randb_t: rank of the low-rank random-basis A(t) term."
        ),
    )
    p.add_argument(
        "--lxf-low-rank-divergence-estimator",
        type=str,
        default="hutchinson",
        choices=["hutchinson", "exact"],
        help=(
            "linear_x_flow_low_rank_t / linear_x_flow_pure_low_rank_t / linear_x_flow_pure_cond_low_rank_t / "
            "linear_x_flow_lr_t_ts / xflow_sir_lrank / xflow_sir_lrank_dia / xflow_sir_lrank_dia_theta / xflow_sir_lrank_scalar / "
            "xflow_sir_lrank_scalar_theta / xflow_sir_pure_lrank: "
            "reduced divergence in z=U^T x "
            "(pure_low_rank_t / low_rank_t / lr_t_ts: "
            "orthonormal U, trace of dh/dz; pure_cond_low_rank_t: tr((U^T U) dh/dz) with U(theta,t) detached in div). "
            "xflow_sir_lrank and xflow_sir_pure_lrank use tr((U^T U) dh/dz) for fixed raw SIR U. "
            "linear_x_flow_lr_t_ts: same Hutchinson/exact trace on h as low_rank_t (b is frozen after mean-regression pretrain); "
            ""
            ""
            "`hutchinson` uses Rademacher probes (default; faster when r is large). "
            "`exact` uses one autograd per output dimension (r calls)."
        ),
    )
    p.add_argument(
        "--lxf-hutchinson-probes",
        type=int,
        default=1,
        help=(
            "linear_x_flow_low_rank_t / linear_x_flow_pure_low_rank_t / linear_x_flow_pure_cond_low_rank_t / "
            "linear_x_flow_lr_t_ts / xflow_sir_lrank / xflow_sir_lrank_dia / xflow_sir_lrank_dia_theta / xflow_sir_lrank_scalar / "
            "xflow_sir_lrank_scalar_theta / xflow_sir_pure_lrank with hutchinson: "
            "number of Rademacher probes per divergence."
        ),
    )
    p.add_argument("--lxf-randb-lambda-a", type=float, default=1e-4, help="scheduled random-basis LXF only: L2 penalty on diagonal a.")
    p.add_argument("--lxf-randb-lambda-s", type=float, default=1e-4, help="scheduled random-basis LXF only: L2 penalty on symmetric S.")
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
            "linear-x-flow only: save the C and DeltaL matrices. C is computed by default for "
            "bin-likelihood Hellinger; this flag is kept for compatibility and still controls "
            "extra C/DeltaL diagnostics when --lxf-analytic-gaussian-hellinger is used."
        ),
    )
    p.add_argument(
        "--lxf-analytic-gaussian-hellinger",
        action="store_true",
        help=(
            "linear-x-flow Gaussian endpoint methods only: use the legacy analytic endpoint "
            "Gaussian Hellinger matrix instead of the default bin-level likelihood estimate."
        ),
    )
    p.add_argument(
        "--lxf-nlpca-ode-steps",
        type=int,
        default=32,
        help=(
            "linear_x_flow_low_rank_t / linear_x_flow_pure_low_rank_t / linear_x_flow_pure_cond_low_rank_t / "
            "linear_x_flow_lr_t_ts only: fixed Euler steps for ODE likelihood."
        ),
    )
    p.add_argument(
        "--lxfs-path-schedule",
        type=str,
        default="cosine",
        choices=["linear", "straight", "cosine", "cos"],
        help="scheduled linear-x-flow only: affine path a(t), b(t) for FM training.",
    )
    p.add_argument("--lxfs-epochs", type=int, default=2000, help="scheduled linear-x-flow only: training epochs.")
    p.add_argument(
        "--lxf-low-rank-t-warmup-epochs",
        type=int,
        default=1000,
        help=(
            "linear_x_flow_low_rank_t: b(t,theta)-only warmup epochs before full lxfs training; set 0 to disable. "
            "linear_x_flow_lr_t_ts: epochs of mean-squared regression pretraining b(theta) to normalized x1 (required >= 1); then b is frozen for lxfs training."
        ),
    )
    p.add_argument("--lxfs-batch-size", type=int, default=1024, help="scheduled linear-x-flow only: training batch size.")
    p.add_argument("--lxfs-lr", type=float, default=1e-4, help="scheduled linear-x-flow only: learning rate.")
    p.add_argument("--lxfs-hidden-dim", type=int, default=128, help="scheduled linear-x-flow only: b_phi MLP hidden width.")
    p.add_argument("--lxfs-depth", type=int, default=3, help="scheduled linear-x-flow only: b_phi MLP depth.")
    p.add_argument("--lxfs-weight-decay", type=float, default=0.0, help="scheduled linear-x-flow only: AdamW weight decay.")
    p.add_argument(
        "--lxfs-t-eps",
        type=float,
        default=0.05,
        help="scheduled linear-x-flow only: bridge time is sampled in [t_eps, 1-t_eps].",
    )
    p.add_argument(
        "--lxfs-solve-jitter",
        type=float,
        default=1e-6,
        help="scheduled linear-x-flow only: jitter for solving A mu=(exp(A)-I)b and Cholesky log likelihood.",
    )
    p.add_argument(
        "--lxfs-early-patience",
        type=int,
        default=300,
        help="scheduled linear-x-flow only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--lxfs-early-min-delta",
        type=float,
        default=1e-4,
        help="scheduled linear-x-flow only: early-stop min delta on smoothed validation FM loss.",
    )
    p.add_argument(
        "--lxfs-early-ema-alpha",
        type=float,
        default=0.05,
        help="scheduled linear-x-flow only: EMA alpha for validation FM loss monitor.",
    )
    p.add_argument(
        "--lxfs-weight-ema-decay",
        type=float,
        default=0.9,
        help="scheduled linear-x-flow only: model-weight EMA decay; <=0 disables weight EMA.",
    )
    p.add_argument(
        "--lxfs-max-grad-norm",
        type=float,
        default=10.0,
        help="scheduled linear-x-flow only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--lxfs-pair-batch-size",
        type=int,
        default=65536,
        help="scheduled linear-x-flow only: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--lxfs-quadrature-steps",
        type=int,
        default=64,
        help="scheduled linear-x-flow only: fixed time grid size for endpoint Gaussian quadrature.",
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
    p.add_argument("--contrastive-epochs", type=int, default=2000, help="contrastive-family methods: training epochs.")
    p.add_argument(
        "--contrastive-batch-size",
        type=int,
        default=256,
        help="contrastive-family methods: minibatch size for soft contrastive cross entropy.",
    )
    p.add_argument("--contrastive-lr", type=float, default=1e-3, help="contrastive-family methods: learning rate.")
    p.add_argument(
        "--contrastive-hidden-dim",
        type=int,
        default=128,
        help="contrastive-family methods: MLP hidden width.",
    )
    p.add_argument("--contrastive-depth", type=int, default=3, help="contrastive-family methods: MLP depth.")
    p.add_argument(
        "--contrastive-weight-decay",
        type=float,
        default=0.0,
        help="contrastive-family methods: AdamW weight decay.",
    )
    p.add_argument(
        "--contrastive-early-patience",
        type=int,
        default=300,
        help="contrastive-family methods: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--contrastive-early-min-delta",
        type=float,
        default=1e-4,
        help="contrastive-family methods: early-stop min delta.",
    )
    p.add_argument(
        "--contrastive-early-ema-alpha",
        type=float,
        default=0.05,
        help="contrastive-family methods: EMA alpha for validation monitor.",
    )
    p.add_argument(
        "--contrastive-max-grad-norm",
        type=float,
        default=10.0,
        help="contrastive-family methods: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--contrastive-pair-batch-size",
        type=int,
        default=65536,
        help="contrastive-family methods: approximate pair budget per C-matrix block (rows*cols).",
    )
    p.add_argument(
        "--contrastive-soft-bandwidth-bins",
        type=int,
        default=10,
        help=(
            "contrastive-soft only: Gaussian theta-kernel bandwidth (raw θ units) is "
            "train_theta_span / (2 * K), where K is this bin count."
        ),
    )
    p.add_argument(
        "--contrastive-soft-score-arch",
        type=str,
        default="normalized_dot",
        choices=[
            "normalized_dot",
            "additive_independent",
        ],
        help=(
            "contrastive-soft only: scalar score architecture. normalized_dot uses "
            "S(x,theta)=alpha normalize(g(x))^T normalize(a(theta)); additive_independent uses "
            "D^{-1} sum_d h_d(x_d)^T a(theta)."
        ),
    )
    p.add_argument(
        "--contrastive-soft-dot-dim",
        type=int,
        default=10,
        help=(
            "contrastive-soft normalized_dot/additive_independent only: shared feature dimension "
            "for dot-product features."
        ),
    )
    p.add_argument(
        "--contrastive-soft-periodic",
        action="store_true",
        help="contrastive-soft only: use circular theta distance in the soft target kernel (scalar θ only).",
    )
    p.add_argument(
        "--contrastive-soft-period",
        type=float,
        default=2.0 * np.pi,
        help="contrastive-soft only: period for circular theta distance when --contrastive-soft-periodic is set.",
    )
    p.add_argument(
        "--contrastive-soft-categorical-beta",
        type=float,
        default=0.0,
        help=(
            "contrastive-soft-categorical only: off-class target weight before row normalization; "
            "0 makes the true class the only positive class."
        ),
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
        "linear-x-flow-t": "linear_x_flow_t",
        "linear_x_flow_t": "linear_x_flow_t",
        "linear-x-flow-scalar-t": "linear_x_flow_scalar_t",
        "linear_x_flow_scalar_t": "linear_x_flow_scalar_t",
        "linear-x-flow-diagonal-theta-t": "linear_x_flow_diagonal_theta_t",
        "linear_x_flow_diagonal_theta_t": "linear_x_flow_diagonal_theta_t",
        "linear-x-flow-low-rank-t": "linear_x_flow_low_rank_t",
        "linear_x_flow_low_rank_t": "linear_x_flow_low_rank_t",
        "xflow-sir-lrank": "xflow_sir_lrank",
        "xflow_sir_lrank": "xflow_sir_lrank",
        "xflow-sir-lrank-dia": "xflow_sir_lrank_dia",
        "xflow_sir_lrank_dia": "xflow_sir_lrank_dia",
        "xflow-sir-lrank-dia-theta": "xflow_sir_lrank_dia_theta",
        "xflow_sir_lrank_dia_theta": "xflow_sir_lrank_dia_theta",
        "xflow-sir-lrank-scalar": "xflow_sir_lrank_scalar",
        "xflow_sir_lrank_scalar": "xflow_sir_lrank_scalar",
        "xflow-sir-lrank-scalar-theta": "xflow_sir_lrank_scalar_theta",
        "xflow_sir_lrank_scalar_theta": "xflow_sir_lrank_scalar_theta",
        "xflow-sir-pure-lrank": "xflow_sir_pure_lrank",
        "xflow_sir_pure_lrank": "xflow_sir_pure_lrank",
        "linear-x-flow-pure-low-rank-t": "linear_x_flow_pure_low_rank_t",
        "linear_x_flow_pure_low_rank_t": "linear_x_flow_pure_low_rank_t",
        "linear-x-flow-pure-cond-low-rank-t": "linear_x_flow_pure_cond_low_rank_t",
        "linear_x_flow_pure_cond_low_rank_t": "linear_x_flow_pure_cond_low_rank_t",
        "linear-x-flow-lr-t-ts": "linear_x_flow_lr_t_ts",
        "linear_x_flow_lr_t_ts": "linear_x_flow_lr_t_ts",
        "linear-x-flow-low-rank-randb-t": "linear_x_flow_low_rank_randb_t",
        "linear_x_flow_low_rank_randb_t": "linear_x_flow_low_rank_randb_t",
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
        "contrastive-soft": "contrastive_soft",
        "contrastive_soft": "contrastive_soft",
        "contrasive-soft": "contrastive_soft",
        "contrasive_soft": "contrastive_soft",
        "contrastive-soft-categorical": "contrastive_soft_categorical",
        "contrastive_soft_categorical": "contrastive_soft_categorical",
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


# Public SIR wrapper tokens -> inner estimator run after projecting x -> z on train.
_SIR_WRAPPER_ALIASES: dict[str, str] = {
    "sir-xflow-lrank-t": "sir_xflow_lrank_t",
    "sir_xflow_lrank_t": "sir_xflow_lrank_t",
    "sir-xflow": "sir_xflow",
    "sir_xflow": "sir_xflow",
    "sir-thetaflow": "sir_thetaflow",
    "sir_thetaflow": "sir_thetaflow",
}

SIR_WRAPPER_INNER_METHOD: dict[str, str] = {
    "sir_xflow_lrank_t": "linear_x_flow_low_rank_t",
    "sir_xflow": "x_flow",
    "sir_thetaflow": "theta_flow",
}


def _normalize_sir_wrapper_method(tfm: str) -> str | None:
    key = str(tfm).strip().lower()
    return _SIR_WRAPPER_ALIASES.get(key)


def _normalize_sir_xflow_method(tfm: str) -> str | None:
    """Backward-compatible alias: any SIR wrapper public token."""
    return _normalize_sir_wrapper_method(tfm)


def _sir_inner_theta_field_method(public_sir: str) -> str:
    pub = str(public_sir).strip().lower()
    inner = SIR_WRAPPER_INNER_METHOD.get(pub)
    if inner is None:
        raise ValueError(f"Unknown SIR wrapper method: {public_sir!r}")
    return inner


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


def _validate_sir_wrapper_cli(args: argparse.Namespace) -> None:
    sir_dim = int(getattr(args, "sir_dim", 0))
    if sir_dim < 1:
        raise ValueError("--sir-dim must be >= 1.")
    if int(getattr(args, "sir_num_bins", 0)) < 2:
        raise ValueError("--sir-num-bins must be >= 2.")
    ridge = float(getattr(args, "sir_ridge", 0.0))
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("--sir-ridge must be finite and > 0.")
    pub = str(getattr(args, "theta_field_method", "")).strip().lower()
    inner = _sir_inner_theta_field_method(pub)
    base_args = argparse.Namespace(**vars(args).copy())
    setattr(base_args, "theta_field_method", inner)
    if inner == "linear_x_flow_low_rank_t":
        _validate_lxf_cli(base_args)
        low_rank_dim = getattr(args, "lxf_low_rank_dim", None)
        low_rank_dim_eff = 3 if low_rank_dim is None else int(low_rank_dim)
        if low_rank_dim_eff > sir_dim:
            raise ValueError("--lxf-low-rank-dim must be <= --sir-dim for sir_xflow_lrank_t.")
    elif inner == "x_flow":
        validate_estimation_args(base_args)
    elif inner == "theta_flow":
        validate_estimation_args(base_args)
    else:
        raise ValueError(f"SIR wrapper inner method not handled in CLI validation: {inner!r}")


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
    scheduled = method in _TIME_LXF_METHODS
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
    low_rank_dim = getattr(args, "lxf_low_rank_dim", None)
    sir_lxf_methods = (
        "xflow_sir_lrank",
        "xflow_sir_lrank_dia",
        "xflow_sir_lrank_dia_theta",
        "xflow_sir_lrank_scalar",
        "xflow_sir_lrank_scalar_theta",
        "xflow_sir_pure_lrank",
    )
    low_rank_methods = (
        "linear_x_flow_low_rank_t",
        "linear_x_flow_pure_low_rank_t",
        "linear_x_flow_pure_cond_low_rank_t",
        "linear_x_flow_lr_t_ts",
        "linear_x_flow_low_rank_randb_t",
    )
    if method in low_rank_methods and low_rank_dim is None:
        low_rank_dim = 3
    if method in (*low_rank_methods, *sir_lxf_methods) and low_rank_dim is not None and int(low_rank_dim) < 1:
        raise ValueError("--lxf-low-rank-dim must be >= 1.")
    if method in sir_lxf_methods:
        if int(getattr(args, "sir_num_bins", 0)) < 2:
            raise ValueError(f"--sir-num-bins must be >= 2 for {method}.")
        ridge = float(getattr(args, "sir_ridge", 0.0))
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ValueError(f"--sir-ridge must be finite and > 0 for {method}.")
    if method == "linear_x_flow_lr_t_ts" and int(getattr(args, "lxf_low_rank_t_warmup_epochs", 0)) < 1:
        raise ValueError(
            "linear_x_flow_lr_t_ts requires --lxf-low-rank-t-warmup-epochs >= 1 "
            "(mean-regression pretrain for b(theta))."
        )
    _lrdiv = str(getattr(args, "lxf_low_rank_divergence_estimator", "hutchinson")).strip().lower()
    if _lrdiv not in ("hutchinson", "exact"):
        raise ValueError("--lxf-low-rank-divergence-estimator must be one of: hutchinson, exact.")
    if int(getattr(args, "lxf_hutchinson_probes", 1)) < 1:
        raise ValueError("--lxf-hutchinson-probes must be >= 1.")
    if method == "linear_x_flow_low_rank_randb_t":
        if float(getattr(args, "lxf_randb_lambda_a", 0.0)) < 0.0:
            raise ValueError("--lxf-randb-lambda-a must be >= 0.")
        if float(getattr(args, "lxf_randb_lambda_s", 0.0)) < 0.0:
            raise ValueError("--lxf-randb-lambda-s must be >= 0.")
    if float(getattr(args, f"{prefix}_weight_decay", 0.0)) < 0.0:
        raise ValueError(f"{label}-weight-decay must be >= 0.")
    te = float(getattr(args, f"{prefix}_t_eps", 1e-3))
    if scheduled:
        if not (0.0 < te < 0.5):
            raise ValueError("--lxfs-t-eps must be in (0, 0.5).")
        path_schedule_from_name(str(getattr(args, "lxfs_path_schedule", "cosine")))
        if int(getattr(args, "lxfs_quadrature_steps", 0)) < 2:
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


def _validate_contrastive_cli(args: argparse.Namespace) -> None:
    from fisher.h_decoding_convergence_methods import contrastive_soft_fourier_settings_from_theta_flow_args

    tfm = str(getattr(args, "theta_field_method", "")).strip().lower()
    cnorm = _normalize_contrastive_method(tfm)
    if cnorm == "contrastive_soft_categorical" and bool(getattr(args, "theta_flow_fourier_state", False)):
        raise ValueError("--theta-flow-fourier-state is not supported for contrastive_soft_categorical.")
    fk, pm, inc_lin = contrastive_soft_fourier_settings_from_theta_flow_args(args)
    if fk > 0 and (not np.isfinite(pm) or pm <= 0.0):
        raise ValueError(
            "--theta-flow-fourier-period-mult must be finite and > 0 when --theta-flow-fourier-state is set "
            f"(effective Fourier harmonics K={fk})."
        )
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
    soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
    soft_arch_aliases = {
        "normalized_dot": "normalized_dot",
        "additive_independent": "additive_independent",
    }
    if soft_arch not in soft_arch_aliases:
        raise ValueError(
            "--contrastive-soft-score-arch must be one of "
            "{'normalized_dot','additive_independent'}."
        )
    setattr(args, "contrastive_soft_score_arch", soft_arch_aliases[soft_arch])
    if int(getattr(args, "contrastive_soft_dot_dim", 0)) < 1:
        raise ValueError("--contrastive-soft-dot-dim must be >= 1.")
    if int(getattr(args, "contrastive_soft_bandwidth_bins", 0)) < 1:
        raise ValueError("--contrastive-soft-bandwidth-bins must be >= 1.")
    period = float(getattr(args, "contrastive_soft_period", 2.0 * np.pi))
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("--contrastive-soft-period must be finite and > 0.")
    beta = float(getattr(args, "contrastive_soft_categorical_beta", 0.0))
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("--contrastive-soft-categorical-beta must be finite and >= 0.")

def _validate_cli(args: argparse.Namespace) -> None:
    tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
    nfr_norm = _normalize_nf_reduction_method(tfm)
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
        or pinf_norm is not None
        or contrastive_norm is not None
        or _normalize_sir_xflow_method(tfm) is not None
    ) else _normalize_gaussian_network_method(tfm)
    flow_ae_norm = _normalize_flow_autoencoder_method(tfm)
    flow_pca_norm = _normalize_flow_pca_method(tfm)
    sir_xflow_norm = _normalize_sir_xflow_method(tfm)
    if contrastive_norm is not None:
        setattr(args, "theta_field_method", contrastive_norm)
        _validate_contrastive_cli(args)
        tfm = contrastive_norm
    elif pinf_norm is not None:
        setattr(args, "theta_field_method", pinf_norm)
        _validate_pinf_cli(args)
        tfm = pinf_norm
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
    elif sir_xflow_norm is not None:
        setattr(args, "theta_field_method", sir_xflow_norm)
        _validate_sir_wrapper_cli(args)
        tfm = sir_xflow_norm
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
        "sir_xflow_lrank_t",
        "sir_xflow",
        "sir_thetaflow",
        "gaussian_x_flow",
        "gaussian_x_flow_diagonal",
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
        "linear_theta_flow",
        "nf_reduction",
        "pi_nf",
        "contrastive_soft",
        "contrastive_soft_categorical",
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
        raw_tfm = str(getattr(args, "theta_field_method", "theta_flow")).strip().lower()
        cnorm = _normalize_contrastive_method(raw_tfm)
        lxf_norm = _normalize_linear_x_flow_method(raw_tfm)
        arch = str(getattr(args, "flow_arch", "mlp")).strip().lower()
        fourier_ok = (
            raw_tfm in ("theta_flow", "theta_flow_autoencoder")
            or lxf_norm in ("linear_x_flow_t", "xflow_sir_lrank")
            or cnorm == "contrastive_soft"
        )
        if not fourier_ok:
            raise ValueError(
                "--theta-flow-fourier-state requires --theta-field-method theta_flow, theta-flow-autoencoder, "
                "linear_x_flow_t, xflow_sir_lrank, or contrastive_soft "
                f"(got {getattr(args, 'theta_field_method', None)!r})."
            )
        if raw_tfm in ("theta_flow", "theta_flow_autoencoder") and arch != "mlp":
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
