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
``--theta-field-method linear-x-flow-diagonal-t`` trains ``v(x,t,theta)=diag(a(t))x+b(t,theta)``
on a scheduled affine probability path and evaluates its quadrature Gaussian likelihood.
``--theta-field-method xflow-sir-lrank`` uses the scheduled full-``x`` low-rank correction
linear X-flow, but fixes ``U`` to raw SIR directions fitted on the train split.
``--theta-field-method xflow-sir-lrank-dia`` uses the same fixed-SIR low-rank correction
with diagonal ``A(t)``.
``--theta-field-method xflow-sir-lrank-dia-theta`` uses the same fixed-SIR low-rank
correction with diagonal ``A(t,\theta)``.
``--theta-field-method xflow-sir-lrank-scalar`` uses the same fixed-SIR low-rank correction
with scalar ``A(t)=a(t)I``.
``--theta-field-method xflow-sir-lrank-scalar-theta`` uses the same fixed-SIR low-rank
correction with scalar ``A(t,\theta)=a(t,\theta)I``.
``--theta-field-method xflow-sir-pure-lrank`` fixes ``U`` to raw SIR directions on the train split
and trains pure velocity ``v(x,t,\theta)=U h(U^\top x,t,\theta)`` (no scheduled linear ``A(t)x`` or ``b``).
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


from fisher import h_decoding_convergence_cli as _conv_cli
from fisher import h_decoding_convergence_methods as _conv_methods
from fisher import h_decoding_convergence_plots as _conv_plots


def _reexport_module_names(module: Any) -> None:
    globals().update({name: value for name, value in vars(module).items() if not name.startswith("__")})


_reexport_module_names(_conv_cli)
_reexport_module_names(_conv_methods)
_reexport_module_names(_conv_plots)

















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
    if str(meta.get("theta_type", "")) == "categorical":
        k_cat = int(meta.get("num_categories", n_bins))
        if n_bins != k_cat:
            raise ValueError(
                f"Categorical dataset requires --num-theta-bins == num_categories ({k_cat}); got {n_bins}."
            )
        theta_scalar_all, theta_ref, edges, edge_lo, edge_hi, bin_idx_all = prepare_categorical_binning_for_convergence(
            theta_raw_all,
            k_cat,
        )
    else:
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
        th_phys, theta_fourier_ref_phys = theta_phys_rows_and_ref_for_fourier(
            theta_raw_all,
            perm,
            int(args.n_ref),
        )
        theta_state_all, theta_fourier_ref_range, theta_fourier_period, theta_fourier_center = _build_theta_fourier_state(
            th_phys,
            theta_ref=theta_fourier_ref_phys,
            k=int(args.theta_flow_fourier_k),
            period_mult=float(args.theta_flow_fourier_period_mult),
            include_linear=bool(args.theta_flow_fourier_include_linear),
        )
        print(
            format_theta_fourier_state_log_message(
                tag="[convergence]",
                state_dim=int(theta_state_all.shape[1]),
                k=int(args.theta_flow_fourier_k),
                ref_range_vec=theta_fourier_ref_range,
                period_vec=theta_fourier_period,
                center_vec=theta_fourier_center,
                period_mult=float(args.theta_flow_fourier_period_mult),
                include_linear=bool(args.theta_flow_fourier_include_linear),
            ),
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
    gt_method = "analytic_gaussian_centers"
    gt_theta_centers = theta_centers_for_analytic_gt(dataset_for_gt, centers)
    try:
        h_gt_mc = estimate_hellinger_sq_grid_centers_analytic(
            dataset_for_gt,
            gt_theta_centers,
            symmetrize=bool(args.gt_hellinger_symmetrize),
        )
    except TypeError:
        gt_method = "mc_likelihood"
        h_gt_mc = estimate_hellinger_sq_one_sided_mc(
            dataset_for_gt,
            centers,
            n_mc=gt_n_mc,
            symmetrize=bool(args.gt_hellinger_symmetrize),
        )
    h_gt_sqrt = _sqrt_h_like(h_gt_mc)
    print(
        f"[convergence] GT Hellinger ({gt_method}) n_bins={n_bins} center_shape={gt_theta_centers.shape} "
        f"legacy_n_mc={gt_n_mc} wall time: {time.time() - t_gt0:.1f}s "
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
    elif tfm == "sir_xflow_lrank_t":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(sir_dim={int(getattr(args, 'sir_dim', 5))}; sir_num_bins={int(getattr(args, 'sir_num_bins', 10))}; "
            "linear_x_flow_low_rank_t in SIR z-space)",
            flush=True,
        )
        print(
            "[convergence] sir_xflow_lrank_t mode fits SIR on train data, projects x to z, "
            "then uses the scheduled linear_x_flow_low_rank_t bin likelihood in projected space.",
            flush=True,
        )
    elif tfm == "sir_xflow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(sir_dim={int(getattr(args, 'sir_dim', 5))}; sir_num_bins={int(getattr(args, 'sir_num_bins', 10))}; "
            "x_flow on SIR z)",
            flush=True,
        )
        print(
            "[convergence] sir_xflow mode fits SIR on train data, projects x to z, "
            "then runs conditional x-flow ODE likelihood on z.",
            flush=True,
        )
    elif tfm == "sir_thetaflow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(sir_dim={int(getattr(args, 'sir_dim', 5))}; sir_num_bins={int(getattr(args, 'sir_num_bins', 10))}; "
            "theta_flow on SIR z)",
            flush=True,
        )
        print(
            "[convergence] sir_thetaflow mode fits SIR on train data, projects x to z, "
            "then runs theta-flow Bayes-ratio estimation conditioning on z.",
            flush=True,
        )
    elif tfm == "xflow_sir_pure_lrank":
        lxf_rank_desc = "auto_90_plus1" if getattr(args, "lxf_low_rank_dim", None) is None else str(int(getattr(args, "lxf_low_rank_dim")))
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(lxf_low_rank_dim={lxf_rank_desc}; "
            f"sir_num_bins={int(getattr(args, 'sir_num_bins', 10))}; raw SIR U; pure U h(U^T x), no A or b)",
            flush=True,
        )
        print(
            "[convergence] xflow_sir_pure_lrank mode fits SIR on train data, fixes raw SIR components as U, "
            "then trains pure velocity v=U h(U^T x,t,theta) in the original x space (no scheduled linear drift).",
            flush=True,
        )
    elif tfm in ("xflow_sir_lrank", "xflow_sir_lrank_dia", "xflow_sir_lrank_dia_theta", "xflow_sir_lrank_scalar", "xflow_sir_lrank_scalar_theta"):
        lxf_rank_desc = "auto_90_plus1" if getattr(args, "lxf_low_rank_dim", None) is None else str(int(getattr(args, "lxf_low_rank_dim")))
        if tfm == "xflow_sir_lrank_dia":
            a_desc = "diagonal A(t)"
        elif tfm == "xflow_sir_lrank_dia_theta":
            a_desc = "diagonal A(theta,t)"
        elif tfm == "xflow_sir_lrank_scalar":
            a_desc = "scalar A(t)"
        elif tfm == "xflow_sir_lrank_scalar_theta":
            a_desc = "scalar A(theta,t)"
        else:
            a_desc = "full A(t)"
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(lxf_low_rank_dim={lxf_rank_desc}; "
            f"sir_num_bins={int(getattr(args, 'sir_num_bins', 10))}; raw SIR U in full x-space; {a_desc})",
            flush=True,
        )
        print(
            f"[convergence] {tfm} mode fits SIR on train data, fixes raw SIR components as U, "
            f"then trains scheduled low-rank correction X-flow with {a_desc} in the original x space.",
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
    elif tfm == "contrastive_soft":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(score_arch={str(getattr(args, 'contrastive_soft_score_arch', 'normalized_dot'))}; "
            f"dot_dim={int(getattr(args, 'contrastive_soft_dot_dim', 10))}; "
            f"hidden_dim={int(getattr(args, 'contrastive_hidden_dim', 128))}; "
            f"depth={int(getattr(args, 'contrastive_depth', 3))}; "
            f"bandwidth_bins={int(getattr(args, 'contrastive_soft_bandwidth_bins', 10))}; "
            f"(raw h = train_theta_range / (2 * bandwidth_bins)); "
            f"periodic={bool(getattr(args, 'contrastive_soft_periodic', False))}; identity x embedding)",
            flush=True,
        )
        print(
            "[convergence] contrastive_soft mode trains a scalar score S(x,theta) with Gaussian-kernel soft positives "
            "over shuffled minibatch theta candidates (Euclidean on z-scored theta when d_theta>1), "
            "then uses C[i,j]=S(x_i,theta_j), DeltaL=C-diag(C), "
            "and one-sided H^2 from exp(DeltaL/2).",
            flush=True,
        )
    elif tfm == "contrastive_soft_categorical":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(score_arch={str(getattr(args, 'contrastive_soft_score_arch', 'normalized_dot'))}; "
            f"dot_dim={int(getattr(args, 'contrastive_soft_dot_dim', 10))}; "
            f"hidden_dim={int(getattr(args, 'contrastive_hidden_dim', 128))}; "
            f"depth={int(getattr(args, 'contrastive_depth', 3))}; "
            f"beta={float(getattr(args, 'contrastive_soft_categorical_beta', 0.0))}; "
            f"theta classes={int(args.num_theta_bins)} one-hot; identity x embedding)",
            flush=True,
        )
        print(
            "[convergence] contrastive_soft_categorical mode trains class-level S(x,e_k) with categorical "
            "soft targets, then expands C[i,j]=S(x_i,e_{y_j}), DeltaL=C-diag(C), "
            "and one-sided H^2 from exp(DeltaL/2).",
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
