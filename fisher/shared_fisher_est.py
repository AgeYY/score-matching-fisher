"""Shared Fisher estimation: score vs decoder vs ground truth (core logic)."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from global_setting import SCORE_VAL_FRACTION

from fisher.data import (
    ToyConditionalGMMNonGaussianDataset,
    ToyConditionalGaussianDataset,
    ToyConditionalGaussianCosineRandampSqrtdDataset,
    ToyConditionalGaussianRandampDataset,
    ToyConditionalGaussianRandampSqrtdDataset,
    ToyConditionalGaussianSqrtdDataset,
    ToyCosSinPiecewiseNoiseDataset,
    ToyLinearPiecewiseNoiseDataset,
)
from fisher.evaluation import evaluate_score_fisher, evaluate_score_fisher_with_prior, parse_sigma_alpha_list
from fisher.h_matrix import HMatrixEstimator, HMatrixResult
from fisher.models import (
    ConditionalScore1D,
    ConditionalScore1DFiLMPerLayer,
    ConditionalThetaFlowVelocity,
    ConditionalThetaFlowVelocityFiLMPerLayer,
    ConditionalThetaFlowVelocitySoftMoE,
    ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer,
    ConditionalThetaFlowVelocityThetaFourierMLP,
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXFlowVelocityIndependentMLP,
    ConditionalXFlowVelocityIndependentThetaFourierMLP,
    ConditionalXFlowVelocityThetaFourierFiLMPerLayer,
    ConditionalXFlowVelocityThetaFourierMLP,
    LocalDecoderLogit,
    PriorScore1D,
    PriorScore1DFiLMPerLayer,
    PriorThetaFlowVelocity,
    PriorThetaFlowVelocityFiLMPerLayer,
    PriorThetaFlowVelocityThetaFourierFiLMPerLayer,
    PriorThetaFlowVelocityThetaFourierMLP,
)
from fisher.dataset_family_recipes import raise_if_legacy_dataset_family, raise_if_removed_dataset_family
from fisher.ctsm_models import (
    PairConditionedTimeScoreNetBase,
    ToyPairConditionedTimeScoreNet,
    ToyPairConditionedTimeScoreNetFiLM,
)
from fisher.ctsm_objectives import ctsm_v_pair_conditioned_loss
from fisher.ctsm_paths import TwoSB
from fisher.shared_dataset_io import (
    SHARED_DATASET_META_KEYS,
    apply_sigma_defaults_for_dataset_family,
    meta_dict_from_args,
)


def effective_theta_fourier_omega_for_prefix(args: Any, prefix: str) -> tuple[float, str]:
    """Compute scalar ``omega`` for ``sin(k * omega * theta)`` / ``cos(...)``.

    ``prefix`` is the argparse stem without ``_omega_mode`` / ``_omega``, e.g.
    ``flow_x_theta_fourier``, ``flow_theta_fourier``, ``flow_prior_theta_fourier``.

    - ``fixed``: use ``{prefix}_omega`` directly (e.g. ``omega=1`` gives period ``2π`` for k=1).
    - ``theta_range`` (default): ``omega_eff = (2π / span) * mult`` where ``mult`` is ``{prefix}_omega``.
    """
    mode_key = f"{prefix}_omega_mode"
    omega_key = f"{prefix}_omega"
    mode = str(getattr(args, mode_key, "theta_range")).strip().lower()
    mult = float(getattr(args, omega_key, 1.0))
    if mode == "fixed":
        return mult, f"omega_mode=fixed omega={mult:.6g}"
    if mode == "theta_range":
        lo = float(getattr(args, "theta_low", -6.0))
        hi = float(getattr(args, "theta_high", 6.0))
        span = hi - lo
        if not (span > 0.0):
            raise ValueError(
                f"{prefix} theta_fourier: omega_mode=theta_range requires theta_high > theta_low "
                "(use dataset meta or --theta-low / --theta-high)."
            )
        omega_eff = (2.0 * math.pi) / span * mult
        return omega_eff, f"omega_mode=theta_range span={span:.6g} mult={mult:.6g} omega_eff={omega_eff:.6g}"
    raise ValueError(f"{mode_key} must be one of {{'fixed', 'theta_range'}} (got {mode!r}).")


def effective_flow_x_theta_fourier_omega(args: Any) -> tuple[float, str]:
    """Same as :func:`effective_theta_fourier_omega_for_prefix` with ``flow_x_theta_fourier`` prefix."""
    return effective_theta_fourier_omega_for_prefix(args, "flow_x_theta_fourier")


def effective_flow_theta_fourier_omega_post(args: Any) -> tuple[float, str]:
    """Posterior theta-flow Fourier omega (``--flow-theta-fourier-*``)."""
    return effective_theta_fourier_omega_for_prefix(args, "flow_theta_fourier")


def effective_flow_theta_fourier_omega_prior(args: Any) -> tuple[float, str]:
    """Prior theta-flow Fourier omega (``--flow-prior-theta-fourier-*``)."""
    return effective_theta_fourier_omega_for_prefix(args, "flow_prior_theta_fourier")
from fisher.trainers import (
    geometric_sigma_schedule,
    train_conditional_theta_flow_model,
    train_conditional_x_flow_model,
    train_local_decoder,
    train_prior_theta_flow_model,
    train_prior_score_model,
    train_prior_score_model_ncsm_continuous,
    train_score_model,
    train_score_model_ncsm_continuous,
)


def require_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")
    return torch.device(name)


def normalize_theta_field_method(method: str) -> str:
    m = str(method).strip().lower()
    if m in ("theta_flow", "theta_path_integral", "x_flow", "ctsm_v"):
        return m
    legacy_names = (
        "flow",
        "flow_likelihood",
        "flow_x_likelihood",
        "dsm",
    )
    if m in legacy_names:
        raise ValueError(
            "Legacy --theta-field-method is removed. Use one of "
            "{'theta_flow', 'theta_path_integral', 'x_flow', 'ctsm_v'}. "
            "theta_flow = theta-space flow ODE log-likelihood Bayes ratios; "
            "theta_path_integral = velocity-to-score plus trapezoid integral along sorted theta."
        )
    raise ValueError(
        "--theta-field-method must be one of {'theta_flow','theta_path_integral','x_flow','ctsm_v'}."
    )


def normalize_flow_arch(args: Any) -> str:
    arch = str(getattr(args, "flow_arch", "mlp")).strip().lower()
    if arch in ("mlp", "soft_moe", "film", "film_fourier"):
        return arch
    raise ValueError("--flow-arch must be one of {'mlp','soft_moe','film','film_fourier'}.")


def build_posterior_score_model(
    args: Any, device: torch.device
) -> ConditionalScore1D | ConditionalScore1DFiLMPerLayer:
    """Instantiate posterior DSM (theta score) as MLP or FiLM (x-trunk + residual FiLM)."""
    sigma_mode = str(getattr(args, "score_sigma_feature_mode", "auto")).lower()
    if sigma_mode == "auto":
        use_log_sigma = bool(getattr(args, "score_noise_mode", "continuous") == "continuous")
    elif sigma_mode == "log":
        use_log_sigma = True
    elif sigma_mode == "linear":
        use_log_sigma = False
    else:
        raise ValueError("--score-sigma-feature-mode must be one of {'auto','log','linear'}.")
    arch = str(getattr(args, "score_arch", "mlp")).lower()
    common = dict(
        x_dim=int(args.x_dim),
        hidden_dim=int(args.score_hidden_dim),
        depth=int(args.score_depth),
        use_log_sigma=use_log_sigma,
        use_layer_norm=bool(getattr(args, "score_use_layer_norm", False)),
        zero_out_init=bool(getattr(args, "score_zero_out_init", False)),
    )
    if arch == "film":
        return ConditionalScore1DFiLMPerLayer(
            **common,
            gated_film=bool(getattr(args, "score_gated_film", False)),
        ).to(device)
    if arch == "mlp":
        return ConditionalScore1D(**common).to(device)
    raise ValueError(f"Unknown --score-arch: {arch!r} (expected 'mlp' or 'film').")


def build_prior_score_model(args: Any, device: torch.device) -> PriorScore1D | PriorScore1DFiLMPerLayer:
    """Instantiate prior DSM as MLP or FiLM (theta-trunk + residual FiLM from theta_tilde,sigma)."""
    sigma_mode = str(getattr(args, "prior_sigma_feature_mode", "auto")).lower()
    if sigma_mode == "auto":
        use_log_sigma = bool(getattr(args, "score_noise_mode", "continuous") == "continuous")
    elif sigma_mode == "log":
        use_log_sigma = True
    elif sigma_mode == "linear":
        use_log_sigma = False
    else:
        raise ValueError("--prior-sigma-feature-mode must be one of {'auto','log','linear'}.")
    arch = str(getattr(args, "prior_score_arch", "mlp")).lower()
    common = dict(
        hidden_dim=int(getattr(args, "prior_hidden_dim", 128)),
        depth=int(getattr(args, "prior_depth", 3)),
        use_log_sigma=use_log_sigma,
        use_layer_norm=bool(getattr(args, "prior_use_layer_norm", False)),
        zero_out_init=bool(getattr(args, "prior_zero_out_init", False)),
    )
    if arch == "film":
        return PriorScore1DFiLMPerLayer(
            **common,
            gated_film=bool(getattr(args, "prior_gated_film", False)),
        ).to(device)
    if arch == "mlp":
        return PriorScore1D(**common).to(device)
    raise ValueError(f"Unknown --prior-score-arch: {arch!r} (expected 'mlp' or 'film').")


def _apply_dsm_stability_preset(args: Any) -> None:
    preset = str(getattr(args, "dsm_stability_preset", "stable_v1")).lower()
    if preset == "legacy":
        return
    if preset != "stable_v1":
        raise ValueError("--dsm-stability-preset must be one of {'legacy','stable_v1'}.")
    # Conservative defaults aimed at robust convergence.
    # Only override values still at legacy defaults so explicit CLI values are respected.
    legacy_to_stable = {
        "score_optimizer": ("adam", "adamw"),
        "prior_optimizer": ("adam", "adamw"),
        "score_weight_decay": (0.0, 1e-4),
        "prior_weight_decay": (0.0, 1e-4),
        "score_lr_scheduler": ("none", "cosine"),
        "prior_lr_scheduler": ("none", "cosine"),
        "score_lr_warmup_frac": (0.0, 0.05),
        "prior_lr_warmup_frac": (0.0, 0.05),
        "score_max_grad_norm": (0.0, 1.0),
        "prior_max_grad_norm": (0.0, 1.0),
        "score_abort_on_nonfinite": (False, True),
        "prior_abort_on_nonfinite": (False, True),
        "score_loss_type": ("mse", "huber"),
        "prior_loss_type": ("mse", "huber"),
        "score_huber_delta": (1.0, 1.0),
        "prior_huber_delta": (1.0, 1.0),
        "score_sigma_sample_mode": ("uniform_log", "uniform_log"),
        "score_sigma_sample_beta": (2.0, 2.0),
    }
    for key, (legacy_val, stable_val) in legacy_to_stable.items():
        cur = getattr(args, key, legacy_val)
        if cur == legacy_val:
            setattr(args, key, stable_val)


def analytic_fisher_curve(
    centers: np.ndarray,
    dataset: ToyConditionalGaussianDataset | ToyCosSinPiecewiseNoiseDataset | ToyLinearPiecewiseNoiseDataset,
) -> np.ndarray:
    t = centers.reshape(-1, 1)
    dmu = dataset.tuning_curve_derivative(t)
    cov = dataset.covariance(t)
    dcov = dataset.covariance_derivative(t)
    inv_cov = np.linalg.inv(cov)
    mean_term = np.einsum("bi,bij,bj->b", dmu, inv_cov, dmu)
    a = np.einsum("bij,bjk->bik", inv_cov, dcov)
    b = np.einsum("bij,bjk->bik", a, inv_cov)
    c = np.einsum("bij,bjk->bik", b, dcov)
    cov_term = 0.5 * np.einsum("bii->b", c)
    fisher = mean_term + cov_term
    return fisher.astype(np.float64)


def gt_fisher_curve_exact_score_mc(
    centers: np.ndarray,
    dataset: ToyConditionalGMMNonGaussianDataset,
    mc_samples_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    fisher = np.full(centers.shape[0], np.nan, dtype=np.float64)
    se = np.full(centers.shape[0], np.nan, dtype=np.float64)
    for i, th in enumerate(centers):
        t = np.full((mc_samples_per_bin, 1), fill_value=float(th), dtype=np.float64)
        x = dataset.sample_x(t)
        s = dataset.score_theta_exact(x, t)
        sq = s**2
        fisher[i] = float(np.mean(sq))
        se[i] = float(np.std(sq, ddof=1) / np.sqrt(mc_samples_per_bin))
    return fisher, se


def compute_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    a = pred[valid]
    b = gt[valid]
    if a.size == 0:
        return {"n_valid": 0.0, "rmse": float("nan"), "mae": float("nan"), "corr": float("nan")}
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
    return {"n_valid": float(a.size), "rmse": rmse, "mae": mae, "corr": corr}


def _save_h_matrix_dsm_artifacts(
    args: Any,
    h_result: HMatrixResult,
    suffix: str,
) -> tuple[str, str, str, str]:
    """Write h_matrix_results*.npz, summary txt, symmetric heatmap, optional DeltaL heatmap."""
    h_npz_path = os.path.join(args.output_dir, f"h_matrix_results{suffix}.npz")
    h_payload: dict[str, Any] = {
        "theta_used": h_result.theta_used,
        "theta_sorted": h_result.theta_sorted,
        "perm": h_result.perm.astype(np.int64),
        "inv_perm": h_result.inv_perm.astype(np.int64),
        "h_directed": h_result.h_directed,
        "h_sym": h_result.h_sym,
        "sigma_eval": np.asarray([h_result.sigma_eval], dtype=np.float64),
        "h_field_method": np.asarray([h_result.field_method], dtype=object),
        "h_eval_scalar_name": np.asarray([h_result.eval_scalar_name], dtype=object),
        "n_samples": np.asarray([h_result.theta_used.size], dtype=np.int32),
        "order_mode": np.asarray([h_result.order_mode], dtype=object),
        "delta_diag_max_abs": np.asarray([h_result.delta_diag_max_abs], dtype=np.float64),
        "h_sym_max_asym_abs": np.asarray([h_result.h_sym_max_asym_abs], dtype=np.float64),
    }
    if h_result.flow_scheduler is not None:
        h_payload["h_flow_scheduler"] = np.asarray([h_result.flow_scheduler], dtype=object)
    if h_result.flow_score_mode is not None:
        h_payload["h_flow_score_mode"] = np.asarray([h_result.flow_score_mode], dtype=object)
    if bool(getattr(args, "h_save_intermediates", False)):
        h_payload["g_matrix"] = h_result.g_matrix
        h_payload["c_matrix"] = h_result.c_matrix
        h_payload["delta_l_matrix"] = h_result.delta_l_matrix
    np.savez(h_npz_path, **h_payload)

    h_summary_path = os.path.join(args.output_dir, f"h_matrix_summary{suffix}.txt")
    with open(h_summary_path, "w", encoding="utf-8") as hf:
        hf.write("H-matrix estimation summary\n")
        hf.write(f"dataset_family: {args.dataset_family}\n")
        hf.write(f"field_method: {h_result.field_method}\n")
        hf.write(f"eval_scalar_name: {h_result.eval_scalar_name}\n")
        hf.write(f"eval_scalar_value: {h_result.sigma_eval}\n")
        hf.write(f"n_samples: {h_result.theta_used.size}\n")
        hf.write(f"order_mode: {h_result.order_mode}\n")
        hf.write(f"h_shape: {h_result.h_sym.shape}\n")
        hf.write(f"h_sym_min: {float(np.min(h_result.h_sym))}\n")
        hf.write(f"h_sym_max: {float(np.max(h_result.h_sym))}\n")
        hf.write(f"h_sym_diag_max_abs: {float(np.max(np.abs(np.diag(h_result.h_sym))))}\n")
        hf.write(f"delta_diag_max_abs: {h_result.delta_diag_max_abs}\n")
        hf.write(f"h_sym_max_asym_abs: {h_result.h_sym_max_asym_abs}\n")
        if h_result.flow_scheduler is not None:
            hf.write(f"flow_scheduler: {h_result.flow_scheduler}\n")
        if h_result.flow_score_mode is not None:
            hf.write(f"flow_score_mode: {h_result.flow_score_mode}\n")
        hf.write(f"h_save_intermediates: {bool(getattr(args, 'h_save_intermediates', False))}\n")

    h_fig_path = os.path.join(args.output_dir, f"h_matrix_sym_heatmap{suffix}.png")
    plt.figure(figsize=(6.2, 5.6))
    im = plt.imshow(h_result.h_sym, aspect="auto", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=r"$H^{sym}_{ij}$")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Symmetric H-matrix heatmap")
    plt.tight_layout()
    plt.savefig(h_fig_path, dpi=180)
    plt.close()

    h_delta_fig_path = ""
    if bool(getattr(args, "h_save_intermediates", False)):
        h_delta_fig_path = os.path.join(args.output_dir, f"delta_l_heatmap{suffix}.png")
        plt.figure(figsize=(6.2, 5.6))
        im = plt.imshow(h_result.delta_l_matrix, aspect="auto", origin="lower")
        plt.colorbar(im, fraction=0.046, pad=0.04, label=r"$\Delta L_{ij}$")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.title("Directed log-ratio heatmap")
        plt.tight_layout()
        plt.savefig(h_delta_fig_path, dpi=180)
        plt.close()

    return h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path


def posterior_proxy_sigma(theta: np.ndarray, x: np.ndarray, l2: float) -> float:
    if l2 < 0.0:
        raise ValueError("score-proxy-l2 must be non-negative.")
    y = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] != y.shape[0]:
        raise ValueError("x must be 2D and match theta rows.")
    x_aug = np.concatenate([np.ones((xx.shape[0], 1), dtype=np.float64), xx], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float64)
    reg[0, 0] = 0.0
    w = np.linalg.solve(xtx + l2 * reg, x_aug.T @ y)
    resid = y - x_aug @ w
    sigma_post = float(np.std(resid.reshape(-1)))
    return max(sigma_post, 1e-8)


def _subset_x_by_theta(
    theta: np.ndarray,
    x: np.ndarray,
    target: float,
    bandwidth: float,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.where(np.abs(theta.reshape(-1) - target) <= bandwidth)[0]
    if idx.size == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    if cap > 0 and idx.size > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    return x[idx]


def decoder_min_ntr_for_fit(min_class_count: int, val_frac: float, min_val_class_size: int) -> int:
    """Smallest balanced per-class train count ``ntr`` such that after the validation holdout, ``nfit >= min_class_count``."""
    if min_class_count < 1:
        return 1
    for ntr in range(1, 500000):
        nval = int(round(float(val_frac) * ntr))
        nval = max(int(min_val_class_size), nval)
        nval = min(nval, ntr - 1)
        if nval < 1:
            continue
        nfit = ntr - nval
        if nfit >= min_class_count:
            return int(ntr)
    return -1


def fit_decoder_from_shared_data(
    centers: np.ndarray,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    epsilon: float,
    bandwidth: float,
    min_class_count: int,
    train_cap: int,
    eval_cap: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    depth: int,
    val_frac: float,
    min_val_class_size: int,
    early_patience: int,
    early_min_delta: float,
    early_ema_alpha: float,
    restore_best: bool,
    device: torch.device,
    log_every: int,
    rng: np.random.Generator,
    *,
    debug_bins: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fisher = np.full(centers.size, np.nan, dtype=np.float64)
    se = np.full(centers.size, np.nan, dtype=np.float64)
    valid = np.zeros(centers.size, dtype=bool)

    n_centers = int(centers.size)
    reasons = np.full(n_centers, "", dtype=object)
    ntr_arr = np.full(n_centers, np.nan, dtype=np.float64)
    nev_arr = np.full(n_centers, np.nan, dtype=np.float64)
    nval_arr = np.full(n_centers, np.nan, dtype=np.float64)
    nfit_arr = np.full(n_centers, np.nan, dtype=np.float64)
    ntr_pos_raw = np.full(n_centers, np.nan, dtype=np.float64)
    ntr_neg_raw = np.full(n_centers, np.nan, dtype=np.float64)
    nev_pos_raw = np.full(n_centers, np.nan, dtype=np.float64)
    nev_neg_raw = np.full(n_centers, np.nan, dtype=np.float64)

    skip_counts: dict[str, int] = {
        "ok": 0,
        "insufficient_counts": 0,
        "invalid_nval": 0,
        "insufficient_fit_after_val": 0,
    }

    min_ntr_fit = decoder_min_ntr_for_fit(min_class_count, val_frac, min_val_class_size)
    nev_max_scan = 0
    ntr_max_scan = 0
    for theta0 in centers:
        tp = float(theta0 + 0.5 * epsilon)
        tm = float(theta0 - 0.5 * epsilon)
        # cap=0 => no subsampling; avoids consuming rng before the main loop.
        ep = _subset_x_by_theta(theta_eval, x_eval, tp, bandwidth, 0, rng)
        em = _subset_x_by_theta(theta_eval, x_eval, tm, bandwidth, 0, rng)
        nev_max_scan = max(nev_max_scan, min(ep.shape[0], em.shape[0]))
        tp_tr = _subset_x_by_theta(theta_train, x_train, tp, bandwidth, 0, rng)
        tm_tr = _subset_x_by_theta(theta_train, x_train, tm, bandwidth, 0, rng)
        ntr_max_scan = max(ntr_max_scan, min(tp_tr.shape[0], tm_tr.shape[0]))
    print(
        "[decoder] preflight: "
        f"min_class_count={min_class_count}, val_frac={val_frac}, min_val_class_size={min_val_class_size} "
        f"=> minimum balanced ntr needed for nfit>={min_class_count} is ~{min_ntr_fit} "
        f"(and need nev>={min_class_count} per class in eval windows)."
    )
    print(
        "[decoder] preflight scan (no training): "
        f"max balanced ntr≈{ntr_max_scan}, max balanced nev≈{nev_max_scan} "
        f"(if max_nev < min_class_count, bins will skip with insufficient_counts)."
    )
    if nev_max_scan < int(min_class_count):
        print(
            "[decoder] WARNING: max balanced eval count nev "
            f"({nev_max_scan}) < --decoder-min-class-count ({min_class_count}). "
            "Lower --decoder-min-class-count, widen --decoder-bandwidth, or use more eval data."
        )

    for i, theta0 in enumerate(centers):
        theta_plus = float(theta0 + 0.5 * epsilon)
        theta_minus = float(theta0 - 0.5 * epsilon)

        xtr_pos = _subset_x_by_theta(theta_train, x_train, theta_plus, bandwidth, train_cap, rng)
        xtr_neg = _subset_x_by_theta(theta_train, x_train, theta_minus, bandwidth, train_cap, rng)
        xev_pos = _subset_x_by_theta(theta_eval, x_eval, theta_plus, bandwidth, eval_cap, rng)
        xev_neg = _subset_x_by_theta(theta_eval, x_eval, theta_minus, bandwidth, eval_cap, rng)

        ntr_pos_raw[i] = float(xtr_pos.shape[0])
        ntr_neg_raw[i] = float(xtr_neg.shape[0])
        nev_pos_raw[i] = float(xev_pos.shape[0])
        nev_neg_raw[i] = float(xev_neg.shape[0])

        ntr = min(xtr_pos.shape[0], xtr_neg.shape[0])
        nev = min(xev_pos.shape[0], xev_neg.shape[0])
        ntr_arr[i] = float(ntr)
        nev_arr[i] = float(nev)
        if ntr < min_class_count or nev < min_class_count:
            reasons[i] = "insufficient_counts"
            skip_counts["insufficient_counts"] += 1
            if debug_bins:
                print(
                    f"[decoder skip {i+1:3d}/{n_centers}] theta0={theta0:+.5f} reason=insufficient_counts "
                    f"ntr_pos={int(ntr_pos_raw[i])} ntr_neg={int(ntr_neg_raw[i])} ntr={ntr} "
                    f"nev_pos={int(nev_pos_raw[i])} nev_neg={int(nev_neg_raw[i])} nev={nev} "
                    f"(need ntr,nev>={min_class_count})"
                )
            continue

        if xtr_pos.shape[0] != ntr:
            xtr_pos = xtr_pos[rng.choice(xtr_pos.shape[0], size=ntr, replace=False)]
        if xtr_neg.shape[0] != ntr:
            xtr_neg = xtr_neg[rng.choice(xtr_neg.shape[0], size=ntr, replace=False)]
        if xev_pos.shape[0] != nev:
            xev_pos = xev_pos[rng.choice(xev_pos.shape[0], size=nev, replace=False)]
        if xev_neg.shape[0] != nev:
            xev_neg = xev_neg[rng.choice(xev_neg.shape[0], size=nev, replace=False)]

        nval = int(round(float(val_frac) * ntr))
        nval = max(int(min_val_class_size), nval)
        nval = min(nval, ntr - 1)
        nval_arr[i] = float(nval)
        if nval < 1:
            reasons[i] = "invalid_nval"
            skip_counts["invalid_nval"] += 1
            if debug_bins:
                print(
                    f"[decoder skip {i+1:3d}/{n_centers}] theta0={theta0:+.5f} reason=invalid_nval "
                    f"ntr={ntr} nval={nval}"
                )
            continue
        nfit = ntr - nval
        nfit_arr[i] = float(nfit)
        if nfit < min_class_count:
            reasons[i] = "insufficient_fit_after_val"
            skip_counts["insufficient_fit_after_val"] += 1
            if debug_bins:
                print(
                    f"[decoder skip {i+1:3d}/{n_centers}] theta0={theta0:+.5f} reason=insufficient_fit_after_val "
                    f"ntr={ntr} nval={nval} nfit={nfit} (need nfit>={min_class_count}); "
                    f"min_ntr_hint~{min_ntr_fit}"
                )
            continue

        perm_pos = rng.permutation(ntr)
        perm_neg = rng.permutation(ntr)
        pos_fit = xtr_pos[perm_pos[:nfit]]
        pos_val = xtr_pos[perm_pos[nfit:]]
        neg_fit = xtr_neg[perm_neg[:nfit]]
        neg_val = xtr_neg[perm_neg[nfit:]]

        xtr = np.concatenate([pos_fit, neg_fit], axis=0)
        ytr = np.concatenate([np.ones(nfit, dtype=np.float64), np.zeros(nfit, dtype=np.float64)], axis=0)
        xval = np.concatenate([pos_val, neg_val], axis=0)
        yval = np.concatenate([np.ones(nval, dtype=np.float64), np.zeros(nval, dtype=np.float64)], axis=0)

        model = LocalDecoderLogit(x_dim=x_train.shape[1], hidden_dim=hidden_dim, depth=depth).to(device)
        _ = train_local_decoder(
            model=model,
            x_train=xtr,
            y_train=ytr,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            x_val=xval,
            y_val=yval,
            early_stopping_patience=early_patience,
            early_stopping_min_delta=early_min_delta,
            early_stopping_ema_alpha=early_ema_alpha,
            restore_best=restore_best,
            log_every=max(1, log_every),
        )
        model.eval()
        with torch.no_grad():
            mix = np.concatenate([xev_pos, xev_neg], axis=0)
            mix_t = torch.from_numpy(mix.astype(np.float32)).to(device)
            logits_mix = model(mix_t).cpu().numpy().reshape(-1)
        fisher_samples = (logits_mix**2) / (epsilon**2)
        fisher[i] = float(np.mean(fisher_samples))
        se[i] = float(np.std(fisher_samples, ddof=1) / np.sqrt(fisher_samples.size))
        valid[i] = True
        reasons[i] = "ok"
        skip_counts["ok"] += 1

        if i == 0 or (i + 1) % log_every == 0 or (i + 1) == centers.size:
            print(
                f"[decoder theta {i+1:3d}/{centers.size}] theta0={theta0:+.3f} "
                f"ntr={ntr} fit={nfit} val={nval} nev={nev} fisher={fisher[i]:.4f}"
            )

    n_valid = int(np.sum(valid))
    skip_non_ok = {k: v for k, v in skip_counts.items() if k != "ok"}
    top_reason = max(skip_non_ok, key=skip_non_ok.get) if skip_non_ok and sum(skip_non_ok.values()) > 0 else ""

    def _stat_line(name: str, arr: np.ndarray, mask: np.ndarray) -> str:
        vals = arr[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return f"{name}: (no finite values)"
        return (
            f"{name}: min={float(np.min(vals)):.1f}, p50={float(np.median(vals)):.1f}, "
            f"max={float(np.max(vals)):.1f}"
        )

    skipped_mask = ~valid
    print(
        "[decoder] summary: "
        f"valid={n_valid}/{n_centers}, "
        f"skipped insufficient_counts={skip_counts['insufficient_counts']}, "
        f"invalid_nval={skip_counts['invalid_nval']}, "
        f"insufficient_fit_after_val={skip_counts['insufficient_fit_after_val']}"
    )
    if np.any(skipped_mask):
        print(f"  among skipped bins: {_stat_line('ntr', ntr_arr, skipped_mask)}")
        print(f"  among skipped bins: {_stat_line('nev', nev_arr, skipped_mask)}")
        finite_nfit = skipped_mask & np.isfinite(nfit_arr)
        if np.any(finite_nfit):
            print(f"  among skipped (nfit known): {_stat_line('nfit', nfit_arr, finite_nfit)}")

    if n_valid == 0:
        print(
            "[decoder] WARNING: zero valid decoder bins. "
            f"Dominant skip reason: {top_reason} (count={skip_non_ok.get(top_reason, 0)}). "
            "Typical fixes: lower --decoder-min-class-count, lower --decoder-min-val-class-size, "
            "widen --decoder-bandwidth, increase data / caps, or reduce --decoder-epsilon."
        )
    elif min_ntr_fit > 0 and np.nanmax(ntr_arr) < float(min_ntr_fit) and skip_counts["insufficient_fit_after_val"] > 0:
        print(
            "[decoder] NOTE: some bins were skipped because nfit < min_class_count after validation holdout. "
            f"Balanced local ntr must reach ~{min_ntr_fit} for current val settings; "
            "consider lowering --decoder-min-val-class-size or --decoder-min-class-count."
        )

    diag: dict[str, Any] = {
        "skip_counts": dict(skip_counts),
        "min_ntr_for_fit": int(min_ntr_fit),
        "reasons": reasons,
        "ntr": ntr_arr,
        "nev": nev_arr,
        "nval": nval_arr,
        "nfit": nfit_arr,
        "ntr_pos_raw": ntr_pos_raw,
        "ntr_neg_raw": ntr_neg_raw,
        "nev_pos_raw": nev_pos_raw,
        "nev_neg_raw": nev_neg_raw,
        "dominant_skip_reason": top_reason,
        "n_valid": n_valid,
    }
    return fisher, se, valid, diag


def build_dataset_from_meta(
    meta: dict[str, Any],
) -> (
    ToyConditionalGaussianDataset
    | ToyConditionalGaussianSqrtdDataset
    | ToyConditionalGaussianCosineRandampSqrtdDataset
    | ToyConditionalGaussianRandampDataset
    | ToyConditionalGaussianRandampSqrtdDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset
):
    family = str(meta["dataset_family"])
    raise_if_removed_dataset_family(family)
    raise_if_legacy_dataset_family(family)
    seed = int(meta["seed"])
    if family in ("cosine_gaussian", "cosine_gaussian_const_noise"):
        return ToyConditionalGaussianDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            tuning_curve_family=str(meta.get("tuning_curve_family", "cosine")),
            vm_mu_amp=float(meta.get("vm_mu_amp", 1.0)),
            vm_kappa=float(meta.get("vm_kappa", 1.0)),
            vm_omega=float(meta.get("vm_omega", 1.0)),
            gauss_mu_amp=float(meta.get("gauss_mu_amp", 1.0)),
            gauss_kappa=float(meta.get("gauss_kappa", 0.2)),
            gauss_omega=float(meta.get("gauss_omega", 1.0)),
            sigma_x1=float(meta["sigma_x1"]),
            sigma_x2=float(meta["sigma_x2"]),
            rho=float(meta["rho"]),
            cov_theta_amp1=float(meta["cov_theta_amp1"]),
            cov_theta_amp2=float(meta["cov_theta_amp2"]),
            cov_theta_amp_rho=float(meta["cov_theta_amp_rho"]),
            cov_theta_freq1=float(meta["cov_theta_freq1"]),
            cov_theta_freq2=float(meta["cov_theta_freq2"]),
            cov_theta_freq_rho=float(meta["cov_theta_freq_rho"]),
            cov_theta_phase1=float(meta["cov_theta_phase1"]),
            cov_theta_phase2=float(meta["cov_theta_phase2"]),
            cov_theta_phase_rho=float(meta["cov_theta_phase_rho"]),
            rho_clip=float(meta["rho_clip"]),
            seed=seed,
        )
    if family == "cosine_gaussian_sqrtd":
        return ToyConditionalGaussianSqrtdDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            tuning_curve_family=str(meta.get("tuning_curve_family", "cosine")),
            vm_mu_amp=float(meta.get("vm_mu_amp", 1.0)),
            vm_kappa=float(meta.get("vm_kappa", 1.0)),
            vm_omega=float(meta.get("vm_omega", 1.0)),
            gauss_mu_amp=float(meta.get("gauss_mu_amp", 1.0)),
            gauss_kappa=float(meta.get("gauss_kappa", 0.2)),
            gauss_omega=float(meta.get("gauss_omega", 1.0)),
            sigma_x1=float(meta["sigma_x1"]),
            sigma_x2=float(meta["sigma_x2"]),
            rho=float(meta["rho"]),
            cov_theta_amp1=float(meta["cov_theta_amp1"]),
            cov_theta_amp2=float(meta["cov_theta_amp2"]),
            cov_theta_amp_rho=float(meta["cov_theta_amp_rho"]),
            cov_theta_freq1=float(meta["cov_theta_freq1"]),
            cov_theta_freq2=float(meta["cov_theta_freq2"]),
            cov_theta_freq_rho=float(meta["cov_theta_freq_rho"]),
            cov_theta_phase1=float(meta["cov_theta_phase1"]),
            cov_theta_phase2=float(meta["cov_theta_phase2"]),
            cov_theta_phase_rho=float(meta["cov_theta_phase_rho"]),
            rho_clip=float(meta["rho_clip"]),
            seed=seed,
        )
    if family == "cosine_gaussian_sqrtd_rand_tune":
        cta_raw = meta.get("cosine_tune_amp_per_dim")
        cta: np.ndarray | None
        if cta_raw is not None:
            cta = np.asarray(cta_raw, dtype=np.float64).reshape(-1)
        else:
            cta = None
        return ToyConditionalGaussianCosineRandampSqrtdDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            tuning_curve_family=str(meta.get("tuning_curve_family", "cosine")),
            vm_mu_amp=float(meta.get("vm_mu_amp", 1.0)),
            vm_kappa=float(meta.get("vm_kappa", 1.0)),
            vm_omega=float(meta.get("vm_omega", 1.0)),
            gauss_mu_amp=float(meta.get("gauss_mu_amp", 1.0)),
            gauss_kappa=float(meta.get("gauss_kappa", 0.2)),
            gauss_omega=float(meta.get("gauss_omega", 1.0)),
            sigma_x1=float(meta["sigma_x1"]),
            sigma_x2=float(meta["sigma_x2"]),
            rho=float(meta["rho"]),
            cov_theta_amp1=float(meta["cov_theta_amp1"]),
            cov_theta_amp2=float(meta["cov_theta_amp2"]),
            cov_theta_amp_rho=float(meta["cov_theta_amp_rho"]),
            cov_theta_freq1=float(meta["cov_theta_freq1"]),
            cov_theta_freq2=float(meta["cov_theta_freq2"]),
            cov_theta_freq_rho=float(meta["cov_theta_freq_rho"]),
            cov_theta_phase1=float(meta["cov_theta_phase1"]),
            cov_theta_phase2=float(meta["cov_theta_phase2"]),
            cov_theta_phase_rho=float(meta["cov_theta_phase_rho"]),
            rho_clip=float(meta["rho_clip"]),
            cosine_tune_amp_low=float(meta.get("cosine_tune_amp_low", 0.5)),
            cosine_tune_amp_high=float(meta.get("cosine_tune_amp_high", 1.5)),
            cosine_tune_amp_per_dim=cta,
            seed=seed,
        )
    if family == "randamp_gaussian":
        amps_raw = meta.get("randamp_mu_amp_per_dim")
        amps: np.ndarray | None
        if amps_raw is not None:
            amps = np.asarray(amps_raw, dtype=np.float64).reshape(-1)
        else:
            amps = None
        return ToyConditionalGaussianRandampDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            tuning_curve_family=str(meta.get("tuning_curve_family", "cosine")),
            vm_mu_amp=float(meta.get("vm_mu_amp", 1.0)),
            vm_kappa=float(meta.get("vm_kappa", 1.0)),
            vm_omega=float(meta.get("vm_omega", 1.0)),
            gauss_mu_amp=float(meta.get("gauss_mu_amp", 1.0)),
            gauss_kappa=float(meta.get("gauss_kappa", 0.2)),
            gauss_omega=float(meta.get("gauss_omega", 1.0)),
            sigma_x1=float(meta["sigma_x1"]),
            sigma_x2=float(meta["sigma_x2"]),
            rho=float(meta["rho"]),
            cov_theta_amp1=float(meta["cov_theta_amp1"]),
            cov_theta_amp2=float(meta["cov_theta_amp2"]),
            cov_theta_amp_rho=float(meta["cov_theta_amp_rho"]),
            cov_theta_freq1=float(meta["cov_theta_freq1"]),
            cov_theta_freq2=float(meta["cov_theta_freq2"]),
            cov_theta_freq_rho=float(meta["cov_theta_freq_rho"]),
            cov_theta_phase1=float(meta["cov_theta_phase1"]),
            cov_theta_phase2=float(meta["cov_theta_phase2"]),
            cov_theta_phase_rho=float(meta["cov_theta_phase_rho"]),
            rho_clip=float(meta["rho_clip"]),
            randamp_mu_low=float(meta.get("randamp_mu_low", 0.5)),
            randamp_mu_high=float(meta.get("randamp_mu_high", 1.5)),
            randamp_kappa=float(meta.get("randamp_kappa", 0.2)),
            randamp_omega=float(meta.get("randamp_omega", 1.0)),
            randamp_mu_amp_per_dim=amps,
            seed=seed,
        )
    if family == "randamp_gaussian_sqrtd":
        amps_raw = meta.get("randamp_mu_amp_per_dim")
        amps_sqrt: np.ndarray | None
        if amps_raw is not None:
            amps_sqrt = np.asarray(amps_raw, dtype=np.float64).reshape(-1)
        else:
            amps_sqrt = None
        gen_x_dim = int(meta["x_dim"])
        if bool(meta.get("pr_autoencoder_embedded", False)):
            gen_x_dim = int(meta.get("pr_autoencoder_z_dim", gen_x_dim))
        return ToyConditionalGaussianRandampSqrtdDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=gen_x_dim,
            tuning_curve_family=str(meta.get("tuning_curve_family", "cosine")),
            vm_mu_amp=float(meta.get("vm_mu_amp", 1.0)),
            vm_kappa=float(meta.get("vm_kappa", 1.0)),
            vm_omega=float(meta.get("vm_omega", 1.0)),
            gauss_mu_amp=float(meta.get("gauss_mu_amp", 1.0)),
            gauss_kappa=float(meta.get("gauss_kappa", 0.2)),
            gauss_omega=float(meta.get("gauss_omega", 1.0)),
            sigma_x1=float(meta["sigma_x1"]),
            sigma_x2=float(meta["sigma_x2"]),
            rho=float(meta["rho"]),
            cov_theta_amp1=float(meta["cov_theta_amp1"]),
            cov_theta_amp2=float(meta["cov_theta_amp2"]),
            cov_theta_amp_rho=float(meta["cov_theta_amp_rho"]),
            cov_theta_freq1=float(meta["cov_theta_freq1"]),
            cov_theta_freq2=float(meta["cov_theta_freq2"]),
            cov_theta_freq_rho=float(meta["cov_theta_freq_rho"]),
            cov_theta_phase1=float(meta["cov_theta_phase1"]),
            cov_theta_phase2=float(meta["cov_theta_phase2"]),
            cov_theta_phase_rho=float(meta["cov_theta_phase_rho"]),
            rho_clip=float(meta["rho_clip"]),
            randamp_mu_low=float(meta.get("randamp_mu_low", 0.5)),
            randamp_mu_high=float(meta.get("randamp_mu_high", 1.5)),
            randamp_kappa=float(meta.get("randamp_kappa", 0.2)),
            randamp_omega=float(meta.get("randamp_omega", 1.0)),
            randamp_mu_amp_per_dim=amps_sqrt,
            seed=seed,
        )
    if family == "cosine_gmm":
        return ToyConditionalGMMNonGaussianDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            tuning_curve_family=str(meta.get("tuning_curve_family", "cosine")),
            vm_mu_amp=float(meta.get("vm_mu_amp", 1.0)),
            vm_kappa=float(meta.get("vm_kappa", 1.0)),
            vm_omega=float(meta.get("vm_omega", 1.0)),
            gauss_mu_amp=float(meta.get("gauss_mu_amp", 1.0)),
            gauss_kappa=float(meta.get("gauss_kappa", 0.2)),
            gauss_omega=float(meta.get("gauss_omega", 1.0)),
            sigma_x1=float(meta["sigma_x1"]),
            sigma_x2=float(meta["sigma_x2"]),
            rho=float(meta["rho"]),
            sep_scale=float(meta["gmm_sep_scale"]),
            sep_freq=float(meta["gmm_sep_freq"]),
            sep_phase=float(meta["gmm_sep_phase"]),
            mix_logit_scale=float(meta["gmm_mix_logit_scale"]),
            mix_bias=float(meta["gmm_mix_bias"]),
            mix_freq=float(meta["gmm_mix_freq"]),
            mix_phase=float(meta["gmm_mix_phase"]),
            seed=seed,
        )
    if family == "cos_sin_piecewise":
        return ToyCosSinPiecewiseNoiseDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            sigma_piecewise_low=float(meta.get("sigma_piecewise_low", 0.1)),
            sigma_piecewise_high=float(meta.get("sigma_piecewise_high", 0.1)),
            theta_zero_to_low=bool(meta.get("theta_zero_to_low", True)),
            seed=seed,
        )
    if family == "linear_piecewise":
        return ToyLinearPiecewiseNoiseDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            linear_k=float(meta.get("linear_k", 1.0)),
            sigma_piecewise_low=float(meta.get("sigma_piecewise_low", 0.1)),
            sigma_piecewise_high=float(meta.get("sigma_piecewise_high", 0.1)),
            linear_sigma_schedule=str(meta.get("linear_sigma_schedule", "sigmoid")),
            linear_sigma_sigmoid_center=float(meta.get("linear_sigma_sigmoid_center", 0.0)),
            linear_sigma_sigmoid_steepness=float(meta.get("linear_sigma_sigmoid_steepness", 2.0)),
            theta_zero_to_low=bool(meta.get("theta_zero_to_low", True)),
            seed=seed,
        )
    raise ValueError(f"Unknown dataset_family: {family}")


def build_dataset_from_args(
    ns: Any,
) -> (
    ToyConditionalGaussianDataset
    | ToyConditionalGaussianSqrtdDataset
    | ToyConditionalGaussianCosineRandampSqrtdDataset
    | ToyConditionalGaussianRandampDataset
    | ToyConditionalGaussianRandampSqrtdDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset
):
    return build_dataset_from_meta(meta_dict_from_args(ns))


def validate_dataset_sample_args(args: Any) -> None:
    """Validate generic dataset args; per-family tuning/noise is fixed via ``apply_family_recipe_to_namespace``."""
    apply_sigma_defaults_for_dataset_family(args)
    _ons = float(getattr(args, "obs_noise_scale", 1.0))
    if not math.isfinite(_ons) or _ons <= 0.0:
        raise ValueError("--obs-noise-scale must be a finite positive number.")
    if args.x_dim < 1:
        raise ValueError("--x-dim must be >= 1.")
    if str(getattr(args, "dataset_family", "")) in (
        "cos_sin_piecewise",
        "linear_piecewise",
    ) and int(args.x_dim) != 2:
        raise ValueError("--dataset-family cos_sin_piecewise / linear_piecewise requires --x-dim 2.")
    if int(args.n_total) < 2:
        raise ValueError("--n-total must be >= 2 for train/validation split.")
    if not (0.0 < float(args.train_frac) <= 1.0):
        raise ValueError("--train-frac must be in (0, 1].")


def validate_estimation_args(args: Any) -> None:
    _apply_dsm_stability_preset(args)
    _tfm_val = normalize_theta_field_method(str(getattr(args, "theta_field_method", "theta_flow")))
    _arch = normalize_flow_arch(args)
    setattr(args, "theta_field_method", _tfm_val)
    setattr(args, "flow_arch", _arch)
    legacy_flow_score_arch = getattr(args, "flow_score_arch", None)
    legacy_flow_prior_arch = getattr(args, "flow_prior_arch", None)
    if legacy_flow_score_arch is not None:
        raise ValueError(
            "Legacy --flow-score-arch is removed. Use --flow-arch {mlp,soft_moe,film,film_fourier}."
        )
    if legacy_flow_prior_arch is not None:
        raise ValueError(
            "Legacy --flow-prior-arch is removed. Use --flow-arch {mlp,soft_moe,film,film_fourier}."
        )
    if args.score_eval_sigmas < 1:
        raise ValueError("--score-eval-sigmas must be >= 1.")
    if args.score_early_patience < 1:
        raise ValueError("--score-early-patience must be >= 1.")
    if args.score_early_min_delta < 0.0:
        raise ValueError("--score-early-min-delta must be non-negative.")
    if not (0.0 < float(args.score_early_ema_alpha) <= 1.0):
        raise ValueError("--score-early-ema-alpha must be in (0, 1].")
    if int(getattr(args, "score_early_ema_warmup_epochs", 0)) < 0:
        raise ValueError("--score-early-ema-warmup-epochs must be >= 0.")
    if args.score_sigma_min_alpha <= 0.0 or args.score_sigma_max_alpha <= 0.0:
        raise ValueError("--score-sigma-min-alpha and --score-sigma-max-alpha must be positive.")
    if float(args.score_sigma_min_alpha) > float(args.score_sigma_max_alpha):
        raise ValueError("--score-sigma-min-alpha must be <= --score-sigma-max-alpha.")
    _sa = str(getattr(args, "score_arch", "mlp")).lower()
    _pa = str(getattr(args, "prior_score_arch", "mlp")).lower()
    if _sa not in ("mlp", "film"):
        raise ValueError("--score-arch must be one of {'mlp','film'}.")
    if _pa not in ("mlp", "film"):
        raise ValueError("--prior-score-arch must be one of {'mlp','film'}.")
    if str(getattr(args, "score_sigma_feature_mode", "auto")).lower() not in ("auto", "log", "linear"):
        raise ValueError("--score-sigma-feature-mode must be one of {'auto','log','linear'}.")
    if str(getattr(args, "prior_sigma_feature_mode", "auto")).lower() not in ("auto", "log", "linear"):
        raise ValueError("--prior-sigma-feature-mode must be one of {'auto','log','linear'}.")
    if str(getattr(args, "score_optimizer", "adamw")).lower() not in ("adam", "adamw"):
        raise ValueError("--score-optimizer must be one of {'adam','adamw'}.")
    if str(getattr(args, "prior_optimizer", "adamw")).lower() not in ("adam", "adamw"):
        raise ValueError("--prior-optimizer must be one of {'adam','adamw'}.")
    if float(getattr(args, "score_weight_decay", 0.0)) < 0.0:
        raise ValueError("--score-weight-decay must be >= 0.")
    if float(getattr(args, "prior_weight_decay", 0.0)) < 0.0:
        raise ValueError("--prior-weight-decay must be >= 0.")
    if str(getattr(args, "score_lr_scheduler", "none")).lower() not in ("none", "cosine"):
        raise ValueError("--score-lr-scheduler must be one of {'none','cosine'}.")
    if str(getattr(args, "prior_lr_scheduler", "none")).lower() not in ("none", "cosine"):
        raise ValueError("--prior-lr-scheduler must be one of {'none','cosine'}.")
    if not (0.0 <= float(getattr(args, "score_lr_warmup_frac", 0.0)) < 1.0):
        raise ValueError("--score-lr-warmup-frac must be in [0,1).")
    if not (0.0 <= float(getattr(args, "prior_lr_warmup_frac", 0.0)) < 1.0):
        raise ValueError("--prior-lr-warmup-frac must be in [0,1).")
    if float(getattr(args, "score_huber_delta", 1.0)) <= 0.0:
        raise ValueError("--score-huber-delta must be positive.")
    if float(getattr(args, "prior_huber_delta", 1.0)) <= 0.0:
        raise ValueError("--prior-huber-delta must be positive.")
    if float(getattr(args, "score_max_grad_norm", 1.0)) < 0.0:
        raise ValueError("--score-max-grad-norm must be >= 0 (0 disables clipping).")
    if float(getattr(args, "prior_max_grad_norm", 1.0)) < 0.0:
        raise ValueError("--prior-max-grad-norm must be >= 0 (0 disables clipping).")
    if str(getattr(args, "score_loss_type", "mse")).lower() not in ("mse", "huber"):
        raise ValueError("--score-loss-type must be one of {'mse','huber'}.")
    if str(getattr(args, "prior_loss_type", "mse")).lower() not in ("mse", "huber"):
        raise ValueError("--prior-loss-type must be one of {'mse','huber'}.")
    if str(getattr(args, "score_sigma_sample_mode", "uniform_log")).lower() not in ("uniform_log", "beta_log"):
        raise ValueError("--score-sigma-sample-mode must be one of {'uniform_log','beta_log'}.")
    if float(getattr(args, "score_sigma_sample_beta", 2.0)) <= 0.0:
        raise ValueError("--score-sigma-sample-beta must be positive.")
    if args.score_proxy_min_mult <= 0.0 or args.score_proxy_max_mult <= 0.0:
        raise ValueError("--score-proxy-min-mult and --score-proxy-max-mult must be positive.")
    if args.score_proxy_min_mult > args.score_proxy_max_mult:
        raise ValueError("--score-proxy-min-mult must be <= --score-proxy-max-mult.")
    if args.score_fixed_sigma <= 0.0:
        raise ValueError("--score-fixed-sigma must be positive.")
    if not (0.0 < args.decoder_val_frac < 1.0):
        raise ValueError("--decoder-val-frac must be in (0, 1).")
    if args.decoder_min_val_class_size < 1:
        raise ValueError("--decoder-min-val-class-size must be >= 1.")
    if args.decoder_early_patience < 1:
        raise ValueError("--decoder-early-patience must be >= 1.")
    if args.decoder_early_min_delta < 0.0:
        raise ValueError("--decoder-early-min-delta must be non-negative.")
    if not (0.0 < float(args.decoder_early_ema_alpha) <= 1.0):
        raise ValueError("--decoder-early-ema-alpha must be in (0, 1].")
    if int(args.decoder_min_class_count) < 1:
        raise ValueError("--decoder-min-class-count must be >= 1.")
    if getattr(args, "prior_epochs", 1) < 1:
        raise ValueError("--prior-epochs must be >= 1.")
    if getattr(args, "prior_early_patience", 1) < 1:
        raise ValueError("--prior-early-patience must be >= 1.")
    if getattr(args, "prior_early_min_delta", 0.0) < 0.0:
        raise ValueError("--prior-early-min-delta must be non-negative.")
    if not (0.0 < float(getattr(args, "prior_early_ema_alpha", 0.05)) <= 1.0):
        raise ValueError("--prior-early-ema-alpha must be in (0, 1].")
    if int(getattr(args, "prior_early_ema_warmup_epochs", 0)) < 0:
        raise ValueError("--prior-early-ema-warmup-epochs must be >= 0.")
    if int(getattr(args, "h_batch_size", 1)) < 1:
        raise ValueError("--h-batch-size must be >= 1.")
    if float(getattr(args, "h_sigma_eval", -1.0)) == 0.0:
        raise ValueError("--h-sigma-eval must be positive, or <= 0 to auto-select sigma_min.")
    if int(getattr(args, "flow_epochs", 1)) < 1:
        raise ValueError("--flow-epochs must be >= 1.")
    if bool(getattr(args, "flow_x_two_stage_mean_theta_pretrain", False)):
        if _tfm_val != "x_flow":
            raise ValueError(
                "--flow-x-two-stage-mean-theta-pretrain is only valid with --theta-field-method x_flow."
            )
        if int(getattr(args, "flow_epochs", 1)) < 2:
            raise ValueError(
                "--flow-x-two-stage-mean-theta-pretrain requires --flow-epochs >= 2 (split floor(E/2) + remainder)."
            )
    if int(getattr(args, "flow_batch_size", 1)) < 1:
        raise ValueError("--flow-batch-size must be >= 1.")
    if float(getattr(args, "flow_lr", 0.0)) <= 0.0:
        raise ValueError("--flow-lr must be positive.")
    if float(getattr(args, "flow_endpoint_loss_weight", 0.0)) < 0.0:
        raise ValueError("--flow-endpoint-loss-weight must be non-negative.")
    if int(getattr(args, "flow_endpoint_steps", 20)) < 1:
        raise ValueError("--flow-endpoint-steps must be >= 1.")
    if int(getattr(args, "flow_early_patience", 1)) < 1:
        raise ValueError("--flow-early-patience must be >= 1.")
    if float(getattr(args, "flow_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--flow-early-min-delta must be non-negative.")
    if not (0.0 < float(getattr(args, "flow_early_ema_alpha", 0.05)) <= 1.0):
        raise ValueError("--flow-early-ema-alpha must be in (0, 1].")
    if not (0.0 <= float(getattr(args, "flow_eval_t", 0.8)) <= 1.0):
        raise ValueError("--flow-eval-t must be in [0, 1].")
    if int(getattr(args, "flow_cond_embed_dim", 16)) < 1:
        raise ValueError("--flow-cond-embed-dim must be >= 1.")
    if int(getattr(args, "flow_cond_embed_depth", 1)) < 1:
        raise ValueError("--flow-cond-embed-depth must be >= 1.")
    if int(getattr(args, "flow_prior_cond_embed_dim", 16)) < 1:
        raise ValueError("--flow-prior-cond-embed-dim must be >= 1.")
    if int(getattr(args, "flow_prior_cond_embed_depth", 1)) < 1:
        raise ValueError("--flow-prior-cond-embed-depth must be >= 1.")
    if _arch not in ("mlp", "soft_moe", "film", "film_fourier"):
        raise ValueError("--flow-arch must be one of {'mlp','soft_moe','film','film_fourier'}.")
    if int(getattr(args, "flow_moe_num_experts", 4)) < 1:
        raise ValueError("--flow-moe-num-experts must be >= 1.")
    if float(getattr(args, "flow_moe_router_temperature", 1.0)) <= 0.0:
        raise ValueError("--flow-moe-router-temperature must be > 0.")
    _fx_k = int(getattr(args, "flow_x_theta_fourier_k", 4))
    _fx_omega_mode = str(getattr(args, "flow_x_theta_fourier_omega_mode", "theta_range")).strip().lower()
    _fx_inc_lin = not bool(getattr(args, "flow_x_theta_fourier_no_linear", False))
    _fx_inc_bias = not bool(getattr(args, "flow_x_theta_fourier_no_bias", False))
    if _fx_k < 0:
        raise ValueError("--flow-x-theta-fourier-k must be >= 0.")
    if _fx_omega_mode not in ("fixed", "theta_range"):
        raise ValueError("--flow-x-theta-fourier-omega-mode must be one of {'fixed', 'theta_range'}.")
    if _tfm_val == "x_flow" and _arch == "soft_moe":
        raise ValueError("--flow-arch soft_moe is supported only for theta_flow/theta_path_integral.")
    if _tfm_val == "x_flow" and _arch == "film_fourier":
        if _fx_omega_mode == "theta_range":
            lo = float(getattr(args, "theta_low", -6.0))
            hi = float(getattr(args, "theta_high", 6.0))
            if not (hi > lo):
                raise ValueError(
                    "flow_x theta_fourier: theta_range omega mode requires theta_high > theta_low "
                    "(after merging dataset metadata)."
                )
        if _fx_k > 0:
            om_eff, _ = effective_flow_x_theta_fourier_omega(args)
            if abs(om_eff) < 1e-12:
                raise ValueError(
                    "Effective Fourier omega is ~0; check --flow-x-theta-fourier-omega and theta span."
                )
        theta_feat_dim = (1 if _fx_inc_bias else 0) + (1 if _fx_inc_lin else 0) + 2 * _fx_k
        if theta_feat_dim < 1:
            raise ValueError(
                "x_flow film_fourier: theta feature dim is 0. Use "
                "--flow-x-theta-fourier-k >= 1 or keep linear/bias features enabled."
            )
    _ft_post_k = int(getattr(args, "flow_theta_fourier_k", 4))
    _ft_post_omega_mode = str(getattr(args, "flow_theta_fourier_omega_mode", "theta_range")).strip().lower()
    _ft_post_inc_lin = not bool(getattr(args, "flow_theta_fourier_no_linear", False))
    _ft_post_inc_bias = not bool(getattr(args, "flow_theta_fourier_no_bias", False))
    _ft_prior_k = int(getattr(args, "flow_prior_theta_fourier_k", 4))
    _ft_prior_omega_mode = str(getattr(args, "flow_prior_theta_fourier_omega_mode", "theta_range")).strip().lower()
    _ft_prior_inc_lin = not bool(getattr(args, "flow_prior_theta_fourier_no_linear", False))
    _ft_prior_inc_bias = not bool(getattr(args, "flow_prior_theta_fourier_no_bias", False))
    if _tfm_val in ("theta_flow", "theta_path_integral") and _arch == "film_fourier":
        if _ft_post_k < 0:
            raise ValueError("--flow-theta-fourier-k must be >= 0.")
        if _ft_post_omega_mode not in ("fixed", "theta_range"):
            raise ValueError("--flow-theta-fourier-omega-mode must be one of {'fixed', 'theta_range'}.")
        if _ft_post_omega_mode == "theta_range":
            lo = float(getattr(args, "theta_low", -6.0))
            hi = float(getattr(args, "theta_high", 6.0))
            if not (hi > lo):
                raise ValueError(
                    "theta-flow posterior theta_fourier: theta_range omega mode requires theta_high > theta_low "
                    "(after merging dataset metadata)."
                )
        if _ft_post_k > 0:
            om_eff, _ = effective_flow_theta_fourier_omega_post(args)
            if abs(om_eff) < 1e-12:
                raise ValueError(
                    "Effective Fourier omega is ~0 for posterior theta-flow; check "
                    "--flow-theta-fourier-omega and theta span."
                )
        _tf_dim_post = (1 if _ft_post_inc_bias else 0) + (1 if _ft_post_inc_lin else 0) + 2 * _ft_post_k
        if _tf_dim_post < 1:
            raise ValueError(
                "theta_flow / theta_path_integral film_fourier (posterior): theta feature dim is 0. Use "
                "--flow-theta-fourier-k >= 1 or keep linear/bias features enabled."
            )
    if _tfm_val in ("theta_flow", "theta_path_integral") and _arch == "film_fourier":
        if _ft_prior_k < 0:
            raise ValueError("--flow-prior-theta-fourier-k must be >= 0.")
        if _ft_prior_omega_mode not in ("fixed", "theta_range"):
            raise ValueError("--flow-prior-theta-fourier-omega-mode must be one of {'fixed', 'theta_range'}.")
        if _ft_prior_omega_mode == "theta_range":
            lo = float(getattr(args, "theta_low", -6.0))
            hi = float(getattr(args, "theta_high", 6.0))
            if not (hi > lo):
                raise ValueError(
                    "theta-flow prior theta_fourier: theta_range omega mode requires theta_high > theta_low "
                    "(after merging dataset metadata)."
                )
        if _ft_prior_k > 0:
            om_eff, _ = effective_flow_theta_fourier_omega_prior(args)
            if abs(om_eff) < 1e-12:
                raise ValueError(
                    "Effective Fourier omega is ~0 for prior theta-flow; check "
                    "--flow-prior-theta-fourier-omega and theta span."
                )
        _tf_dim_prior = (1 if _ft_prior_inc_bias else 0) + (1 if _ft_prior_inc_lin else 0) + 2 * _ft_prior_k
        if _tf_dim_prior < 1:
            raise ValueError(
                "theta_flow / theta_path_integral film_fourier (prior): theta feature dim is 0. Use "
                "--flow-prior-theta-fourier-k >= 1 or keep linear/bias features enabled."
            )
    if (
        bool(getattr(args, "compute_h_matrix", False))
        and not bool(getattr(args, "prior_enable", True))
        and _tfm_val != "x_flow"
    ):
        raise ValueError("--compute-h-matrix requires prior score; do not use --no-prior-score.")
    if bool(getattr(args, "skip_shared_fisher_gt_compare", False)):
        if not bool(getattr(args, "compute_h_matrix", False)):
            raise ValueError("--skip-shared-fisher-gt-compare requires --compute-h-matrix.")
        if not bool(getattr(args, "prior_enable", True)) and _tfm_val != "x_flow":
            raise ValueError("--skip-shared-fisher-gt-compare requires prior score; do not use --no-prior-score.")
    if int(getattr(args, "ctsm_epochs", 1)) < 1:
        raise ValueError("--ctsm-epochs must be >= 1.")
    if int(getattr(args, "ctsm_batch_size", 1)) < 1:
        raise ValueError("--ctsm-batch-size must be >= 1.")
    if float(getattr(args, "ctsm_lr", 0.0)) <= 0.0:
        raise ValueError("--ctsm-lr must be positive.")
    if int(getattr(args, "ctsm_hidden_dim", 1)) < 1:
        raise ValueError("--ctsm-hidden-dim must be >= 1.")
    if float(getattr(args, "ctsm_two_sb_var", 0.0)) <= 0.0:
        raise ValueError("--ctsm-two-sb-var must be positive.")
    if str(getattr(args, "ctsm_path_schedule", "linear")).strip().lower() not in ("linear", "cosine"):
        raise ValueError("--ctsm-path-schedule must be one of {'linear','cosine'}.")
    if float(getattr(args, "ctsm_path_eps", 0.0)) <= 0.0:
        raise ValueError("--ctsm-path-eps must be positive.")
    if float(getattr(args, "ctsm_t_eps", -1.0)) < 0.0 or float(getattr(args, "ctsm_t_eps", -1.0)) >= 0.5:
        raise ValueError("--ctsm-t-eps must be in [0, 0.5).")
    if int(getattr(args, "ctsm_int_n_time", 1)) < 2:
        raise ValueError("--ctsm-int-n-time must be >= 2.")
    if float(getattr(args, "ctsm_m_scale", 0.0)) <= 0.0:
        raise ValueError("--ctsm-m-scale must be positive.")
    if float(getattr(args, "ctsm_delta_scale", 0.0)) <= 0.0:
        raise ValueError("--ctsm-delta-scale must be positive.")
    _ca = str(getattr(args, "ctsm_arch", "film")).strip().lower()
    if _ca not in ("mlp", "film"):
        raise ValueError("--ctsm-arch must be one of {'mlp','film'}.")
    if int(getattr(args, "ctsm_film_depth", 1)) < 1:
        raise ValueError("--ctsm-film-depth must be >= 1.")
    if float(getattr(args, "ctsm_weight_decay", 0.0)) < 0.0:
        raise ValueError("--ctsm-weight-decay must be >= 0.")


def _save_dsm_score_prior_training_losses_npz(
    output_dir: str,
    *,
    theta_all: np.ndarray,
    theta_score_fit: np.ndarray,
    theta_score_val: np.ndarray,
    score_split: str,
    score_train_losses: np.ndarray,
    score_val_losses: np.ndarray,
    score_val_monitor_losses: np.ndarray,
    score_best_epoch: int,
    score_stopped_epoch: int,
    score_stopped_early: bool,
    score_best_val_loss: float,
    prior_enable: bool,
    prior_train_losses: np.ndarray,
    prior_val_losses: np.ndarray,
    prior_val_monitor_losses: np.ndarray,
    prior_best_epoch: int,
    prior_stopped_epoch: int,
    prior_stopped_early: bool,
    prior_best_val_loss: float,
    score_has_nonfinite: bool = False,
    score_grad_norm_mean: float = float("nan"),
    score_grad_norm_max: float = float("nan"),
    score_param_norm_final: float = float("nan"),
    score_n_clipped_steps: int = 0,
    score_n_total_steps: int = 0,
    score_lr_last: float = float("nan"),
    prior_has_nonfinite: bool = False,
    prior_grad_norm_mean: float = float("nan"),
    prior_grad_norm_max: float = float("nan"),
    prior_param_norm_final: float = float("nan"),
    prior_n_clipped_steps: int = 0,
    prior_n_total_steps: int = 0,
    prior_lr_last: float = float("nan"),
    theta_field_method: str = "theta_flow",
) -> str:
    """Write per-run training curves for posterior and optional prior (DSM score or theta-flow)."""
    path = os.path.join(output_dir, "score_prior_training_losses.npz")
    tfm = str(theta_field_method).strip().lower()
    np.savez_compressed(
        path,
        theta_field_method=np.asarray([tfm], dtype=object),
        n_theta_all=np.int64(np.asarray(theta_all).shape[0]),
        n_score_fit=np.int64(theta_score_fit.shape[0]),
        n_score_val=np.int64(theta_score_val.shape[0]),
        score_split=np.asarray([str(score_split)], dtype=object),
        score_train_losses=np.asarray(score_train_losses, dtype=np.float64),
        score_val_losses=np.asarray(score_val_losses, dtype=np.float64),
        score_val_monitor_losses=np.asarray(score_val_monitor_losses, dtype=np.float64),
        score_best_epoch=np.int64(score_best_epoch),
        score_stopped_epoch=np.int64(score_stopped_epoch),
        score_stopped_early=np.bool_(score_stopped_early),
        score_best_val_smooth=np.float64(score_best_val_loss),
        score_has_nonfinite=np.bool_(score_has_nonfinite),
        score_grad_norm_mean=np.float64(score_grad_norm_mean),
        score_grad_norm_max=np.float64(score_grad_norm_max),
        score_param_norm_final=np.float64(score_param_norm_final),
        score_n_clipped_steps=np.int64(score_n_clipped_steps),
        score_n_total_steps=np.int64(score_n_total_steps),
        score_lr_last=np.float64(score_lr_last),
        prior_enable=np.bool_(prior_enable),
        prior_train_losses=np.asarray(prior_train_losses, dtype=np.float64),
        prior_val_losses=np.asarray(prior_val_losses, dtype=np.float64),
        prior_val_monitor_losses=np.asarray(prior_val_monitor_losses, dtype=np.float64),
        prior_best_epoch=np.int64(prior_best_epoch),
        prior_stopped_epoch=np.int64(prior_stopped_epoch),
        prior_stopped_early=np.bool_(prior_stopped_early),
        prior_best_val_smooth=np.float64(prior_best_val_loss),
        prior_has_nonfinite=np.bool_(prior_has_nonfinite),
        prior_grad_norm_mean=np.float64(prior_grad_norm_mean),
        prior_grad_norm_max=np.float64(prior_grad_norm_max),
        prior_param_norm_final=np.float64(prior_param_norm_final),
        prior_n_clipped_steps=np.int64(prior_n_clipped_steps),
        prior_n_total_steps=np.int64(prior_n_total_steps),
        prior_lr_last=np.float64(prior_lr_last),
    )
    return path


def _save_theta_flow_model_checkpoint(
    *,
    output_dir: str,
    filename: str,
    model: torch.nn.Module,
    model_role: str,
    theta_field_method: str,
    flow_arch: str,
    flow_scheduler: str,
    theta_dim_flow: int,
    model_hparams: dict[str, Any],
    args: Any,
) -> str:
    """Save a theta-flow model checkpoint with enough metadata to reconstruct the module."""
    path = os.path.join(output_dir, filename)
    payload = {
        "checkpoint_version": 1,
        "model_role": str(model_role),
        "theta_field_method": str(theta_field_method),
        "flow_arch": str(flow_arch),
        "flow_scheduler": str(flow_scheduler),
        "theta_dim_flow": int(theta_dim_flow),
        "x_dim": int(getattr(args, "x_dim", -1)),
        "theta_flow_onehot_state": bool(getattr(args, "theta_flow_onehot_state", False)),
        "theta_flow_fourier_state": bool(getattr(args, "theta_flow_fourier_state", False)),
        "theta_flow_fourier_k": int(getattr(args, "theta_flow_fourier_k", 0)),
        "theta_flow_fourier_period_mult": float(getattr(args, "theta_flow_fourier_period_mult", 0.0)),
        "theta_flow_fourier_include_linear": bool(getattr(args, "theta_flow_fourier_include_linear", False)),
        "model_hparams": dict(model_hparams),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def merge_meta_into_args(meta: dict[str, Any], est_ns: Any) -> Any:
    """Merge dataset ``meta`` into CLI args.

    Only keys that belong to shared-dataset metadata (see ``SHARED_DATASET_META_KEYS``)
    may override argparse defaults; other keys in ``meta`` are ignored so estimation
    flags like ``theta_field_method`` cannot be accidentally overwritten.
    """
    out = vars(est_ns).copy()
    for k, v in meta.items():
        if k == "version":
            continue
        if k in SHARED_DATASET_META_KEYS:
            out[k] = v
    return SimpleNamespace(**out)


def train_pair_conditioned_ctsm_v_model(
    *,
    model: PairConditionedTimeScoreNetBase,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    device: torch.device,
    log_every: int,
    two_sb_var: float,
    path_schedule: str = "linear",
    path_eps: float = 1e-12,
    factor: float,
    t_eps: float,
    theta_val: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 1000,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
    val_batches_per_epoch: int = 8,
) -> dict[str, Any]:
    theta_fit_np = np.asarray(theta_train, dtype=np.float32).reshape(-1, 1)
    x_fit_np = np.asarray(x_train, dtype=np.float32)
    if x_fit_np.ndim != 2 or x_fit_np.shape[0] != theta_fit_np.shape[0]:
        raise ValueError("CTSM-v training expects x_train shape (N,d) and theta_train length N.")
    if theta_fit_np.shape[0] < 2:
        raise ValueError("CTSM-v training requires at least 2 samples.")

    theta_val_np = None
    x_val_np = None
    if theta_val is not None and x_val is not None:
        theta_val_np = np.asarray(theta_val, dtype=np.float32).reshape(-1, 1)
        x_val_np = np.asarray(x_val, dtype=np.float32)
        if x_val_np.ndim != 2 or x_val_np.shape[0] != theta_val_np.shape[0]:
            raise ValueError("CTSM-v validation expects x_val shape (N,d) and theta_val length N.")
        if theta_val_np.shape[0] < 2:
            theta_val_np = None
            x_val_np = None

    prob_path = TwoSB(
        dim=int(x_fit_np.shape[1]),
        var=float(two_sb_var),
        scheduler=str(path_schedule),
        eps=float(path_eps),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    theta_fit_t = torch.from_numpy(theta_fit_np).to(device)
    x_fit_t = torch.from_numpy(x_fit_np).to(device)
    theta_val_t = torch.from_numpy(theta_val_np).to(device) if theta_val_np is not None else None
    x_val_t = torch.from_numpy(x_val_np).to(device) if x_val_np is not None else None

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    ema_monitor: float | None = None
    patience_bad = 0
    stopped_early = False
    has_nonfinite = False

    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    n_total_steps = 0

    n_fit = int(theta_fit_t.shape[0])
    n_val = int(theta_val_t.shape[0]) if theta_val_t is not None else 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ia = torch.randint(0, n_fit, (int(batch_size),), device=device)
        ib = torch.randint(0, n_fit, (int(batch_size),), device=device)
        x0 = x_fit_t[ia]
        x1 = x_fit_t[ib]
        a = theta_fit_t[ia]
        b = theta_fit_t[ib]
        loss = ctsm_v_pair_conditioned_loss(
            model=model,
            prob_path=prob_path,
            x0=x0,
            x1=x1,
            a=a,
            b=b,
            factor=float(factor),
            t_eps=float(t_eps),
        )
        if not torch.isfinite(loss):
            has_nonfinite = True
            raise RuntimeError(f"Non-finite CTSM-v train loss at epoch {epoch}: {float(loss.detach().cpu())!r}")
        opt.zero_grad(set_to_none=True)
        loss.backward()

        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            gn = float(p.grad.detach().norm(2).item())
            grad_sq += gn * gn
        grad_norm = float(math.sqrt(grad_sq))
        grad_norm_sum += grad_norm
        grad_norm_max = max(grad_norm_max, grad_norm)
        n_total_steps += 1

        opt.step()
        train_loss = float(loss.detach().cpu().item())
        train_losses.append(train_loss)

        if theta_val_t is None or x_val_t is None:
            val_loss = train_loss
        else:
            model.eval()
            vb = max(1, int(val_batches_per_epoch))
            v_losses: list[float] = []
            with torch.no_grad():
                for _ in range(vb):
                    ia_v = torch.randint(0, n_val, (int(batch_size),), device=device)
                    ib_v = torch.randint(0, n_val, (int(batch_size),), device=device)
                    x0_v = x_val_t[ia_v]
                    x1_v = x_val_t[ib_v]
                    a_v = theta_val_t[ia_v]
                    b_v = theta_val_t[ib_v]
                    lv = ctsm_v_pair_conditioned_loss(
                        model=model,
                        prob_path=prob_path,
                        x0=x0_v,
                        x1=x1_v,
                        a=a_v,
                        b=b_v,
                        factor=float(factor),
                        t_eps=float(t_eps),
                    )
                    if not torch.isfinite(lv):
                        has_nonfinite = True
                        raise RuntimeError(
                            f"Non-finite CTSM-v validation loss at epoch {epoch}: {float(lv.detach().cpu())!r}"
                        )
                    v_losses.append(float(lv.detach().cpu().item()))
            val_loss = float(np.mean(v_losses))

        val_losses.append(val_loss)
        if ema_monitor is None:
            ema_monitor = val_loss
        else:
            alpha = float(early_stopping_ema_alpha)
            ema_monitor = alpha * val_loss + (1.0 - alpha) * ema_monitor
        val_monitor_losses.append(float(ema_monitor))

        improved = float(ema_monitor) < (best_val_loss - float(early_stopping_min_delta))
        if improved:
            best_val_loss = float(ema_monitor)
            best_epoch = int(epoch)
            patience_bad = 0
            if restore_best:
                best_state = deepcopy(model.state_dict())
        else:
            patience_bad += 1

        if epoch % max(1, int(log_every)) == 0:
            print(
                "[ctsm_v] "
                f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f} "
                f"val_ema={float(ema_monitor):.6f} best_epoch={best_epoch}"
            )
        if patience_bad >= int(early_stopping_patience):
            stopped_early = True
            break

    stopped_epoch = int(len(train_losses))
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    param_sq = 0.0
    with torch.no_grad():
        for p in model.parameters():
            pn = float(p.detach().norm(2).item())
            param_sq += pn * pn
    param_norm_final = float(math.sqrt(param_sq))

    if best_epoch <= 0:
        best_epoch = stopped_epoch
        if np.isfinite(np.asarray(val_monitor_losses, dtype=np.float64)).any():
            best_val_loss = float(np.nanmin(np.asarray(val_monitor_losses, dtype=np.float64)))
        else:
            best_val_loss = float("nan")

    return {
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_monitor_losses": np.asarray(val_monitor_losses, dtype=np.float64),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "best_val_loss": float(best_val_loss),
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(grad_norm_sum / max(1, n_total_steps)),
        "grad_norm_max": float(grad_norm_max),
        "param_norm_final": float(param_norm_final),
        "n_clipped_steps": int(0),
        "n_total_steps": int(n_total_steps),
        "lr_last": float(opt.param_groups[0]["lr"]),
    }


def build_conditional_x_velocity_model(
    *,
    flow_arch: str,
    args: Any,
    device: torch.device,
) -> torch.nn.Module:
    arch = str(flow_arch).strip().lower()
    if arch == "mlp":
        return ConditionalXFlowVelocity(
            x_dim=int(args.x_dim),
            hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
            depth=int(getattr(args, "flow_depth", 3)),
            use_logit_time=True,
        ).to(device)
    if arch == "film":
        return ConditionalXFlowVelocityFiLMPerLayer(
            x_dim=int(args.x_dim),
            hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
            depth=int(getattr(args, "flow_depth", 3)),
            use_logit_time=True,
            use_layer_norm=bool(getattr(args, "flow_use_layer_norm", False)),
            gated_film=bool(getattr(args, "flow_gated_film", False)),
            zero_out_init=bool(getattr(args, "flow_zero_out_init", False)),
            cond_embed_dim=int(getattr(args, "flow_cond_embed_dim", 16)),
            cond_embed_depth=int(getattr(args, "flow_cond_embed_depth", 1)),
            cond_embed_act=str(getattr(args, "flow_cond_embed_act", "silu")),
        ).to(device)
    if arch == "film_fourier":
        om_eff, _ = effective_flow_x_theta_fourier_omega(args)
        return ConditionalXFlowVelocityThetaFourierFiLMPerLayer(
            x_dim=int(args.x_dim),
            hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
            depth=int(getattr(args, "flow_depth", 3)),
            use_logit_time=True,
            theta_fourier_k=int(getattr(args, "flow_x_theta_fourier_k", 4)),
            theta_fourier_omega=float(om_eff),
            theta_fourier_include_linear=not bool(getattr(args, "flow_x_theta_fourier_no_linear", False)),
            theta_fourier_include_bias=not bool(getattr(args, "flow_x_theta_fourier_no_bias", False)),
        ).to(device)
    raise ValueError("--flow-arch must be one of {'mlp','film','film_fourier'}.")


def run_shared_fisher_estimation(
    args: Any,
    dataset: ToyConditionalGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset,
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    rng: np.random.Generator,
) -> None:
    _apply_dsm_stability_preset(args)
    run_seed = int(getattr(args, "seed", 7))
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    device = require_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    theta_score_fit = np.asarray(theta_train, dtype=np.float64)
    x_score_fit = np.asarray(x_train, dtype=np.float64)
    theta_score_val = np.asarray(theta_validation, dtype=np.float64)
    x_score_val = np.asarray(x_validation, dtype=np.float64)
    theta_prior_fit_override = getattr(args, "theta_flow_prior_theta_train_override", None)
    theta_prior_val_override = getattr(args, "theta_flow_prior_theta_validation_override", None)
    theta_prior_all_override = getattr(args, "theta_flow_prior_theta_all_override", None)
    if theta_score_fit.shape[0] < 1 or theta_score_val.shape[0] < 1:
        raise ValueError(
            "Shared Fisher estimation requires non-empty train and validation splits. "
            "Regenerate the dataset with 0 < --train-frac < 1 (see make_dataset.py)."
        )
    print(
        "[score_data] "
        f"train={theta_score_fit.shape[0]} validation={theta_score_val.shape[0]}"
    )
    print(
        "[score_train] "
        f"fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
        "(dataset train_idx / validation_idx)"
    )

    theta_field_method = normalize_theta_field_method(str(getattr(args, "theta_field_method", "theta_flow")))
    flow_arch = normalize_flow_arch(args)
    if theta_field_method == "ctsm_v":
        if not bool(getattr(args, "compute_h_matrix", False)):
            raise RuntimeError("ctsm_v requires --compute-h-matrix to produce output artifacts.")
        theta_std = float(np.std(theta_score_fit))
        print(
            "[ctsm_v] "
            f"arch={str(getattr(args, 'ctsm_arch', 'film')).strip().lower()} "
            f"fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
            f"x_dim={int(args.x_dim)} theta_std={theta_std:.6f} "
            f"two_sb_var={float(getattr(args, 'ctsm_two_sb_var', 2.0)):.6f} "
            f"path_schedule={str(getattr(args, 'ctsm_path_schedule', 'linear')).strip().lower()} "
            f"path_eps={float(getattr(args, 'ctsm_path_eps', 1e-12)):.1e} "
            f"factor={float(getattr(args, 'ctsm_factor', 1.0)):.6f} "
            f"t_eps={float(getattr(args, 'ctsm_t_eps', 1e-5)):.2e} "
            f"int_n_time={int(getattr(args, 'ctsm_int_n_time', 300))} "
            f"adamw_wd={float(getattr(args, 'ctsm_weight_decay', 0.0)):.3e}"
        )
        _ctsm_arch = str(getattr(args, "ctsm_arch", "film")).strip().lower()
        if _ctsm_arch == "film":
            ctsm_model = ToyPairConditionedTimeScoreNetFiLM(
                dim=int(args.x_dim),
                hidden_dim=int(getattr(args, "ctsm_hidden_dim", 256)),
                depth=int(getattr(args, "ctsm_film_depth", 3)),
                m_scale=float(getattr(args, "ctsm_m_scale", 1.0)),
                delta_scale=float(getattr(args, "ctsm_delta_scale", 0.5)),
                use_logit_time=not bool(getattr(args, "ctsm_raw_time", False)),
                gated_film=bool(getattr(args, "ctsm_gated_film", False)),
            ).to(device)
        elif _ctsm_arch == "mlp":
            ctsm_model = ToyPairConditionedTimeScoreNet(
                dim=int(args.x_dim),
                hidden_dim=int(getattr(args, "ctsm_hidden_dim", 256)),
                m_scale=float(getattr(args, "ctsm_m_scale", 1.0)),
                delta_scale=float(getattr(args, "ctsm_delta_scale", 0.5)),
            ).to(device)
        else:
            raise ValueError("--ctsm-arch must be one of {'mlp','film'}.")
        ctsm_out = train_pair_conditioned_ctsm_v_model(
            model=ctsm_model,
            theta_train=theta_score_fit,
            x_train=x_score_fit,
            epochs=int(getattr(args, "ctsm_epochs", 8000)),
            batch_size=int(getattr(args, "ctsm_batch_size", 512)),
            lr=float(getattr(args, "ctsm_lr", 2e-3)),
            weight_decay=float(getattr(args, "ctsm_weight_decay", 0.0)),
            device=device,
            log_every=max(1, int(args.log_every)),
            two_sb_var=float(getattr(args, "ctsm_two_sb_var", 2.0)),
            path_schedule=str(getattr(args, "ctsm_path_schedule", "linear")),
            path_eps=float(getattr(args, "ctsm_path_eps", 1e-12)),
            factor=float(getattr(args, "ctsm_factor", 1.0)),
            t_eps=float(getattr(args, "ctsm_t_eps", 1e-5)),
            theta_val=theta_score_val,
            x_val=x_score_val,
            early_stopping_patience=int(getattr(args, "flow_early_patience", 1000)),
            early_stopping_min_delta=float(getattr(args, "flow_early_min_delta", 1e-4)),
            early_stopping_ema_alpha=float(getattr(args, "flow_early_ema_alpha", 0.05)),
            restore_best=bool(getattr(args, "flow_restore_best", True)),
        )
        post_train_losses = np.asarray(ctsm_out["train_losses"], dtype=np.float64)
        post_val_losses = np.asarray(ctsm_out["val_losses"], dtype=np.float64)
        post_val_monitor_losses = np.asarray(ctsm_out.get("val_monitor_losses", []), dtype=np.float64)
        post_best_epoch = int(ctsm_out["best_epoch"])
        post_stopped_epoch = int(ctsm_out["stopped_epoch"])
        post_stopped_early = bool(ctsm_out["stopped_early"])
        post_best_val_loss = float(ctsm_out["best_val_loss"])

        post_loss_fig = os.path.join(args.output_dir, "score_loss_vs_epoch.png")
        epochs_arr = np.arange(1, post_train_losses.size + 1)
        plt.figure(figsize=(8.8, 5.0))
        plt.plot(epochs_arr, post_train_losses, color="#1f77b4", linewidth=2.0, label="CTSM-v train loss")
        if post_val_losses.size == post_train_losses.size and np.any(np.isfinite(post_val_losses)):
            plt.plot(epochs_arr, post_val_losses, color="#d62728", linewidth=2.0, label="CTSM-v val loss")
        if post_val_monitor_losses.size == post_train_losses.size and np.any(np.isfinite(post_val_monitor_losses)):
            plt.plot(
                epochs_arr,
                post_val_monitor_losses,
                color="#ff7f0e",
                linewidth=2.0,
                linestyle="--",
                label=f"CTSM-v val EMA (α={getattr(args, 'flow_early_ema_alpha', 0.05):g})",
            )
        if 1 <= post_best_epoch <= post_train_losses.size:
            plt.axvline(post_best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {post_best_epoch}")
        if 1 <= post_stopped_epoch <= post_train_losses.size:
            plt.axvline(
                post_stopped_epoch,
                color="#9467bd",
                linestyle=":",
                linewidth=1.6,
                label=f"Stop epoch {post_stopped_epoch}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Pair-conditioned CTSM-v training")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(post_loss_fig, dpi=180)
        plt.close()

        h_eval = float(getattr(args, "ctsm_t_eps", 1e-5))
        theta_h_matrix = np.asarray(theta_all, dtype=np.float64)
        x_h_matrix = np.asarray(x_all, dtype=np.float64)
        print(
            "[h_matrix] "
            "enabled=True field=ctsm_v "
            f"ctsm_arch={str(getattr(args, 'ctsm_arch', 'film')).strip().lower()} "
            f"ctsm_path_schedule={str(getattr(args, 'ctsm_path_schedule', 'linear')).strip().lower()} "
            f"ctsm_t_eps={h_eval:.6g} ctsm_int_n_time={int(getattr(args, 'ctsm_int_n_time', 300))} "
            f"n_theta_x={int(theta_h_matrix.shape[0])} (train+validation full pool) "
            f"restore_original_order={bool(getattr(args, 'h_restore_original_order', False))} "
            f"pair_batch_size={int(getattr(args, 'h_batch_size', 65536))}"
        )
        h_estimator = HMatrixEstimator(
            model_post=ctsm_model,
            model_prior=None,
            sigma_eval=h_eval,
            device=device,
            pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
            field_method="ctsm_v",
            flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
            ctsm_int_n_time=int(getattr(args, "ctsm_int_n_time", 300)),
            ctsm_t_eps=float(getattr(args, "ctsm_t_eps", 1e-5)),
        )
        h_result = h_estimator.run(
            theta=theta_h_matrix,
            x=x_h_matrix,
            restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
        )

        suffix = "_non_gauss" if args.dataset_family == "cosine_gmm" else "_theta_cov"
        h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path = _save_h_matrix_dsm_artifacts(args, h_result, suffix)
        print(
            "[summary] ctsm_v mode completed (pair-conditioned CTSM-v bridge score integrated over t for per-pair DeltaL)."
        )
        print("Saved artifacts:")
        print(f"  - {post_loss_fig}")
        print(f"  - {h_npz_path}")
        print(f"  - {h_summary_path}")
        print(f"  - {h_fig_path}")
        if h_delta_fig_path:
            print(f"  - {h_delta_fig_path}")
        prior_empty = np.asarray([], dtype=np.float64)
        tnpz = _save_dsm_score_prior_training_losses_npz(
            args.output_dir,
            theta_all=theta_all,
            theta_score_fit=theta_score_fit,
            theta_score_val=theta_score_val,
            score_split="dataset_train_validation",
            score_train_losses=post_train_losses,
            score_val_losses=post_val_losses,
            score_val_monitor_losses=post_val_monitor_losses,
            score_best_epoch=post_best_epoch,
            score_stopped_epoch=post_stopped_epoch,
            score_stopped_early=post_stopped_early,
            score_best_val_loss=post_best_val_loss,
            prior_enable=False,
            prior_train_losses=prior_empty,
            prior_val_losses=prior_empty,
            prior_val_monitor_losses=prior_empty,
            prior_best_epoch=0,
            prior_stopped_epoch=0,
            prior_stopped_early=False,
            prior_best_val_loss=float("nan"),
            score_has_nonfinite=bool(ctsm_out.get("has_nonfinite", False)),
            score_grad_norm_mean=float(ctsm_out.get("grad_norm_mean", float("nan"))),
            score_grad_norm_max=float(ctsm_out.get("grad_norm_max", float("nan"))),
            score_param_norm_final=float(ctsm_out.get("param_norm_final", float("nan"))),
            score_n_clipped_steps=int(ctsm_out.get("n_clipped_steps", 0)),
            score_n_total_steps=int(ctsm_out.get("n_total_steps", 0)),
            score_lr_last=float(ctsm_out.get("lr_last", float("nan"))),
            prior_has_nonfinite=False,
            prior_grad_norm_mean=float("nan"),
            prior_grad_norm_max=float("nan"),
            prior_param_norm_final=float("nan"),
            prior_n_clipped_steps=0,
            prior_n_total_steps=0,
            prior_lr_last=float("nan"),
            theta_field_method="ctsm_v",
        )
        print(f"[training_losses] saved {tnpz}")
        return

    if theta_field_method == "x_flow":
        if not bool(getattr(args, "compute_h_matrix", False)):
            raise RuntimeError(f"{theta_field_method} requires --compute-h-matrix to produce output artifacts.")
        theta_std = float(np.std(theta_score_fit))
        flow_score_arch = str(flow_arch).strip().lower()
        _xf_extra = ""
        if flow_score_arch == "film":
            _xf_extra = (
                " film"
                f" theta_embed_dim={int(getattr(args, 'flow_cond_embed_dim', 16))}"
                f" theta_embed_depth={int(getattr(args, 'flow_cond_embed_depth', 1))}"
                f" theta_embed_act={str(getattr(args, 'flow_cond_embed_act', 'silu'))}"
            )
        elif flow_score_arch == "film_fourier":
            _om_eff, _om_log = effective_flow_x_theta_fourier_omega(args)
            _xf_extra = (
                f" theta_fourier_k={int(getattr(args, 'flow_x_theta_fourier_k', 4))}"
                f" {_om_log}"
                f" theta_fourier_omega_in_net={float(_om_eff):.6g}"
                f" theta_fourier_linear={not bool(getattr(args, 'flow_x_theta_fourier_no_linear', False))}"
                f" theta_fourier_bias={not bool(getattr(args, 'flow_x_theta_fourier_no_bias', False))}"
            )
            _xf_extra = " film_x_trunk" + _xf_extra
        _xf_twostage = bool(getattr(args, "flow_x_two_stage_mean_theta_pretrain", False)) and theta_field_method == "x_flow"
        _e_tot = int(getattr(args, "flow_epochs", 10000))
        _e1 = _e_tot // 2
        _e2 = _e_tot - _e1
        print(
            f"[{theta_field_method}] "
            f"fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
            f"scheduler={getattr(args, 'flow_scheduler', 'cosine')} method={theta_field_method} "
            f"x_dim={int(args.x_dim)} theta_std={theta_std:.6f}"
            f" two_stage_mean_theta_pretrain={_xf_twostage}"
            + (
                f" (stage1_epochs={_e1} stage2_epochs={_e2})"
                if _xf_twostage
                else ""
            )
            + f"{_xf_extra}"
        )
        x_flow_model = build_conditional_x_velocity_model(
            flow_arch=flow_score_arch,
            args=args,
            device=device,
        )
        post_train_out = train_conditional_x_flow_model(
            model=x_flow_model,
            theta_train=theta_score_fit,
            x_train=x_score_fit,
            epochs=int(getattr(args, "flow_epochs", 10000)),
            batch_size=int(getattr(args, "flow_batch_size", 256)),
            lr=float(getattr(args, "flow_lr", 1e-3)),
            device=device,
            log_every=max(1, args.log_every),
            theta_val=theta_score_val,
            x_val=x_score_val,
            early_stopping_patience=int(getattr(args, "flow_early_patience", 1000)),
            early_stopping_min_delta=float(getattr(args, "flow_early_min_delta", 1e-4)),
            early_stopping_ema_alpha=float(getattr(args, "flow_early_ema_alpha", 0.05)),
            restore_best=bool(getattr(args, "flow_restore_best", True)),
            scheduler_name=str(getattr(args, "flow_scheduler", "cosine")),
            two_stage_mean_theta_pretrain=_xf_twostage,
        )
        if theta_field_method == "x_flow" and post_train_out.get("flow_x_two_stage"):
            print(
                "[x_flow] two-stage training finished: "
                f"theta_mean_pretrain={float(post_train_out['theta_mean_pretrain']):.6f} "
                f"stage_boundary_epoch={int(post_train_out['stage_boundary_epoch'])} "
                f"best_epoch={int(post_train_out['best_epoch'])} "
                f"best_val_smooth={float(post_train_out['best_val_loss']):.6f}"
            )
        post_train_losses = np.asarray(post_train_out["train_losses"], dtype=np.float64)
        post_val_losses = np.asarray(post_train_out["val_losses"], dtype=np.float64)
        post_val_monitor_losses = np.asarray(post_train_out.get("val_monitor_losses", []), dtype=np.float64)
        post_best_epoch = int(post_train_out["best_epoch"])
        post_stopped_epoch = int(post_train_out["stopped_epoch"])
        post_stopped_early = bool(post_train_out["stopped_early"])
        post_best_val_loss = float(post_train_out["best_val_loss"])

        post_loss_fig = os.path.join(args.output_dir, "score_loss_vs_epoch.png")
        epochs_arr = np.arange(1, post_train_losses.size + 1)
        _train_label = "X-flow"
        plt.figure(figsize=(8.8, 5.0))
        plt.plot(epochs_arr, post_train_losses, color="#1f77b4", linewidth=2.0, label=f"{_train_label} train loss")
        if post_val_losses.size == post_train_losses.size and np.any(np.isfinite(post_val_losses)):
            plt.plot(epochs_arr, post_val_losses, color="#d62728", linewidth=2.0, label=f"{_train_label} val loss")
        if post_val_monitor_losses.size == post_train_losses.size and np.any(np.isfinite(post_val_monitor_losses)):
            plt.plot(
                epochs_arr,
                post_val_monitor_losses,
                color="#ff7f0e",
                linewidth=2.0,
                linestyle="--",
                label=f"{_train_label} val EMA (α={getattr(args, 'flow_early_ema_alpha', 0.05):g})",
            )
        if theta_field_method == "x_flow" and post_train_out.get("flow_x_two_stage"):
            sb = int(post_train_out["stage_boundary_epoch"])
            if 1 <= sb < post_train_losses.size:
                plt.axvline(
                    sb,
                    color="#7f7f7f",
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.9,
                    label=f"Two-stage boundary (end stage 1, epoch {sb})",
                )
        if 1 <= post_best_epoch <= post_train_losses.size:
            plt.axvline(post_best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {post_best_epoch}")
        if 1 <= post_stopped_epoch <= post_train_losses.size:
            plt.axvline(
                post_stopped_epoch,
                color="#9467bd",
                linestyle=":",
                linewidth=1.6,
                label=f"Stop epoch {post_stopped_epoch}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Conditional x-flow training (x_flow)")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(post_loss_fig, dpi=180)
        plt.close()

        theta_h_matrix = np.asarray(theta_all, dtype=np.float64)
        x_h_matrix = np.asarray(x_all, dtype=np.float64)
        h_eval = 1.0
        print(
            "[h_matrix] "
            f"enabled=True field={theta_field_method} "
            f"flow_ode_t_span={h_eval:.6f} "
            f"n_theta_x={int(theta_h_matrix.shape[0])} (train+validation full pool) "
            f"restore_original_order={bool(getattr(args, 'h_restore_original_order', False))} "
            f"pair_batch_size={int(getattr(args, 'h_batch_size', 65536))}"
        )
        h_estimator = HMatrixEstimator(
            model_post=x_flow_model,
            model_prior=None,
            sigma_eval=h_eval,
            device=device,
            pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
            field_method="flow_x_likelihood",
            flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
        )
        h_result = h_estimator.run(
            theta=theta_h_matrix,
            x=x_h_matrix,
            restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
        )

        suffix = "_non_gauss" if args.dataset_family == "cosine_gmm" else "_theta_cov"
        h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path = _save_h_matrix_dsm_artifacts(
            args, h_result, suffix
        )
        print(
            "[summary] x_flow mode completed (H-matrix only path; "
            "ODESolver.compute_likelihood on conditional x-flow for log p(x|theta))."
        )
        print("Saved artifacts:")
        print(f"  - {post_loss_fig}")
        print(f"  - {h_npz_path}")
        print(f"  - {h_summary_path}")
        print(f"  - {h_fig_path}")
        if h_delta_fig_path:
            print(f"  - {h_delta_fig_path}")
        prior_empty = np.asarray([], dtype=np.float64)
        tnpz = _save_dsm_score_prior_training_losses_npz(
            args.output_dir,
            theta_all=theta_all,
            theta_score_fit=theta_score_fit,
            theta_score_val=theta_score_val,
            score_split="dataset_train_validation",
            score_train_losses=post_train_losses,
            score_val_losses=post_val_losses,
            score_val_monitor_losses=post_val_monitor_losses,
            score_best_epoch=post_best_epoch,
            score_stopped_epoch=post_stopped_epoch,
            score_stopped_early=post_stopped_early,
            score_best_val_loss=post_best_val_loss,
            prior_enable=False,
            prior_train_losses=prior_empty,
            prior_val_losses=prior_empty,
            prior_val_monitor_losses=prior_empty,
            prior_best_epoch=0,
            prior_stopped_epoch=0,
            prior_stopped_early=False,
            prior_best_val_loss=float("nan"),
            score_has_nonfinite=bool(post_train_out.get("has_nonfinite", False)),
            score_grad_norm_mean=float(post_train_out.get("grad_norm_mean", float("nan"))),
            score_grad_norm_max=float(post_train_out.get("grad_norm_max", float("nan"))),
            score_param_norm_final=float(post_train_out.get("param_norm_final", float("nan"))),
            score_n_clipped_steps=int(post_train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=int(post_train_out.get("n_total_steps", 0)),
            score_lr_last=float(post_train_out.get("lr_last", float("nan"))),
            prior_has_nonfinite=False,
            prior_grad_norm_mean=float("nan"),
            prior_grad_norm_max=float("nan"),
            prior_param_norm_final=float("nan"),
            prior_n_clipped_steps=0,
            prior_n_total_steps=0,
            prior_lr_last=float("nan"),
            theta_field_method=theta_field_method,
        )
        print(f"[training_losses] saved {tnpz}")
        return

    if theta_field_method in ("theta_flow", "theta_path_integral"):
        if not bool(getattr(args, "prior_enable", True)):
            raise ValueError("theta_flow and theta_path_integral currently require prior model enabled.")
        flow_eval_t = float(getattr(args, "flow_eval_t", 0.8))
        if not (0.0 <= flow_eval_t <= 1.0):
            raise ValueError("--flow-eval-t must be in [0, 1].")
        theta_std = float(np.std(theta_score_fit))
        theta_dim_flow = int(theta_score_fit.shape[1]) if theta_score_fit.ndim >= 2 else 1
        theta_prior_fit = (
            np.asarray(theta_prior_fit_override, dtype=np.float64)
            if theta_prior_fit_override is not None
            else theta_score_fit
        )
        theta_prior_val = (
            np.asarray(theta_prior_val_override, dtype=np.float64)
            if theta_prior_val_override is not None
            else theta_score_val
        )
        theta_prior_all = (
            np.asarray(theta_prior_all_override, dtype=np.float64)
            if theta_prior_all_override is not None
            else np.asarray(theta_all, dtype=np.float64)
        )
        if theta_prior_fit.shape[0] != theta_score_fit.shape[0]:
            raise ValueError("theta_flow prior override train rows must match posterior train rows.")
        if theta_prior_val.shape[0] != theta_score_val.shape[0]:
            raise ValueError("theta_flow prior override validation rows must match posterior validation rows.")
        if theta_prior_all.shape[0] != np.asarray(theta_all).shape[0]:
            raise ValueError("theta_flow prior override all rows must match theta_all rows.")
        theta_dim_prior_flow = int(theta_prior_fit.shape[1]) if theta_prior_fit.ndim >= 2 else 1
        flow_score_arch = str(flow_arch).strip().lower()
        flow_prior_arch = "mlp" if flow_score_arch == "soft_moe" else str(flow_arch).strip().lower()
        if theta_dim_prior_flow != 1 and flow_prior_arch != "mlp":
            raise ValueError(
                "theta_flow prior override with non-scalar theta currently supports prior flow_arch=mlp only "
                f"(got prior={flow_prior_arch!r}, theta_dim_prior={theta_dim_prior_flow})."
            )
        theta_onehot_state = bool(getattr(args, "theta_flow_onehot_state", False))
        if theta_onehot_state:
            if theta_field_method != "theta_flow":
                raise ValueError("--theta-flow-onehot-state requires theta_field_method=theta_flow.")
            if flow_score_arch != "mlp" or flow_prior_arch != "mlp":
                raise ValueError(
                    "--theta-flow-onehot-state currently supports flow_arch=mlp only "
                    f"(got posterior={flow_score_arch!r}, prior={flow_prior_arch!r})."
                )
            if theta_dim_flow < 2:
                raise ValueError(
                    "--theta-flow-onehot-state expects theta dimension >= 2 after preprocessing; "
                    f"got theta_dim={theta_dim_flow}."
                )
        theta_repr = f"one_hot[{theta_dim_flow}]" if theta_onehot_state else ("scalar" if theta_dim_flow == 1 else f"vector[{theta_dim_flow}]")
        print(
            f"[{theta_field_method}] "
            f"fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
            f"scheduler={getattr(args, 'flow_scheduler', 'cosine')} method={theta_field_method} "
            f"t_eval={flow_eval_t:.6f} "
            f"theta_std={theta_std:.6f} theta_dim={theta_dim_flow} theta_repr={theta_repr} "
            f"prior_theta_dim={theta_dim_prior_flow}"
        )
        _xf_post = ""
        if flow_score_arch == "film":
            _xf_post = (
                f" post_film theta_embed_dim={int(getattr(args, 'flow_cond_embed_dim', 16))}"
                f" theta_embed_depth={int(getattr(args, 'flow_cond_embed_depth', 1))}"
                f" theta_embed_act={str(getattr(args, 'flow_cond_embed_act', 'silu'))}"
            )
        elif flow_score_arch == "film_fourier":
            _om_eff, _om_log = effective_flow_theta_fourier_omega_post(args)
            _xf_post = (
                f" theta_fourier_post_k={int(getattr(args, 'flow_theta_fourier_k', 4))}"
                f" {_om_log}"
                f" theta_fourier_post_omega_in_net={float(_om_eff):.6g}"
                f" theta_fourier_post_linear={not bool(getattr(args, 'flow_theta_fourier_no_linear', False))}"
                f" theta_fourier_post_bias={not bool(getattr(args, 'flow_theta_fourier_no_bias', False))}"
            )
        elif flow_score_arch == "soft_moe":
            _xf_post = (
                f" moe_post_shared_backbone=true moe_post_linear_expert_heads=true"
                f" moe_post_num_experts={int(getattr(args, 'flow_moe_num_experts', 4))}"
                f" moe_post_router_temp={float(getattr(args, 'flow_moe_router_temperature', 1.0)):.6g}"
            )
        _xf_prior = ""
        if flow_prior_arch == "film":
            _xf_prior = (
                f" prior_film theta_embed_dim={int(getattr(args, 'flow_prior_cond_embed_dim', 16))}"
                f" theta_embed_depth={int(getattr(args, 'flow_prior_cond_embed_depth', 1))}"
                f" theta_embed_act={str(getattr(args, 'flow_prior_cond_embed_act', 'silu'))}"
            )
        elif flow_prior_arch == "film_fourier":
            _om_eff_p, _om_log_p = effective_flow_theta_fourier_omega_prior(args)
            _xf_prior = (
                f" theta_fourier_prior_k={int(getattr(args, 'flow_prior_theta_fourier_k', 4))}"
                f" {_om_log_p}"
                f" theta_fourier_prior_omega_in_net={float(_om_eff_p):.6g}"
                f" theta_fourier_prior_linear={not bool(getattr(args, 'flow_prior_theta_fourier_no_linear', False))}"
                f" theta_fourier_prior_bias={not bool(getattr(args, 'flow_prior_theta_fourier_no_bias', False))}"
            )
        print(
            f"[{theta_field_method}] arch "
            f"posterior={flow_score_arch} prior={flow_prior_arch} "
            f"gated_post={bool(getattr(args, 'flow_gated_film', False))} "
            f"gated_prior={bool(getattr(args, 'flow_prior_gated_film', False))} "
            f"ln_post={bool(getattr(args, 'flow_use_layer_norm', False))} "
            f"ln_prior={bool(getattr(args, 'flow_prior_use_layer_norm', False))} "
            f"zero_out_post={bool(getattr(args, 'flow_zero_out_init', False))} "
            f"zero_out_prior={bool(getattr(args, 'flow_prior_zero_out_init', False))} "
            f"{_xf_post}{_xf_prior}"
        )

        if flow_score_arch == "mlp":
            post_ckpt_hparams = {
                "x_dim": int(args.x_dim),
                "hidden_dim": int(getattr(args, "flow_hidden_dim", 128)),
                "depth": int(getattr(args, "flow_depth", 3)),
                "use_logit_time": True,
                "theta_dim": int(theta_dim_flow),
            }
            post_model = ConditionalThetaFlowVelocity(
                x_dim=args.x_dim,
                hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
                depth=int(getattr(args, "flow_depth", 3)),
                use_logit_time=True,
                theta_dim=theta_dim_flow,
            ).to(device)
        elif flow_score_arch == "soft_moe":
            post_ckpt_hparams = {
                "x_dim": int(args.x_dim),
                "hidden_dim": int(getattr(args, "flow_hidden_dim", 128)),
                "depth": int(getattr(args, "flow_depth", 3)),
                "use_logit_time": True,
                "theta_dim": int(theta_dim_flow),
                "num_experts": int(getattr(args, "flow_moe_num_experts", 4)),
                "router_temperature": float(getattr(args, "flow_moe_router_temperature", 1.0)),
                "moe_shared_backbone": True,
                "moe_linear_expert_heads": True,
            }
            post_model = ConditionalThetaFlowVelocitySoftMoE(
                x_dim=args.x_dim,
                hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
                depth=int(getattr(args, "flow_depth", 3)),
                use_logit_time=True,
                theta_dim=theta_dim_flow,
                num_experts=int(getattr(args, "flow_moe_num_experts", 4)),
                router_temperature=float(getattr(args, "flow_moe_router_temperature", 1.0)),
            ).to(device)
        elif flow_score_arch == "film":
            post_ckpt_hparams = {
                "x_dim": int(args.x_dim),
                "hidden_dim": int(getattr(args, "flow_hidden_dim", 128)),
                "depth": int(getattr(args, "flow_depth", 3)),
                "use_logit_time": True,
                "use_layer_norm": bool(getattr(args, "flow_use_layer_norm", False)),
                "gated_film": bool(getattr(args, "flow_gated_film", False)),
                "zero_out_init": bool(getattr(args, "flow_zero_out_init", False)),
                "cond_embed_dim": int(getattr(args, "flow_cond_embed_dim", 16)),
                "cond_embed_depth": int(getattr(args, "flow_cond_embed_depth", 1)),
                "cond_embed_act": str(getattr(args, "flow_cond_embed_act", "silu")),
            }
            post_model = ConditionalThetaFlowVelocityFiLMPerLayer(
                x_dim=int(args.x_dim),
                hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
                depth=int(getattr(args, "flow_depth", 3)),
                use_logit_time=True,
                use_layer_norm=bool(getattr(args, "flow_use_layer_norm", False)),
                gated_film=bool(getattr(args, "flow_gated_film", False)),
                zero_out_init=bool(getattr(args, "flow_zero_out_init", False)),
                cond_embed_dim=int(getattr(args, "flow_cond_embed_dim", 16)),
                cond_embed_depth=int(getattr(args, "flow_cond_embed_depth", 1)),
                cond_embed_act=str(getattr(args, "flow_cond_embed_act", "silu")),
            ).to(device)
        elif flow_score_arch == "film_fourier":
            _om_eff, _ = effective_flow_theta_fourier_omega_post(args)
            post_ckpt_hparams = {
                "x_dim": int(args.x_dim),
                "hidden_dim": int(getattr(args, "flow_hidden_dim", 128)),
                "depth": int(getattr(args, "flow_depth", 3)),
                "use_logit_time": True,
                "use_layer_norm": bool(getattr(args, "flow_use_layer_norm", False)),
                "gated_film": bool(getattr(args, "flow_gated_film", False)),
                "zero_out_init": bool(getattr(args, "flow_zero_out_init", False)),
                "theta_fourier_k": int(getattr(args, "flow_theta_fourier_k", 4)),
                "theta_fourier_omega": float(_om_eff),
                "theta_fourier_include_linear": not bool(getattr(args, "flow_theta_fourier_no_linear", False)),
                "theta_fourier_include_bias": not bool(getattr(args, "flow_theta_fourier_no_bias", False)),
            }
            post_model = ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer(
                x_dim=int(args.x_dim),
                hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
                depth=int(getattr(args, "flow_depth", 3)),
                use_logit_time=True,
                use_layer_norm=bool(getattr(args, "flow_use_layer_norm", False)),
                gated_film=bool(getattr(args, "flow_gated_film", False)),
                zero_out_init=bool(getattr(args, "flow_zero_out_init", False)),
                theta_fourier_k=int(getattr(args, "flow_theta_fourier_k", 4)),
                theta_fourier_omega=float(_om_eff),
                theta_fourier_include_linear=not bool(getattr(args, "flow_theta_fourier_no_linear", False)),
                theta_fourier_include_bias=not bool(getattr(args, "flow_theta_fourier_no_bias", False)),
            ).to(device)
        else:
            raise ValueError("--flow-arch must be one of {'mlp','soft_moe','film','film_fourier'}.")
        post_train_out = train_conditional_theta_flow_model(
            model=post_model,
            theta_train=theta_score_fit,
            x_train=x_score_fit,
            epochs=int(getattr(args, "flow_epochs", 10000)),
            batch_size=int(getattr(args, "flow_batch_size", 256)),
            lr=float(getattr(args, "flow_lr", 1e-3)),
            device=device,
            log_every=max(1, args.log_every),
            theta_val=theta_score_val,
            x_val=x_score_val,
            early_stopping_patience=int(getattr(args, "flow_early_patience", 1000)),
            early_stopping_min_delta=float(getattr(args, "flow_early_min_delta", 1e-4)),
            early_stopping_ema_alpha=float(getattr(args, "flow_early_ema_alpha", 0.05)),
            restore_best=bool(getattr(args, "flow_restore_best", True)),
            scheduler_name=str(getattr(args, "flow_scheduler", "cosine")),
            endpoint_loss_weight=(
                float(getattr(args, "flow_endpoint_loss_weight", 0.0))
                if theta_field_method == "theta_flow"
                else 0.0
            ),
            endpoint_ode_steps=int(getattr(args, "flow_endpoint_steps", 20)),
        )
        post_train_losses = np.asarray(post_train_out["train_losses"], dtype=np.float64)
        post_val_losses = np.asarray(post_train_out["val_losses"], dtype=np.float64)
        post_val_monitor_losses = np.asarray(post_train_out.get("val_monitor_losses", []), dtype=np.float64)
        post_best_epoch = int(post_train_out["best_epoch"])
        post_stopped_epoch = int(post_train_out["stopped_epoch"])
        post_stopped_early = bool(post_train_out["stopped_early"])
        post_best_val_loss = float(post_train_out["best_val_loss"])

        post_loss_fig = os.path.join(args.output_dir, "score_loss_vs_epoch.png")
        epochs_arr = np.arange(1, post_train_losses.size + 1)
        plt.figure(figsize=(8.8, 5.0))
        plt.plot(epochs_arr, post_train_losses, color="#1f77b4", linewidth=2.0, label="Theta-flow train loss")
        if post_val_losses.size == post_train_losses.size and np.any(np.isfinite(post_val_losses)):
            plt.plot(epochs_arr, post_val_losses, color="#d62728", linewidth=2.0, label="Theta-flow val loss")
        if post_val_monitor_losses.size == post_train_losses.size and np.any(np.isfinite(post_val_monitor_losses)):
            plt.plot(
                epochs_arr,
                post_val_monitor_losses,
                color="#ff7f0e",
                linewidth=2.0,
                linestyle="--",
                label=f"Theta-flow val EMA (α={getattr(args, 'flow_early_ema_alpha', 0.05):g})",
            )
        if 1 <= post_best_epoch <= post_train_losses.size:
            plt.axvline(post_best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {post_best_epoch}")
        if 1 <= post_stopped_epoch <= post_train_losses.size:
            plt.axvline(
                post_stopped_epoch,
                color="#9467bd",
                linestyle=":",
                linewidth=1.6,
                label=f"Stop epoch {post_stopped_epoch}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Conditional theta-flow training")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(post_loss_fig, dpi=180)
        plt.close()

        if flow_prior_arch == "mlp":
            prior_ckpt_hparams = {
                "hidden_dim": int(getattr(args, "prior_hidden_dim", 128)),
                "depth": int(getattr(args, "prior_depth", 3)),
                "use_logit_time": True,
                "theta_dim": int(theta_dim_prior_flow),
            }
            prior_model_flow = PriorThetaFlowVelocity(
                hidden_dim=int(getattr(args, "prior_hidden_dim", 128)),
                depth=int(getattr(args, "prior_depth", 3)),
                use_logit_time=True,
                theta_dim=theta_dim_prior_flow,
            ).to(device)
        elif flow_prior_arch == "film":
            prior_ckpt_hparams = {
                "hidden_dim": int(getattr(args, "prior_hidden_dim", 128)),
                "depth": int(getattr(args, "prior_depth", 3)),
                "use_logit_time": True,
                "use_layer_norm": bool(getattr(args, "flow_prior_use_layer_norm", False)),
                "gated_film": bool(getattr(args, "flow_prior_gated_film", False)),
                "zero_out_init": bool(getattr(args, "flow_prior_zero_out_init", False)),
                "cond_embed_dim": int(getattr(args, "flow_prior_cond_embed_dim", 16)),
                "cond_embed_depth": int(getattr(args, "flow_prior_cond_embed_depth", 1)),
                "cond_embed_act": str(getattr(args, "flow_prior_cond_embed_act", "silu")),
            }
            prior_model_flow = PriorThetaFlowVelocityFiLMPerLayer(
                hidden_dim=int(getattr(args, "prior_hidden_dim", 128)),
                depth=int(getattr(args, "prior_depth", 3)),
                use_logit_time=True,
                use_layer_norm=bool(getattr(args, "flow_prior_use_layer_norm", False)),
                gated_film=bool(getattr(args, "flow_prior_gated_film", False)),
                zero_out_init=bool(getattr(args, "flow_prior_zero_out_init", False)),
                cond_embed_dim=int(getattr(args, "flow_prior_cond_embed_dim", 16)),
                cond_embed_depth=int(getattr(args, "flow_prior_cond_embed_depth", 1)),
                cond_embed_act=str(getattr(args, "flow_prior_cond_embed_act", "silu")),
            ).to(device)
        elif flow_prior_arch == "film_fourier":
            _om_eff_p, _ = effective_flow_theta_fourier_omega_prior(args)
            prior_ckpt_hparams = {
                "hidden_dim": int(getattr(args, "prior_hidden_dim", 128)),
                "depth": int(getattr(args, "prior_depth", 3)),
                "use_logit_time": True,
                "use_layer_norm": bool(getattr(args, "flow_prior_use_layer_norm", False)),
                "gated_film": bool(getattr(args, "flow_prior_gated_film", False)),
                "zero_out_init": bool(getattr(args, "flow_prior_zero_out_init", False)),
                "theta_fourier_k": int(getattr(args, "flow_prior_theta_fourier_k", 4)),
                "theta_fourier_omega": float(_om_eff_p),
                "theta_fourier_include_linear": not bool(getattr(args, "flow_prior_theta_fourier_no_linear", False)),
                "theta_fourier_include_bias": not bool(getattr(args, "flow_prior_theta_fourier_no_bias", False)),
            }
            prior_model_flow = PriorThetaFlowVelocityThetaFourierFiLMPerLayer(
                hidden_dim=int(getattr(args, "prior_hidden_dim", 128)),
                depth=int(getattr(args, "prior_depth", 3)),
                use_logit_time=True,
                use_layer_norm=bool(getattr(args, "flow_prior_use_layer_norm", False)),
                gated_film=bool(getattr(args, "flow_prior_gated_film", False)),
                zero_out_init=bool(getattr(args, "flow_prior_zero_out_init", False)),
                theta_fourier_k=int(getattr(args, "flow_prior_theta_fourier_k", 4)),
                theta_fourier_omega=float(_om_eff_p),
                theta_fourier_include_linear=not bool(getattr(args, "flow_prior_theta_fourier_no_linear", False)),
                theta_fourier_include_bias=not bool(getattr(args, "flow_prior_theta_fourier_no_bias", False)),
            ).to(device)
        else:
            raise ValueError("--flow-arch must be one of {'mlp','film','film_fourier'}.")
        prior_train_out = train_prior_theta_flow_model(
            model=prior_model_flow,
            theta_train=theta_prior_fit,
            epochs=int(getattr(args, "prior_epochs", 10000)),
            batch_size=int(getattr(args, "prior_batch_size", 256)),
            lr=float(getattr(args, "prior_lr", 1e-3)),
            device=device,
            log_every=max(1, args.log_every),
            theta_val=theta_prior_val,
            early_stopping_patience=int(getattr(args, "prior_early_patience", 1000)),
            early_stopping_min_delta=float(getattr(args, "prior_early_min_delta", 1e-4)),
            early_stopping_ema_alpha=float(getattr(args, "prior_early_ema_alpha", 0.05)),
            restore_best=bool(getattr(args, "prior_restore_best", True)),
            scheduler_name=str(getattr(args, "flow_scheduler", "cosine")),
        )
        prior_train_losses = np.asarray(prior_train_out["train_losses"], dtype=np.float64)
        prior_val_losses = np.asarray(prior_train_out["val_losses"], dtype=np.float64)
        prior_val_monitor_losses = np.asarray(prior_train_out.get("val_monitor_losses", []), dtype=np.float64)
        prior_best_epoch = int(prior_train_out["best_epoch"])
        prior_stopped_epoch = int(prior_train_out["stopped_epoch"])
        prior_stopped_early = bool(prior_train_out["stopped_early"])
        prior_best_val_loss = float(prior_train_out["best_val_loss"])

        prior_loss_fig = os.path.join(args.output_dir, "prior_score_loss_vs_epoch.png")
        epochs_prior = np.arange(1, prior_train_losses.size + 1)
        plt.figure(figsize=(8.8, 5.0))
        plt.plot(epochs_prior, prior_train_losses, color="#1f77b4", linewidth=2.0, label="Prior theta-flow train loss")
        if prior_val_losses.size == prior_train_losses.size and np.any(np.isfinite(prior_val_losses)):
            plt.plot(epochs_prior, prior_val_losses, color="#d62728", linewidth=2.0, label="Prior theta-flow val loss")
        if prior_val_monitor_losses.size == prior_train_losses.size and np.any(np.isfinite(prior_val_monitor_losses)):
            plt.plot(
                epochs_prior,
                prior_val_monitor_losses,
                color="#ff7f0e",
                linewidth=2.0,
                linestyle="--",
                label=f"Prior val EMA (α={getattr(args, 'prior_early_ema_alpha', 0.05):g})",
            )
        if 1 <= prior_best_epoch <= prior_train_losses.size:
            plt.axvline(prior_best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {prior_best_epoch}")
        if 1 <= prior_stopped_epoch <= prior_train_losses.size:
            plt.axvline(
                prior_stopped_epoch,
                color="#9467bd",
                linestyle=":",
                linewidth=1.6,
                label=f"Stop epoch {prior_stopped_epoch}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Prior theta-flow training")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prior_loss_fig, dpi=180)
        plt.close()

        post_ckpt_path = _save_theta_flow_model_checkpoint(
            output_dir=args.output_dir,
            filename="theta_flow_posterior_checkpoint.pt",
            model=post_model,
            model_role="posterior",
            theta_field_method=theta_field_method,
            flow_arch=flow_score_arch,
            flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
            theta_dim_flow=theta_dim_flow,
            model_hparams=post_ckpt_hparams,
            args=args,
        )
        prior_ckpt_path = _save_theta_flow_model_checkpoint(
            output_dir=args.output_dir,
            filename="theta_flow_prior_checkpoint.pt",
            model=prior_model_flow,
            model_role="prior",
            theta_field_method=theta_field_method,
            flow_arch=flow_prior_arch,
            flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
            theta_dim_flow=theta_dim_prior_flow,
            model_hparams=prior_ckpt_hparams,
            args=args,
        )
        print(f"[theta_flow_ckpt] saved posterior checkpoint: {post_ckpt_path}")
        print(f"[theta_flow_ckpt] saved prior checkpoint: {prior_ckpt_path}")

        # H matrix: full per-run pool (train ∪ validation); fitting stays on train split with val early stop.
        theta_h_matrix = np.asarray(theta_all, dtype=np.float64)
        x_h_matrix = np.asarray(x_all, dtype=np.float64)

        h_result: HMatrixResult | None = None
        if bool(getattr(args, "compute_h_matrix", False)):
            h_eval = flow_eval_t
            _h_field = "theta_flow" if theta_field_method == "theta_flow" else "theta_path_integral"
            print(
                "[h_matrix] "
                f"enabled=True field={_h_field} "
                f"t_eval={h_eval:.6f} "
                f"n_theta_x={int(theta_h_matrix.shape[0])} (train+validation full pool) "
                f"restore_original_order={bool(getattr(args, 'h_restore_original_order', False))} "
                f"pair_batch_size={int(getattr(args, 'h_batch_size', 65536))}"
            )
            h_estimator = HMatrixEstimator(
                model_post=post_model,
                model_prior=prior_model_flow,
                sigma_eval=h_eval,
                device=device,
                pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
                field_method=_h_field,
                flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
            )
            h_result = h_estimator.run(
                theta=theta_h_matrix,
                x=x_h_matrix,
                theta_prior=theta_prior_all,
                restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
            )

        suffix = "_non_gauss" if args.dataset_family == "cosine_gmm" else "_theta_cov"
        if h_result is not None:
            h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path = _save_h_matrix_dsm_artifacts(
                args, h_result, suffix
            )
            if theta_field_method == "theta_flow":
                print(
                    "[summary] theta_flow mode completed (H-matrix only path; "
                    "ODESolver.compute_likelihood on conditional theta-flow for log p(theta|x) minus prior; "
                    "Bayes-ratio matrix)."
                )
            else:
                print(
                    "[summary] theta_path_integral mode completed (H-matrix only path; "
                    "velocity converted to score via path.velocity_to_epsilon and s=-eps/sigma_t; "
                    "trapezoid integral along sorted theta)."
                )
            print("Saved artifacts:")
            print(f"  - {post_loss_fig}")
            print(f"  - {prior_loss_fig}")
            print(f"  - {post_ckpt_path}")
            print(f"  - {prior_ckpt_path}")
            print(f"  - {h_npz_path}")
            print(f"  - {h_summary_path}")
            print(f"  - {h_fig_path}")
            if h_delta_fig_path:
                print(f"  - {h_delta_fig_path}")
            tnpz = _save_dsm_score_prior_training_losses_npz(
                args.output_dir,
                theta_all=theta_all,
                theta_score_fit=theta_score_fit,
                theta_score_val=theta_score_val,
                score_split="dataset_train_validation",
                score_train_losses=post_train_losses,
                score_val_losses=post_val_losses,
                score_val_monitor_losses=post_val_monitor_losses,
                score_best_epoch=post_best_epoch,
                score_stopped_epoch=post_stopped_epoch,
                score_stopped_early=post_stopped_early,
                score_best_val_loss=post_best_val_loss,
                prior_enable=True,
                prior_train_losses=prior_train_losses,
                prior_val_losses=prior_val_losses,
                prior_val_monitor_losses=prior_val_monitor_losses,
                prior_best_epoch=prior_best_epoch,
                prior_stopped_epoch=prior_stopped_epoch,
                prior_stopped_early=prior_stopped_early,
                prior_best_val_loss=prior_best_val_loss,
                score_has_nonfinite=bool(post_train_out.get("has_nonfinite", False)),
                score_grad_norm_mean=float(post_train_out.get("grad_norm_mean", float("nan"))),
                score_grad_norm_max=float(post_train_out.get("grad_norm_max", float("nan"))),
                score_param_norm_final=float(post_train_out.get("param_norm_final", float("nan"))),
                score_n_clipped_steps=int(post_train_out.get("n_clipped_steps", 0)),
                score_n_total_steps=int(post_train_out.get("n_total_steps", 0)),
                score_lr_last=float(post_train_out.get("lr_last", float("nan"))),
                prior_has_nonfinite=bool(prior_train_out.get("has_nonfinite", False)),
                prior_grad_norm_mean=float(prior_train_out.get("grad_norm_mean", float("nan"))),
                prior_grad_norm_max=float(prior_train_out.get("grad_norm_max", float("nan"))),
                prior_param_norm_final=float(prior_train_out.get("param_norm_final", float("nan"))),
                prior_n_clipped_steps=int(prior_train_out.get("n_clipped_steps", 0)),
                prior_n_total_steps=int(prior_train_out.get("n_total_steps", 0)),
                prior_lr_last=float(prior_train_out.get("lr_last", float("nan"))),
                theta_field_method=theta_field_method,
            )
            print(f"[training_losses] saved {tnpz}")
            return

        raise RuntimeError(
            "theta_flow and theta_path_integral require --compute-h-matrix to produce output artifacts."
        )

    theta_std = float(np.std(theta_score_fit))
    sigma_post = float("nan")
    sigma_min: float | None = None
    sigma_max: float | None = None
    sigma_base: float | None = None
    if args.score_sigma_scale_mode == "theta_std":
        sigma_base = theta_std
        sigma_min = args.score_sigma_min_alpha * sigma_base
        sigma_max = args.score_sigma_max_alpha * sigma_base
        print(f"[sigma_scale] mode=theta_std theta_std={theta_std:.6f}")
    elif args.score_sigma_scale_mode == "posterior_proxy":
        sigma_post = posterior_proxy_sigma(theta_score_fit, x_score_fit, l2=args.score_proxy_l2)
        sigma_base = sigma_post
        sigma_min = args.score_proxy_min_mult * sigma_post
        sigma_max = args.score_proxy_max_mult * sigma_post
        print(
            "[sigma_scale] "
            f"mode=posterior_proxy theta_std={theta_std:.6f} "
            f"sigma_post={sigma_post:.6f} l2={args.score_proxy_l2:g} "
            f"mult=[{args.score_proxy_min_mult:g},{args.score_proxy_max_mult:g}]"
        )
    else:
        sigma_min = args.score_fixed_sigma
        sigma_max = args.score_fixed_sigma
        sigma_base = args.score_fixed_sigma
        print(f"[sigma_scale] mode=fixed sigma={args.score_fixed_sigma:.6f} theta_std={theta_std:.6f}")

    score_model = build_posterior_score_model(args, device)
    _sa = str(getattr(args, "score_arch", "mlp")).lower()
    print(
        "[score_model] "
        f"arch={_sa} "
        f"hidden_dim={int(args.score_hidden_dim)} depth={int(args.score_depth)} "
        f"noise_mode={args.score_noise_mode}"
        + (
            " film=x_trunk_residual_film(theta_tilde,sigma)"
            if _sa == "film"
            else ""
        )
    )
    if args.score_noise_mode == "continuous":
        sigma_values = geometric_sigma_schedule(
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            n_levels=args.score_eval_sigmas,
            descending=True,
        )
        print(
            f"[score:continuous] "
            f"sigma_min={min(float(sigma_min), float(sigma_max)):.6f}, "
            f"sigma_max={max(float(sigma_min), float(sigma_max)):.6f}, "
            f"eval_sigma_values={sigma_values.tolist()}"
        )
        score_train_out = train_score_model_ncsm_continuous(
            model=score_model,
            theta_train=theta_score_fit,
            x_train=x_score_fit,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            epochs=args.score_epochs,
            batch_size=args.score_batch_size,
            lr=args.score_lr,
            device=device,
            log_every=max(1, args.log_every),
            theta_val=theta_score_val,
            x_val=x_score_val,
            early_stopping_patience=args.score_early_patience,
            early_stopping_min_delta=args.score_early_min_delta,
            early_stopping_ema_alpha=float(args.score_early_ema_alpha),
            early_stopping_ema_warmup_epochs=int(getattr(args, "score_early_ema_warmup_epochs", 0)),
            restore_best=args.score_restore_best,
            optimizer_name=str(getattr(args, "score_optimizer", "adamw")),
            weight_decay=float(getattr(args, "score_weight_decay", 1e-4)),
            max_grad_norm=float(getattr(args, "score_max_grad_norm", 1.0)),
            lr_scheduler=str(getattr(args, "score_lr_scheduler", "cosine")),
            lr_warmup_frac=float(getattr(args, "score_lr_warmup_frac", 0.05)),
            loss_type=str(getattr(args, "score_loss_type", "huber")),
            huber_delta=float(getattr(args, "score_huber_delta", 1.0)),
            normalize_by_sigma=bool(getattr(args, "score_normalize_by_sigma", False)),
            abort_on_nonfinite=bool(getattr(args, "score_abort_on_nonfinite", True)),
            sigma_sample_mode=str(getattr(args, "score_sigma_sample_mode", "uniform_log")),
            sigma_sample_beta=float(getattr(args, "score_sigma_sample_beta", 2.0)),
        )
    else:
        if args.score_sigma_scale_mode == "fixed":
            sigma_values = np.full((args.score_eval_sigmas,), fill_value=float(sigma_base), dtype=np.float64)
        else:
            sigma_alpha = parse_sigma_alpha_list(args.score_sigma_alpha_list)
            sigma_values = sigma_alpha * float(sigma_base)
        print(f"[score:discrete] sigma_values={sigma_values.tolist()}")
        score_train_out = train_score_model(
            model=score_model,
            theta_train=theta_score_fit,
            x_train=x_score_fit,
            sigma_values=sigma_values,
            epochs=args.score_epochs,
            batch_size=args.score_batch_size,
            lr=args.score_lr,
            device=device,
            log_every=max(1, args.log_every),
            theta_val=theta_score_val,
            x_val=x_score_val,
            early_stopping_patience=args.score_early_patience,
            early_stopping_min_delta=args.score_early_min_delta,
            early_stopping_ema_alpha=float(args.score_early_ema_alpha),
            early_stopping_ema_warmup_epochs=int(getattr(args, "score_early_ema_warmup_epochs", 0)),
            restore_best=args.score_restore_best,
            optimizer_name=str(getattr(args, "score_optimizer", "adamw")),
            weight_decay=float(getattr(args, "score_weight_decay", 1e-4)),
            max_grad_norm=float(getattr(args, "score_max_grad_norm", 1.0)),
            lr_scheduler=str(getattr(args, "score_lr_scheduler", "cosine")),
            lr_warmup_frac=float(getattr(args, "score_lr_warmup_frac", 0.05)),
            loss_type=str(getattr(args, "score_loss_type", "huber")),
            huber_delta=float(getattr(args, "score_huber_delta", 1.0)),
            normalize_by_sigma=bool(getattr(args, "score_normalize_by_sigma", False)),
            abort_on_nonfinite=bool(getattr(args, "score_abort_on_nonfinite", True)),
        )
    score_train_losses = np.asarray(score_train_out["train_losses"], dtype=np.float64)
    score_val_losses = np.asarray(score_train_out["val_losses"], dtype=np.float64)
    score_val_monitor_losses = np.asarray(score_train_out.get("val_monitor_losses", []), dtype=np.float64)
    best_epoch = int(score_train_out["best_epoch"])
    stopped_epoch = int(score_train_out["stopped_epoch"])
    stopped_early = bool(score_train_out["stopped_early"])
    best_val_loss = float(score_train_out["best_val_loss"])
    prior_enable = bool(getattr(args, "prior_enable", True))
    print(
        "[score_early_stop] "
        f"stopped_early={stopped_early} stopped_epoch={stopped_epoch} "
        f"best_epoch={best_epoch} best_val_smooth={best_val_loss:.6f} "
        f"ema_alpha={args.score_early_ema_alpha} "
        f"ema_warmup_epochs={int(getattr(args, 'score_early_ema_warmup_epochs', 0))} "
        f"restore_best={args.score_restore_best}"
    )

    loss_fig_path = os.path.join(args.output_dir, "score_loss_vs_epoch.png")
    epochs_arr = np.arange(1, score_train_losses.size + 1)
    plt.figure(figsize=(8.8, 5.0))
    plt.plot(epochs_arr, score_train_losses, color="#1f77b4", linewidth=2.0, label="Score train loss")
    if score_val_losses.size == score_train_losses.size and np.any(np.isfinite(score_val_losses)):
        plt.plot(epochs_arr, score_val_losses, color="#d62728", linewidth=2.0, label="Score val loss")
    if score_val_monitor_losses.size == score_train_losses.size and np.any(np.isfinite(score_val_monitor_losses)):
        plt.plot(
            epochs_arr,
            score_val_monitor_losses,
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
            label=f"Score val EMA (α={args.score_early_ema_alpha:g})",
        )
    if 1 <= best_epoch <= score_train_losses.size:
        plt.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {best_epoch}")
    if 1 <= stopped_epoch <= score_train_losses.size:
        plt.axvline(
            stopped_epoch,
            color="#9467bd",
            linestyle=":",
            linewidth=1.6,
            label=f"Stop epoch {stopped_epoch}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Score Training and Validation Loss")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=180)
    plt.close()
    if not prior_enable:
        tnpz = _save_dsm_score_prior_training_losses_npz(
            args.output_dir,
            theta_all=theta_all,
            theta_score_fit=theta_score_fit,
            theta_score_val=theta_score_val,
            score_split="dataset_train_validation",
            score_train_losses=score_train_losses,
            score_val_losses=score_val_losses,
            score_val_monitor_losses=score_val_monitor_losses,
            score_best_epoch=best_epoch,
            score_stopped_epoch=stopped_epoch,
            score_stopped_early=stopped_early,
            score_best_val_loss=best_val_loss,
            prior_enable=False,
            prior_train_losses=np.empty(0, dtype=np.float64),
            prior_val_losses=np.empty(0, dtype=np.float64),
            prior_val_monitor_losses=np.empty(0, dtype=np.float64),
            prior_best_epoch=0,
            prior_stopped_epoch=0,
            prior_stopped_early=False,
            prior_best_val_loss=float("nan"),
            score_has_nonfinite=bool(score_train_out.get("has_nonfinite", False)),
            score_grad_norm_mean=float(score_train_out.get("grad_norm_mean", float("nan"))),
            score_grad_norm_max=float(score_train_out.get("grad_norm_max", float("nan"))),
            score_param_norm_final=float(score_train_out.get("param_norm_final", float("nan"))),
            score_n_clipped_steps=int(score_train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=int(score_train_out.get("n_total_steps", 0)),
            score_lr_last=float(score_train_out.get("lr_last", float("nan"))),
        )
        print(f"[training_losses] saved {tnpz}")
    eval_low = args.theta_low + args.eval_margin
    eval_high = args.theta_high - args.eval_margin
    theta_score_fisher_eval, x_score_fisher_eval = theta_score_val, x_score_val
    print(
        "[score_fisher_eval] "
        f"validation n={theta_score_fisher_eval.shape[0]}"
    )

    fisher_mode = str(getattr(args, "fisher_score_mode", "posterior_minus_prior"))
    fisher_mode_display = fisher_mode if prior_enable else "posterior_only (prior disabled)"
    prior_train_out: dict[str, Any] | None = None
    prior_train_losses = np.asarray([], dtype=np.float64)
    prior_val_losses = np.asarray([], dtype=np.float64)
    prior_val_monitor_losses = np.asarray([], dtype=np.float64)
    prior_best_epoch = 0
    prior_stopped_epoch = 0
    prior_stopped_early = False
    prior_best_val_loss = float("nan")
    prior_fig_path = ""
    score_eval_wp: Any | None = None
    score_eval: Any | None = None
    prior_model: PriorScore1D | PriorScore1DFiLMPerLayer | None = None
    h_result: HMatrixResult | None = None
    h_sigma_eval = float("nan")

    if prior_enable:
        prior_model = build_prior_score_model(args, device)
        _pa = str(getattr(args, "prior_score_arch", "mlp")).lower()
        print(
            "[prior_model] "
            f"arch={_pa} "
            f"hidden_dim={int(getattr(args, 'prior_hidden_dim', 128))} depth={int(getattr(args, 'prior_depth', 3))}"
            + (
                " film=theta_trunk_residual_film(theta_tilde,sigma)"
                if _pa == "film"
                else ""
            )
        )
        print(
            "[prior_train] "
            f"fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
            f"noise_mode={args.score_noise_mode} (same σ schedule as posterior)"
        )
        if args.score_noise_mode == "continuous":
            prior_train_out = train_prior_score_model_ncsm_continuous(
                model=prior_model,
                theta_train=theta_score_fit,
                sigma_min=float(sigma_min),
                sigma_max=float(sigma_max),
                epochs=int(getattr(args, "prior_epochs", 10000)),
                batch_size=int(getattr(args, "prior_batch_size", 256)),
                lr=float(getattr(args, "prior_lr", 1e-3)),
                device=device,
                log_every=max(1, args.log_every),
                theta_val=theta_score_val,
                early_stopping_patience=int(getattr(args, "prior_early_patience", 1000)),
                early_stopping_min_delta=float(getattr(args, "prior_early_min_delta", 1e-4)),
                early_stopping_ema_alpha=float(getattr(args, "prior_early_ema_alpha", 0.05)),
                early_stopping_ema_warmup_epochs=int(getattr(args, "prior_early_ema_warmup_epochs", 0)),
                restore_best=bool(getattr(args, "prior_restore_best", True)),
                optimizer_name=str(getattr(args, "prior_optimizer", "adamw")),
                weight_decay=float(getattr(args, "prior_weight_decay", 1e-4)),
                max_grad_norm=float(getattr(args, "prior_max_grad_norm", 1.0)),
                lr_scheduler=str(getattr(args, "prior_lr_scheduler", "cosine")),
                lr_warmup_frac=float(getattr(args, "prior_lr_warmup_frac", 0.05)),
                loss_type=str(getattr(args, "prior_loss_type", "huber")),
                huber_delta=float(getattr(args, "prior_huber_delta", 1.0)),
                normalize_by_sigma=bool(getattr(args, "prior_normalize_by_sigma", False)),
                abort_on_nonfinite=bool(getattr(args, "prior_abort_on_nonfinite", True)),
                sigma_sample_mode=str(getattr(args, "score_sigma_sample_mode", "uniform_log")),
                sigma_sample_beta=float(getattr(args, "score_sigma_sample_beta", 2.0)),
            )
        else:
            prior_train_out = train_prior_score_model(
                model=prior_model,
                theta_train=theta_score_fit,
                sigma_values=sigma_values,
                epochs=int(getattr(args, "prior_epochs", 10000)),
                batch_size=int(getattr(args, "prior_batch_size", 256)),
                lr=float(getattr(args, "prior_lr", 1e-3)),
                device=device,
                log_every=max(1, args.log_every),
                theta_val=theta_score_val,
                early_stopping_patience=int(getattr(args, "prior_early_patience", 1000)),
                early_stopping_min_delta=float(getattr(args, "prior_early_min_delta", 1e-4)),
                early_stopping_ema_alpha=float(getattr(args, "prior_early_ema_alpha", 0.05)),
                early_stopping_ema_warmup_epochs=int(getattr(args, "prior_early_ema_warmup_epochs", 0)),
                restore_best=bool(getattr(args, "prior_restore_best", True)),
                optimizer_name=str(getattr(args, "prior_optimizer", "adamw")),
                weight_decay=float(getattr(args, "prior_weight_decay", 1e-4)),
                max_grad_norm=float(getattr(args, "prior_max_grad_norm", 1.0)),
                lr_scheduler=str(getattr(args, "prior_lr_scheduler", "cosine")),
                lr_warmup_frac=float(getattr(args, "prior_lr_warmup_frac", 0.05)),
                loss_type=str(getattr(args, "prior_loss_type", "huber")),
                huber_delta=float(getattr(args, "prior_huber_delta", 1.0)),
                normalize_by_sigma=bool(getattr(args, "prior_normalize_by_sigma", False)),
                abort_on_nonfinite=bool(getattr(args, "prior_abort_on_nonfinite", True)),
            )
        prior_train_losses = np.asarray(prior_train_out["train_losses"], dtype=np.float64)
        prior_val_losses = np.asarray(prior_train_out["val_losses"], dtype=np.float64)
        prior_val_monitor_losses = np.asarray(prior_train_out.get("val_monitor_losses", []), dtype=np.float64)
        prior_best_epoch = int(prior_train_out["best_epoch"])
        prior_stopped_epoch = int(prior_train_out["stopped_epoch"])
        prior_stopped_early = bool(prior_train_out["stopped_early"])
        prior_best_val_loss = float(prior_train_out["best_val_loss"])
        print(
            "[prior_early_stop] "
            f"stopped_early={prior_stopped_early} stopped_epoch={prior_stopped_epoch} "
            f"best_epoch={prior_best_epoch} best_val_smooth={prior_best_val_loss:.6f} "
            f"ema_alpha={getattr(args, 'prior_early_ema_alpha', 0.05)} "
            f"ema_warmup_epochs={int(getattr(args, 'prior_early_ema_warmup_epochs', 0))} "
            f"restore_best={getattr(args, 'prior_restore_best', True)}"
        )
        prior_fig_path = os.path.join(args.output_dir, "prior_score_loss_vs_epoch.png")
        epochs_prior = np.arange(1, prior_train_losses.size + 1)
        plt.figure(figsize=(8.8, 5.0))
        plt.plot(epochs_prior, prior_train_losses, color="#1f77b4", linewidth=2.0, label="Prior score train loss")
        if prior_val_losses.size == prior_train_losses.size and np.any(np.isfinite(prior_val_losses)):
            plt.plot(epochs_prior, prior_val_losses, color="#d62728", linewidth=2.0, label="Prior score val loss")
        if prior_val_monitor_losses.size == prior_train_losses.size and np.any(np.isfinite(prior_val_monitor_losses)):
            plt.plot(
                epochs_prior,
                prior_val_monitor_losses,
                color="#ff7f0e",
                linewidth=2.0,
                linestyle="--",
                label=f"Prior val EMA (α={getattr(args, 'prior_early_ema_alpha', 0.05):g})",
            )
        if 1 <= prior_best_epoch <= prior_train_losses.size:
            plt.axvline(prior_best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {prior_best_epoch}")
        if 1 <= prior_stopped_epoch <= prior_train_losses.size:
            plt.axvline(
                prior_stopped_epoch,
                color="#9467bd",
                linestyle=":",
                linewidth=1.6,
                label=f"Stop epoch {prior_stopped_epoch}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Prior score (unconditional DSM) training")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prior_fig_path, dpi=180)
        plt.close()
        tnpz = _save_dsm_score_prior_training_losses_npz(
            args.output_dir,
            theta_all=theta_all,
            theta_score_fit=theta_score_fit,
            theta_score_val=theta_score_val,
            score_split="dataset_train_validation",
            score_train_losses=score_train_losses,
            score_val_losses=score_val_losses,
            score_val_monitor_losses=score_val_monitor_losses,
            score_best_epoch=best_epoch,
            score_stopped_epoch=stopped_epoch,
            score_stopped_early=stopped_early,
            score_best_val_loss=best_val_loss,
            prior_enable=True,
            prior_train_losses=prior_train_losses,
            prior_val_losses=prior_val_losses,
            prior_val_monitor_losses=prior_val_monitor_losses,
            prior_best_epoch=prior_best_epoch,
            prior_stopped_epoch=prior_stopped_epoch,
            prior_stopped_early=prior_stopped_early,
            prior_best_val_loss=prior_best_val_loss,
            score_has_nonfinite=bool(score_train_out.get("has_nonfinite", False)),
            score_grad_norm_mean=float(score_train_out.get("grad_norm_mean", float("nan"))),
            score_grad_norm_max=float(score_train_out.get("grad_norm_max", float("nan"))),
            score_param_norm_final=float(score_train_out.get("param_norm_final", float("nan"))),
            score_n_clipped_steps=int(score_train_out.get("n_clipped_steps", 0)),
            score_n_total_steps=int(score_train_out.get("n_total_steps", 0)),
            score_lr_last=float(score_train_out.get("lr_last", float("nan"))),
            prior_has_nonfinite=bool(prior_train_out.get("has_nonfinite", False)),
            prior_grad_norm_mean=float(prior_train_out.get("grad_norm_mean", float("nan"))),
            prior_grad_norm_max=float(prior_train_out.get("grad_norm_max", float("nan"))),
            prior_param_norm_final=float(prior_train_out.get("param_norm_final", float("nan"))),
            prior_n_clipped_steps=int(prior_train_out.get("n_clipped_steps", 0)),
            prior_n_total_steps=int(prior_train_out.get("n_total_steps", 0)),
            prior_lr_last=float(prior_train_out.get("lr_last", float("nan"))),
        )
        print(f"[training_losses] saved {tnpz}")

        if bool(getattr(args, "skip_shared_fisher_gt_compare", False)):
            if not bool(getattr(args, "compute_h_matrix", False)):
                raise RuntimeError("skip_shared_fisher_gt_compare requires compute_h_matrix")
            suffix = "_non_gauss" if args.dataset_family == "cosine_gmm" else "_theta_cov"
            sigma_min_eval = float(np.min(np.asarray(sigma_values, dtype=np.float64)))
            h_sigma_eval = float(args.h_sigma_eval) if float(args.h_sigma_eval) > 0.0 else sigma_min_eval
            print(
                "[h_matrix] "
                f"enabled=True sigma_eval={h_sigma_eval:.6f} "
                f"n_theta_x={int(np.asarray(theta_all).shape[0])} (train+validation full pool) "
                f"restore_original_order={bool(getattr(args, 'h_restore_original_order', False))} "
                f"pair_batch_size={int(getattr(args, 'h_batch_size', 65536))}"
            )
            h_estimator = HMatrixEstimator(
                model_post=score_model,
                model_prior=prior_model,
                sigma_eval=h_sigma_eval,
                device=device,
                pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
            )
            h_result_skip = h_estimator.run(
                theta=np.asarray(theta_all, dtype=np.float64),
                x=np.asarray(x_all, dtype=np.float64),
                restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
            )
            print(
                "[h_matrix] "
                f"done n={h_result_skip.theta_used.size} "
                f"delta_diag_max_abs={h_result_skip.delta_diag_max_abs:.3e} "
                f"h_sym_max_asym_abs={h_result_skip.h_sym_max_asym_abs:.3e}"
            )
            h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path = _save_h_matrix_dsm_artifacts(
                args, h_result_skip, suffix
            )
            print(
                "[summary] DSM skip_shared_fisher_gt_compare: H-matrix only; "
                "skipped Fisher curves, decoder training, and GT Fisher comparison artifacts."
            )
            print("Saved artifacts:")
            print(f"  - {tnpz}")
            print(f"  - {loss_fig_path}")
            print(f"  - {prior_fig_path}")
            print(f"  - {h_npz_path}")
            print(f"  - {h_summary_path}")
            print(f"  - {h_fig_path}")
            if h_delta_fig_path:
                print(f"  - {h_delta_fig_path}")
            return

        score_eval_wp = evaluate_score_fisher_with_prior(
            model_post=score_model,
            model_prior=prior_model,
            theta_eval=theta_score_fisher_eval,
            x_eval=x_score_fisher_eval,
            dataset=dataset,
            sigma_values=sigma_values,
            fd_delta=args.fd_delta,
            n_bins=args.n_bins,
            min_bin_count=args.score_min_bin_count,
            eval_low=eval_low,
            eval_high=eval_high,
            device=device,
        )
        centers = score_eval_wp.curves_combined.centers
    else:
        if bool(getattr(args, "skip_shared_fisher_gt_compare", False)):
            raise ValueError(
                "--skip-shared-fisher-gt-compare requires prior score training (cannot estimate H without prior)."
            )
        score_eval = evaluate_score_fisher(
            model=score_model,
            theta_eval=theta_score_fisher_eval,
            x_eval=x_score_fisher_eval,
            dataset=dataset,
            sigma_values=sigma_values,
            fd_delta=args.fd_delta,
            n_bins=args.n_bins,
            min_bin_count=args.score_min_bin_count,
            eval_low=eval_low,
            eval_high=eval_high,
            device=device,
        )
        centers = score_eval.curves.centers

    if bool(getattr(args, "compute_h_matrix", False)):
        if prior_model is None:
            raise RuntimeError("compute_h_matrix requires a trained prior model.")
        sigma_min_eval = float(np.min(np.asarray(sigma_values, dtype=np.float64)))
        h_sigma_eval = float(args.h_sigma_eval) if float(args.h_sigma_eval) > 0.0 else sigma_min_eval
        print(
            "[h_matrix] "
            f"enabled=True sigma_eval={h_sigma_eval:.6f} "
            f"n_theta_x={int(np.asarray(theta_all).shape[0])} (train+validation full pool) "
            f"restore_original_order={bool(getattr(args, 'h_restore_original_order', False))} "
            f"pair_batch_size={int(getattr(args, 'h_batch_size', 65536))}"
        )
        h_estimator = HMatrixEstimator(
            model_post=score_model,
            model_prior=prior_model,
            sigma_eval=h_sigma_eval,
            device=device,
            pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
        )
        h_result = h_estimator.run(
            theta=np.asarray(theta_all, dtype=np.float64),
            x=np.asarray(x_all, dtype=np.float64),
            restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
        )
        print(
            "[h_matrix] "
            f"done n={h_result.theta_used.size} "
            f"delta_diag_max_abs={h_result.delta_diag_max_abs:.3e} "
            f"h_sym_max_asym_abs={h_result.h_sym_max_asym_abs:.3e}"
        )

    dec_theta_validation = theta_validation
    dec_x_validation = x_validation

    decoder_fisher, decoder_se, decoder_valid, decoder_diag = fit_decoder_from_shared_data(
        centers=centers,
        theta_train=theta_train,
        x_train=x_train,
        theta_eval=dec_theta_validation,
        x_eval=dec_x_validation,
        epsilon=args.decoder_epsilon,
        bandwidth=args.decoder_bandwidth,
        min_class_count=args.decoder_min_class_count,
        train_cap=args.decoder_train_cap,
        eval_cap=args.decoder_eval_cap,
        epochs=args.decoder_epochs,
        batch_size=args.decoder_batch_size,
        lr=args.decoder_lr,
        hidden_dim=args.decoder_hidden_dim,
        depth=args.decoder_depth,
        val_frac=args.decoder_val_frac,
        min_val_class_size=args.decoder_min_val_class_size,
        early_patience=args.decoder_early_patience,
        early_min_delta=args.decoder_early_min_delta,
        early_ema_alpha=float(args.decoder_early_ema_alpha),
        restore_best=args.decoder_restore_best,
        device=device,
        log_every=max(1, args.log_every),
        rng=rng,
        debug_bins=bool(getattr(args, "decoder_debug_bins", False)),
    )

    if args.dataset_family in (
        "cosine_gaussian",
        "cosine_gaussian_const_noise",
        "cosine_gaussian_sqrtd",
        "cosine_gaussian_sqrtd_rand_tune",
        "randamp_gaussian",
        "randamp_gaussian_sqrtd",
        "cos_sin_piecewise",
        "linear_piecewise",
    ):
        gt = analytic_fisher_curve(centers, dataset)
        gt_se = np.full_like(gt, np.nan)
    else:
        gt, gt_se = gt_fisher_curve_exact_score_mc(
            centers=centers,
            dataset=dataset,
            mc_samples_per_bin=args.gt_mc_samples_per_bin,
        )
    if prior_enable and score_eval_wp is not None:
        score_valid_post = np.isfinite(score_eval_wp.curves_posterior.fisher_model) & score_eval_wp.curves_posterior.valid
        score_valid_comb = np.isfinite(score_eval_wp.curves_combined.fisher_model) & score_eval_wp.curves_combined.valid
        score_metrics_post = compute_metrics(score_eval_wp.curves_posterior.fisher_model, gt, score_valid_post)
        score_metrics_comb = compute_metrics(score_eval_wp.curves_combined.fisher_model, gt, score_valid_comb)
        if fisher_mode == "posterior_minus_prior":
            score_metrics = score_metrics_comb
            score_valid = score_valid_comb
        else:
            score_metrics = score_metrics_post
            score_valid = score_valid_post
    else:
        assert score_eval is not None
        score_valid = np.isfinite(score_eval.curves.fisher_model) & score_eval.curves.valid
        score_metrics = compute_metrics(score_eval.curves.fisher_model, gt, score_valid)
        score_metrics_post = score_metrics
        score_metrics_comb = score_metrics
        score_valid_post = score_valid
        score_valid_comb = score_valid

    decoder_metrics = compute_metrics(decoder_fisher, gt, decoder_valid)

    decoder_diag_path = os.path.join(args.output_dir, "decoder_bin_diagnostics.txt")
    with open(decoder_diag_path, "w", encoding="utf-8") as df:
        df.write("Per-bin decoder Fisher diagnostics (local two-class windows)\n")
        df.write(
            f"epsilon={args.decoder_epsilon} bandwidth={args.decoder_bandwidth} "
            f"min_class_count={args.decoder_min_class_count}\n"
        )
        df.write(
            f"val_frac={args.decoder_val_frac} min_val_class_size={args.decoder_min_val_class_size} "
            f"min_ntr_for_fit_hint={decoder_diag['min_ntr_for_fit']}\n"
        )
        df.write(f"skip_counts={decoder_diag['skip_counts']}\n")
        df.write(
            "columns: i theta0 reason ntr_pos_raw ntr_neg_raw nev_pos_raw nev_neg_raw "
            "ntr nev nval nfit valid\n"
        )
        for i in range(int(centers.size)):
            r = str(decoder_diag["reasons"][i])
            df.write(
                f"{i} {float(centers[i]):+.6f} {r} "
                f"{int(decoder_diag['ntr_pos_raw'][i])} {int(decoder_diag['ntr_neg_raw'][i])} "
                f"{int(decoder_diag['nev_pos_raw'][i])} {int(decoder_diag['nev_neg_raw'][i])} "
                f"{decoder_diag['ntr'][i]:.1f} {decoder_diag['nev'][i]:.1f} "
                f"{decoder_diag['nval'][i]:.1f} {decoder_diag['nfit'][i]:.1f} "
                f"{int(decoder_valid[i])}\n"
            )

    suffix = "_non_gauss" if args.dataset_family == "cosine_gmm" else "_theta_cov"
    h_npz_path = ""
    h_summary_path = ""
    h_fig_path = ""
    h_delta_fig_path = ""
    if h_result is not None:
        h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path = _save_h_matrix_dsm_artifacts(
            args, h_result, suffix
        )

    fig_path = os.path.join(args.output_dir, f"fisher_curve_shared_dataset_vs_gt{suffix}.png")
    plt.figure(figsize=(9.0, 5.6))
    plt.plot(centers, gt, color="black", linewidth=2.6, label="GT Fisher")
    if np.any(np.isfinite(gt_se)):
        plt.fill_between(centers, gt - 1.96 * gt_se, gt + 1.96 * gt_se, color="black", alpha=0.10, linewidth=0.0)
    if prior_enable and score_eval_wp is not None:
        plt.plot(
            centers[score_valid_post],
            score_eval_wp.curves_posterior.fisher_model[score_valid_post],
            color="#aec7e8",
            linewidth=2.0,
            linestyle="--",
            label=r"Posterior score only ($\sigma_{\min}$)",
        )
        if np.any(np.isfinite(score_eval_wp.curves_posterior.se_model[score_valid_post])):
            plt.fill_between(
                centers[score_valid_post],
                score_eval_wp.curves_posterior.fisher_model[score_valid_post]
                - 1.96 * score_eval_wp.curves_posterior.se_model[score_valid_post],
                score_eval_wp.curves_posterior.fisher_model[score_valid_post]
                + 1.96 * score_eval_wp.curves_posterior.se_model[score_valid_post],
                color="#aec7e8",
                alpha=0.10,
                linewidth=0.0,
            )
        plt.plot(
            centers[score_valid_comb],
            score_eval_wp.curves_combined.fisher_model[score_valid_comb],
            color="#1f77b4",
            linewidth=2.2,
            label=r"Likelihood score (post $-$ prior, $\sigma_{\min}$)",
        )
        if np.any(np.isfinite(score_eval_wp.curves_combined.se_model[score_valid_comb])):
            plt.fill_between(
                centers[score_valid_comb],
                score_eval_wp.curves_combined.fisher_model[score_valid_comb]
                - 1.96 * score_eval_wp.curves_combined.se_model[score_valid_comb],
                score_eval_wp.curves_combined.fisher_model[score_valid_comb]
                + 1.96 * score_eval_wp.curves_combined.se_model[score_valid_comb],
                color="#1f77b4",
                alpha=0.12,
                linewidth=0.0,
            )
    else:
        assert score_eval is not None
        plt.plot(
            centers[score_valid],
            score_eval.curves.fisher_model[score_valid],
            color="#1f77b4",
            linewidth=2.2,
            label=r"Score matching (at $\sigma_{\min}$)",
        )
        if np.any(np.isfinite(score_eval.curves.se_model[score_valid])):
            plt.fill_between(
                centers[score_valid],
                score_eval.curves.fisher_model[score_valid] - 1.96 * score_eval.curves.se_model[score_valid],
                score_eval.curves.fisher_model[score_valid] + 1.96 * score_eval.curves.se_model[score_valid],
                color="#1f77b4",
                alpha=0.12,
                linewidth=0.0,
            )
    plt.plot(
        centers[decoder_valid],
        decoder_fisher[decoder_valid],
        color="#2ca02c",
        linewidth=2.2,
        label="Decoder local classification (shared data)",
    )
    if np.any(np.isfinite(decoder_se[decoder_valid])):
        plt.fill_between(
            centers[decoder_valid],
            decoder_fisher[decoder_valid] - 1.96 * decoder_se[decoder_valid],
            decoder_fisher[decoder_valid] + 1.96 * decoder_se[decoder_valid],
            color="#2ca02c",
            alpha=0.12,
            linewidth=0.0,
        )
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fisher information")
    plt.title(
        f"Shared Dataset Comparison ({args.dataset_family}): "
        f"{'Post/prior score vs ' if prior_enable else ''}Decoder vs GT"
    )
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    npz_path = os.path.join(args.output_dir, f"shared_dataset_compare_curves{suffix}.npz")
    if prior_enable and score_eval_wp is not None:
        fisher_primary = (
            score_eval_wp.curves_combined.fisher_model
            if fisher_mode == "posterior_minus_prior"
            else score_eval_wp.curves_posterior.fisher_model
        )
        fisher_primary_se = (
            score_eval_wp.curves_combined.se_model
            if fisher_mode == "posterior_minus_prior"
            else score_eval_wp.curves_posterior.se_model
        )
        fisher_primary_valid = (
            score_valid_comb.astype(np.int32)
            if fisher_mode == "posterior_minus_prior"
            else score_valid_post.astype(np.int32)
        )
        np.savez(
            npz_path,
            centers=centers,
            fisher_gt=gt,
            fisher_gt_se=gt_se,
            fisher_score=fisher_primary,
            fisher_score_se=fisher_primary_se,
            fisher_score_valid=fisher_primary_valid,
            fisher_score_posterior=score_eval_wp.curves_posterior.fisher_model,
            fisher_score_posterior_se=score_eval_wp.curves_posterior.se_model,
            fisher_score_posterior_valid=score_valid_post.astype(np.int32),
            fisher_score_combined=score_eval_wp.curves_combined.fisher_model,
            fisher_score_combined_se=score_eval_wp.curves_combined.se_model,
            fisher_score_combined_valid=score_valid_comb.astype(np.int32),
            fisher_score_mode=np.asarray([fisher_mode_display], dtype=object),
            prior_enable=np.asarray([1], dtype=np.int32),
            fisher_decoder=decoder_fisher,
            fisher_decoder_se=decoder_se,
            fisher_decoder_valid=decoder_valid.astype(np.int32),
            score_losses=score_train_losses,
            score_train_losses=score_train_losses,
            score_val_losses=score_val_losses,
            score_val_smooth_losses=score_val_monitor_losses,
            score_best_epoch=np.asarray([best_epoch], dtype=np.int32),
            score_stopped_epoch=np.asarray([stopped_epoch], dtype=np.int32),
            score_stopped_early=np.asarray([int(stopped_early)], dtype=np.int32),
            score_best_val_loss=np.asarray([best_val_loss], dtype=np.float64),
            score_has_nonfinite=np.asarray([int(bool(score_train_out.get("has_nonfinite", False)))], dtype=np.int32),
            score_grad_norm_mean=np.asarray([float(score_train_out.get("grad_norm_mean", float("nan")))], dtype=np.float64),
            score_grad_norm_max=np.asarray([float(score_train_out.get("grad_norm_max", float("nan")))], dtype=np.float64),
            score_param_norm_final=np.asarray([float(score_train_out.get("param_norm_final", float("nan")))], dtype=np.float64),
            score_n_clipped_steps=np.asarray([int(score_train_out.get("n_clipped_steps", 0))], dtype=np.int32),
            score_n_total_steps=np.asarray([int(score_train_out.get("n_total_steps", 0))], dtype=np.int32),
            prior_train_losses=prior_train_losses,
            prior_val_losses=prior_val_losses,
            prior_val_smooth_losses=prior_val_monitor_losses,
            prior_best_epoch=np.asarray([prior_best_epoch], dtype=np.int32),
            prior_stopped_epoch=np.asarray([prior_stopped_epoch], dtype=np.int32),
            prior_stopped_early=np.asarray([int(prior_stopped_early)], dtype=np.int32),
            prior_best_val_loss=np.asarray([prior_best_val_loss], dtype=np.float64),
            prior_has_nonfinite=np.asarray([int(bool(prior_train_out.get("has_nonfinite", False)))], dtype=np.int32),
            prior_grad_norm_mean=np.asarray([float(prior_train_out.get("grad_norm_mean", float("nan")))], dtype=np.float64),
            prior_grad_norm_max=np.asarray([float(prior_train_out.get("grad_norm_max", float("nan")))], dtype=np.float64),
            prior_param_norm_final=np.asarray([float(prior_train_out.get("param_norm_final", float("nan")))], dtype=np.float64),
            prior_n_clipped_steps=np.asarray([int(prior_train_out.get("n_clipped_steps", 0))], dtype=np.int32),
            prior_n_total_steps=np.asarray([int(prior_train_out.get("n_total_steps", 0))], dtype=np.int32),
        )
    else:
        assert score_eval is not None
        np.savez(
            npz_path,
            centers=centers,
            fisher_gt=gt,
            fisher_gt_se=gt_se,
            fisher_score=score_eval.curves.fisher_model,
            fisher_score_se=score_eval.curves.se_model,
            fisher_score_valid=score_valid.astype(np.int32),
            fisher_score_mode=np.asarray([fisher_mode_display], dtype=object),
            prior_enable=np.asarray([0], dtype=np.int32),
            fisher_decoder=decoder_fisher,
            fisher_decoder_se=decoder_se,
            fisher_decoder_valid=decoder_valid.astype(np.int32),
            score_losses=score_train_losses,
            score_train_losses=score_train_losses,
            score_val_losses=score_val_losses,
            score_val_smooth_losses=score_val_monitor_losses,
            score_best_epoch=np.asarray([best_epoch], dtype=np.int32),
            score_stopped_epoch=np.asarray([stopped_epoch], dtype=np.int32),
            score_stopped_early=np.asarray([int(stopped_early)], dtype=np.int32),
            score_best_val_loss=np.asarray([best_val_loss], dtype=np.float64),
            score_has_nonfinite=np.asarray([int(bool(score_train_out.get("has_nonfinite", False)))], dtype=np.int32),
            score_grad_norm_mean=np.asarray([float(score_train_out.get("grad_norm_mean", float("nan")))], dtype=np.float64),
            score_grad_norm_max=np.asarray([float(score_train_out.get("grad_norm_max", float("nan")))], dtype=np.float64),
            score_param_norm_final=np.asarray([float(score_train_out.get("param_norm_final", float("nan")))], dtype=np.float64),
            score_n_clipped_steps=np.asarray([int(score_train_out.get("n_clipped_steps", 0))], dtype=np.int32),
            score_n_total_steps=np.asarray([int(score_train_out.get("n_total_steps", 0))], dtype=np.int32),
        )

    metrics_path = os.path.join(args.output_dir, f"metrics_vs_gt{suffix}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Shared dataset Fisher comparison against GT\n")
        f.write(f"dataset_family: {args.dataset_family}\n")
        f.write(f"x_dim: {args.x_dim}\n")
        f.write(f"n_total: {args.n_total}\n")
        f.write(f"train_frac: {args.train_frac}\n")
        f.write("score_split: dataset_train_idx / validation_idx (NPZ splits)\n")
        f.write(
            "score_data_counts: "
            f"train={int(np.asarray(theta_train).shape[0])}, "
            f"validation={int(np.asarray(theta_validation).shape[0])}\n"
        )
        f.write(
            "score_fit_val_counts: "
            f"fit={theta_score_fit.shape[0]}, val={theta_score_val.shape[0]}\n"
        )
        f.write(f"gt_mc_samples_per_bin: {args.gt_mc_samples_per_bin}\n")
        f.write(
            "score_early_stopping: "
            f"patience={args.score_early_patience}, min_delta={args.score_early_min_delta}, "
            f"ema_alpha={args.score_early_ema_alpha}, "
            f"restore_best={args.score_restore_best}, stopped_early={stopped_early}, "
            f"best_epoch={best_epoch}, stopped_epoch={stopped_epoch}, best_val_smooth={best_val_loss}\n"
        )
        f.write(
            "score_stability: "
            f"preset={getattr(args, 'dsm_stability_preset', 'stable_v1')}, "
            f"optimizer={getattr(args, 'score_optimizer', 'adamw')}, "
            f"wd={getattr(args, 'score_weight_decay', 0.0)}, "
            f"scheduler={getattr(args, 'score_lr_scheduler', 'none')}, "
            f"warmup_frac={getattr(args, 'score_lr_warmup_frac', 0.0)}, "
            f"max_grad_norm={getattr(args, 'score_max_grad_norm', 0.0)}, "
            f"loss_type={getattr(args, 'score_loss_type', 'mse')}, "
            f"huber_delta={getattr(args, 'score_huber_delta', 1.0)}, "
            f"normalize_by_sigma={getattr(args, 'score_normalize_by_sigma', False)}, "
            f"sigma_sample_mode={getattr(args, 'score_sigma_sample_mode', 'uniform_log')}, "
            f"sigma_sample_beta={getattr(args, 'score_sigma_sample_beta', 2.0)}, "
            f"has_nonfinite={bool(score_train_out.get('has_nonfinite', False))}, "
            f"grad_norm_mean={float(score_train_out.get('grad_norm_mean', float('nan'))):.6g}, "
            f"grad_norm_max={float(score_train_out.get('grad_norm_max', float('nan'))):.6g}, "
            f"n_clipped_steps={int(score_train_out.get('n_clipped_steps', 0))}/"
            f"{int(score_train_out.get('n_total_steps', 0))}\n"
        )
        f.write(f"score_noise_mode: {args.score_noise_mode}\n")
        f.write(f"score_sigma_scale_mode: {args.score_sigma_scale_mode}\n")
        f.write(
            "score_fisher_eval_method: "
            f"sigma_min_direct, sigma_eval_used={float(np.min(sigma_values))}\n"
        )
        f.write(f"prior_enable: {prior_enable}\n")
        f.write(f"fisher_score_mode: {fisher_mode_display}\n")
        if prior_enable and score_eval_wp is not None:
            f.write(
                "prior_early_stopping: "
                f"patience={getattr(args, 'prior_early_patience', 1000)}, "
                f"min_delta={getattr(args, 'prior_early_min_delta', 1e-4)}, "
                f"ema_alpha={getattr(args, 'prior_early_ema_alpha', 0.05)}, "
                f"restore_best={getattr(args, 'prior_restore_best', True)}, "
                f"stopped_early={prior_stopped_early}, best_epoch={prior_best_epoch}, "
                f"stopped_epoch={prior_stopped_epoch}, best_val_smooth={prior_best_val_loss}\n"
            )
            f.write(
                "prior_stability: "
                f"optimizer={getattr(args, 'prior_optimizer', 'adamw')}, "
                f"wd={getattr(args, 'prior_weight_decay', 0.0)}, "
                f"scheduler={getattr(args, 'prior_lr_scheduler', 'none')}, "
                f"warmup_frac={getattr(args, 'prior_lr_warmup_frac', 0.0)}, "
                f"max_grad_norm={getattr(args, 'prior_max_grad_norm', 0.0)}, "
                f"loss_type={getattr(args, 'prior_loss_type', 'mse')}, "
                f"huber_delta={getattr(args, 'prior_huber_delta', 1.0)}, "
                f"normalize_by_sigma={getattr(args, 'prior_normalize_by_sigma', False)}, "
                f"has_nonfinite={bool(prior_train_out.get('has_nonfinite', False))}, "
                f"grad_norm_mean={float(prior_train_out.get('grad_norm_mean', float('nan'))):.6g}, "
                f"grad_norm_max={float(prior_train_out.get('grad_norm_max', float('nan'))):.6g}, "
                f"n_clipped_steps={int(prior_train_out.get('n_clipped_steps', 0))}/"
                f"{int(prior_train_out.get('n_total_steps', 0))}\n"
            )
            f.write(
                "score_posterior_vs_gt: "
                f"valid={int(score_metrics_post['n_valid'])}/{args.n_bins}, "
                f"rmse={score_metrics_post['rmse']:.6f}, "
                f"mae={score_metrics_post['mae']:.6f}, "
                f"corr={score_metrics_post['corr']:.6f}\n"
            )
            f.write(
                "score_combined_post_minus_prior_vs_gt: "
                f"valid={int(score_metrics_comb['n_valid'])}/{args.n_bins}, "
                f"rmse={score_metrics_comb['rmse']:.6f}, "
                f"mae={score_metrics_comb['mae']:.6f}, "
                f"corr={score_metrics_comb['corr']:.6f}\n"
            )
        f.write(f"theta_std_train: {theta_std}\n")
        if np.isfinite(sigma_post):
            f.write(f"sigma_post_proxy: {sigma_post}\n")
        if args.score_noise_mode == "continuous":
            f.write(
                "score_sigma_continuous: "
                f"sigma_min={min(float(sigma_min), float(sigma_max))}, "
                f"sigma_max={max(float(sigma_min), float(sigma_max))}, "
                f"eval_levels={args.score_eval_sigmas}, "
                f"alpha_min={args.score_sigma_min_alpha}, alpha_max={args.score_sigma_max_alpha}, "
                f"proxy_l2={args.score_proxy_l2}, proxy_mult=[{args.score_proxy_min_mult},{args.score_proxy_max_mult}], "
                f"fixed_sigma={args.score_fixed_sigma}\n"
            )
        else:
            f.write(f"score_sigma_alpha_list: {args.score_sigma_alpha_list}\n")
            f.write(f"score_sigma_discrete_values: {sigma_values.tolist()}\n")
        f.write(f"decoder_epsilon: {args.decoder_epsilon}\n")
        f.write(f"decoder_bandwidth: {args.decoder_bandwidth}\n")
        f.write(
            "decoder_early_stopping: "
            f"val_frac={args.decoder_val_frac}, min_val_class_size={args.decoder_min_val_class_size}, "
            f"patience={args.decoder_early_patience}, min_delta={args.decoder_early_min_delta}, "
            f"ema_alpha={args.decoder_early_ema_alpha}, restore_best={args.decoder_restore_best}\n"
        )
        f.write(f"decoder_min_ntr_for_fit_hint: {decoder_diag['min_ntr_for_fit']}\n")
        f.write(f"decoder_skip_counts: {decoder_diag['skip_counts']}\n")
        f.write(f"decoder_dominant_skip_reason: {decoder_diag['dominant_skip_reason']}\n")
        f.write(f"decoder_bin_diagnostics_file: {decoder_diag_path}\n")
        if h_result is not None:
            f.write(f"h_matrix_enabled: True\n")
            f.write(f"h_sigma_eval: {h_result.sigma_eval}\n")
            f.write(f"h_order_mode: {h_result.order_mode}\n")
            f.write(f"h_n_samples: {h_result.theta_used.size}\n")
            f.write(f"h_delta_diag_max_abs: {h_result.delta_diag_max_abs}\n")
            f.write(f"h_sym_max_asym_abs: {h_result.h_sym_max_asym_abs}\n")
            f.write(f"h_matrix_npz: {h_npz_path}\n")
            f.write(f"h_matrix_summary: {h_summary_path}\n")
            f.write(f"h_matrix_heatmap: {h_fig_path}\n")
            if h_delta_fig_path:
                f.write(f"delta_l_heatmap: {h_delta_fig_path}\n")
        if args.dataset_family in (
            "cosine_gaussian",
            "cosine_gaussian_const_noise",
            "cosine_gaussian_sqrtd",
            "cosine_gaussian_sqrtd_rand_tune",
            "randamp_gaussian",
            "randamp_gaussian_sqrtd",
        ):
            _a = 0.5 * (float(args.cov_theta_amp1) + float(args.cov_theta_amp2))
            f.write(
                "cov_theta: "
                f"alpha_mean_activity=({args.cov_theta_amp1}+{args.cov_theta_amp2})/2={_a:.6g}, "
                f"amp_rho={args.cov_theta_amp_rho}, "
                f"freq1={args.cov_theta_freq1}, freq2={args.cov_theta_freq2}, freq_rho={args.cov_theta_freq_rho}, "
                f"phase1={args.cov_theta_phase1}, phase2={args.cov_theta_phase2}, phase_rho={args.cov_theta_phase_rho}, "
                f"rho_clip={args.rho_clip}\n"
            )
        elif args.dataset_family == "cos_sin_piecewise":
            f.write(
                "cos_sin_piecewise: "
                f"sigma_piecewise_low={args.sigma_piecewise_low}, "
                f"sigma_piecewise_high={args.sigma_piecewise_high}, "
                f"theta_zero_to_low={args.theta_zero_to_low}\n"
            )
        elif args.dataset_family == "linear_piecewise":
            f.write(
                "linear_piecewise: "
                f"linear_k={args.linear_k}, "
                f"sigma_piecewise_low={args.sigma_piecewise_low}, "
                f"sigma_piecewise_high={args.sigma_piecewise_high}, "
                f"linear_sigma_schedule={getattr(args, 'linear_sigma_schedule', 'linear')}, "
                f"linear_sigma_sigmoid_center={getattr(args, 'linear_sigma_sigmoid_center', 0.0)}, "
                f"linear_sigma_sigmoid_steepness={getattr(args, 'linear_sigma_sigmoid_steepness', 2.0)}, "
                f"theta_zero_to_low={args.theta_zero_to_low}\n"
            )
        else:
            f.write(
                "gmm_theta: "
                f"sep_scale={args.gmm_sep_scale}, sep_freq={args.gmm_sep_freq}, sep_phase={args.gmm_sep_phase}, "
                f"mix_logit_scale={args.gmm_mix_logit_scale}, mix_bias={args.gmm_mix_bias}, "
                f"mix_freq={args.gmm_mix_freq}, mix_phase={args.gmm_mix_phase}, rho_clip={args.rho_clip}\n"
            )
        f.write(
            "score_vs_gt_primary: "
            f"(mode={fisher_mode_display}) "
            f"valid={int(score_metrics['n_valid'])}/{args.n_bins}, "
            f"rmse={score_metrics['rmse']:.6f}, "
            f"mae={score_metrics['mae']:.6f}, "
            f"corr={score_metrics['corr']:.6f}\n"
        )
        f.write(
            "decoder_vs_gt: "
            f"valid={int(decoder_metrics['n_valid'])}/{args.n_bins}, "
            f"rmse={decoder_metrics['rmse']:.6f}, "
            f"mae={decoder_metrics['mae']:.6f}, "
            f"corr={decoder_metrics['corr']:.6f}\n"
        )

    print("[summary]")
    print(
        f"  score vs GT (primary, mode={fisher_mode_display}): "
        f"valid={int(score_metrics['n_valid'])}/{args.n_bins}, "
        f"rmse={score_metrics['rmse']:.4f}, mae={score_metrics['mae']:.4f}, corr={score_metrics['corr']:.4f}"
    )
    if prior_enable and score_eval_wp is not None:
        print(
            "  score posterior-only vs GT: "
            f"valid={int(score_metrics_post['n_valid'])}/{args.n_bins}, "
            f"rmse={score_metrics_post['rmse']:.4f}, mae={score_metrics_post['mae']:.4f}, "
            f"corr={score_metrics_post['corr']:.4f}"
        )
        print(
            "  score combined (post - prior) vs GT: "
            f"valid={int(score_metrics_comb['n_valid'])}/{args.n_bins}, "
            f"rmse={score_metrics_comb['rmse']:.4f}, mae={score_metrics_comb['mae']:.4f}, "
            f"corr={score_metrics_comb['corr']:.4f}"
        )
    print(
        "  decoder vs GT: "
        f"valid={int(decoder_metrics['n_valid'])}/{args.n_bins}, "
        f"rmse={decoder_metrics['rmse']:.4f}, mae={decoder_metrics['mae']:.4f}, corr={decoder_metrics['corr']:.4f}"
    )
    print("Saved artifacts:")
    print(f"  - {loss_fig_path}")
    if prior_enable and prior_fig_path:
        print(f"  - {prior_fig_path}")
    print(f"  - {fig_path}")
    print(f"  - {npz_path}")
    print(f"  - {metrics_path}")
    print(f"  - {decoder_diag_path}")
    if h_result is not None:
        print(f"  - {h_npz_path}")
        print(f"  - {h_summary_path}")
        print(f"  - {h_fig_path}")
        if h_delta_fig_path:
            print(f"  - {h_delta_fig_path}")
