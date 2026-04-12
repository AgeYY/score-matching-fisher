"""Shared Fisher estimation: score vs decoder vs ground truth (core logic)."""

from __future__ import annotations

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
    LocalDecoderLogit,
    PriorScore1D,
    PriorScore1DFiLMPerLayer,
    PriorThetaFlowVelocity,
)
from fisher.shared_dataset_io import (
    SHARED_DATASET_META_KEYS,
    apply_sigma_defaults_for_dataset_family,
    meta_dict_from_args,
)
from fisher.trainers import (
    geometric_sigma_schedule,
    train_conditional_theta_flow_model,
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
    | ToyConditionalGaussianRandampDataset
    | ToyConditionalGaussianRandampSqrtdDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset
):
    family = str(meta["dataset_family"])
    seed = int(meta["seed"])
    if family == "gaussian":
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
    if family == "gaussian_sqrtd":
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
    if family == "gaussian_randamp":
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
    if family == "gaussian_randamp_sqrtd":
        amps_raw = meta.get("randamp_mu_amp_per_dim")
        amps_sqrt: np.ndarray | None
        if amps_raw is not None:
            amps_sqrt = np.asarray(amps_raw, dtype=np.float64).reshape(-1)
        else:
            amps_sqrt = None
        return ToyConditionalGaussianRandampSqrtdDataset(
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
            randamp_mu_amp_per_dim=amps_sqrt,
            seed=seed,
        )
    if family == "gmm_non_gauss":
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
    if family == "cos_sin_piecewise_noise":
        return ToyCosSinPiecewiseNoiseDataset(
            theta_low=float(meta["theta_low"]),
            theta_high=float(meta["theta_high"]),
            x_dim=int(meta["x_dim"]),
            sigma_piecewise_low=float(meta.get("sigma_piecewise_low", 0.1)),
            sigma_piecewise_high=float(meta.get("sigma_piecewise_high", 0.1)),
            theta_zero_to_low=bool(meta.get("theta_zero_to_low", True)),
            seed=seed,
        )
    if family == "linear_piecewise_noise":
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
    | ToyConditionalGaussianRandampDataset
    | ToyConditionalGaussianRandampSqrtdDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset
):
    return build_dataset_from_meta(meta_dict_from_args(ns))


def validate_dataset_sample_args(args: Any) -> None:
    apply_sigma_defaults_for_dataset_family(args)
    if getattr(args, "tuning_curve_family", "cosine") not in ("cosine", "von_mises_raw", "gaussian_raw"):
        raise ValueError('--tuning-curve-family must be "cosine", "von_mises_raw", or "gaussian_raw".')
    if getattr(args, "tuning_curve_family", "cosine") == "von_mises_raw":
        if float(getattr(args, "vm_kappa", 0.0)) < 0.0:
            raise ValueError("--vm-kappa must be non-negative for von_mises_raw.")
        if float(getattr(args, "vm_mu_amp", 0.0)) <= 0.0:
            raise ValueError("--vm-mu-amp must be positive for von_mises_raw.")
    elif getattr(args, "tuning_curve_family", "cosine") == "gaussian_raw":
        if float(getattr(args, "gauss_kappa", 0.0)) < 0.0:
            raise ValueError("--gauss-kappa must be non-negative for gaussian_raw.")
        if float(getattr(args, "gauss_mu_amp", 0.0)) <= 0.0:
            raise ValueError("--gauss-mu-amp must be positive for gaussian_raw.")
    if str(getattr(args, "dataset_family", "")) in ("gaussian_randamp", "gaussian_randamp_sqrtd"):
        _rlo = float(getattr(args, "randamp_mu_low", 0.5))
        _rhi = float(getattr(args, "randamp_mu_high", 1.5))
        if not (_rlo < _rhi):
            raise ValueError(
                "gaussian_randamp / gaussian_randamp_sqrtd require --randamp-mu-low < --randamp-mu-high."
            )
        if float(getattr(args, "randamp_kappa", 0.0)) < 0.0:
            raise ValueError("--randamp-kappa must be non-negative for gaussian_randamp families.")
    if args.x_dim < 2:
        raise ValueError("--x-dim must be >= 2.")
    if str(getattr(args, "dataset_family", "")) in (
        "cos_sin_piecewise_noise",
        "linear_piecewise_noise",
    ) and int(args.x_dim) != 2:
        raise ValueError("--dataset-family cos_sin_piecewise_noise / linear_piecewise_noise requires --x-dim 2.")
    if float(getattr(args, "sigma_piecewise_low", 0.0)) <= 0.0:
        raise ValueError("--sigma-piecewise-low must be positive.")
    if float(getattr(args, "sigma_piecewise_high", 0.0)) <= 0.0:
        raise ValueError("--sigma-piecewise-high must be positive.")
    if str(getattr(args, "dataset_family", "")) == "linear_piecewise_noise":
        if str(getattr(args, "linear_sigma_schedule", "linear")).lower() == "sigmoid":
            if float(getattr(args, "linear_sigma_sigmoid_steepness", 0.0)) <= 0.0:
                raise ValueError("--linear-sigma-sigmoid-steepness must be positive when --linear-sigma-schedule is sigmoid.")
    if int(args.n_total) < 2:
        raise ValueError("--n-total must be >= 2 for train/eval split.")
    if not (0.0 < float(args.train_frac) <= 1.0):
        raise ValueError("--train-frac must be in (0, 1].")


def validate_estimation_args(args: Any) -> None:
    _apply_dsm_stability_preset(args)
    if str(getattr(args, "theta_field_method", "dsm")) not in ("dsm", "flow"):
        raise ValueError("--theta-field-method must be one of {'dsm', 'flow'}.")
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
    if int(getattr(args, "flow_batch_size", 1)) < 1:
        raise ValueError("--flow-batch-size must be >= 1.")
    if float(getattr(args, "flow_lr", 0.0)) <= 0.0:
        raise ValueError("--flow-lr must be positive.")
    if int(getattr(args, "flow_early_patience", 1)) < 1:
        raise ValueError("--flow-early-patience must be >= 1.")
    if float(getattr(args, "flow_early_min_delta", 0.0)) < 0.0:
        raise ValueError("--flow-early-min-delta must be non-negative.")
    if not (0.0 < float(getattr(args, "flow_early_ema_alpha", 0.05)) <= 1.0):
        raise ValueError("--flow-early-ema-alpha must be in (0, 1].")
    if not (0.0 <= float(getattr(args, "flow_eval_t", 0.9)) <= 1.0):
        raise ValueError("--flow-eval-t must be in [0, 1].")
    if bool(getattr(args, "compute_h_matrix", False)) and not bool(getattr(args, "prior_enable", True)):
        raise ValueError("--compute-h-matrix requires prior score; do not use --no-prior-score.")
    if bool(getattr(args, "skip_shared_fisher_gt_compare", False)):
        if not bool(getattr(args, "compute_h_matrix", False)):
            raise ValueError("--skip-shared-fisher-gt-compare requires --compute-h-matrix.")
        if not bool(getattr(args, "prior_enable", True)):
            raise ValueError("--skip-shared-fisher-gt-compare requires prior score; do not use --no-prior-score.")


def _save_dsm_score_prior_training_losses_npz(
    output_dir: str,
    *,
    theta_all: np.ndarray,
    theta_score_fit: np.ndarray,
    theta_score_val: np.ndarray,
    score_data_mode: str,
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
    theta_field_method: str = "dsm",
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
        score_data_mode=np.asarray([str(score_data_mode)], dtype=object),
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
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    rng: np.random.Generator,
) -> None:
    _apply_dsm_stability_preset(args)
    run_seed = int(getattr(args, "seed", 7))
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    device = require_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.score_data_mode == "full":
        theta_score_train, x_score_train = theta_all, x_all
        theta_score_eval, x_score_eval = theta_all, x_all
    else:
        theta_score_train, x_score_train = theta_train, x_train
        theta_score_eval, x_score_eval = theta_eval, x_eval
    print(
        "[score_data] "
        f"mode={args.score_data_mode} "
        f"train={theta_score_train.shape[0]} eval={theta_score_eval.shape[0]}"
    )
    if args.score_val_source == "eval_set":
        theta_score_fit = theta_score_train
        x_score_fit = x_score_train
        theta_score_val = theta_score_eval
        x_score_val = x_score_eval
        if theta_score_fit.shape[0] < 1 or theta_score_val.shape[0] < 1:
            raise ValueError("score train/eval split must have non-empty train and eval sets.")
        print(
            "[score_train] "
            f"val_source=eval_set fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]}"
        )
    else:
        n_score_total = theta_score_train.shape[0]
        if n_score_total < 2:
            raise ValueError("score training requires at least 2 samples for train/validation split.")
        n_score_val = int(round(SCORE_VAL_FRACTION * n_score_total))
        n_score_val = max(1, min(n_score_val, n_score_total - 1))
        score_perm = rng.permutation(n_score_total)
        score_val_idx = score_perm[:n_score_val]
        score_fit_idx = score_perm[n_score_val:]
        theta_score_fit = theta_score_train[score_fit_idx]
        x_score_fit = x_score_train[score_fit_idx]
        theta_score_val = theta_score_train[score_val_idx]
        x_score_val = x_score_train[score_val_idx]
        print(
            "[score_train] "
            f"val_source=train_split fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
            f"val_target_frac={SCORE_VAL_FRACTION} "
            f"val_frac_eff={theta_score_val.shape[0]/n_score_total:.4f}"
        )

    theta_field_method = str(getattr(args, "theta_field_method", "dsm")).strip().lower()
    if theta_field_method == "flow":
        if not bool(getattr(args, "prior_enable", True)):
            raise ValueError("theta_field_method=flow currently requires prior model enabled.")
        flow_eval_t = float(getattr(args, "flow_eval_t", 0.9))
        if not (0.0 <= flow_eval_t <= 1.0):
            raise ValueError("--flow-eval-t must be in [0, 1].")
        theta_std = float(np.std(theta_score_fit))
        print(
            "[theta_flow] "
            f"fit={theta_score_fit.shape[0]} val={theta_score_val.shape[0]} "
            f"scheduler={getattr(args, 'flow_scheduler', 'cosine')} t_eval={flow_eval_t:.6f} "
            f"theta_std={theta_std:.6f}"
        )

        post_model = ConditionalThetaFlowVelocity(
            x_dim=args.x_dim,
            hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
            depth=int(getattr(args, "flow_depth", 3)),
            use_logit_time=True,
        ).to(device)
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

        prior_model_flow = PriorThetaFlowVelocity(
            hidden_dim=int(getattr(args, "prior_hidden_dim", 128)),
            depth=int(getattr(args, "prior_depth", 3)),
            use_logit_time=True,
        ).to(device)
        prior_train_out = train_prior_theta_flow_model(
            model=prior_model_flow,
            theta_train=theta_score_fit,
            epochs=int(getattr(args, "prior_epochs", 10000)),
            batch_size=int(getattr(args, "prior_batch_size", 256)),
            lr=float(getattr(args, "prior_lr", 1e-3)),
            device=device,
            log_every=max(1, args.log_every),
            theta_val=theta_score_val,
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

        if args.score_fisher_eval_data == "full":
            theta_score_fisher_eval, x_score_fisher_eval = theta_all, x_all
        else:
            theta_score_fisher_eval, x_score_fisher_eval = theta_score_eval, x_score_eval
        if theta_score_fisher_eval.shape[0] == 0:
            raise ValueError(
                "--score-fisher-eval-data score_eval requires non-empty theta_eval/x_eval; "
                "use --train-frac < 1 or --score-fisher-eval-data full."
            )

        h_result: HMatrixResult | None = None
        if bool(getattr(args, "compute_h_matrix", False)):
            h_eval = flow_eval_t
            print(
                "[h_matrix] "
                f"enabled=True field=flow t_eval={h_eval:.6f} "
                f"restore_original_order={bool(getattr(args, 'h_restore_original_order', False))} "
                f"pair_batch_size={int(getattr(args, 'h_batch_size', 65536))}"
            )
            h_estimator = HMatrixEstimator(
                model_post=post_model,
                model_prior=prior_model_flow,
                sigma_eval=h_eval,
                device=device,
                pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
                field_method="flow",
                flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
            )
            h_result = h_estimator.run(
                theta=theta_score_fisher_eval,
                x=x_score_fisher_eval,
                restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
            )

        suffix = "_non_gauss" if args.dataset_family == "gmm_non_gauss" else "_theta_cov"
        if h_result is not None:
            h_npz_path, h_summary_path, h_fig_path, h_delta_fig_path = _save_h_matrix_dsm_artifacts(
                args, h_result, suffix
            )
            print(
                "[summary] flow mode completed (H-matrix only path; "
                "velocity converted to score via path.velocity_to_epsilon and s=-eps/sigma_t)."
            )
            print("Saved artifacts:")
            print(f"  - {post_loss_fig}")
            print(f"  - {prior_loss_fig}")
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
                score_data_mode=str(args.score_data_mode),
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
                theta_field_method="flow",
            )
            print(f"[training_losses] saved {tnpz}")
            return

        raise RuntimeError("theta_field_method=flow requires --compute-h-matrix to produce output artifacts.")

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
            score_data_mode=str(args.score_data_mode),
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
    if args.score_fisher_eval_data == "full":
        theta_score_fisher_eval, x_score_fisher_eval = theta_all, x_all
    else:
        theta_score_fisher_eval, x_score_fisher_eval = theta_score_eval, x_score_eval
    if theta_score_fisher_eval.shape[0] == 0:
        raise ValueError(
            "--score-fisher-eval-data score_eval requires non-empty theta_eval/x_eval; "
            "use --train-frac < 1 or --score-fisher-eval-data full."
        )
    print(
        "[score_fisher_eval] "
        f"data={args.score_fisher_eval_data} n={theta_score_fisher_eval.shape[0]}"
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
            score_data_mode=str(args.score_data_mode),
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
            suffix = "_non_gauss" if args.dataset_family == "gmm_non_gauss" else "_theta_cov"
            sigma_min_eval = float(np.min(np.asarray(sigma_values, dtype=np.float64)))
            h_sigma_eval = float(args.h_sigma_eval) if float(args.h_sigma_eval) > 0.0 else sigma_min_eval
            print(
                "[h_matrix] "
                f"enabled=True sigma_eval={h_sigma_eval:.6f} "
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
                theta=theta_score_fisher_eval,
                x=x_score_fisher_eval,
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
            theta=theta_score_fisher_eval,
            x=x_score_fisher_eval,
            restore_original_order=bool(getattr(args, "h_restore_original_order", False)),
        )
        print(
            "[h_matrix] "
            f"done n={h_result.theta_used.size} "
            f"delta_diag_max_abs={h_result.delta_diag_max_abs:.3e} "
            f"h_sym_max_asym_abs={h_result.h_sym_max_asym_abs:.3e}"
        )

    dec_theta_eval = theta_eval
    dec_x_eval = x_eval
    if dec_theta_eval.shape[0] == 0:
        dec_theta_eval = theta_train
        dec_x_eval = x_train
        print(
            "[decoder] theta_eval empty; using theta_train/x_train for decoder eval windows "
            "(train_frac=1 has no held-out split)."
        )

    decoder_fisher, decoder_se, decoder_valid, decoder_diag = fit_decoder_from_shared_data(
        centers=centers,
        theta_train=theta_train,
        x_train=x_train,
        theta_eval=dec_theta_eval,
        x_eval=dec_x_eval,
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
        "gaussian",
        "gaussian_sqrtd",
        "gaussian_randamp",
        "gaussian_randamp_sqrtd",
        "cos_sin_piecewise_noise",
        "linear_piecewise_noise",
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

    suffix = "_non_gauss" if args.dataset_family == "gmm_non_gauss" else "_theta_cov"
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
        f.write(f"score_data_mode: {args.score_data_mode}\n")
        f.write(f"score_fisher_eval_data: {args.score_fisher_eval_data}\n")
        f.write(
            "score_data_counts: "
            f"train={theta_score_train.shape[0]}, eval={theta_score_eval.shape[0]}\n"
        )
        f.write(
            "score_fit_val_counts: "
            f"fit={theta_score_fit.shape[0]}, val={theta_score_val.shape[0]}, "
            f"val_source={args.score_val_source}"
        )
        if args.score_val_source == "train_split":
            f.write(
                f", val_target_frac={SCORE_VAL_FRACTION}, "
                f"val_frac_eff={theta_score_val.shape[0] / max(theta_score_train.shape[0], 1):.6f}\n"
            )
        else:
            f.write("\n")
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
        if args.dataset_family in ("gaussian", "gaussian_sqrtd", "gaussian_randamp", "gaussian_randamp_sqrtd"):
            _a = 0.5 * (float(args.cov_theta_amp1) + float(args.cov_theta_amp2))
            f.write(
                "cov_theta: "
                f"alpha_mean_activity=({args.cov_theta_amp1}+{args.cov_theta_amp2})/2={_a:.6g}, "
                f"amp_rho={args.cov_theta_amp_rho}, "
                f"freq1={args.cov_theta_freq1}, freq2={args.cov_theta_freq2}, freq_rho={args.cov_theta_freq_rho}, "
                f"phase1={args.cov_theta_phase1}, phase2={args.cov_theta_phase2}, phase_rho={args.cov_theta_phase_rho}, "
                f"rho_clip={args.rho_clip}\n"
            )
        elif args.dataset_family == "cos_sin_piecewise_noise":
            f.write(
                "cos_sin_piecewise_noise: "
                f"sigma_piecewise_low={args.sigma_piecewise_low}, "
                f"sigma_piecewise_high={args.sigma_piecewise_high}, "
                f"theta_zero_to_low={args.theta_zero_to_low}\n"
            )
        elif args.dataset_family == "linear_piecewise_noise":
            f.write(
                "linear_piecewise_noise: "
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
