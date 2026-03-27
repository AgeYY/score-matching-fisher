from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from fisher.data import ToyConditionalGMMNonGaussianDataset, ToyConditionalGaussianDataset
from fisher.models import ConditionalScore1D, LocalDecoderLogit


def parse_sigma_alpha_list(items: list[float]) -> np.ndarray:
    arr = np.asarray(items, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("sigma alpha list must have at least 2 values.")
    if np.any(arr <= 0):
        raise ValueError("all sigma alpha values must be positive.")
    return np.unique(arr)[::-1]


def log_p_x_given_theta(x: np.ndarray, theta: np.ndarray, dataset: ToyConditionalGaussianDataset) -> np.ndarray:
    if hasattr(dataset, "log_p_x_given_theta"):
        return dataset.log_p_x_given_theta(x, theta)
    mu = dataset.tuning_curve(theta)
    delta = x - mu
    cov = dataset.covariance(theta)
    inv_cov = np.linalg.inv(cov)
    quad = np.einsum("ni,nij,nj->n", delta, inv_cov, delta)
    _, logdet = np.linalg.slogdet(cov)
    d = x.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def finite_difference_score(
    x: np.ndarray,
    theta: np.ndarray,
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
    delta: float,
) -> np.ndarray:
    theta_plus = theta + delta
    theta_minus = theta - delta
    lp = log_p_x_given_theta(x, theta_plus, dataset)
    lm = log_p_x_given_theta(x, theta_minus, dataset)
    return ((lp - lm) / (2.0 * delta)).reshape(-1)


@dataclass
class BinnedStats:
    centers: np.ndarray
    mean: np.ndarray
    se: np.ndarray
    counts: np.ndarray
    valid: np.ndarray


@dataclass
class BinnedFisher:
    centers: np.ndarray
    fisher_model: np.ndarray
    fisher_fd: np.ndarray
    se_model: np.ndarray
    se_fd: np.ndarray
    counts: np.ndarray
    valid: np.ndarray


def bin_mean_and_se(
    theta: np.ndarray,
    values: np.ndarray,
    theta_low: float,
    theta_high: float,
    n_bins: int,
    min_count: int,
) -> BinnedStats:
    bins = np.linspace(theta_low, theta_high, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(theta.reshape(-1), bins) - 1
    in_range = (idx >= 0) & (idx < n_bins)
    idx = idx[in_range]
    vals = values.reshape(-1)[in_range]

    mean = np.full(n_bins, np.nan, dtype=np.float64)
    se = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        counts[b] = c
        if c < min_count:
            continue
        vv = vals[mask]
        mean[b] = float(np.mean(vv))
        se[b] = float(np.std(vv, ddof=1) / np.sqrt(c)) if c > 1 else np.nan

    valid = np.isfinite(mean)
    return BinnedStats(centers=centers, mean=mean, se=se, counts=counts, valid=valid)


def make_theta_centers(theta_low: float, theta_high: float, n_bins: int) -> np.ndarray:
    bins = np.linspace(theta_low, theta_high, n_bins + 1)
    return 0.5 * (bins[:-1] + bins[1:])


def nw_regress_scalar_1d(
    theta: np.ndarray,
    values: np.ndarray,
    centers: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bandwidth <= 0.0:
        raise ValueError("NW bandwidth must be positive.")
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    if t.size != v.size:
        raise ValueError("theta and values must have the same number of samples.")
    if t.size < 1:
        raise ValueError("theta and values must be non-empty.")

    diff = (t[:, None] - c[None, :]) / bandwidth
    logw = -0.5 * (diff**2)
    # Normalize in log space for stability.
    logw = logw - np.max(logw, axis=0, keepdims=True)
    w = np.exp(logw)
    z = np.sum(w, axis=0, keepdims=True)
    z = np.maximum(z, 1e-30)
    w = w / z

    mean = np.sum(w * v[:, None], axis=0)
    centered = v[:, None] - mean[None, :]
    var = np.sum(w * (centered**2), axis=0)
    neff = 1.0 / np.maximum(np.sum(w**2, axis=0), 1e-30)
    se = np.sqrt(np.maximum(var, 0.0) / np.maximum(neff, 1e-30))
    return mean.astype(np.float64), se.astype(np.float64), neff.astype(np.float64)


def gp_regress_scalar_1d(
    theta: np.ndarray,
    values: np.ndarray,
    centers: np.ndarray,
    max_fit_points: int,
    length_scale: float,
    length_scale_scale: str,
    theta_low: float,
    theta_high: float,
    white_noise: float,
    alpha: float,
    normalize_y: bool,
    optimizer_restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if max_fit_points < 1:
        raise ValueError("gp max_fit_points must be >= 1.")
    if length_scale <= 0.0:
        raise ValueError("gp length_scale must be positive.")
    if length_scale_scale not in {"absolute", "theta_range"}:
        raise ValueError("gp length_scale_scale must be one of {'absolute','theta_range'}.")
    if white_noise <= 0.0:
        raise ValueError("gp white_noise must be positive.")
    if alpha < 0.0:
        raise ValueError("gp alpha must be non-negative.")
    if optimizer_restarts < 0:
        raise ValueError("gp optimizer_restarts must be >= 0.")

    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    if t.size != v.size:
        raise ValueError("theta and values must have the same number of samples.")
    if t.size < 2:
        raise ValueError("need at least 2 samples for GP regression.")

    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    if t.size < 2:
        raise ValueError("need at least 2 finite samples for GP regression.")

    rng = np.random.default_rng(seed)
    if t.size > max_fit_points:
        idx = rng.choice(t.size, size=max_fit_points, replace=False)
        t = t[idx]
        v = v[idx]
    n_fit = int(t.size)

    if length_scale_scale == "theta_range":
        ls = length_scale * float(theta_high - theta_low)
    else:
        ls = length_scale
    ls = max(ls, 1e-6)
    white = max(white_noise, 1e-8)
    ls_lb = max(ls * 1e-3, 1e-6)
    ls_ub = max(ls * 1e3, ls_lb * 10.0)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=ls, length_scale_bounds=(ls_lb, ls_ub))
        + WhiteKernel(noise_level=white, noise_level_bounds=(1e-8, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=max(alpha, 0.0),
        normalize_y=normalize_y,
        n_restarts_optimizer=optimizer_restarts,
        random_state=seed,
    )
    gp.fit(t.reshape(-1, 1), v)
    mean, std = gp.predict(c.reshape(-1, 1), return_std=True)
    return mean.astype(np.float64), std.astype(np.float64), n_fit


def extrapolate_sigma2_to_zero(
    sigma_values: np.ndarray, fisher_per_sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(sigma_values, dtype=np.float64) ** 2
    y = np.asarray(fisher_per_sigma, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != x.size:
        raise ValueError("fisher_per_sigma must have shape (n_sigma, n_bins).")

    n_bins = y.shape[1]
    intercept = np.full(n_bins, np.nan, dtype=np.float64)
    slope = np.full(n_bins, np.nan, dtype=np.float64)
    r2 = np.full(n_bins, np.nan, dtype=np.float64)

    # Fixed-noise mode: if all sigma values are (numerically) identical,
    # return the single-noise Fisher estimate directly.
    if np.allclose(x, x[0], rtol=0.0, atol=1e-15):
        for b in range(n_bins):
            yy = y[:, b]
            mask = np.isfinite(yy)
            if int(mask.sum()) < 1:
                continue
            intercept[b] = float(np.mean(yy[mask]))
        return intercept, slope, r2

    for b in range(n_bins):
        yy = y[:, b]
        mask = np.isfinite(yy)
        if int(mask.sum()) < 2:
            continue
        xx = x[mask]
        yb = yy[mask]
        p = np.polyfit(xx, yb, deg=1)
        slope[b] = float(p[0])
        intercept[b] = float(p[1])
        pred = slope[b] * xx + intercept[b]
        ss_res = float(np.sum((yb - pred) ** 2))
        ss_tot = float(np.sum((yb - np.mean(yb)) ** 2))
        r2[b] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return intercept, slope, r2


def compute_curve_metrics(model: np.ndarray, fd: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    a = model[valid]
    b = fd[valid]
    if a.size == 0:
        return {
            "n_valid_bins": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "relative_rmse": float("nan"),
            "corr": float("nan"),
        }
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    rel = float(rmse / (np.mean(np.abs(b)) + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
    return {
        "n_valid_bins": float(a.size),
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": rel,
        "corr": corr,
    }


@dataclass
class ScoreEvalResult:
    curves: BinnedFisher
    fisher_per_sigma: np.ndarray
    se_per_sigma: np.ndarray
    sigma_values: np.ndarray
    slope: np.ndarray
    r2: np.ndarray
    metrics: dict[str, float]


def evaluate_score_fisher(
    model: ConditionalScore1D,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
    sigma_values: np.ndarray,
    fd_delta: float,
    n_bins: int,
    min_bin_count: int,
    eval_low: float,
    eval_high: float,
    device: torch.device,
    estimator: str = "binned",
    nw_bandwidth: float = 0.1,
    nw_bandwidth_scale: str = "theta_range",
    nw_min_effective_count: float = 80.0,
    gp_max_fit_points: int = 2500,
    gp_length_scale: float = 0.1,
    gp_length_scale_scale: str = "theta_range",
    gp_white_noise: float = 1e-3,
    gp_alpha: float = 1e-6,
    gp_normalize_y: bool = True,
    gp_optimizer_restarts: int = 0,
    gp_seed: int = 7,
) -> ScoreEvalResult:
    if estimator not in {"binned", "nw", "gp"}:
        raise ValueError("estimator must be one of {'binned','nw','gp'}.")
    if nw_bandwidth_scale not in {"absolute", "theta_range"}:
        raise ValueError("nw_bandwidth_scale must be one of {'absolute','theta_range'}.")
    if nw_bandwidth <= 0.0:
        raise ValueError("nw_bandwidth must be positive.")
    if nw_min_effective_count < 0.0:
        raise ValueError("nw_min_effective_count must be non-negative.")

    with torch.no_grad():
        t_eval_t = torch.from_numpy(theta_eval.astype(np.float32)).to(device)
        x_eval_t = torch.from_numpy(x_eval.astype(np.float32)).to(device)
        score_model_by_sigma = []
        for s in sigma_values:
            pred = model.predict_score(t_eval_t, x_eval_t, sigma_eval=float(s)).cpu().numpy().reshape(-1)
            score_model_by_sigma.append(pred)
    score_model_by_sigma_arr = np.stack(score_model_by_sigma, axis=0)
    score_fd = finite_difference_score(x_eval, theta_eval, dataset, delta=fd_delta)

    centers_ref = make_theta_centers(eval_low, eval_high, n_bins)
    theta_flat = theta_eval.reshape(-1)
    in_range = (theta_flat >= eval_low) & (theta_flat <= eval_high)
    theta_in = theta_flat[in_range]
    score_fd_in = score_fd.reshape(-1)[in_range]
    score_model_by_sigma_in = score_model_by_sigma_arr[:, in_range]

    if estimator == "binned":
        fd_stats = bin_mean_and_se(
            theta=theta_in,
            values=score_fd_in**2,
            theta_low=eval_low,
            theta_high=eval_high,
            n_bins=n_bins,
            min_count=min_bin_count,
        )
        counts_ref = fd_stats.counts
    elif estimator == "nw":
        if nw_bandwidth_scale == "theta_range":
            bw = nw_bandwidth * float(eval_high - eval_low)
        else:
            bw = nw_bandwidth
        fd_mean, fd_se, fd_neff = nw_regress_scalar_1d(
            theta=theta_in,
            values=score_fd_in**2,
            centers=centers_ref,
            bandwidth=bw,
        )
        if nw_min_effective_count == 0.0:
            fd_valid = np.isfinite(fd_mean)
        else:
            fd_valid = np.isfinite(fd_mean) & np.isfinite(fd_neff) & (fd_neff >= nw_min_effective_count)
        fd_stats = BinnedStats(
            centers=centers_ref,
            mean=fd_mean,
            se=fd_se,
            counts=np.rint(fd_neff).astype(np.int64),
            valid=fd_valid,
        )
        counts_ref = fd_neff
    else:
        fd_mean, fd_std, fd_nfit = gp_regress_scalar_1d(
            theta=theta_in,
            values=score_fd_in**2,
            centers=centers_ref,
            max_fit_points=gp_max_fit_points,
            length_scale=gp_length_scale,
            length_scale_scale=gp_length_scale_scale,
            theta_low=eval_low,
            theta_high=eval_high,
            white_noise=gp_white_noise,
            alpha=gp_alpha,
            normalize_y=gp_normalize_y,
            optimizer_restarts=gp_optimizer_restarts,
            seed=gp_seed,
        )
        fd_valid = np.isfinite(fd_mean) & np.isfinite(fd_std)
        fd_stats = BinnedStats(
            centers=centers_ref,
            mean=fd_mean,
            se=fd_std,
            counts=np.full(centers_ref.size, fill_value=fd_nfit, dtype=np.int64),
            valid=fd_valid,
        )
        counts_ref = np.full(centers_ref.size, fill_value=fd_nfit, dtype=np.float64)

    fisher_per_sigma = []
    se_per_sigma = []
    for k in range(score_model_by_sigma_arr.shape[0]):
        if estimator == "binned":
            st = bin_mean_and_se(
                theta=theta_in,
                values=score_model_by_sigma_in[k] ** 2,
                theta_low=eval_low,
                theta_high=eval_high,
                n_bins=n_bins,
                min_count=min_bin_count,
            )
            fisher_per_sigma.append(st.mean)
            se_per_sigma.append(st.se)
        elif estimator == "nw":
            mean_k, se_k, _ = nw_regress_scalar_1d(
                theta=theta_in,
                values=score_model_by_sigma_in[k] ** 2,
                centers=centers_ref,
                bandwidth=bw,
            )
            fisher_per_sigma.append(mean_k)
            se_per_sigma.append(se_k)
        else:
            mean_k, std_k, nfit_k = gp_regress_scalar_1d(
                theta=theta_in,
                values=score_model_by_sigma_in[k] ** 2,
                centers=centers_ref,
                max_fit_points=gp_max_fit_points,
                length_scale=gp_length_scale,
                length_scale_scale=gp_length_scale_scale,
                theta_low=eval_low,
                theta_high=eval_high,
                white_noise=gp_white_noise,
                alpha=gp_alpha,
                normalize_y=gp_normalize_y,
                optimizer_restarts=gp_optimizer_restarts,
                seed=gp_seed + 1000 + k,
            )
            fisher_per_sigma.append(mean_k)
            se_per_sigma.append(std_k)
            counts_ref = np.full(centers_ref.size, fill_value=nfit_k, dtype=np.float64)
    fisher_per_sigma_arr = np.stack(fisher_per_sigma, axis=0)
    se_per_sigma_arr = np.stack(se_per_sigma, axis=0)

    fisher0, slope, r2 = extrapolate_sigma2_to_zero(sigma_values=sigma_values, fisher_per_sigma=fisher_per_sigma_arr)
    se_model0 = se_per_sigma_arr[-1]
    valid = np.isfinite(fisher0) & fd_stats.valid

    curves = BinnedFisher(
        centers=centers_ref,
        fisher_model=fisher0,
        fisher_fd=fd_stats.mean,
        se_model=se_model0,
        se_fd=fd_stats.se,
        counts=np.asarray(counts_ref),
        valid=valid,
    )
    metrics = compute_curve_metrics(curves.fisher_model, curves.fisher_fd, curves.valid)
    return ScoreEvalResult(
        curves=curves,
        fisher_per_sigma=fisher_per_sigma_arr,
        se_per_sigma=se_per_sigma_arr,
        sigma_values=sigma_values,
        slope=slope,
        r2=r2,
        metrics=metrics,
    )


@dataclass
class DecoderLocalEval:
    fisher_decoder: float
    se_decoder: float
    fisher_fd: float
    logits_pos: np.ndarray
    logits_neg: np.ndarray


def evaluate_local_decoder(
    model: LocalDecoderLogit,
    x_eval_pos: np.ndarray,
    x_eval_neg: np.ndarray,
    epsilon: float,
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
    theta0: float,
    n_eval_local: int,
    fd_delta: float,
    device: torch.device,
) -> DecoderLocalEval:
    model.eval()
    x_eval_mix = np.concatenate([x_eval_pos, x_eval_neg], axis=0)
    with torch.no_grad():
        pos_t = torch.from_numpy(x_eval_pos.astype(np.float32)).to(device)
        neg_t = torch.from_numpy(x_eval_neg.astype(np.float32)).to(device)
        mix_t = torch.from_numpy(x_eval_mix.astype(np.float32)).to(device)
        logits_pos = model(pos_t).cpu().numpy().reshape(-1)
        logits_neg = model(neg_t).cpu().numpy().reshape(-1)
        logits_mix = model(mix_t).cpu().numpy().reshape(-1)
    fisher_decoder_samples = (logits_mix**2) / (epsilon**2)
    fisher_decoder = float(np.mean(fisher_decoder_samples))
    se_decoder = float(np.std(fisher_decoder_samples, ddof=1) / np.sqrt(fisher_decoder_samples.size))

    t_fd = np.full((2 * n_eval_local, 1), theta0, dtype=np.float64)
    x_fd = dataset.sample_x(t_fd)
    score_fd = finite_difference_score(x_fd, t_fd, dataset, delta=fd_delta)
    fisher_fd = float(np.mean(score_fd**2))

    return DecoderLocalEval(
        fisher_decoder=fisher_decoder,
        se_decoder=se_decoder,
        fisher_fd=fisher_fd,
        logits_pos=logits_pos,
        logits_neg=logits_neg,
    )
