from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

def log_p_gaussian_mvnormal_from_cov(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Multivariate Gaussian log-density for batched ``x`` with per-row mean and covariance.

    Shapes: ``x`` and ``mu`` are ``(n, d)``; ``cov`` is ``(n, d, d)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mu = np.asarray(mu, dtype=np.float64).reshape(x.shape[0], x.shape[1])
    cov = np.asarray(cov, dtype=np.float64)
    delta = x - mu
    inv_cov = np.linalg.inv(cov)
    quad = np.einsum("ni,nij,nj->n", delta, inv_cov, delta)
    _, logdet = np.linalg.slogdet(cov)
    d = float(x.shape[1])
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


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
from fisher.models import (
    ConditionalScore1D,
    ConditionalScore1DFiLMPerLayer,
    LocalDecoderLogit,
    PriorScore1D,
    PriorScore1DFiLMPerLayer,
)


def parse_sigma_alpha_list(items: list[float]) -> np.ndarray:
    arr = np.asarray(items, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("sigma alpha list must have at least 2 values.")
    if np.any(arr <= 0):
        raise ValueError("all sigma alpha values must be positive.")
    return np.unique(arr)[::-1]


def log_p_x_given_theta(
    x: np.ndarray,
    theta: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGaussianSqrtdDataset
    | ToyConditionalGaussianCosineRandampSqrtdDataset
    | ToyConditionalGaussianRandampDataset
    | ToyConditionalGaussianRandampSqrtdDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
) -> np.ndarray:
    if hasattr(dataset, "log_p_x_given_theta"):
        return dataset.log_p_x_given_theta(x, theta)
    mu = dataset.tuning_curve(theta)
    cov = dataset.covariance(theta)
    return log_p_gaussian_mvnormal_from_cov(x, mu, cov)


def finite_difference_score(
    x: np.ndarray,
    theta: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset,
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
    model: ConditionalScore1D | ConditionalScore1DFiLMPerLayer,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset,
    sigma_values: np.ndarray,
    fd_delta: float,
    n_bins: int,
    min_bin_count: int,
    eval_low: float,
    eval_high: float,
    device: torch.device,
) -> ScoreEvalResult:
    with torch.no_grad():
        t_eval_t = torch.from_numpy(theta_eval.astype(np.float32)).to(device)
        x_eval_t = torch.from_numpy(x_eval.astype(np.float32)).to(device)
        score_model_by_sigma = []
        for s in sigma_values:
            pred = model.predict_score(t_eval_t, x_eval_t, sigma_eval=float(s)).cpu().numpy().reshape(-1)
            score_model_by_sigma.append(pred)
    score_model_by_sigma_arr = np.stack(score_model_by_sigma, axis=0)
    score_fd = finite_difference_score(x_eval, theta_eval, dataset, delta=fd_delta)

    fd_stats = bin_mean_and_se(
        theta=theta_eval,
        values=score_fd**2,
        theta_low=eval_low,
        theta_high=eval_high,
        n_bins=n_bins,
        min_count=min_bin_count,
    )

    fisher_per_sigma = []
    se_per_sigma = []
    counts_ref = None
    centers_ref = None
    for k in range(score_model_by_sigma_arr.shape[0]):
        st = bin_mean_and_se(
            theta=theta_eval,
            values=score_model_by_sigma_arr[k] ** 2,
            theta_low=eval_low,
            theta_high=eval_high,
            n_bins=n_bins,
            min_count=min_bin_count,
        )
        fisher_per_sigma.append(st.mean)
        se_per_sigma.append(st.se)
        if counts_ref is None:
            counts_ref = st.counts
            centers_ref = st.centers
    fisher_per_sigma_arr = np.stack(fisher_per_sigma, axis=0)
    se_per_sigma_arr = np.stack(se_per_sigma, axis=0)

    k_min = int(np.argmin(np.asarray(sigma_values, dtype=np.float64)))
    fisher0 = fisher_per_sigma_arr[k_min]
    se_model0 = se_per_sigma_arr[k_min]
    slope = np.full(fisher0.shape, np.nan, dtype=np.float64)
    r2 = np.full(fisher0.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(fisher0) & fd_stats.valid

    curves = BinnedFisher(
        centers=centers_ref,
        fisher_model=fisher0,
        fisher_fd=fd_stats.mean,
        se_model=se_model0,
        se_fd=fd_stats.se,
        counts=counts_ref,
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
class ScoreEvalWithPriorResult:
    """Posterior-only and posterior-minus-prior Fisher curves vs finite-difference likelihood score."""

    curves_posterior: BinnedFisher
    curves_combined: BinnedFisher
    fisher_per_sigma_posterior: np.ndarray
    fisher_per_sigma_combined: np.ndarray
    se_per_sigma_posterior: np.ndarray
    se_per_sigma_combined: np.ndarray
    sigma_values: np.ndarray
    slope: np.ndarray
    r2: np.ndarray
    metrics_posterior: dict[str, float]
    metrics_combined: dict[str, float]


def evaluate_score_fisher_with_prior(
    model_post: ConditionalScore1D | ConditionalScore1DFiLMPerLayer,
    model_prior: PriorScore1D | PriorScore1DFiLMPerLayer,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset,
    sigma_values: np.ndarray,
    fd_delta: float,
    n_bins: int,
    min_bin_count: int,
    eval_low: float,
    eval_high: float,
    device: torch.device,
) -> ScoreEvalWithPriorResult:
    with torch.no_grad():
        t_eval_t = torch.from_numpy(theta_eval.astype(np.float32)).to(device)
        x_eval_t = torch.from_numpy(x_eval.astype(np.float32)).to(device)
        score_post_by_sigma = []
        score_prior_by_sigma = []
        for s in sigma_values:
            sp = model_post.predict_score(t_eval_t, x_eval_t, sigma_eval=float(s)).cpu().numpy().reshape(-1)
            spr = model_prior.predict_score(t_eval_t, sigma_eval=float(s)).cpu().numpy().reshape(-1)
            score_post_by_sigma.append(sp)
            score_prior_by_sigma.append(spr)
    score_post_arr = np.stack(score_post_by_sigma, axis=0)
    score_prior_arr = np.stack(score_prior_by_sigma, axis=0)
    score_total_arr = score_post_arr - score_prior_arr

    score_fd = finite_difference_score(x_eval, theta_eval, dataset, delta=fd_delta)

    fd_stats = bin_mean_and_se(
        theta=theta_eval,
        values=score_fd**2,
        theta_low=eval_low,
        theta_high=eval_high,
        n_bins=n_bins,
        min_count=min_bin_count,
    )

    fisher_post_per_sigma: list[np.ndarray] = []
    fisher_combined_per_sigma: list[np.ndarray] = []
    se_post_per_sigma: list[np.ndarray] = []
    se_combined_per_sigma: list[np.ndarray] = []
    counts_ref = None
    centers_ref = None
    for k in range(score_post_arr.shape[0]):
        st_post = bin_mean_and_se(
            theta=theta_eval,
            values=score_post_arr[k] ** 2,
            theta_low=eval_low,
            theta_high=eval_high,
            n_bins=n_bins,
            min_count=min_bin_count,
        )
        st_comb = bin_mean_and_se(
            theta=theta_eval,
            values=score_total_arr[k] ** 2,
            theta_low=eval_low,
            theta_high=eval_high,
            n_bins=n_bins,
            min_count=min_bin_count,
        )
        fisher_post_per_sigma.append(st_post.mean)
        fisher_combined_per_sigma.append(st_comb.mean)
        se_post_per_sigma.append(st_post.se)
        se_combined_per_sigma.append(st_comb.se)
        if counts_ref is None:
            counts_ref = st_post.counts
            centers_ref = st_post.centers

    fisher_per_sigma_posterior = np.stack(fisher_post_per_sigma, axis=0)
    fisher_per_sigma_combined = np.stack(fisher_combined_per_sigma, axis=0)
    se_per_sigma_posterior = np.stack(se_post_per_sigma, axis=0)
    se_per_sigma_combined = np.stack(se_combined_per_sigma, axis=0)

    k_min = int(np.argmin(np.asarray(sigma_values, dtype=np.float64)))
    fisher_post0 = fisher_per_sigma_posterior[k_min]
    fisher_comb0 = fisher_per_sigma_combined[k_min]
    se_post0 = se_per_sigma_posterior[k_min]
    se_comb0 = se_per_sigma_combined[k_min]
    slope = np.full(fisher_post0.shape, np.nan, dtype=np.float64)
    r2 = np.full(fisher_post0.shape, np.nan, dtype=np.float64)
    valid_post = np.isfinite(fisher_post0) & fd_stats.valid
    valid_comb = np.isfinite(fisher_comb0) & fd_stats.valid

    curves_posterior = BinnedFisher(
        centers=centers_ref,
        fisher_model=fisher_post0,
        fisher_fd=fd_stats.mean,
        se_model=se_post0,
        se_fd=fd_stats.se,
        counts=counts_ref,
        valid=valid_post,
    )
    curves_combined = BinnedFisher(
        centers=centers_ref,
        fisher_model=fisher_comb0,
        fisher_fd=fd_stats.mean,
        se_model=se_comb0,
        se_fd=fd_stats.se,
        counts=counts_ref,
        valid=valid_comb,
    )
    metrics_posterior = compute_curve_metrics(curves_posterior.fisher_model, curves_posterior.fisher_fd, curves_posterior.valid)
    metrics_combined = compute_curve_metrics(curves_combined.fisher_model, curves_combined.fisher_fd, curves_combined.valid)
    return ScoreEvalWithPriorResult(
        curves_posterior=curves_posterior,
        curves_combined=curves_combined,
        fisher_per_sigma_posterior=fisher_per_sigma_posterior,
        fisher_per_sigma_combined=fisher_per_sigma_combined,
        se_per_sigma_posterior=se_per_sigma_posterior,
        se_per_sigma_combined=se_per_sigma_combined,
        sigma_values=sigma_values,
        slope=slope,
        r2=r2,
        metrics_posterior=metrics_posterior,
        metrics_combined=metrics_combined,
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
    dataset: ToyConditionalGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset,
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
