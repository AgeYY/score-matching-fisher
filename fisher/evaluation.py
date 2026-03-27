from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

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

    fisher0, slope, r2 = extrapolate_sigma2_to_zero(sigma_values=sigma_values, fisher_per_sigma=fisher_per_sigma_arr)
    se_model0 = se_per_sigma_arr[-1]
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
