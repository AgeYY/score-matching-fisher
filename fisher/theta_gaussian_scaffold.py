"""Gaussian-posterior scaffold utilities for scalar-theta conditional flows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


def _as_theta_col(theta: np.ndarray) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim == 1:
        th = th.reshape(-1, 1)
    if th.ndim != 2 or int(th.shape[1]) != 1:
        raise ValueError("Gaussian posterior scaffold v1 requires scalar theta with shape (N,) or (N,1).")
    return th


def _logsumexp_np(a: np.ndarray) -> float:
    z = np.asarray(a, dtype=np.float64)
    m = float(np.max(z))
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(z - m))))


def _fit_binned_likelihood(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    n_bins: int,
    variance_floor: float,
    theta_low: float | None,
    theta_high: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    th = _as_theta_col(theta_train).reshape(-1)
    x = np.asarray(x_train, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x_train must be 2D.")
    if int(x.shape[0]) != int(th.size):
        raise ValueError("theta_train and x_train must have the same number of rows.")
    nb = int(n_bins)
    vf = float(variance_floor)
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    if not np.isfinite(vf) or vf <= 0.0:
        raise ValueError("variance_floor must be a finite positive number.")
    lo = float(np.min(th) if theta_low is None else theta_low)
    hi = float(np.max(th) if theta_high is None else theta_high)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError("theta_low/theta_high must be finite with theta_high > theta_low.")

    edges = np.linspace(lo, hi, nb + 1, dtype=np.float64)
    bin_idx = np.searchsorted(edges, th, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, nb - 1)
    x_dim = int(x.shape[1])
    means = np.zeros((nb, x_dim), dtype=np.float64)
    counts = np.bincount(bin_idx, minlength=nb)
    global_mean = np.mean(x, axis=0)
    for b in range(nb):
        idx = np.flatnonzero(bin_idx == b)
        if idx.size > 0:
            means[b] = np.mean(x[idx], axis=0)
        else:
            means[b] = global_mean
    nonempty = np.flatnonzero(counts > 0)
    if nonempty.size > 0:
        for b in np.flatnonzero(counts == 0):
            nearest = int(nonempty[np.argmin(np.abs(nonempty - int(b)))])
            means[int(b)] = means[nearest]

    residuals = x - means[bin_idx]
    global_residual_var = np.maximum(np.mean(residuals**2, axis=0), vf)
    vars_ = np.broadcast_to(global_residual_var.reshape(1, x_dim), (nb, x_dim)).copy()
    return edges, means, vars_, counts


@dataclass
class ThetaDiscreteScaffold:
    """Binned diagonal-Gaussian posterior with discrete theta-bin source.

    The approximate posterior is represented directly as bin masses
    ``q0(bin | x) ∝ p_g(x | bin)``. Source samples are bin centers, and source
    log likelihoods use the corresponding piecewise-constant density on bins.
    """

    bin_edges: np.ndarray
    bin_means: np.ndarray
    bin_vars: np.ndarray
    variance_floor: float
    source_eps: float = 0.0

    @classmethod
    def fit(
        cls,
        *,
        theta_train: np.ndarray,
        x_train: np.ndarray,
        n_bins: int = 10,
        variance_floor: float = 1e-6,
        source_eps: float = 0.0,
        theta_low: float | None = None,
        theta_high: float | None = None,
    ) -> "ThetaDiscreteScaffold":
        eps = float(source_eps)
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("source_eps must be finite and >= 0.")
        edges, means, vars_, _ = _fit_binned_likelihood(
            theta_train=theta_train,
            x_train=x_train,
            n_bins=n_bins,
            variance_floor=variance_floor,
            theta_low=theta_low,
            theta_high=theta_high,
        )
        return cls(
            bin_edges=edges,
            bin_means=means,
            bin_vars=vars_,
            variance_floor=float(variance_floor),
            source_eps=eps,
        )

    @property
    def theta_low(self) -> float:
        return float(self.bin_edges[0])

    @property
    def theta_high(self) -> float:
        return float(self.bin_edges[-1])

    @property
    def bin_centers(self) -> np.ndarray:
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def bin_widths(self) -> np.ndarray:
        return np.diff(np.asarray(self.bin_edges, dtype=np.float64))

    def _theta_bin_indices(self, theta: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.bin_edges, np.asarray(theta, dtype=np.float64), side="right") - 1
        return np.clip(idx, 0, int(self.bin_means.shape[0]) - 1)

    def discretize_theta_np(self, theta: np.ndarray) -> np.ndarray:
        th = _as_theta_col(theta).reshape(-1)
        idx = self._theta_bin_indices(th)
        return self.bin_centers[idx].reshape(-1, 1)

    def log_likelihood_bins(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        if x2.ndim != 2:
            raise ValueError("x must be 1D or 2D.")
        mean_g = np.asarray(self.bin_means, dtype=np.float64)
        var_g = np.maximum(np.asarray(self.bin_vars, dtype=np.float64), float(self.variance_floor))
        diff = x2[:, None, :] - mean_g[None, :, :]
        log_det = np.sum(np.log(2.0 * math.pi * var_g), axis=1)
        maha = np.sum(diff * diff / var_g[None, :, :], axis=2)
        return -0.5 * (maha + log_det.reshape(1, -1))

    def q0_bins(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logw = self.log_likelihood_bins(x)
        out = np.zeros_like(logw, dtype=np.float64)
        for i in range(logw.shape[0]):
            lse = _logsumexp_np(logw[i])
            if np.isfinite(lse):
                out[i] = np.exp(logw[i] - lse)
            else:
                out[i] = 1.0 / float(logw.shape[1])
        return out, logw

    def sample_matched_np(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        th = _as_theta_col(theta).reshape(-1)
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim != 2 or int(x2.shape[0]) != int(th.size):
            raise ValueError("x must be 2D with the same number of rows as theta.")
        gen = np.random.default_rng() if rng is None else rng
        q, _ = self.q0_bins(x2)
        samples = np.zeros_like(th, dtype=np.float64)
        branch_ids = np.zeros(th.shape[0], dtype=np.int64)
        centers = self.bin_centers
        for i in range(th.size):
            probs = np.asarray(q[i], dtype=np.float64)
            probs = probs / max(float(np.sum(probs)), 1e-300)
            b = int(gen.choice(probs.size, p=probs))
            branch_ids[i] = b
            samples[i] = float(centers[b])
        return samples.reshape(-1, 1), branch_ids

    def sample_matched_torch(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        device = theta.device
        dtype = theta.dtype
        samples, _ = self.sample_matched_np(
            theta.detach().cpu().numpy(),
            x.detach().cpu().numpy(),
        )
        return torch.as_tensor(samples, device=device, dtype=dtype).reshape_as(theta)

    def log_prob_np(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        th = _as_theta_col(theta).reshape(-1)
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        if x2.ndim != 2 or int(x2.shape[0]) != int(th.size):
            raise ValueError("x must be 2D with the same number of rows as theta.")
        q, _ = self.q0_bins(x2)
        idx = self._theta_bin_indices(th)
        widths = np.maximum(self.bin_widths[idx], 1e-300)
        mass = np.maximum(q[np.arange(th.size), idx], 1e-300)
        return np.log(mass) - np.log(widths)

    def log_prob_torch(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Torch version of :meth:`log_prob_np` for scaffold-NLL theta-flow training."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if theta.ndim != 2 or int(theta.shape[1]) != 1:
            raise ValueError("ThetaDiscreteScaffold.log_prob_torch requires scalar theta with shape (N,) or (N,1).")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or int(x.shape[0]) != int(theta.shape[0]):
            raise ValueError("x must be 2D with the same number of rows as theta.")

        device = theta.device
        dtype = theta.dtype
        means = torch.as_tensor(self.bin_means, device=device, dtype=dtype)
        vars_ = torch.clamp(
            torch.as_tensor(self.bin_vars, device=device, dtype=dtype),
            min=float(self.variance_floor),
        )
        edges = torch.as_tensor(self.bin_edges, device=device, dtype=dtype)
        widths_all = torch.clamp(edges[1:] - edges[:-1], min=torch.finfo(dtype).tiny)

        diff = x[:, None, :] - means[None, :, :]
        log_det = torch.sum(torch.log(2.0 * math.pi * vars_), dim=1)
        maha = torch.sum(diff * diff / vars_[None, :, :], dim=2)
        logw = -0.5 * (maha + log_det.reshape(1, -1))
        logq = logw - torch.logsumexp(logw, dim=1, keepdim=True)

        idx = torch.bucketize(theta.reshape(-1), edges, right=True) - 1
        idx = torch.clamp(idx, 0, int(means.shape[0]) - 1)
        rows = torch.arange(theta.shape[0], device=device)
        return logq[rows, idx] - torch.log(widths_all[idx])

    def to_npz_payload(self) -> dict[str, Any]:
        return {
            "bin_edges": np.asarray(self.bin_edges, dtype=np.float64),
            "bin_centers": np.asarray(self.bin_centers, dtype=np.float64),
            "bin_widths": np.asarray(self.bin_widths, dtype=np.float64),
            "bin_means": np.asarray(self.bin_means, dtype=np.float64),
            "bin_vars": np.asarray(self.bin_vars, dtype=np.float64),
            "variance_floor": np.float64(self.variance_floor),
            "source_eps": np.float64(self.source_eps),
            "scaffold_kind": np.asarray(["discrete_q0"], dtype=object),
        }

    @classmethod
    def from_npz(cls, path: str) -> "ThetaDiscreteScaffold":
        z = np.load(path, allow_pickle=True)
        return cls(
            bin_edges=np.asarray(z["bin_edges"], dtype=np.float64),
            bin_means=np.asarray(z["bin_means"], dtype=np.float64),
            bin_vars=np.asarray(z["bin_vars"], dtype=np.float64),
            variance_floor=float(np.asarray(z["variance_floor"]).reshape(-1)[0]),
            source_eps=float(np.asarray(z["source_eps"]).reshape(-1)[0]) if "source_eps" in z.files else 0.0,
        )


@dataclass
class ThetaGaussianScaffold:
    """Binned diagonal Gaussian likelihood plus k-Gaussian posterior sampler.

    The approximate posterior first lives on theta bins:
    q0(bin | x) proportional to p_g(x | bin), with a uniform theta-bin prior.
    Each bin has its own x-mean, and all bins share one diagonal x-variance
    estimated from the pooled within-bin residuals.
    For each x, a 1D Gaussian mixture is fit to those bin masses and used as the
    theta-flow source/base distribution.
    """

    bin_edges: np.ndarray
    bin_means: np.ndarray
    bin_vars: np.ndarray
    n_components: int
    em_steps: int
    variance_floor: float
    min_branch_mass: float
    source_eps: float

    @classmethod
    def fit(
        cls,
        *,
        theta_train: np.ndarray,
        x_train: np.ndarray,
        n_bins: int = 10,
        grid_size: int = 512,
        n_components: int = 3,
        em_steps: int = 20,
        variance_floor: float = 1e-6,
        min_branch_mass: float = 1e-4,
        source_eps: float = 1e-6,
        theta_low: float | None = None,
        theta_high: float | None = None,
    ) -> "ThetaGaussianScaffold":
        nb = int(n_bins)
        gs = int(grid_size)
        nc = int(n_components)
        es = int(em_steps)
        vf = float(variance_floor)
        mb = float(min_branch_mass)
        eps = float(source_eps)
        if nb < 1:
            raise ValueError("n_bins must be >= 1.")
        if gs < 8:
            raise ValueError("grid_size must be >= 8.")
        if nc < 1:
            raise ValueError("n_components must be >= 1.")
        if nc > nb:
            raise ValueError("n_components must be <= n_bins.")
        if es < 1:
            raise ValueError("em_steps must be >= 1.")
        if not np.isfinite(vf) or vf <= 0.0:
            raise ValueError("variance_floor must be a finite positive number.")
        if not np.isfinite(mb) or mb < 0.0:
            raise ValueError("min_branch_mass must be finite and >= 0.")
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("source_eps must be finite and >= 0.")
        edges, means, vars_, _ = _fit_binned_likelihood(
            theta_train=theta_train,
            x_train=x_train,
            n_bins=nb,
            variance_floor=vf,
            theta_low=theta_low,
            theta_high=theta_high,
        )

        return cls(
            bin_edges=edges,
            bin_means=means,
            bin_vars=vars_,
            n_components=nc,
            em_steps=es,
            variance_floor=vf,
            min_branch_mass=mb,
            source_eps=eps,
        )

    @property
    def theta_low(self) -> float:
        return float(self.bin_edges[0])

    @property
    def theta_high(self) -> float:
        return float(self.bin_edges[-1])

    @property
    def bin_centers(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.bin_edges[:-1], dtype=np.float64) + np.asarray(self.bin_edges[1:], dtype=np.float64))

    @property
    def theta_variance_floor(self) -> float:
        widths = np.diff(np.asarray(self.bin_edges, dtype=np.float64))
        if widths.size < 1:
            return max(float(self.source_eps) ** 2, 1e-12)
        # Uniform-bin variance is a natural lower scale for one bin's theta mass.
        return max(float(np.median(widths) ** 2 / 12.0), float(self.source_eps) ** 2, 1e-12)

    def _theta_bin_indices(self, theta: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.bin_edges, np.asarray(theta, dtype=np.float64), side="right") - 1
        return np.clip(idx, 0, int(self.bin_means.shape[0]) - 1)

    def log_likelihood_bins(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        if x2.ndim != 2:
            raise ValueError("x must be 1D or 2D.")
        mean_g = np.asarray(self.bin_means, dtype=np.float64)
        var_g = np.maximum(np.asarray(self.bin_vars, dtype=np.float64), float(self.variance_floor))
        diff = x2[:, None, :] - mean_g[None, :, :]
        log_det = np.sum(np.log(2.0 * math.pi * var_g), axis=1)
        maha = np.sum(diff * diff / var_g[None, :, :], axis=2)
        return -0.5 * (maha + log_det.reshape(1, -1))

    def q0_bins(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logw = self.log_likelihood_bins(x)
        out = np.zeros_like(logw, dtype=np.float64)
        for i in range(logw.shape[0]):
            lse = _logsumexp_np(logw[i])
            if np.isfinite(lse):
                out[i] = np.exp(logw[i] - lse)
            else:
                out[i] = 1.0 / float(logw.shape[1])
        return out, logw

    # Backward-compatible alias for older tests/callers; now this is 10-bin posterior mass.
    def q0_grid(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.q0_bins(x)

    @staticmethod
    def _normal_logpdf(theta: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(2.0 * math.pi * vars_) + (theta[..., None] - means) ** 2 / vars_)

    @staticmethod
    def _std_normal_cdf(z: np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=np.float64)
        return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))

    def _component_log_trunc_norm(self, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(np.asarray(vars_, dtype=np.float64), 1e-300))
        lo = (float(self.theta_low + self.source_eps) - np.asarray(means, dtype=np.float64)) / std
        hi = (float(self.theta_high - self.source_eps) - np.asarray(means, dtype=np.float64)) / std
        z = np.maximum(self._std_normal_cdf(hi) - self._std_normal_cdf(lo), 1e-300)
        return np.log(z)

    def _initial_component_means(self, weights: np.ndarray) -> np.ndarray:
        centers = self.bin_centers
        k = int(self.n_components)
        cdf = np.cumsum(np.asarray(weights, dtype=np.float64))
        total = float(cdf[-1]) if cdf.size else 0.0
        if not np.isfinite(total) or total <= 0.0:
            return np.linspace(float(centers[0]), float(centers[-1]), k, dtype=np.float64)
        cdf = cdf / total
        qs = (np.arange(k, dtype=np.float64) + 0.5) / float(k)
        return np.interp(qs, cdf, centers)

    def _fit_mixture_one(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = self.bin_centers
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        k = int(self.n_components)
        floor = float(self.theta_variance_floor)
        if w.size != centers.size:
            raise ValueError("weights must have one value per theta bin.")
        if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0.0:
            w = np.full_like(centers, 1.0 / float(centers.size), dtype=np.float64)
        else:
            w = np.maximum(w, 0.0)
            w = w / max(float(np.sum(w)), 1e-300)
        means = self._initial_component_means(w)
        global_var = float(np.sum(w * (centers - float(np.sum(w * centers))) ** 2))
        vars_ = np.full(k, max(global_var, floor), dtype=np.float64)
        pis = np.full(k, 1.0 / float(k), dtype=np.float64)
        for _ in range(int(self.em_steps)):
            log_comp = np.log(np.maximum(pis, 1e-300))[None, :] + self._normal_logpdf(centers, means, vars_)
            row_max = np.max(log_comp, axis=1, keepdims=True)
            resp = np.exp(log_comp - row_max)
            resp = resp / np.maximum(np.sum(resp, axis=1, keepdims=True), 1e-300)
            weighted_resp = w[:, None] * resp
            eff = np.sum(weighted_resp, axis=0)
            dead = eff <= 1e-12
            eff_safe = np.maximum(eff, 1e-12)
            new_means = np.sum(weighted_resp * centers[:, None], axis=0) / eff_safe
            new_vars = np.sum(weighted_resp * (centers[:, None] - new_means[None, :]) ** 2, axis=0) / eff_safe
            if np.any(dead):
                new_means[dead] = self._initial_component_means(w)[dead]
                new_vars[dead] = max(global_var, floor)
                eff[dead] = 1e-12
            means = np.clip(new_means, self.theta_low, self.theta_high)
            vars_ = np.maximum(new_vars, floor)
            pis = eff / max(float(np.sum(eff)), 1e-300)
        order = np.argsort(means)
        return pis[order], means[order], vars_[order]

    def mixture_params_np(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, _ = self.q0_bins(x)
        pis = np.zeros((q.shape[0], int(self.n_components)), dtype=np.float64)
        means = np.zeros_like(pis)
        vars_ = np.zeros_like(pis)
        for i in range(q.shape[0]):
            pis[i], means[i], vars_[i] = self._fit_mixture_one(q[i])
        return pis, means, vars_

    def branch_for_theta(self, theta_value: float, pis: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> int:
        th = np.asarray([float(theta_value)], dtype=np.float64)
        log_resp = np.log(np.maximum(np.asarray(pis, dtype=np.float64), 1e-300)) + self._normal_logpdf(
            th,
            np.asarray(means, dtype=np.float64),
            np.asarray(vars_, dtype=np.float64),
        )[0] - self._component_log_trunc_norm(np.asarray(means, dtype=np.float64), np.asarray(vars_, dtype=np.float64))
        return int(np.argmax(log_resp))

    def sample_matched_np(self, theta: np.ndarray, x: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        th = _as_theta_col(theta).reshape(-1)
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim != 2 or int(x2.shape[0]) != int(th.size):
            raise ValueError("x must be 2D with the same number of rows as theta.")
        gen = np.random.default_rng() if rng is None else rng
        pis, means, vars_ = self.mixture_params_np(x2)
        samples = np.zeros_like(th, dtype=np.float64)
        branch_ids = np.zeros(th.shape[0], dtype=np.int64)
        for i in range(th.size):
            k = self.branch_for_theta(float(th[i]), pis[i], means[i], vars_[i])
            branch_ids[i] = int(k)
            lo = float(self.theta_low + self.source_eps)
            hi = float(self.theta_high - self.source_eps)
            val = float("nan")
            for _ in range(64):
                draw = float(gen.normal(loc=float(means[i, k]), scale=float(np.sqrt(vars_[i, k]))))
                if lo <= draw <= hi:
                    val = draw
                    break
            if not np.isfinite(val):
                val = float(np.clip(float(means[i, k]), lo, hi))
            samples[i] = val
        return samples.reshape(-1, 1), branch_ids

    def sample_matched_torch(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        device = theta.device
        dtype = theta.dtype
        samples, _ = self.sample_matched_np(
            theta.detach().cpu().numpy(),
            x.detach().cpu().numpy(),
        )
        return torch.as_tensor(samples, device=device, dtype=dtype).reshape_as(theta)

    def log_prob_np(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        th = _as_theta_col(theta).reshape(-1)
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        if x2.ndim != 2 or int(x2.shape[0]) != int(th.size):
            raise ValueError("x must be 2D with the same number of rows as theta.")
        unique_x, inv = np.unique(x2, axis=0, return_inverse=True)
        pis_u, means_u, vars_u = self.mixture_params_np(unique_x)
        out = np.full(th.size, -np.inf, dtype=np.float64)
        for i in range(th.size):
            u = int(inv[i])
            log_comp = np.log(np.maximum(pis_u[u], 1e-300)) + self._normal_logpdf(
                np.asarray([float(th[i])], dtype=np.float64),
                means_u[u],
                vars_u[u],
            )[0] - self._component_log_trunc_norm(means_u[u], vars_u[u])
            out[i] = _logsumexp_np(log_comp)
        return out

    def to_npz_payload(self) -> dict[str, Any]:
        return {
            "bin_edges": np.asarray(self.bin_edges, dtype=np.float64),
            "bin_centers": np.asarray(self.bin_centers, dtype=np.float64),
            "bin_means": np.asarray(self.bin_means, dtype=np.float64),
            "bin_vars": np.asarray(self.bin_vars, dtype=np.float64),
            "n_components": np.int64(self.n_components),
            "em_steps": np.int64(self.em_steps),
            "variance_floor": np.float64(self.variance_floor),
            "theta_variance_floor": np.float64(self.theta_variance_floor),
            "min_branch_mass": np.float64(self.min_branch_mass),
            "source_eps": np.float64(self.source_eps),
        }

    @classmethod
    def from_npz(cls, path: str) -> "ThetaGaussianScaffold":
        z = np.load(path, allow_pickle=True)
        if "bin_edges" in z.files:
            bin_edges = np.asarray(z["bin_edges"], dtype=np.float64)
        else:
            theta_grid = np.asarray(z["theta_grid"], dtype=np.float64)
            n_bins = int(np.asarray(z["bin_means"]).shape[0])
            bin_edges = np.linspace(float(theta_grid[0]), float(theta_grid[-1]), n_bins + 1, dtype=np.float64)
        return cls(
            bin_edges=bin_edges,
            bin_means=np.asarray(z["bin_means"], dtype=np.float64),
            bin_vars=np.asarray(z["bin_vars"], dtype=np.float64),
            n_components=int(np.asarray(z["n_components"]).reshape(-1)[0]) if "n_components" in z.files else min(3, int(np.asarray(z["bin_means"]).shape[0])),
            em_steps=int(np.asarray(z["em_steps"]).reshape(-1)[0]) if "em_steps" in z.files else 20,
            variance_floor=float(np.asarray(z["variance_floor"]).reshape(-1)[0]),
            min_branch_mass=float(np.asarray(z["min_branch_mass"]).reshape(-1)[0]),
            source_eps=float(np.asarray(z["source_eps"]).reshape(-1)[0]),
        )
