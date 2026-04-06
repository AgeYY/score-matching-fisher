"""H-matrix estimation from learned posterior/prior theta scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fisher.models import ConditionalScore1D, PriorScore1D


@dataclass
class HMatrixResult:
    theta_used: np.ndarray
    theta_sorted: np.ndarray
    perm: np.ndarray
    inv_perm: np.ndarray
    g_matrix: np.ndarray
    c_matrix: np.ndarray
    delta_l_matrix: np.ndarray
    h_directed: np.ndarray
    h_sym: np.ndarray
    sigma_eval: float
    order_mode: str
    delta_diag_max_abs: float
    h_sym_max_asym_abs: float


class HMatrixEstimator:
    """Estimate directed and symmetric H-matrices from score models."""

    def __init__(
        self,
        *,
        model_post: ConditionalScore1D,
        model_prior: PriorScore1D,
        sigma_eval: float,
        device: torch.device,
        pair_batch_size: int = 65536,
    ) -> None:
        if pair_batch_size < 1:
            raise ValueError("pair_batch_size must be >= 1.")
        if sigma_eval <= 0.0:
            raise ValueError("sigma_eval must be positive.")
        self.model_post = model_post
        self.model_prior = model_prior
        self.sigma_eval = float(sigma_eval)
        self.device = device
        self.pair_batch_size = int(pair_batch_size)

    @staticmethod
    def sort_by_theta(theta: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta_col = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if theta_col.shape[0] != x2.shape[0]:
            raise ValueError("theta and x must have the same number of rows.")
        perm = np.argsort(theta_col.reshape(-1), kind="mergesort")
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(perm.size, dtype=perm.dtype)
        theta_sorted = theta_col[perm]
        x_sorted = x2[perm]
        return theta_sorted, x_sorted, perm, inv_perm

    def compute_g_matrix(self, theta_sorted: np.ndarray, x_sorted: np.ndarray) -> np.ndarray:
        n = int(theta_sorted.shape[0])
        if n < 1:
            raise ValueError("Need at least one sample to compute H-matrix.")
        row_block = max(1, int(self.pair_batch_size // n))
        theta_grid_col = np.asarray(theta_sorted, dtype=np.float32).reshape(n, 1)
        g = np.zeros((n, n), dtype=np.float64)
        self.model_post.eval()
        self.model_prior.eval()
        with torch.no_grad():
            for i0 in range(0, n, row_block):
                i1 = min(n, i0 + row_block)
                xb = np.asarray(x_sorted[i0:i1], dtype=np.float32)
                b = int(i1 - i0)
                theta_tile = np.tile(theta_grid_col, (b, 1))
                x_rep = np.repeat(xb, repeats=n, axis=0)
                theta_t = torch.from_numpy(theta_tile).to(self.device)
                x_t = torch.from_numpy(x_rep).to(self.device)
                s_post = self.model_post.predict_score(theta_t, x_t, sigma_eval=self.sigma_eval).cpu().numpy().reshape(b, n)
                s_prior = self.model_prior.predict_score(theta_t, sigma_eval=self.sigma_eval).cpu().numpy().reshape(b, n)
                g[i0:i1, :] = (s_post - s_prior).astype(np.float64)
        return g

    @staticmethod
    def compute_c_matrix(theta_sorted: np.ndarray, g_matrix: np.ndarray) -> np.ndarray:
        theta_flat = np.asarray(theta_sorted, dtype=np.float64).reshape(-1)
        if g_matrix.ndim != 2 or g_matrix.shape[0] != theta_flat.size or g_matrix.shape[1] != theta_flat.size:
            raise ValueError("g_matrix must have shape (N, N) matching theta_sorted.")
        dtheta = np.diff(theta_flat).reshape(1, -1)
        trapezoids = 0.5 * dtheta * (g_matrix[:, :-1] + g_matrix[:, 1:])
        c = np.zeros_like(g_matrix, dtype=np.float64)
        if trapezoids.shape[1] > 0:
            c[:, 1:] = np.cumsum(trapezoids, axis=1)
        return c

    @staticmethod
    def compute_delta_l(c_matrix: np.ndarray) -> np.ndarray:
        if c_matrix.ndim != 2 or c_matrix.shape[0] != c_matrix.shape[1]:
            raise ValueError("c_matrix must be square.")
        diag = np.diag(c_matrix).reshape(-1, 1)
        return c_matrix - diag

    @staticmethod
    def compute_h_directed(delta_l_matrix: np.ndarray) -> np.ndarray:
        z = 0.5 * np.asarray(delta_l_matrix, dtype=np.float64)
        z = np.clip(z, -60.0, 60.0)
        h_directed = 1.0 - (1.0 / np.cosh(z))
        np.fill_diagonal(h_directed, 0.0)
        return h_directed

    @staticmethod
    def symmetrize(h_directed: np.ndarray) -> np.ndarray:
        return 0.5 * (h_directed + h_directed.T)

    @staticmethod
    def _permute_back(mat_sorted: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
        return mat_sorted[np.ix_(inv_perm, inv_perm)]

    def run(self, theta: np.ndarray, x: np.ndarray, *, restore_original_order: bool = False) -> HMatrixResult:
        theta_col = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        theta_sorted, x_sorted, perm, inv_perm = self.sort_by_theta(theta_col, x)
        if np.any(np.diff(theta_sorted.reshape(-1)) < 0.0):
            raise ValueError("sort_by_theta produced a non-monotone theta sequence.")

        g_sorted = self.compute_g_matrix(theta_sorted, x_sorted)
        c_sorted = self.compute_c_matrix(theta_sorted, g_sorted)
        delta_sorted = self.compute_delta_l(c_sorted)
        h_dir_sorted = self.compute_h_directed(delta_sorted)
        h_sym_sorted = self.symmetrize(h_dir_sorted)

        delta_diag_max_abs = float(np.max(np.abs(np.diag(delta_sorted))))
        h_sym_max_asym_abs = float(np.max(np.abs(h_sym_sorted - h_sym_sorted.T)))

        if restore_original_order:
            theta_used = theta_col.reshape(-1)
            g_used = self._permute_back(g_sorted, inv_perm)
            c_used = self._permute_back(c_sorted, inv_perm)
            delta_used = self._permute_back(delta_sorted, inv_perm)
            h_dir_used = self._permute_back(h_dir_sorted, inv_perm)
            h_sym_used = self._permute_back(h_sym_sorted, inv_perm)
            order_mode = "original"
        else:
            theta_used = theta_sorted.reshape(-1)
            g_used = g_sorted
            c_used = c_sorted
            delta_used = delta_sorted
            h_dir_used = h_dir_sorted
            h_sym_used = h_sym_sorted
            order_mode = "sorted"

        return HMatrixResult(
            theta_used=theta_used,
            theta_sorted=theta_sorted.reshape(-1),
            perm=perm,
            inv_perm=inv_perm,
            g_matrix=g_used,
            c_matrix=c_used,
            delta_l_matrix=delta_used,
            h_directed=h_dir_used,
            h_sym=h_sym_used,
            sigma_eval=self.sigma_eval,
            order_mode=order_mode,
            delta_diag_max_abs=delta_diag_max_abs,
            h_sym_max_asym_abs=h_sym_max_asym_abs,
        )
