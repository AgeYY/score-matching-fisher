"""Unit tests for classifier-initialized Hellinger helper in study_h_decoding_convergence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from fisher.shared_dataset_io import SharedDatasetBundle


_REPO = Path(__file__).resolve().parent.parent
_BIN = _REPO / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import study_h_decoding_convergence as shc  # noqa: E402


class TestClassifierInitializedHellinger(unittest.TestCase):
    def test_classifier_initialized_matrix_shape_symmetry_range_sparse(self) -> None:
        rng = np.random.default_rng(123)
        n_bins = 4
        n_per_bin_full = [30, 26, 18, 5]
        x_dim = 3

        x_all_parts: list[np.ndarray] = []
        bin_all_parts: list[np.ndarray] = []
        for b, n_b in enumerate(n_per_bin_full):
            mu = np.array([float(b), -0.3 * float(b), 0.2 * float(b)], dtype=np.float64)
            x_b = rng.normal(loc=mu, scale=0.25, size=(int(n_b), x_dim))
            x_all_parts.append(x_b)
            bin_all_parts.append(np.full(int(n_b), int(b), dtype=np.int64))
        x_all = np.vstack(x_all_parts)
        bin_all = np.concatenate(bin_all_parts)

        n_total = int(x_all.shape[0])
        n_train = int(0.7 * n_total)
        tr_idx = np.arange(0, n_train, dtype=np.int64)
        va_idx = np.arange(n_train, n_total, dtype=np.int64)

        x_train = np.asarray(x_all[tr_idx], dtype=np.float64)
        bin_train = np.asarray(bin_all[tr_idx], dtype=np.int64)

        # Ensure at least one sparse pair in train due to low-count final bin.
        sparse_bin = n_bins - 1
        self.assertLess(int(np.sum(bin_train == sparse_bin)), 8)

        theta_dummy = np.zeros((n_total, 1), dtype=np.float64)
        subset = shc.SweepSubset(
            bundle=SharedDatasetBundle(
                meta={"seed": 123, "train_frac": 0.7},
                theta_all=theta_dummy,
                x_all=np.asarray(x_all, dtype=np.float64),
                train_idx=tr_idx,
                validation_idx=va_idx,
                theta_train=np.asarray(theta_dummy[tr_idx], dtype=np.float64),
                x_train=x_train,
                theta_validation=np.asarray(theta_dummy[va_idx], dtype=np.float64),
                x_validation=np.asarray(x_all[va_idx], dtype=np.float64),
            ),
            bin_all=np.asarray(bin_all, dtype=np.int64),
            bin_train=bin_train,
            bin_validation=np.asarray(bin_all[va_idx], dtype=np.int64),
        )

        h2 = shc._classifier_initialized_hellinger_sq_binned(
            subset,
            n_bins,
            min_class_count=8,
            random_state=7,
        )

        self.assertEqual(h2.shape, (n_bins, n_bins))
        self.assertTrue(np.allclose(h2, h2.T, equal_nan=True))
        self.assertTrue(np.allclose(np.diag(h2), 0.0))

        finite = np.isfinite(h2)
        self.assertTrue(np.all((h2[finite] >= 0.0) & (h2[finite] <= 1.0)))
        self.assertTrue(np.any(~finite))


if __name__ == "__main__":
    unittest.main()
