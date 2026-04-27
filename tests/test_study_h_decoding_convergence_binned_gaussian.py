from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from fisher.shared_dataset_io import SharedDatasetBundle


def _load_module():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_binned_gaussian", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subset(mod, x_all: np.ndarray, bin_all: np.ndarray):
    n = int(x_all.shape[0])
    theta = np.arange(n, dtype=np.float64).reshape(-1, 1)
    bundle = SharedDatasetBundle(
        meta={},
        theta_all=theta,
        x_all=np.asarray(x_all, dtype=np.float64),
        train_idx=np.arange(n, dtype=np.int64),
        validation_idx=np.asarray([], dtype=np.int64),
        theta_train=theta,
        x_train=np.asarray(x_all, dtype=np.float64),
        theta_validation=np.empty((0, 1), dtype=np.float64),
        x_validation=np.empty((0, int(x_all.shape[1])), dtype=np.float64),
    )
    return mod.SweepSubset(
        bundle=bundle,
        bin_all=np.asarray(bin_all, dtype=np.int64),
        bin_train=np.asarray(bin_all, dtype=np.int64),
        bin_validation=np.asarray([], dtype=np.int64),
    )


def test_binned_gaussian_hellinger_matches_shared_diagonal_formula() -> None:
    mod = _load_module()
    x_all = np.asarray([[0.0], [0.0], [2.0], [2.0]], dtype=np.float64)
    subset = _subset(mod, x_all=x_all, bin_all=np.asarray([0, 0, 1, 1]))
    h2 = mod._binned_gaussian_hellinger_sq(subset, n_bins=2, variance_floor=1.0)
    expected = 1.0 - math.exp(-0.125 * 4.0)
    assert h2.shape == (2, 2)
    assert np.allclose(np.diag(h2), 0.0)
    assert np.isclose(float(h2[0, 1]), expected)
    assert np.isclose(float(h2[1, 0]), expected)


def test_binned_gaussian_hellinger_fills_empty_bins_and_stays_finite() -> None:
    mod = _load_module()
    x_all = np.asarray([[0.0, 1.0], [3.0, 4.0]], dtype=np.float64)
    subset = _subset(mod, x_all=x_all, bin_all=np.asarray([0, 2]))
    h2 = mod._binned_gaussian_hellinger_sq(subset, n_bins=3, variance_floor=1e-3)
    assert h2.shape == (3, 3)
    assert np.allclose(np.diag(h2), 0.0)
    assert np.isfinite(h2).all()
    assert np.all((h2 >= 0.0) & (h2 <= 1.0))
