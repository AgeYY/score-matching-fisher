from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "bin" / "visualize_h_matrix_binned.py"
_SPEC = importlib.util.spec_from_file_location("visualize_h_matrix_binned", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[str(_SPEC.name)] = _MODULE
_SPEC.loader.exec_module(_MODULE)

average_matrix_by_bins = _MODULE.average_matrix_by_bins
config_from_args = _MODULE.config_from_args
hellinger_acc_lb_from_binned_h_squared = _MODULE.hellinger_acc_lb_from_binned_h_squared
hellinger_acc_ub_from_binned_h_squared = _MODULE.hellinger_acc_ub_from_binned_h_squared
matrix_corr_offdiag = _MODULE.matrix_corr_offdiag
matrix_corr_offdiag_pearson = _MODULE.matrix_corr_offdiag_pearson
impute_offdiag_nan_mean = _MODULE.impute_offdiag_nan_mean
pairwise_bin_logistic_accuracy_train_val = _MODULE.pairwise_bin_logistic_accuracy_train_val
parse_args = _MODULE.parse_args
run_binned_visualization = _MODULE.run_binned_visualization
theta_bin_edges = _MODULE.theta_bin_edges
theta_to_bin_index = _MODULE.theta_to_bin_index


def _dataset_meta_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)
    ns = parser.parse_args([])
    ns.dataset_family = "cosine_gaussian"
    ns.seed = 13
    ns.theta_low = -1.0
    ns.theta_high = 1.0
    ns.x_dim = 2
    ns.n_total = 12
    ns.train_frac = 0.5
    return ns


def _write_fixture_dataset_and_h(tmp_path: Path) -> tuple[Path, Path]:
    ns = _dataset_meta_defaults()
    meta = meta_dict_from_args(ns)

    theta_all = np.linspace(-1.0, 1.0, num=12, dtype=np.float64).reshape(-1, 1)
    x_all = np.concatenate([theta_all, theta_all**2], axis=1)
    train_idx = np.arange(0, 6, dtype=np.int64)
    validation_idx = np.arange(6, 12, dtype=np.int64)

    dataset_npz = tmp_path / "shared_dataset.npz"
    save_shared_dataset_npz(
        dataset_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )

    th = theta_all.reshape(-1)
    h_sym = np.abs(th[:, None] - th[None, :])
    h_sym = h_sym / float(np.max(h_sym))

    h_npz = tmp_path / "h_matrix_results_theta_cov.npz"
    np.savez_compressed(
        h_npz,
        h_sym=h_sym,
        theta_used=th,
        h_field_method=np.asarray(["theta_flow"], dtype=object),
        h_eval_scalar_name=np.asarray(["sigma_eval"], dtype=object),
        sigma_eval=np.asarray([0.1], dtype=np.float64),
    )
    return dataset_npz, h_npz


def test_theta_bin_helpers() -> None:
    theta = np.asarray([-1.0, -0.2, 0.1, 1.0], dtype=np.float64)
    edges, lo, hi = theta_bin_edges(theta, n_bins=2)
    idx = theta_to_bin_index(theta, edges, n_bins=2)

    assert lo == -1.0
    assert hi == 1.0
    assert edges.shape == (3,)
    assert idx.tolist() == [0, 0, 1, 1]


def test_pairwise_clf_uses_train_then_scores_full_eval_pool() -> None:
    """Fit split uses train rows; evaluation can use the full pooled rows."""
    rng = np.random.default_rng(0)
    n_tr = 60
    x_tr = rng.standard_normal((n_tr, 2))
    th_tr = np.concatenate(
        [
            rng.uniform(-1.0, -0.34, size=20),
            rng.uniform(-0.33, 0.33, size=20),
            rng.uniform(0.34, 1.0, size=20),
        ]
    )
    # Eval pool has only one sample in the middle bin. Old validation-only gating would
    # reject all pairs; new fit-train/eval-pool policy should still return finite pairs.
    x_ev = np.vstack(
        [
            rng.standard_normal((10, 2)),
            rng.standard_normal((1, 2)),
            rng.standard_normal((10, 2)),
        ]
    )
    th_ev = np.concatenate(
        [
            rng.uniform(-1.0, -0.34, size=10),
            rng.uniform(-0.33, 0.33, size=1),
            rng.uniform(0.34, 1.0, size=10),
        ]
    )
    edges, _, _ = theta_bin_edges(np.concatenate([th_tr, th_ev]), n_bins=3)
    bi_tr = theta_to_bin_index(th_tr, edges, 3)
    bi_ev = theta_to_bin_index(th_ev, edges, 3)
    acc, valid, _, stats = pairwise_bin_logistic_accuracy_train_val(
        x_tr,
        bi_tr,
        x_ev,
        bi_ev,
        3,
        min_class_count=5,
        random_state=7,
    )
    assert acc.shape == (3, 3)
    assert stats["ok_pairs"] > 0
    assert np.any(np.isfinite(acc[~np.eye(3, dtype=bool)]))
    assert bool(valid[0, 2])


def test_average_matrix_by_bins() -> None:
    mat = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float64,
    )
    bin_idx = np.asarray([0, 0, 1, 1], dtype=np.int64)
    out, counts = average_matrix_by_bins(mat, bin_idx, n_bins=2)

    assert out.shape == (2, 2)
    assert counts.tolist() == [[4, 4], [4, 4]]
    assert np.isclose(out[0, 0], np.mean(mat[:2, :2]))
    assert np.isclose(out[0, 1], np.mean(mat[:2, 2:]))


def test_hellinger_bounds_and_correlation() -> None:
    h2 = np.asarray([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    lb = hellinger_acc_lb_from_binned_h_squared(h2)
    ub = hellinger_acc_ub_from_binned_h_squared(h2)

    assert np.isnan(lb[0, 0]) and np.isnan(ub[1, 1])
    assert np.isclose(lb[0, 1], 0.625)
    assert float(ub[0, 1]) > float(lb[0, 1])

    a = np.asarray([[np.nan, 0.2, 0.8], [0.2, np.nan, 0.6], [0.8, 0.6, np.nan]], dtype=np.float64)
    b = np.asarray([[np.nan, 0.1, 0.9], [0.1, np.nan, 0.5], [0.9, 0.5, np.nan]], dtype=np.float64)
    corr = matrix_corr_offdiag(a, b)
    assert np.isfinite(corr)
    assert corr > 0.0


def test_impute_offdiag_nan_mean_then_pearson_matches_full_offdiag() -> None:
    """NaN off-diagonals in estimate are filled with finite off-diag mean before Pearson r."""
    n = 4
    a = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(a, np.nan)
    # Finite off-diagonal entries: mean = (0.5 + 0.6 + 0.7) / 3 = 0.6
    a[0, 1] = a[1, 0] = 0.5
    a[0, 2] = a[2, 0] = 0.6
    a[1, 2] = a[2, 1] = 0.7
    # Rest off-diagonal stay nan -> impute to 0.6
    b = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(b, np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                b[i, j] = float(i) + 0.1 * float(j)

    imp = impute_offdiag_nan_mean(a)
    off = ~np.eye(n, dtype=bool)
    assert np.all(np.isfinite(imp[off]))
    assert np.all(np.isnan(np.diag(imp)))
    mu = float(np.mean(a[off & np.isfinite(a)]))
    assert np.allclose(imp[off & ~np.isfinite(a)], mu)

    r_imputed = matrix_corr_offdiag_pearson(imp, b)
    # Build fully finite pair vectors by same imputation on a then mask finite(b) - b is finite offdiag
    mask = off & np.isfinite(b)
    av = imp[mask]
    bv = b[mask]
    r_direct = float(np.corrcoef(av, bv)[0, 1])
    assert np.isclose(r_imputed, r_direct, rtol=0.0, atol=1e-12)


def test_integration_gpu_smoke_or_explicit_cuda_error(tmp_path: Path) -> None:
    dataset_npz, h_npz = _write_fixture_dataset_and_h(tmp_path)
    out_dir = tmp_path / "out"

    args = parse_args(
        [
            "--dataset-npz",
            str(dataset_npz),
            "--h-only",
            "--h-matrix-npz",
            str(h_npz),
            "--num-theta-bins",
            "4",
            "--clf-min-class-count",
            "1",
            "--gt-approx-n-total",
            "64",
            "--output-dir",
            str(out_dir),
            "--device",
            "cuda",
            "--no-sssd",
        ]
    )
    config = config_from_args(args)

    if not torch.cuda.is_available():
        _tc = unittest.TestCase()
        with _tc.assertRaises(RuntimeError):
            run_binned_visualization(config)
        return

    result = run_binned_visualization(config)
    assert Path(result.artifacts.out_npz).is_file()
    assert Path(result.artifacts.summary_path).is_file()
    assert Path(result.artifacts.combo_fig_path).is_file()

    with np.load(result.artifacts.out_npz, allow_pickle=True) as data:
        assert "h_binned" in data.files
        assert "clf_accuracy_binned" in data.files
        assert int(data["num_theta_bins"].reshape(-1)[0]) == 4
