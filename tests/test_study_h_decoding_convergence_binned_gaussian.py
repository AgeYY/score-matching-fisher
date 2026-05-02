from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

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


def test_theta2_grid_binning_flattens_row_major() -> None:
    mod = _load_module()
    theta = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    grid = mod.prepare_theta2_grid_binning_for_convergence(
        theta,
        np.arange(theta.shape[0], dtype=np.int64),
        n_ref=theta.shape[0],
        n_bins_x=2,
        n_bins_y=2,
    )
    assert grid.grid_shape == (2, 2)
    assert grid.centers.shape == (4, 2)
    np.testing.assert_array_equal(grid.bin_idx_all, np.asarray([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(grid.centers[0], [-0.5, -0.5])
    np.testing.assert_allclose(grid.centers[3], [0.5, 0.5])


def test_binned_gaussian_linear_x_flow_alias_and_estimate_one_smoke(tmp_path: Path) -> None:
    mod = _load_module()
    assert mod._normalize_linear_x_flow_method("bin-gaussian-linear-x-flow-diagonal") == "bin_gaussian_linear_x_flow_diagonal"
    assert mod._normalize_linear_x_flow_method("bin_lxf_diagonal") == "bin_gaussian_linear_x_flow_diagonal"

    theta_all = np.asarray([0.0, 0.1, 1.0, 1.1, 2.0, 2.1], dtype=np.float64).reshape(-1, 1)
    x_all = np.asarray(
        [[0.0, 0.1], [0.1, -0.1], [1.0, 0.9], [1.1, 1.1], [2.0, -0.9], [2.1, -1.1]],
        dtype=np.float64,
    )
    train_idx = np.asarray([0, 1, 2, 3], dtype=np.int64)
    val_idx = np.asarray([4, 5], dtype=np.int64)
    bundle = SharedDatasetBundle(
        meta={},
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=val_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[val_idx],
        x_validation=x_all[val_idx],
    )
    parser = mod.build_parser()
    args = parser.parse_args(["--dataset-npz", str(tmp_path / "dummy_dataset.npz")])
    args.theta_field_method = "bin-gaussian-linear-x-flow-diagonal"
    args.device = "cpu"
    args.output_dir = str(tmp_path)
    args.num_theta_bins = 3
    args.n_ref = 6
    args.lxf_epochs = 2
    args.lxf_batch_size = 2
    args.lxf_hidden_dim = 8
    args.lxf_depth = 1
    args.lxf_early_patience = 0
    args.lxf_weight_ema_decay = 0.0
    args.lxf_restore_best = False
    args.lxf_pair_batch_size = 64
    args.lxf_save_c_matrix = True
    mod._validate_cli(args)

    loaded, _, dev = mod._estimate_one(
        args=args,
        meta={"seed": 0},
        bundle=bundle,
        output_dir=str(tmp_path),
        n_bins=3,
        bin_train=np.asarray([0, 0, 1, 1], dtype=np.int64),
        bin_validation=np.asarray([2, 2], dtype=np.int64),
        bin_all=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
    )
    assert dev == torch.device("cpu")
    assert loaded.h_sym.shape == (6, 6)
    assert np.isfinite(loaded.h_sym).all()
    z = np.load(tmp_path / "h_matrix_results_theta_cov.npz", allow_pickle=True)
    assert z["c_matrix"].shape == (6, 6)
    assert np.isfinite(z["c_matrix"]).all()
    assert bool(np.asarray(z["lxf_fm_train"]).reshape(-1)[0]) is True
    assert bool(np.asarray(z["lxf_a_fixed"]).reshape(-1)[0]) is True
    assert str(np.asarray(z["lxf_a_source"], dtype=object).reshape(-1)[0]) == "binned_gaussian_shared_diagonal_covariance"
    assert "lxf_shared_variance" in z.files
    assert "lxf_bin_counts" in z.files
    assert "lxf_normalized_bin_means" in z.files
    assert "lxf_variance_floor" in z.files
    loss = np.load(tmp_path / "score_prior_training_losses.npz", allow_pickle=True)
    assert bool(np.asarray(loss["lxf_fm_train"]).reshape(-1)[0]) is True
    assert bool(np.asarray(loss["lxf_a_fixed"]).reshape(-1)[0]) is True
    assert loss["score_train_losses"].size > 0
    assert np.isfinite(loss["score_train_losses"]).all()
