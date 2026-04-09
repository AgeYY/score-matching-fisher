from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "bin" / "visualize_h_matrix_binned.py"
_SPEC = importlib.util.spec_from_file_location("visualize_h_matrix_binned", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

average_matrix_by_bins = _MODULE.average_matrix_by_bins
config_from_args = _MODULE.config_from_args
hellinger_acc_lb_from_binned_h_squared = _MODULE.hellinger_acc_lb_from_binned_h_squared
hellinger_acc_ub_from_binned_h_squared = _MODULE.hellinger_acc_ub_from_binned_h_squared
matrix_corr_offdiag = _MODULE.matrix_corr_offdiag
parse_args = _MODULE.parse_args
run_binned_visualization = _MODULE.run_binned_visualization
theta_bin_edges = _MODULE.theta_bin_edges
theta_to_bin_index = _MODULE.theta_to_bin_index


def _dataset_meta_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)
    ns = parser.parse_args([])
    ns.dataset_family = "gaussian"
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
    eval_idx = np.arange(6, 12, dtype=np.int64)

    dataset_npz = tmp_path / "shared_dataset.npz"
    save_shared_dataset_npz(
        dataset_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        eval_idx=eval_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_eval=theta_all[eval_idx],
        x_eval=x_all[eval_idx],
    )

    th = theta_all.reshape(-1)
    h_sym = np.abs(th[:, None] - th[None, :])
    h_sym = h_sym / float(np.max(h_sym))

    h_npz = tmp_path / "h_matrix_results_theta_cov.npz"
    np.savez_compressed(
        h_npz,
        h_sym=h_sym,
        theta_used=th,
        h_field_method=np.asarray(["dsm"], dtype=object),
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
            "--clf-test-frac",
            "0.5",
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
        with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
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
