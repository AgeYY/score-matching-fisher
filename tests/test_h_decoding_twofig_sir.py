import argparse

import numpy as np
import pytest

from fisher.h_decoding_convergence_methods import _subset_bundle
from fisher.h_decoding_twofig import SIR_FIRST_DEFAULT_ROWS, _project_sir_first_subset, build_sir_first_parser
from fisher.shared_dataset_io import SharedDatasetBundle


def _tiny_bundle() -> SharedDatasetBundle:
    n = 12
    theta = np.linspace(-3.0, 3.0, n, dtype=np.float64).reshape(-1, 1)
    x = np.column_stack(
        [
            theta[:, 0],
            theta[:, 0] ** 2,
            np.sin(theta[:, 0]),
            np.cos(theta[:, 0]),
        ]
    ).astype(np.float64)
    meta = {"version": 2, "dataset_family": "synthetic", "seed": 7, "x_dim": 4, "train_frac": 0.5}
    return SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(6, dtype=np.int64),
        validation_idx=np.arange(6, n, dtype=np.int64),
        theta_train=theta[:6],
        x_train=x[:6],
        theta_validation=theta[6:],
        x_validation=x[6:],
    )


def test_sir_first_parser_help_defaults() -> None:
    parser = build_sir_first_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_sir_first_parser_defaults_without_help() -> None:
    parser = build_sir_first_parser()
    args = parser.parse_args(["--dataset-npz", "dummy.npz"])
    assert args.sir_first is True
    assert args.theta_field_rows == SIR_FIRST_DEFAULT_ROWS
    assert args.n_list == "80,200,400,600"


def test_project_sir_first_subset_is_per_n_and_preserves_bins(tmp_path) -> None:
    bundle = _tiny_bundle()
    meta = bundle.meta
    perm = np.arange(bundle.theta_all.shape[0], dtype=np.int64)
    bin_idx_all = np.arange(bundle.theta_all.shape[0], dtype=np.int64) % 4
    args = argparse.Namespace(sir_dim=2, sir_num_bins=4, sir_ridge=1e-6)

    subset_8 = _subset_bundle(bundle, perm, 8, meta, bin_idx_all=bin_idx_all)
    projected_8, meta_8, path_8 = _project_sir_first_subset(
        subset=subset_8,
        theta_fit_subset=subset_8,
        args=args,
        n=8,
        sir_projection_root=str(tmp_path),
    )
    subset_10 = _subset_bundle(bundle, perm, 10, meta, bin_idx_all=bin_idx_all)
    projected_10, meta_10, path_10 = _project_sir_first_subset(
        subset=subset_10,
        theta_fit_subset=subset_10,
        args=args,
        n=10,
        sir_projection_root=str(tmp_path),
    )

    assert projected_8.bundle.x_train.shape == (4, 2)
    assert projected_8.bundle.x_validation.shape == (4, 2)
    assert projected_8.bundle.x_all.shape == (8, 2)
    assert projected_10.bundle.x_train.shape == (5, 2)
    assert projected_10.bundle.x_validation.shape == (5, 2)
    assert projected_10.bundle.x_all.shape == (10, 2)
    assert np.array_equal(projected_8.bin_all, subset_8.bin_all)
    assert np.array_equal(projected_8.bin_train, subset_8.bin_train)
    assert np.array_equal(projected_8.bin_validation, subset_8.bin_validation)
    assert not np.array_equal(np.asarray(meta_8["sir_x_mean"]), np.asarray(meta_10["sir_x_mean"]))

    with np.load(path_8, allow_pickle=False) as data:
        assert bool(data["sir_enabled"])
        assert data["sir_components"].shape == (4, 2)
        assert data["sir_x_mean"].shape == (4,)
        assert data["sir_theta_edges"].shape == (1, 5)
    with np.load(path_10, allow_pickle=False) as data:
        assert data["sir_components"].shape == (4, 2)
