import argparse

import numpy as np
import pytest

from fisher.h_decoding_convergence_methods import (
    _build_theta_fourier_state,
    _subset_bundle,
    prepare_theta2_grid_binning_for_convergence,
)
from fisher.h_decoding_twofig import (
    SIR_FIRST_DEFAULT_ROWS,
    _parse_theta_field_rows,
    _project_sir_first_subset,
    _validate_cli_for_rows,
    build_parser,
    build_sir_first_parser,
)
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


def _parse_twofig_args(extra: list[str]) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(["--dataset-npz", "dummy.npz", *extra])


@pytest.mark.parametrize("row", ["linear-x-flow-t", "xflow-sir-lrank", "contrastive_soft"])
def test_theta_fourier_state_allowed_for_requested_rows(row: str) -> None:
    args = _parse_twofig_args(["--theta-flow-fourier-state", "--theta-field-rows", row])
    rows = _parse_theta_field_rows(args)
    _validate_cli_for_rows(args, rows)


@pytest.mark.parametrize("row", ["linear-x-flow-t", "xflow-sir-lrank", "contrastive_soft"])
def test_theta_onehot_state_still_rejected_for_rows(row: str) -> None:
    args = _parse_twofig_args(["--theta-flow-onehot-state", "--theta-field-rows", row])
    rows = _parse_theta_field_rows(args)
    with pytest.raises(ValueError, match="theta-flow-onehot-state"):
        _validate_cli_for_rows(args, rows)


@pytest.mark.parametrize(
    "row",
    ["linear-x-flow-t", "xflow-sir-lrank", "contrastive_soft"],
)
def test_theta_fourier_state_cli_ok_with_theta2_grid(row: str) -> None:
    args = _parse_twofig_args(
        [
            "--theta-flow-fourier-state",
            "--theta-field-rows",
            row,
            "--theta-binning-mode",
            "theta2_grid",
        ]
    )
    rows = _parse_theta_field_rows(args)
    _validate_cli_for_rows(args, rows)


def test_theta2_grid_fourier_builder_shape_and_meta_vectors() -> None:
    rng = np.random.default_rng(0)
    th = rng.uniform(-1.0, 1.0, size=(40, 2)).astype(np.float64)
    perm = rng.permutation(40)
    grid = prepare_theta2_grid_binning_for_convergence(th, perm, n_ref=20, n_bins_x=4, n_bins_y=5)
    fou, rr, pp, cc = _build_theta_fourier_state(
        th,
        theta_ref=grid.theta_ref,
        k=2,
        period_mult=2.0,
        include_linear=True,
    )
    assert fou.shape == (40, 2 * (1 + 4))
    assert rr.shape == (2,) and pp.shape == (2,) and cc.shape == (2,)
    assert np.all(np.isfinite(fou))


def test_build_theta_fourier_state_1d_row_matches_column_and_legacy_scalar_fourier() -> None:
    from fisher.contrastive_llr import theta_scalar_fourier_columns

    th = np.linspace(-1.0, 1.0, 15, dtype=np.float64)
    ref = th[:8]
    row, rvec, pvec, cvec = _build_theta_fourier_state(th, theta_ref=ref, k=3, period_mult=2.0, include_linear=True)
    col, r2, p2, c2 = _build_theta_fourier_state(
        th.reshape(-1, 1), theta_ref=ref.reshape(-1, 1), k=3, period_mult=2.0, include_linear=True
    )
    assert np.allclose(row, col)
    assert np.allclose(rvec, r2) and np.allclose(pvec, p2) and np.allclose(cvec, c2)
    legacy = theta_scalar_fourier_columns(th, ref, k=3, period_mult=2.0, include_linear=True)
    assert np.allclose(row, legacy)


def test_build_theta_fourier_constant_coordinate_is_finite() -> None:
    th = np.column_stack([np.linspace(0.0, 1.0, 10), np.full(10, 3.0, dtype=np.float64)])
    fou, *_ = _build_theta_fourier_state(th, theta_ref=th[:5], k=2, period_mult=2.0, include_linear=False)
    assert np.all(np.isfinite(fou))


def test_subset_bundle_passes_fourier_theta_state_width() -> None:
    bundle = _tiny_bundle()
    perm = np.arange(bundle.theta_all.shape[0], dtype=np.int64)
    bin_idx_all = np.arange(bundle.theta_all.shape[0], dtype=np.int64) % 4
    theta_state_all, _, _, _ = _build_theta_fourier_state(
        bundle.theta_all,
        theta_ref=bundle.theta_all[:8],
        k=3,
        period_mult=2.0,
        include_linear=True,
    )
    subset = _subset_bundle(
        bundle,
        perm,
        8,
        bundle.meta,
        bin_idx_all=bin_idx_all,
        theta_state_all=theta_state_all,
    )
    assert subset.bundle.theta_all.shape == (8, 7)
    assert subset.bundle.theta_train.shape == (4, 7)
    assert subset.bundle.theta_validation.shape == (4, 7)


def test_subset_bundle_passes_fourier_theta_state_width_2d_native_theta() -> None:
    n = 12
    theta = np.column_stack([np.linspace(-1.0, 1.0, n), np.linspace(2.0, 3.0, n)]).astype(np.float64)
    x = np.column_stack([theta[:, 0], theta[:, 1], np.sin(theta[:, 0])]).astype(np.float64)
    meta = {"version": 2, "dataset_family": "synthetic", "seed": 1, "x_dim": 3, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
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
    perm = np.arange(n, dtype=np.int64)
    bin_idx_all = np.arange(n, dtype=np.int64) % 4
    theta_state_all, _, _, _ = _build_theta_fourier_state(
        theta,
        theta_ref=theta[:8],
        k=2,
        period_mult=2.0,
        include_linear=False,
    )
    assert theta_state_all.shape[1] == 8
    subset = _subset_bundle(bundle, perm, 8, meta, bin_idx_all=bin_idx_all, theta_state_all=theta_state_all)
    assert subset.bundle.theta_all.shape == (8, 8)
