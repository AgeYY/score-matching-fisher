"""Unit tests for categorical H-decoding twofig helpers."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.h_decoding_categorical_twofig import (
    binary_classifier_hellinger_sqrt,
    build_parser,
    h_sq_category_from_sample_directed,
    h_sq_directed_from_delta_l,
    hellinger_gt_sq_category_matrix,
    main,
    parse_methods,
    _save_method_training_loss_npz,
    _validation_only_work_sweep_subset,
)
from fisher.h_matrix import HMatrixEstimator
from fisher import h_decoding_convergence as conv
from fisher.h_decoding_convergence_methods import prepare_categorical_binning_for_convergence
from fisher.shared_dataset_io import load_shared_dataset_npz


def test_parse_methods_aliases_and_dedup() -> None:
    assert parse_methods("x-flow, X_FLOW, binary-classifier") == ["x_flow", "binary_classifier"]
    assert parse_methods("bin-gaussian, bin_gaussian, x_flow") == ["bin_gaussian", "x_flow"]
    with pytest.raises(ValueError, match="Unknown method"):
        parse_methods("theta_flow")


def test_decode_accuracy_color_limits() -> None:
    from fisher.h_decoding_twofig import _decode_accuracy_color_limits

    a = np.array([[np.nan, 0.55], [0.55, np.nan]])
    vmin, vmax = _decode_accuracy_color_limits(a)
    assert vmin == pytest.approx(0.5) and vmax == 1.0
    b = np.array([[np.nan, 0.35], [0.35, np.nan]])
    vmin2, vmax2 = _decode_accuracy_color_limits(b)
    assert vmin2 == pytest.approx(0.35) and vmax2 == 1.0


def test_hellinger_gt_sq_category_matrix_shape_diag_bounds() -> None:
    ds = ToyCategoricalRandomMoGDataset(x_dim=2, num_categories=4, seed=0)
    h2 = hellinger_gt_sq_category_matrix(ds)
    assert h2.shape == (4, 4)
    assert np.allclose(np.diag(h2), 0.0)
    assert float(np.min(h2)) >= 0.0
    assert float(np.max(h2)) <= 1.0 + 1e-9


def test_category_aggregation_small_delta_l() -> None:
    n = 6
    k = 2
    rng = np.random.default_rng(0)
    c = rng.normal(size=(n, n)).astype(np.float64)
    delta_l = HMatrixEstimator.compute_delta_l(c)
    h_dir = h_sq_directed_from_delta_l(delta_l)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    h_cat = h_sq_category_from_sample_directed(h_dir, labels, k_cat=k)
    assert h_cat.shape == (k, k)
    assert np.allclose(np.diag(h_cat), 0.0)
    assert np.all(h_cat >= 0.0)


def test_binary_classifier_hellinger_sqrt_shape_diag_bounds() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(loc=-1.0, scale=0.1, size=(8, 2))
    x1 = rng.normal(loc=1.0, scale=0.1, size=(8, 2))
    x_all = np.vstack([x0, x1]).astype(np.float64)
    bins = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    args = SimpleNamespace(clf_random_state=0, run_seed=0, clf_min_class_count=2, clf_max_iter=200)

    h = binary_classifier_hellinger_sqrt(
        args,
        x_train=x_all,
        bins_train=bins,
        x_all=x_all,
        bins_all=bins,
        k_cat=2,
    )
    assert h.shape == (2, 2)
    assert np.allclose(np.diag(h), 0.0)
    assert float(np.min(h)) >= 0.0
    assert float(np.max(h)) <= 1.0


def test_cli_defaults_from_parser() -> None:
    p = build_parser()
    ns = p.parse_args([])
    assert int(ns.num_categories) == 2
    assert ns.n_list == "80,200,400,600"
    assert int(ns.n_ref) == 600
    assert str(ns.device) == "cuda"


def test_main_rejects_num_categories_lt2() -> None:
    with pytest.raises(ValueError, match="num-categories"):
        main(["--num-categories", "1", "--device", "cpu"])


def test_visualization_only_cached_shapes(tmp_path: Path) -> None:
    k = 3
    methods = ["binary_classifier"]
    ns = [20, 40]
    h_gt = np.full((k, k), 0.1, dtype=np.float64)
    np.fill_diagonal(h_gt, 0.0)
    dec_ref = np.full((k, k), 0.5, dtype=np.float64)
    np.fill_diagonal(dec_ref, np.nan)
    h_sw = np.random.RandomState(1).rand(len(methods), len(ns), k, k).astype(np.float64)
    dec_sw = np.random.RandomState(2).rand(len(ns), k, k).astype(np.float64)
    corr_h = np.random.RandomState(3).rand(len(methods), len(ns))
    nmse_h = np.random.RandomState(4).rand(len(methods), len(ns))
    corr_d = np.random.RandomState(5).rand(len(ns))
    nmse_d = np.random.RandomState(6).rand(len(ns))
    fake_ds = tmp_path / "fake.npz"
    fake_ds.write_bytes(b"not a real npz")
    npz_path = tmp_path / "h_decoding_categorical_twofig_results.npz"
    np.savez_compressed(
        str(npz_path),
        n=np.asarray(ns, dtype=np.int64),
        n_ref=np.int64(40),
        num_categories=np.int64(k),
        method_names=np.asarray(methods, dtype=object),
        theta_bin_centers=np.arange(k, dtype=np.float64).reshape(-1, 1),
        h_gt_sqrt=h_gt,
        decode_ref=dec_ref,
        decode_sweep=dec_sw,
        h_sqrt_sweep=h_sw,
        corr_h_vs_gt=corr_h,
        nmse_h_vs_gt=nmse_h,
        corr_decode_vs_ref=corr_d,
        nmse_decode_vs_ref=nmse_d,
        native_dataset_npz=np.asarray([str(fake_ds.resolve())], dtype=object),
        hellinger_eval_split=np.asarray(["all"], dtype=object),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    npz_path.rename(out_dir / "h_decoding_categorical_twofig_results.npz")
    loss_root = out_dir / "training_losses" / "binary_classifier"
    loss_root.mkdir(parents=True)
    for n in ns:
        _save_method_training_loss_npz(
            loss_root / f"n_{int(n):06d}.npz",
            method_name="binary_classifier",
            result={},
        )
    argv = [
        "--visualization-only",
        "--output-dir",
        str(out_dir),
        "--dataset-npz",
        str(fake_ds),
        "--num-categories",
        str(k),
        "--n-ref",
        "40",
        "--n-list",
        "20,40",
        "--methods",
        "binary_classifier",
        "--device",
        "cpu",
    ]
    main(argv)
    assert (out_dir / "h_decoding_categorical_twofig_sweep.svg").is_file()
    assert (out_dir / "h_decoding_categorical_twofig_gt.svg").is_file()
    assert (out_dir / "h_decoding_categorical_twofig_training_losses_panel.svg").is_file()


def test_build_parser_hellinger_eval_split() -> None:
    p = build_parser()
    assert p.parse_args([]).hellinger_eval_split == "all"
    assert p.parse_args(["--hellinger-eval-split", "validation", "--device", "cpu"]).hellinger_eval_split == "validation"


def test_validation_only_work_sweep_subset_aligns_with_bundle_validation() -> None:
    repo = Path(__file__).resolve().parent.parent
    npz = repo / "data" / "random_mog_categorical_xdim2_default" / "random_mog_categorical.npz"
    if not npz.is_file():
        pytest.skip("benchmark-cate native NPZ not in checkout")
    bundle = load_shared_dataset_npz(npz)
    meta = dict(bundle.meta)
    k_cat = int(meta.get("num_categories", 5))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(bundle.theta_all, k_cat)
    perm = np.arange(int(bundle.theta_all.shape[0]), dtype=np.int64)
    subset_w = conv._subset_bundle(bundle, perm, 200, meta, bin_idx_all=bin_idx_all)
    vs = _validation_only_work_sweep_subset(subset_w)
    assert int(vs.bundle.x_all.shape[0]) == int(subset_w.bundle.x_validation.shape[0])
    assert np.array_equal(vs.bin_all, subset_w.bin_validation)
