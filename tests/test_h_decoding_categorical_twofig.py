"""Unit tests for categorical H-decoding twofig helpers."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.h_decoding_categorical_twofig import (
    build_parser,
    h_sq_category_from_sample_directed,
    h_sq_directed_from_delta_l,
    hellinger_gt_sq_category_matrix,
    main,
    parse_methods,
    theta_flow_categorical_hellinger_sqrt,
    _validate_cached_cli,
    _save_method_training_loss_npz,
    _validation_only_work_sweep_subset,
    _fit_lda_weighted_projection,
    _fit_pca_weighted_projection,
    _fit_pls_weighted_projection,
)
from fisher.h_matrix import HMatrixEstimator
from fisher import h_decoding_convergence as conv
from fisher.h_decoding_convergence_methods import prepare_categorical_binning_for_convergence
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.svg_utils import concatenate_svgs_horizontally, concatenate_svgs_horizontally_to_png


def _write_tiny_svg(path: Path, *, width: int = 10, height: int = 8) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"></svg>',
        encoding="utf-8",
    )
    return str(path)


def test_concatenate_svgs_horizontally_viewbox_order(tmp_path: Path) -> None:
    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"
    out = tmp_path / "out.svg"
    a.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 20"><rect id="a" width="10" height="20"/></svg>',
        encoding="utf-8",
    )
    b.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="30pt" height="15pt"><circle id="b" cx="5" cy="5" r="5"/></svg>',
        encoding="utf-8",
    )

    got = concatenate_svgs_horizontally([a, b], out, spacing=5.0)

    assert got == str(out)
    root = ET.parse(out).getroot()
    assert root.attrib["viewBox"] == "0 0 45 20"
    cols = [child for child in list(root) if child.tag.endswith("svg")]
    assert [col.attrib["x"] for col in cols] == ["0", "15"]
    assert [col.attrib["viewBox"] for col in cols] == ["0 0 10 20", "0 0 30 15"]


def test_concatenate_svgs_horizontally_to_png_uses_dpi(tmp_path: Path) -> None:
    from PIL import Image

    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"
    out = tmp_path / "out.png"
    a.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 36"><rect width="72" height="36"/></svg>',
        encoding="utf-8",
    )
    b.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72"><circle cx="36" cy="36" r="20"/></svg>',
        encoding="utf-8",
    )

    got = concatenate_svgs_horizontally_to_png([a, b], out, spacing=0.0, dpi=300)

    assert got == str(out)
    with Image.open(out) as im:
        assert im.format == "PNG"
        assert im.size == (600, 300)


def test_parse_methods_aliases_and_dedup() -> None:
    assert parse_methods("x-flow, X_FLOW, binary-classifier") == ["x_flow", "binary_classifier"]
    assert parse_methods("bin-gaussian, bin_gaussian, x_flow") == ["bin_gaussian", "x_flow"]
    assert parse_methods("bin_gaussian_cate") == ["bin_gaussian"]
    assert parse_methods("theta-flow-cate, theta_flow_cate, thetaflow-cate") == ["theta_flow_cate"]
    assert parse_methods("ctsm-v, ctsm_v, ctsm") == ["ctsm_v"]
    assert parse_methods("lda-ctsm-v, lda_ctsm_v, ldactsm-v") == ["lda_ctsm_v"]
    assert parse_methods("pls-ctsm-v, pls_ctsm_v, plsctsm-v") == ["pls_ctsm_v"]
    assert parse_methods("pca-ctsm-v, pca_ctsm_v, pcactsm-v") == ["pca_ctsm_v"]
    with pytest.raises(ValueError, match="Unknown method"):
        parse_methods("theta_flow")


def test_fit_lda_weighted_projection_shapes_weights_and_order() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0, 0.1],
            [-1.0, 0.2, -0.1],
            [1.0, 0.0, 0.0],
            [2.0, -0.1, 0.2],
        ],
        dtype=np.float64,
    )
    bins_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_val = x_train[:2] + 0.5
    x_all = x_train.copy()

    z_train, z_val, z_all, meta = _fit_lda_weighted_projection(
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        x_all=x_all,
        shrinkage=1e-2,
        eps=1e-6,
        max_dim=2,
    )

    assert z_train.shape == (4, 2)
    assert z_val.shape == (2, 2)
    assert z_all.shape == (4, 2)
    weights = np.asarray(meta["lda_ctsm_weights"], dtype=np.float64)
    eigvals = np.asarray(meta["lda_ctsm_eigenvalues"], dtype=np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(2.0)
    assert np.all(eigvals[:-1] >= eigvals[1:] - 1e-12)
    assert np.asarray(meta["lda_ctsm_transform"]).shape == (2, 3)


def test_fit_pls_weighted_projection_shapes_weights_and_metadata() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0, 0.1],
            [-1.0, 0.2, -0.1],
            [1.0, 0.0, 0.0],
            [2.0, -0.1, 0.2],
            [2.5, 0.1, -0.2],
        ],
        dtype=np.float64,
    )
    bins_train = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    x_val = x_train[:2] + 0.25
    x_all = x_train.copy()

    z_train, z_val, z_all, meta = _fit_pls_weighted_projection(
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        x_all=x_all,
        eps=1e-6,
        max_dim=2,
        scale_x=True,
        max_iter=100,
        tol=1e-7,
    )

    assert z_train.shape == (5, 2)
    assert z_val.shape == (2, 2)
    assert z_all.shape == (5, 2)
    weights = np.asarray(meta["pls_ctsm_weights"], dtype=np.float64)
    scores = np.asarray(meta["pls_ctsm_covariance_scores"], dtype=np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(2.0)
    assert np.all(scores >= -1e-12)
    assert np.asarray(meta["pls_ctsm_transform"]).shape == (2, 3)
    assert bool(meta["pls_ctsm_scale_x"])


def test_fit_pca_weighted_projection_shapes_weights_and_metadata() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0, 0.1],
            [-1.0, 0.2, -0.1],
            [1.0, 0.0, 0.0],
            [2.0, -0.1, 0.2],
        ],
        dtype=np.float64,
    )
    x_val = x_train[:2] + 0.25
    x_all = x_train.copy()

    z_train, z_val, z_all, meta = _fit_pca_weighted_projection(
        x_train=x_train,
        x_val=x_val,
        x_all=x_all,
        eps=1e-6,
        max_dim=2,
        scale_x=True,
    )

    assert z_train.shape == (4, 2)
    assert z_val.shape == (2, 2)
    assert z_all.shape == (4, 2)
    weights = np.asarray(meta["pca_ctsm_weights"], dtype=np.float64)
    explained = np.asarray(meta["pca_ctsm_explained_variance"], dtype=np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(2.0)
    assert np.all(explained[:-1] >= explained[1:] - 1e-12)
    assert np.asarray(meta["pca_ctsm_components"]).shape == (3, 2)
    assert np.asarray(meta["pca_ctsm_transform"]).shape == (2, 3)
    assert bool(meta["pca_ctsm_scale_x"])


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


def test_ctsm_pair_models_accept_one_hot_theta() -> None:
    from fisher.ctsm_models import ToyPairConditionedTimeScoreNet, ToyPairConditionedTimeScoreNetFiLM

    x = np.zeros((4, 2), dtype=np.float32)
    t = np.full((4, 1), 0.5, dtype=np.float32)
    m = np.eye(3, dtype=np.float32)[[0, 1, 2, 0]]
    d = np.eye(3, dtype=np.float32)[[1, 2, 0, 1]] - m
    for cls in (ToyPairConditionedTimeScoreNet, ToyPairConditionedTimeScoreNetFiLM):
        model = cls(dim=2, hidden_dim=8, theta_dim=3)
        y = model.forward_full(
            torch.from_numpy(x),
            torch.from_numpy(t),
            torch.from_numpy(m),
            torch.from_numpy(d),
        )
        assert tuple(y.shape) == (4, 2)


def test_h_matrix_ctsm_accepts_one_hot_theta() -> None:
    from fisher.ctsm_models import ToyPairConditionedTimeScoreNet

    theta = np.eye(3, dtype=np.float64)
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    model = ToyPairConditionedTimeScoreNet(dim=2, hidden_dim=8, theta_dim=3)
    est = HMatrixEstimator(
        model_post=model,
        model_prior=None,
        sigma_eval=1e-5,
        field_method="ctsm_v",
        device=torch.device("cpu"),
        pair_batch_size=64,
        ctsm_int_n_time=3,
    )
    got = est.compute_ctsm_delta_l_matrix(theta, x)
    assert got.shape == (3, 3)
    assert np.allclose(np.diag(got), 0.0)


def test_lda_ctsm_v_dispatch_transforms_x_before_ctsm(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray] = {}

    def fake_ctsm(*args, **kwargs):
        seen["x_train"] = np.asarray(kwargs["x_train"], dtype=np.float64)
        seen["x_val"] = np.asarray(kwargs["x_val"], dtype=np.float64)
        seen["x_all"] = np.asarray(kwargs["x_all"], dtype=np.float64)
        n = int(seen["x_all"].shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_ctsm)
    args = SimpleNamespace(lda_ctsm_shrinkage=1e-2, lda_ctsm_eps=1e-6, lda_ctsm_max_dim=1)
    theta = np.eye(2, dtype=np.float64)[[0, 0, 1, 1]]
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.1],
        ],
        dtype=np.float64,
    )
    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="lda_ctsm_v",
        theta_train=theta,
        x_train=x,
        theta_val=theta[:2],
        x_val=x[:2],
        theta_all=theta,
        x_all=x,
        bins_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bins_val=np.array([0, 0], dtype=np.int64),
        bins_all=np.array([0, 0, 1, 1], dtype=np.int64),
        k_cat=2,
    )

    assert seen["x_train"].shape == (4, 1)
    assert seen["x_val"].shape == (2, 1)
    assert seen["x_all"].shape == (4, 1)
    assert "lda_ctsm_weights" in result
    assert str(result["ctsm_theta_encoding"][0]) == "one_hot_lda_weighted_x"


def test_pls_ctsm_v_dispatch_transforms_x_before_ctsm(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray] = {}

    def fake_ctsm(*args, **kwargs):
        seen["x_train"] = np.asarray(kwargs["x_train"], dtype=np.float64)
        seen["x_val"] = np.asarray(kwargs["x_val"], dtype=np.float64)
        seen["x_all"] = np.asarray(kwargs["x_all"], dtype=np.float64)
        n = int(seen["x_all"].shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_ctsm)
    args = SimpleNamespace(
        pls_ctsm_eps=1e-6,
        pls_ctsm_max_dim=1,
        pls_ctsm_scale_x=False,
        pls_ctsm_max_iter=100,
        pls_ctsm_tol=1e-6,
    )
    theta = np.eye(2, dtype=np.float64)[[0, 0, 1, 1, 1]]
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.1],
            [2.4, -0.2],
        ],
        dtype=np.float64,
    )
    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="pls_ctsm_v",
        theta_train=theta,
        x_train=x,
        theta_val=theta[:2],
        x_val=x[:2],
        theta_all=theta,
        x_all=x,
        bins_train=np.array([0, 0, 1, 1, 1], dtype=np.int64),
        bins_val=np.array([0, 0], dtype=np.int64),
        bins_all=np.array([0, 0, 1, 1, 1], dtype=np.int64),
        k_cat=2,
    )

    assert seen["x_train"].shape == (5, 1)
    assert seen["x_val"].shape == (2, 1)
    assert seen["x_all"].shape == (5, 1)
    assert "pls_ctsm_weights" in result
    assert str(result["ctsm_theta_encoding"][0]) == "one_hot_pls_weighted_x"


def test_pca_ctsm_v_dispatch_transforms_x_before_ctsm(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray] = {}

    def fake_ctsm(*args, **kwargs):
        seen["x_train"] = np.asarray(kwargs["x_train"], dtype=np.float64)
        seen["x_val"] = np.asarray(kwargs["x_val"], dtype=np.float64)
        seen["x_all"] = np.asarray(kwargs["x_all"], dtype=np.float64)
        n = int(seen["x_all"].shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_ctsm)
    args = SimpleNamespace(
        pca_ctsm_eps=1e-6,
        pca_ctsm_max_dim=1,
        pca_ctsm_scale_x=False,
    )
    theta = np.eye(2, dtype=np.float64)[[0, 0, 1, 1]]
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.1],
        ],
        dtype=np.float64,
    )
    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="pca_ctsm_v",
        theta_train=theta,
        x_train=x,
        theta_val=theta[:2],
        x_val=x[:2],
        theta_all=theta,
        x_all=x,
        bins_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bins_val=np.array([0, 0], dtype=np.int64),
        bins_all=np.array([0, 0, 1, 1], dtype=np.int64),
        k_cat=2,
    )

    assert seen["x_train"].shape == (4, 1)
    assert seen["x_val"].shape == (2, 1)
    assert seen["x_all"].shape == (4, 1)
    assert "pca_ctsm_weights" in result
    assert str(result["ctsm_theta_encoding"][0]) == "one_hot_pca_weighted_x"


def test_theta_flow_categorical_helper_shape_symmetry_zero_diag(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    def fake_train(*args, **kwargs):
        return {"train_losses": [1.0, 0.5], "val_losses": [1.2, 0.7], "val_monitor_losses": [1.2, 0.8]}

    class FakeEstimator:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, theta, x, *, restore_original_order=False):
            labels = np.asarray(theta, dtype=np.int64).reshape(-1)
            h = np.zeros((labels.size, labels.size), dtype=np.float64)
            h[labels[:, None] != labels[None, :]] = 0.25
            return SimpleNamespace(h_directed=h)

    monkeypatch.setattr(cat, "_build_theta_flow_post_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cat, "_build_theta_flow_prior_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cat, "train_conditional_theta_flow_model", fake_train)
    monkeypatch.setattr(cat, "train_prior_theta_flow_model", fake_train)
    monkeypatch.setattr(cat, "HMatrixEstimator", FakeEstimator)
    args = SimpleNamespace(
        clf_min_class_count=2,
        flow_epochs=2,
        flow_batch_size=4,
        flow_lr=1e-3,
        flow_early_patience=10,
        flow_early_min_delta=0.0,
        flow_early_ema_alpha=0.5,
        flow_restore_best=True,
        flow_scheduler="cosine",
        flow_fm_t_eps=0.05,
        flow_likelihood_finetune_epochs=0,
        theta_flow_posterior_only_likelihood=False,
        prior_epochs=2,
        prior_batch_size=4,
        prior_lr=1e-3,
        prior_early_patience=10,
        prior_early_min_delta=0.0,
        prior_early_ema_alpha=0.5,
        prior_restore_best=True,
        h_batch_size=64,
        flow_ode_steps=8,
        flow_likelihood_exact_divergence=False,
        log_every=100,
    )
    x_train = np.arange(18, dtype=np.float64).reshape(9, 2)
    bins_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    x_val = np.arange(12, dtype=np.float64).reshape(6, 2)
    bins_val = np.array([0, 0, 1, 1, 2, 2])
    result = theta_flow_categorical_hellinger_sqrt(
        args,
        dev=SimpleNamespace(),
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        bins_val=bins_val,
        x_all=x_train,
        bins_all=bins_train,
        k_cat=3,
    )
    h = result["h_sqrt"]
    assert h.shape == (3, 3)
    assert np.allclose(h, h.T, equal_nan=True)
    assert np.allclose(np.diag(h), 0.0)
    assert np.allclose(h[np.triu_indices(3, 1)], 0.5)
    assert int(result["theta_flow_cate_num_valid_pairs"]) == 3


def test_theta_flow_categorical_degeneracy_skips_low_train_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    monkeypatch.setattr(cat, "_build_theta_flow_post_model", lambda *args, **kwargs: pytest.fail("should skip"))
    args = SimpleNamespace(clf_min_class_count=3)
    result = theta_flow_categorical_hellinger_sqrt(
        args,
        dev=SimpleNamespace(),
        x_train=np.zeros((3, 2)),
        bins_train=np.array([0, 0, 1]),
        x_val=np.zeros((2, 2)),
        bins_val=np.array([0, 1]),
        x_all=np.zeros((3, 2)),
        bins_all=np.array([0, 0, 1]),
        k_cat=2,
    )
    h = result["h_sqrt"]
    assert np.isnan(h[0, 1]) and np.isnan(h[1, 0])
    assert int(result["theta_flow_cate_num_valid_pairs"]) == 0
    assert int(result["theta_flow_cate_num_skipped_pairs"]) == 1


def test_cli_defaults_from_parser() -> None:
    p = build_parser()
    ns = p.parse_args([])
    assert int(ns.num_categories) == 2
    assert ns.n_list == "80,200,400,600"
    assert int(ns.n_ref) == 10000
    assert str(ns.device) == "cuda"


def test_main_rejects_num_categories_lt2() -> None:
    with pytest.raises(ValueError, match="num-categories"):
        main(["--num-categories", "1", "--device", "cpu"])


def test_visualization_only_cached_shapes(tmp_path: Path) -> None:
    k = 3
    methods = ["binary_classifier", "theta_flow_cate"]
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
        eval_split=np.asarray(["all"], dtype=object),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    npz_path.rename(out_dir / "h_decoding_categorical_twofig_results.npz")
    for method in methods:
        loss_root = out_dir / "training_losses" / method
        loss_root.mkdir(parents=True)
        for n in ns:
            _save_method_training_loss_npz(
                loss_root / f"n_{int(n):06d}.npz",
                method_name=method,
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
        "binary_classifier,theta-flow-cate",
        "--device",
        "cpu",
    ]
    main(argv)
    assert (out_dir / "h_decoding_categorical_twofig_sweep.svg").is_file()
    assert (out_dir / "h_decoding_categorical_twofig_corr_nmse.svg").is_file()
    assert (out_dir / "h_decoding_categorical_twofig_training_losses_panel.svg").is_file()
    all_columns = out_dir / "h_decoding_categorical_twofig_all_columns.png"
    assert all_columns.is_file()
    summary = (out_dir / "h_decoding_categorical_twofig_summary.txt").read_text(encoding="utf-8")
    assert f"h_decoding_categorical_twofig_all_columns.png: {all_columns.resolve()}" in summary


def test_build_parser_eval_split() -> None:
    p = build_parser()
    assert p.parse_args([]).eval_split == "all"
    assert p.parse_args(["--eval-split", "validation", "--device", "cpu"]).eval_split == "validation"
    with pytest.raises(SystemExit):
        p.parse_args(["--hellinger-eval-split", "validation", "--device", "cpu"])
    with pytest.raises(SystemExit):
        p.parse_args(["--binary-classifier-h-on-full-pool"])


def test_visualization_only_cache_eval_split_validation() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="validation",
    )
    cached = {
        "n": np.asarray([20, 40], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
        "eval_split": np.asarray(["validation"], dtype=object),
    }
    _validate_cached_cli(args, cached, ["binary_classifier"], [20, 40])


def test_visualization_only_cache_legacy_all_only() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="all",
    )
    cached = {
        "n": np.asarray([20], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
    }
    _validate_cached_cli(args, cached, ["binary_classifier"], [20])
    cached["hellinger_eval_split"] = np.asarray(["all"], dtype=object)
    _validate_cached_cli(args, cached, ["binary_classifier"], [20])


def test_visualization_only_cache_rejects_legacy_validation() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="all",
    )
    cached = {
        "n": np.asarray([20], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
        "hellinger_eval_split": np.asarray(["validation"], dtype=object),
    }
    with pytest.raises(ValueError, match="legacy hellinger_eval_split"):
        _validate_cached_cli(args, cached, ["binary_classifier"], [20])


def test_visualization_only_cache_rejects_requested_validation_against_legacy() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="validation",
    )
    cached = {
        "n": np.asarray([20], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
    }
    with pytest.raises(ValueError, match="no eval_split metadata"):
        _validate_cached_cli(args, cached, ["binary_classifier"], [20])


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


def test_eval_split_validation_controls_classifier_and_decoding_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    theta = np.eye(2, dtype=np.float64)[np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)]
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    meta = {"dataset_family": "random_mog_categorical", "theta_type": "categorical", "num_categories": 2, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=x[:3],
        theta_validation=theta[3:],
        x_validation=x[3:],
    )
    seen: dict[str, list] = {"method": [], "llr": [], "decode": []}

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    def fake_pairwise(**kwargs):
        x_eval = kwargs.get("decode_x_all")
        bins_eval = kwargs.get("decode_bin_all")
        if x_eval is None:
            x_eval = kwargs["subset"].bundle.x_all
        if bins_eval is None:
            bins_eval = kwargs["subset"].bin_all
        seen["decode"].append(
            (
                len(kwargs["subset"].bundle.x_train),
                len(np.asarray(x_eval)),
                tuple(np.asarray(bins_eval, dtype=np.int64).tolist()),
            )
        )
        return np.array([[np.nan, 0.5], [0.5, np.nan]])

    monkeypatch.setattr(cat.conv, "_pairwise_clf_from_bundle", fake_pairwise)
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    def fake_train_one_method(*args, **kwargs):
        seen["method"].append((len(kwargs["x_all"]), tuple(kwargs["bins_all"].tolist())))
        return {"delta_l": np.zeros((len(kwargs["x_all"]), len(kwargs["x_all"])), dtype=np.float64)}

    monkeypatch.setattr(cat, "_train_one_method", fake_train_one_method)

    dbg = cat._debug_categorical_module()
    orig_llr = dbg._llr_comparison_metrics

    def fake_llr(delta, true_delta):
        seen["llr"].append((int(np.asarray(delta).shape[0]), tuple(np.asarray(true_delta).shape)))
        return orig_llr(delta, true_delta)

    monkeypatch.setattr(dbg, "_llr_comparison_metrics", fake_llr)

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "binary_classifier",
            "--eval-split",
            "validation",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    assert len(seen["method"]) == 1
    assert len(seen["decode"]) == 2
    assert seen["decode"][-1][0] == 3
    assert seen["decode"][-1][1] == 3
    assert seen["method"][0][0] == 3
    assert seen["decode"][-1][2] == seen["method"][0][1]
    assert any(shape == (3, 3) for _, shape in seen["llr"])

    with np.load(tmp_path / "out" / "h_decoding_categorical_twofig_results.npz", allow_pickle=True) as z:
        assert str(z["eval_split"][0]) == "validation"
        assert "hellinger_eval_split" not in z.files
        assert "classifier_llr_h_eval_pool" not in z.files
        assert "binary_classifier_h_on_full_pool" not in z.files
        assert "decode_hellinger_ref_sqrt" not in z.files
        assert "decode_hellinger_sweep_sqrt" not in z.files
        assert "corr_decode_hellinger_vs_gt" not in z.files
        assert "nmse_decode_hellinger_vs_gt" not in z.files


def test_eval_split_all_controls_classifier_and_decoding_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    theta = np.eye(2, dtype=np.float64)[np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)]
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    meta = {"dataset_family": "random_mog_categorical", "theta_type": "categorical", "num_categories": 2, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=x[:3],
        theta_validation=theta[3:],
        x_validation=x[3:],
    )
    seen: dict[str, list] = {"method": [], "decode": []}

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))
    def fake_pairwise(**kwargs):
        x_eval = kwargs.get("decode_x_all")
        bins_eval = kwargs.get("decode_bin_all")
        if x_eval is None:
            x_eval = kwargs["subset"].bundle.x_all
        if bins_eval is None:
            bins_eval = kwargs["subset"].bin_all
        seen["decode"].append(
            (
                len(kwargs["subset"].bundle.x_train),
                len(np.asarray(x_eval)),
                tuple(np.asarray(bins_eval, dtype=np.int64).tolist()),
            )
        )
        return np.array([[np.nan, 0.5], [0.5, np.nan]])

    monkeypatch.setattr(cat.conv, "_pairwise_clf_from_bundle", fake_pairwise)
    monkeypatch.setattr(
        cat,
        "_train_one_method",
        lambda *args, **kwargs: seen["method"].append((len(kwargs["x_all"]), tuple(kwargs["bins_all"].tolist())))
        or {"delta_l": np.zeros((len(kwargs["x_all"]), len(kwargs["x_all"])), dtype=np.float64)},
    )

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "binary_classifier",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    assert seen["decode"][-1][0] == 3
    assert seen["decode"][-1][1] == 6
    assert seen["method"][0][0] == 6


def test_pr_project_classifier_rows_use_work_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    theta = np.eye(2, dtype=np.float64)[labels]
    native_x = 1000.0 + np.arange(12, dtype=np.float64).reshape(6, 2)
    work_x = np.arange(24, dtype=np.float64).reshape(6, 4)
    native_meta = {
        "dataset_family": "random_mog_categorical",
        "theta_type": "categorical",
        "num_categories": 2,
        "train_frac": 0.5,
    }
    work_meta = dict(native_meta)
    work_meta.update({"pr_autoencoder_embedded": True, "pr_autoencoder_z_dim": 2})
    native_bundle = SharedDatasetBundle(
        meta=native_meta,
        theta_all=theta,
        x_all=native_x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=native_x[:3],
        theta_validation=theta[3:],
        x_validation=native_x[3:],
    )
    work_bundle = SharedDatasetBundle(
        meta=work_meta,
        theta_all=theta,
        x_all=work_x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=work_x[:3],
        theta_validation=theta[3:],
        x_validation=work_x[3:],
    )
    calls = {"load": 0, "decode": [], "method": []}

    def fake_load(path):
        calls["load"] += 1
        return native_bundle if calls["load"] == 1 else work_bundle

    def record_shape(arr: np.ndarray) -> tuple[int, int, float]:
        a = np.asarray(arr, dtype=np.float64)
        return (int(a.shape[0]), int(a.shape[1]), float(np.max(a)))

    def fake_pairwise(**kwargs):
        calls["decode"].append((record_shape(kwargs["decode_x_train"]), record_shape(kwargs["decode_x_all"])))
        return np.array([[np.nan, 0.5], [0.5, np.nan]])

    def fake_train_one_method(*args, **kwargs):
        calls["method"].append((record_shape(kwargs["x_train"]), record_shape(kwargs["x_all"])))
        return {"delta_l": np.zeros((len(kwargs["x_all"]), len(kwargs["x_all"])), dtype=np.float64)}

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "_ensure_pr_projected_npz", lambda *args, **kwargs: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", fake_load)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    monkeypatch.setattr(cat.conv, "_pairwise_clf_from_bundle", fake_pairwise)
    monkeypatch.setattr(cat, "_train_one_method", fake_train_one_method)
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "native.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--pr-project",
            "--pr-output-npz",
            str(tmp_path / "work.npz"),
            "--decode-source-npz",
            str(tmp_path / "missing-source.npz"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "binary_classifier",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    assert len(calls["decode"]) == 2
    assert len(calls["method"]) == 1
    for group in ("decode", "method"):
        for train_shape, eval_shape in calls[group]:
            assert train_shape[1] == 4
            assert eval_shape[1] == 4
            assert train_shape[2] < 1000.0
            assert eval_shape[2] < 1000.0


def test_binary_classifier_method_still_writes_method_row_on_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    rng = np.random.default_rng(0)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 20)
    theta = np.eye(2, dtype=np.float64)[labels]
    x = np.empty((labels.size, 2), dtype=np.float64)
    x[labels == 0] = rng.normal(loc=-1.0, scale=0.2, size=(int(np.sum(labels == 0)), 2))
    x[labels == 1] = rng.normal(loc=1.0, scale=0.2, size=(int(np.sum(labels == 1)), 2))
    meta = {"dataset_family": "random_mog_categorical", "theta_type": "categorical", "num_categories": 2, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(20, dtype=np.int64),
        validation_idx=np.arange(20, 40, dtype=np.int64),
        theta_train=theta[:20],
        x_train=x[:20],
        theta_validation=theta[20:],
        x_validation=x[20:],
    )

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "40",
            "--n-list",
            "40",
            "--methods",
            "binary_classifier",
            "--clf-min-class-count",
            "2",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    with np.load(tmp_path / "out" / "h_decoding_categorical_twofig_results.npz", allow_pickle=True) as z:
        methods = [str(x) for x in z["method_names"]]
        i = methods.index("binary_classifier")
        h = np.asarray(z["h_sqrt_sweep"][i, 0], dtype=np.float64)
        assert h.shape == (2, 2)
        assert np.allclose(np.diag(h), 0.0)
        assert np.isfinite(h[np.triu_indices(2, 1)]).all()
        assert "h_sq_lr_affinity_sweep" not in z.files
        assert "corr_h_lr_affinity_vs_gt" not in z.files
        assert "nmse_h_lr_affinity_vs_gt" not in z.files
        assert "decode_hellinger_ref_sqrt" not in z.files
        assert "decode_hellinger_sweep_sqrt" not in z.files
        assert "corr_decode_hellinger_vs_gt" not in z.files
        assert "nmse_decode_hellinger_vs_gt" not in z.files


def test_pairwise_clf_from_bundle_decode_bin_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    subset = SimpleNamespace(
        bundle=SimpleNamespace(
            x_train=np.zeros((4, 2), dtype=np.float64),
            x_all=np.zeros((6, 2), dtype=np.float64),
        ),
        bin_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bin_all=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    seen = {}

    def fake_accuracy(x_tr, bin_tr, x_ev, bin_ev, n_bins, **kwargs):
        seen["x_ev_rows"] = int(np.asarray(x_ev).shape[0])
        seen["bin_ev"] = tuple(np.asarray(bin_ev, dtype=np.int64).tolist())
        return np.array([[np.nan, 0.75], [0.75, np.nan]]), None, None, None

    monkeypatch.setattr(conv.vhb, "pairwise_bin_logistic_accuracy_train_val", fake_accuracy)
    got = conv._pairwise_clf_from_bundle(
        args=SimpleNamespace(),
        meta={},
        subset=subset,
        output_dir=str(tmp_path),
        n_bins=2,
        clf_min_class_count=2,
        clf_random_state=0,
        decode_x_all=np.ones((2, 2), dtype=np.float64),
        decode_bin_all=np.array([0, 1], dtype=np.int64),
    )
    assert got[0, 1] == pytest.approx(0.75)
    assert seen == {"x_ev_rows": 2, "bin_ev": (0, 1)}


def test_pairwise_clf_from_bundle_decode_bin_override_row_mismatch(tmp_path: Path) -> None:
    subset = SimpleNamespace(
        bundle=SimpleNamespace(
            x_train=np.zeros((4, 2), dtype=np.float64),
            x_all=np.zeros((6, 2), dtype=np.float64),
        ),
        bin_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bin_all=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="decode_x_all rows 3 != decode_bin_all 2"):
        conv._pairwise_clf_from_bundle(
            args=SimpleNamespace(),
            meta={},
            subset=subset,
            output_dir=str(tmp_path),
            n_bins=2,
            clf_min_class_count=2,
            clf_random_state=0,
            decode_x_all=np.ones((3, 2), dtype=np.float64),
            decode_bin_all=np.array([0, 1], dtype=np.int64),
        )
