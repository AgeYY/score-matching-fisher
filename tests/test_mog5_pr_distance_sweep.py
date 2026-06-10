from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fisher.dataset_visualization import plot_mog5_native_scatter_covariance
from fisher.shared_dataset_io import save_shared_dataset_npz

ALL_METRICS = (
    "squared_euclidean",
    "cosine",
    "correlation",
    "mahalanobis_sq",
    "symmetric_kl",
)

def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "compare_mog5_pr_distance_sweeps.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distance_sweeps", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_case_npz(path: Path, *, offset: float = 0.0, metrics: tuple[str, ...] = ALL_METRICS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = np.asarray(metrics)
    labels = np.asarray(["category_0", "category_1", "category_2"])
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    classical = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    flow = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    gt = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    for metric_idx in range(len(metrics)):
        for i, j in pairs:
            gt_val = offset + 10.0 + metric_idx
            classical_val = gt_val + float(i + 1)
            flow_val = gt_val - float(j + 2)
            gt[metric_idx, i, j] = gt[metric_idx, j, i] = gt_val
            classical[metric_idx, i, j] = classical[metric_idx, j, i] = classical_val
            flow[metric_idx, i, j] = flow[metric_idx, j, i] = flow_val
    np.savez_compressed(
        path,
        metric_names=metric_names,
        condition_labels=labels,
        pair_indices=pairs,
        classical_matrices=classical,
        flow_matching_matrices=flow,
        ground_truth_matrices=gt,
    )
    return path


def _write_flow_loss_npz(path: Path, *, scale: float = 1.0, include_val_monitor: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_losses = np.asarray([1.0, 0.65, 0.42, 0.35], dtype=np.float64) * float(scale)
    val_losses = np.asarray([1.1, 0.7, 0.5, 0.55], dtype=np.float64) * float(scale)
    val_monitor_losses = np.asarray([1.1, 0.82, 0.66, 0.60], dtype=np.float64) * float(scale)
    payload = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": np.asarray([3], dtype=np.int64),
        "stopped_epoch": np.asarray([4], dtype=np.int64),
        "stopped_early": np.asarray([True]),
    }
    if include_val_monitor:
        payload["val_monitor_losses"] = val_monitor_losses
    np.savez_compressed(
        path,
        **payload,
    )
    return path


def _write_native_mog5_npz(path: Path, *, n_total: int = 25, x_dim: int = 2) -> Path:
    k = 5
    x_dim = int(x_dim)
    labels = np.arange(n_total, dtype=np.int64) % k
    theta_all = np.eye(k, dtype=np.float64)[labels]
    means2 = np.asarray(
        [
            [-2.0, -1.0],
            [0.0, -1.4],
            [2.0, -0.8],
            [-1.0, 1.3],
            [1.4, 1.2],
        ],
        dtype=np.float64,
    )
    variances2 = np.asarray(
        [
            [0.20, 0.35],
            [0.30, 0.25],
            [0.28, 0.40],
            [0.35, 0.22],
            [0.24, 0.32],
        ],
        dtype=np.float64,
    )
    means = means2
    variances = variances2
    if x_dim > 2:
        extra = np.arange(1, x_dim - 1, dtype=np.float64)
        means = np.column_stack([means2, 0.1 * means2[:, :1] + extra])
        variances = np.column_stack([variances2, np.tile(0.18 + 0.02 * extra, (k, 1))])
    offsets = np.column_stack(
        [
            0.05 * np.sin(np.arange(n_total, dtype=np.float64)),
            0.05 * np.cos(np.arange(n_total, dtype=np.float64)),
        ]
    )
    if x_dim > 2:
        extra_offsets = np.column_stack(
            [0.01 * (j + 1) * np.sin(np.arange(n_total, dtype=np.float64) + j) for j in range(x_dim - 2)]
        )
        offsets = np.column_stack([offsets, extra_offsets])
    x_all = means[labels] + offsets
    train_idx = np.arange(n_total // 2, dtype=np.int64)
    validation_idx = np.arange(n_total // 2, n_total, dtype=np.int64)
    save_shared_dataset_npz(
        path,
        meta={
            "dataset_family": "random_mog_categorical",
            "theta_type": "categorical",
            "theta_encoding": "one_hot",
            "num_categories": k,
            "x_dim": x_dim,
            "n_total": n_total,
            "mog_component_means": means.tolist(),
            "mog_component_variances": variances.tolist(),
        },
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )
    return path


def _write_zero_gt_case_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = np.asarray(["squared_euclidean"])
    labels = np.asarray(["category_0", "category_1"])
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    classical = np.asarray([[[0.0, 2.0], [2.0, 0.0]]], dtype=np.float64)
    flow = np.asarray([[[0.0, -3.0], [-3.0, 0.0]]], dtype=np.float64)
    gt = np.zeros((1, 2, 2), dtype=np.float64)
    np.savez_compressed(
        path,
        metric_names=metrics,
        condition_labels=labels,
        pair_indices=pairs,
        classical_matrices=classical,
        flow_matching_matrices=flow,
        ground_truth_matrices=gt,
    )
    return path


def _fake_ground_truth_payload(metrics: tuple[str, ...] = ALL_METRICS) -> dict[str, object]:
    labels = ("category_0", "category_1", "category_2")
    matrices = np.zeros((len(metrics), len(labels), len(labels)), dtype=np.float64)
    for metric_idx in range(len(metrics)):
        values = np.asarray(
            [
                [0.0, 1.0 + metric_idx, 2.0 + metric_idx],
                [1.0 + metric_idx, 0.0, 3.0 + metric_idx],
                [2.0 + metric_idx, 3.0 + metric_idx, 0.0],
            ],
            dtype=np.float64,
        )
        matrices[metric_idx] = values
    return {
        "metric_names": tuple(metrics),
        "condition_labels": labels,
        "ground_truth_matrices": matrices,
        "n_total": 1000,
        "pr_dim": 2,
        "pr_projected": True,
        "pr_dim_label": "pr2",
        "native_npz": "random_mog_categorical.npz",
    }


def _stub_baseline_ground_truth(monkeypatch: pytest.MonkeyPatch, mod) -> None:
    def fake_baseline(args, metrics):
        return _fake_ground_truth_payload(tuple(str(metric) for metric in metrics))

    monkeypatch.setattr(mod, "compute_baseline_ground_truth_rdms", fake_baseline)


def test_parser_defaults() -> None:
    mod = _load_cli_module()
    repo_root = Path(__file__).resolve().parent.parent
    args = mod.build_parser().parse_args([])

    assert args.n_list == [100, 550, 1000, 1550]
    assert args.native_x_dim == 3
    assert args.pr_dim is None
    assert not hasattr(args, "pr_dim_list")
    assert args.n_total == 100000
    assert args.device == "cuda"
    assert args.metric == "all"
    assert mod.resolve_metric_names(args) == ALL_METRICS
    assert args.early_ema_alpha == pytest.approx(0.05)
    assert args.yscale == "linear"
    assert args.loss_yscale == "linear"
    assert args.n_repeats == 5
    assert args.output_dir == repo_root / "data" / "mog5_native_xdim3_distance_sweeps"

    native2_args = mod.build_parser().parse_args(["--native-x-dim", "2", "--pr-dim", "none"])
    assert native2_args.output_dir == repo_root / "data" / "mog5_native_distance_sweeps"

    pr2_args = mod.build_parser().parse_args(["--native-x-dim", "2", "--pr-dim", "2"])
    assert pr2_args.pr_dim == 2
    assert pr2_args.output_dir == repo_root / "data" / "mog5_pr_distance_sweeps"

    native3_args = mod.build_parser().parse_args(["--native-x-dim", "3", "--pr-dim", "none"])
    assert native3_args.output_dir == repo_root / "data" / "mog5_native_xdim3_distance_sweeps"

    native3_pr_args = mod.build_parser().parse_args(["--native-x-dim", "3", "--pr-dim", "5"])
    assert native3_pr_args.output_dir == repo_root / "data" / "mog5_native_xdim3_pr5_distance_sweeps"

    invalid_args = mod.build_parser().parse_args(["--native-x-dim", "3", "--pr-dim", "2"])
    with pytest.raises(ValueError, match="--pr-dim must be >= native x_dim=3"):
        mod.validate_args(invalid_args)


def test_aggregate_mean_pairwise_abs_errors(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100"])
    data = {
        (100, -1): mod._load_case_cache(_write_case_npz(tmp_path / "case_a.npz", offset=0.0)),
    }

    aggregate, rows = mod.aggregate_sweeps(args=args, case_data=data)

    assert aggregate["n_sweep_classical_matrices"].shape == (1, 5, 3, 3)
    assert "pr_dim_list" not in aggregate
    assert "pr_dim_sweep_classical_matrices" not in aggregate
    assert {r["axis"] for r in rows} == {"n_total"}
    n_rows = [r for r in rows if r["axis"] == "n_total" and r["metric"] == "squared_euclidean"]
    classical_errors = [r["abs_error"] for r in n_rows if r["estimator"] == "classical"]
    flow_errors = [r["abs_error"] for r in n_rows if r["estimator"] == "flow_matching"]
    classical_rel_errors = [r["rel_error"] for r in n_rows if r["estimator"] == "classical"]
    flow_rel_errors = [r["rel_error"] for r in n_rows if r["estimator"] == "flow_matching"]
    assert np.mean(classical_errors) == pytest.approx((1.0 + 1.0 + 2.0) / 3.0)
    assert np.mean(flow_errors) == pytest.approx((3.0 + 4.0 + 4.0) / 3.0)
    assert np.mean(classical_rel_errors) == pytest.approx(((1.0 / 10.0) + (1.0 / 10.0) + (2.0 / 10.0)) / 3.0)
    assert np.mean(flow_rel_errors) == pytest.approx(((3.0 / 10.0) + (4.0 / 10.0) + (4.0 / 10.0)) / 3.0)


def test_repeat_case_paths_and_single_case_args(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100", "--n-repeats", "5", "--seed", "11"])

    cases = mod._unique_cases(args)
    assert len(cases) == 5
    assert [case[2] for case in cases] == [0, 1, 2, 3, 4]
    assert [mod._repeat_seed(args, case[2]) for case in cases] == [11, 12, 13, 14, 15]

    repeat_dir = mod.case_output_dir(
        n_total=100,
        pr_dim=args.pr_dim,
        case_output_name="distance_comparison_flow_skl",
        native_x_dim=3,
        repeat_idx=3,
        n_repeats=5,
    )
    assert repeat_dir.parts[-2:] == ("repeat_03", "distance_comparison_flow_skl")

    single_dir = mod.case_output_dir(
        n_total=100,
        pr_dim=args.pr_dim,
        case_output_name="distance_comparison_flow_skl",
        native_x_dim=3,
        repeat_idx=0,
        n_repeats=1,
    )
    assert single_dir.parts[-1] == "distance_comparison_flow_skl"
    assert "repeat_00" not in single_dir.parts

    case_args = mod._single_case_args(
        args,
        n_total=100,
        pr_dim=args.pr_dim,
        output_dir=tmp_path / "dataset" / "repeat_02" / "distance_comparison_flow_skl",
        repeat_idx=2,
        native_template_npz=tmp_path / "template.npz",
    )
    assert case_args.seed == 13
    assert case_args.dataset_dir == tmp_path / "dataset" / "repeat_02"
    assert case_args.native_template_npz == tmp_path / "template.npz"

    single_repeat_args = mod.build_parser().parse_args(["--n-list", "100", "--n-repeats", "1"])
    assert len(mod._unique_cases(single_repeat_args)) == 1


def test_native_aggregate_metadata_and_npz(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--native-x-dim", "2", "--pr-dim", "none", "--n-list", "100"])
    data = {
        (100, -1): mod._load_case_cache(_write_case_npz(tmp_path / "case_native.npz", offset=0.0)),
    }

    aggregate, rows = mod.aggregate_sweeps(args=args, case_data=data)
    out = mod.write_aggregate_npz(tmp_path / "sweep.npz", aggregate)

    assert aggregate["pr_projected"] is False
    assert aggregate["pr_dim"] is None
    assert aggregate["pr_dim_label"] == "native"
    assert aggregate["native_x_dim"] == 2
    assert rows[0]["pr_projected"] is False
    assert rows[0]["pr_dim"] is None
    assert rows[0]["pr_dim_label"] == "native"
    assert rows[0]["native_x_dim"] == 2
    with np.load(out, allow_pickle=False) as npz:
        np.testing.assert_array_equal(npz["pr_dim"], [-1])
        np.testing.assert_array_equal(npz["native_x_dim"], [2])
        np.testing.assert_array_equal(npz["pr_projected"], [False])
        assert tuple(str(v) for v in npz["pr_dim_label"].tolist()) == ("native",)


def test_repeat_aggregate_arrays_and_csv_rows(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100,200", "--n-repeats", "2", "--seed", "31"])
    case_data = {
        (100, -1, 0): mod._load_case_cache(_write_case_npz(tmp_path / "case_100_r0.npz", offset=0.0)),
        (100, -1, 1): mod._load_case_cache(_write_case_npz(tmp_path / "case_100_r1.npz", offset=2.0)),
        (200, -1, 0): mod._load_case_cache(_write_case_npz(tmp_path / "case_200_r0.npz", offset=4.0)),
        (200, -1, 1): mod._load_case_cache(_write_case_npz(tmp_path / "case_200_r1.npz", offset=6.0)),
    }

    aggregate, rows = mod.aggregate_sweeps(args=args, case_data=case_data)
    out = mod.write_aggregate_npz(tmp_path / "sweep.npz", aggregate)

    assert aggregate["n_repeat_classical_matrices"].shape == (2, 2, 5, 3, 3)
    np.testing.assert_allclose(
        aggregate["n_sweep_classical_matrices"],
        np.mean(aggregate["n_repeat_classical_matrices"], axis=1),
    )
    repeat_errors = mod._mean_pair_error_curve(
        aggregate["n_repeat_flow_matching_matrices"],
        aggregate["n_repeat_ground_truth_matrices"],
        aggregate["pair_indices"],
        metric_idx=0,
        relative=False,
    )
    assert repeat_errors.shape == (2, 2)
    np.testing.assert_allclose(repeat_errors, np.full((2, 2), 11.0 / 3.0))
    assert {row["repeat_idx"] for row in rows} == {0, 1}
    assert {row["repeat_seed"] for row in rows} == {31, 32}
    with np.load(out, allow_pickle=False) as npz:
        assert npz["n_repeat_classical_matrices"].shape == (2, 2, 5, 3, 3)
        np.testing.assert_array_equal(npz["repeat_seeds"], [31, 32])


def test_relative_error_uses_denominator_floor_for_zero_ground_truth(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100", "--metric", "squared_euclidean"])
    data = {
        (100, -1): mod._load_case_cache(_write_zero_gt_case_npz(tmp_path / "case_a.npz")),
    }

    _, rows = mod.aggregate_sweeps(args=args, case_data=data)

    n_rows = [r for r in rows if r["axis"] == "n_total"]
    classical = next(r for r in n_rows if r["estimator"] == "classical")
    flow = next(r for r in n_rows if r["estimator"] == "flow_matching")
    assert classical["rel_error"] == pytest.approx(2.0 / mod.REL_ERROR_DENOM_FLOOR)
    assert flow["rel_error"] == pytest.approx(3.0 / mod.REL_ERROR_DENOM_FLOOR)


def test_mean_pair_error_curve_returns_raw_curve_values(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100,200"])
    case_100 = mod._load_case_cache(_write_case_npz(tmp_path / "case_100.npz", offset=0.0))
    case_200 = mod._load_case_cache(_write_case_npz(tmp_path / "case_200.npz", offset=10.0))
    aggregate, _ = mod.aggregate_sweeps(args=args, case_data={(100, -1): case_100, (200, -1): case_200})

    yvals = mod._mean_pair_error_curve(
        aggregate["n_sweep_flow_matching_matrices"],
        aggregate["n_sweep_ground_truth_matrices"],
        aggregate["pair_indices"],
        metric_idx=0,
        relative=False,
    )

    np.testing.assert_allclose(yvals, [11.0 / 3.0, 11.0 / 3.0])


def test_plot_sweep_error_writes_per_metric_panel_grid(tmp_path: Path) -> None:
    mod = _load_cli_module()
    metrics = ("squared_euclidean", "cosine", "symmetric_kl")
    args = mod.build_parser().parse_args(["--n-list", "100,200"])
    case_100 = mod._load_case_cache(_write_case_npz(tmp_path / "case_100.npz", offset=0.0, metrics=metrics))
    case_200 = mod._load_case_cache(_write_case_npz(tmp_path / "case_200.npz", offset=5.0, metrics=metrics))
    aggregate, _ = mod.aggregate_sweeps(args=args, case_data={(100, -1): case_100, (200, -1): case_200})

    abs_svg, abs_png = mod.plot_sweep_error(
        aggregate,
        svg_path=tmp_path / "mog5_pr_distance_sweep_abs_error.svg",
        png_path=tmp_path / "mog5_pr_distance_sweep_abs_error.png",
        yscale="linear",
        relative=False,
    )
    rel_svg, rel_png = mod.plot_sweep_error(
        aggregate,
        svg_path=tmp_path / "mog5_pr_distance_sweep_rel_error.svg",
        png_path=tmp_path / "mog5_pr_distance_sweep_rel_error.png",
        yscale="log",
        relative=True,
    )

    for path in (abs_svg, abs_png, rel_svg, rel_png):
        assert path.is_file()
        assert path.stat().st_size > 0
    abs_text = abs_svg.read_text(encoding="utf-8")
    rel_text = rel_svg.read_text(encoding="utf-8")
    expected_description = "layout=2x3;metrics=squared_euclidean,cosine,symmetric_kl;rows=classical+flow,flow_only"
    assert expected_description in abs_text
    assert expected_description in rel_text
    assert "Mean absolute error" in abs_text
    assert "Mean relative absolute error" in rel_text


def test_load_flow_loss_cache_reads_val_monitor_losses(tmp_path: Path) -> None:
    mod = _load_cli_module()
    path = _write_flow_loss_npz(tmp_path / "flow_loss.npz", scale=2.0)

    out = mod._load_flow_loss_cache(path)

    np.testing.assert_allclose(out["train_losses"], [2.0, 1.3, 0.84, 0.7])
    np.testing.assert_allclose(out["val_losses"], [2.2, 1.4, 1.0, 1.1])
    np.testing.assert_allclose(out["val_monitor_losses"], [2.2, 1.64, 1.32, 1.2])
    assert out["best_epoch"] == 3
    assert out["stopped_epoch"] == 4
    assert out["stopped_early"] is True


def test_load_flow_loss_cache_allows_legacy_without_val_monitor_losses(tmp_path: Path) -> None:
    mod = _load_cli_module()
    path = _write_flow_loss_npz(tmp_path / "flow_loss.npz", include_val_monitor=False)

    out = mod._load_flow_loss_cache(path)

    assert "val_monitor_losses" not in out
    np.testing.assert_allclose(out["train_losses"], [1.0, 0.65, 0.42, 0.35])
    np.testing.assert_allclose(out["val_losses"], [1.1, 0.7, 0.5, 0.55])


def test_plot_mog5_native_scatter_covariance_writes_svg_and_png(tmp_path: Path) -> None:
    native_npz = _write_native_mog5_npz(tmp_path / "random_mog_categorical.npz")

    svg, png = plot_mog5_native_scatter_covariance(
        native_npz,
        svg_path=tmp_path / "figure.svg",
        png_path=tmp_path / "figure.png",
        max_points=12,
    )

    assert svg.is_file()
    assert png.is_file()
    assert svg.stat().st_size > 0
    assert png.stat().st_size > 0
    svg_text = svg.read_text(encoding="utf-8")
    assert "<image" not in svg_text


def test_plot_mog5_native_scatter_covariance_accepts_3d_x1_x2_view(tmp_path: Path) -> None:
    native_npz = _write_native_mog5_npz(tmp_path / "random_mog_categorical_3d.npz", x_dim=3)

    svg, png = plot_mog5_native_scatter_covariance(
        native_npz,
        svg_path=tmp_path / "figure3d.svg",
        png_path=tmp_path / "figure3d.png",
        max_points=12,
    )

    assert svg.is_file()
    assert png.is_file()
    assert svg.stat().st_size > 0
    assert png.stat().st_size > 0


def test_plot_ground_truth_rdms_writes_requested_metric_panels(tmp_path: Path) -> None:
    mod = _load_cli_module()
    ground_truth = _fake_ground_truth_payload(("cosine",))

    svg, png = mod.plot_ground_truth_rdms(
        ground_truth,
        svg_path=tmp_path / "gt_rdms.svg",
        png_path=tmp_path / "gt_rdms.png",
    )

    assert svg.is_file()
    assert png.is_file()
    assert svg.stat().st_size > 0
    assert png.stat().st_size > 0
    svg_text = svg.read_text(encoding="utf-8")
    assert "metrics=cosine" in svg_text
    assert "squared_euclidean" not in svg_text
    assert "correlation" not in svg_text


def test_plot_ground_truth_rdms_uses_canonical_metric_order(tmp_path: Path) -> None:
    mod = _load_cli_module()
    metrics = ("symmetric_kl", "squared_euclidean", "cosine", "mahalanobis_sq", "correlation")
    ground_truth = _fake_ground_truth_payload(metrics)

    svg, png = mod.plot_ground_truth_rdms(
        ground_truth,
        svg_path=tmp_path / "gt_rdms.svg",
        png_path=tmp_path / "gt_rdms.png",
    )

    assert svg.is_file()
    assert png.is_file()
    assert "metrics=correlation,cosine,squared_euclidean,mahalanobis_sq,symmetric_kl" in svg.read_text(
        encoding="utf-8"
    )


def test_cache_hits_do_not_rerun_and_duplicate_case_is_deduped(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    calls: list[object] = []
    real_single = mod._load_single_case_module()

    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_1000_None" / mod.RESULTS_NAME)

    class FakeSingle:
        @staticmethod
        def build_parser():
            return real_single.build_parser()

        @staticmethod
        def run(args):
            calls.append(args)
            out = Path(args.output_dir) / mod.RESULTS_NAME
            _write_case_npz(out)
            return {"results_npz": out}

    monkeypatch.setattr(mod, "_load_single_case_module", lambda: FakeSingle)
    args = mod.build_parser().parse_args(
        [
            "--n-list",
            "1000,2000,1000",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--n-repeats",
            "1",
            "--yscale",
            "linear",
        ]
    )
    mod.run(args)

    assert len(calls) == 1
    assert calls[0].n_total == 2000
    assert calls[0].pr_dim is None


def test_visualization_only_missing_cache_fails(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    args = mod.build_parser().parse_args(
        ["--visualization-only", "--n-list", "100", "--n-repeats", "1", "--output-dir", str(tmp_path / "sweep")]
    )

    with pytest.raises(FileNotFoundError, match="visualization-only"):
        mod.run(args)


def test_visualization_only_writes_outputs_from_fake_caches(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_None" / mod.RESULTS_NAME)
    _write_native_mog5_npz(tmp_path / "random_mog_categorical.npz")
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    outputs = mod.run(args)

    assert outputs["results_npz"].is_file()
    assert outputs["errors_csv"].is_file()
    assert outputs["figure_svg"].is_file()
    assert outputs["figure_png"].is_file()
    assert outputs["abs_error_figure_svg"].is_file()
    assert outputs["abs_error_figure_png"].is_file()
    assert outputs["rel_error_figure_svg"].is_file()
    assert outputs["rel_error_figure_png"].is_file()
    assert outputs["ground_truth_rdms_figure_svg"].is_file()
    assert outputs["ground_truth_rdms_figure_png"].is_file()
    assert outputs["dataset_figure_svg"].is_file()
    assert outputs["dataset_figure_png"].is_file()
    assert outputs["summary_json"].is_file()
    with np.load(outputs["results_npz"], allow_pickle=False) as data:
        assert data["n_sweep_classical_matrices"].shape == (1, 5, 3, 3)
        assert "pr_dim_list" not in data.files
        assert "pr_dim_sweep_classical_matrices" not in data.files
    with outputs["errors_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert "rel_error" in rows[0]
    assert {row["axis"] for row in rows} == {"n_total"}
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["config"]["abs_error_yscale"] == "linear"
    assert summary["config"]["rel_error_yscale"] == "linear"
    assert summary["config"]["metric"] == "all"
    assert summary["config"]["metrics"] == list(ALL_METRICS)
    assert "pr_dim_list" not in summary["config"]
    assert "abs_error_figure_svg" in summary["outputs"]
    assert "rel_error_figure_svg" in summary["outputs"]
    assert "ground_truth_rdms_figure_svg" in summary["outputs"]
    assert "ground_truth_rdms_figure_png" in summary["outputs"]
    assert "dataset_figure_svg" in summary["outputs"]
    assert "dataset_figure_png" in summary["outputs"]
    assert "flow_loss_figure_svg" not in summary["outputs"]


def test_skip_dataset_viz_omits_sweep_dataset_figure(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_None" / mod.RESULTS_NAME)
    _write_native_mog5_npz(tmp_path / "random_mog_categorical.npz")
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--skip-dataset-viz",
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    outputs = mod.run(args)

    assert "dataset_figure_svg" not in outputs
    assert "dataset_figure_png" not in outputs
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert "dataset_figure_svg" not in summary["outputs"]
    assert "dataset_figure_png" not in summary["outputs"]


def test_native_visualization_only_uses_native_cache_and_summary(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{mod._pr_dim_label(pr_dim)}",
    )
    _write_case_npz(tmp_path / "case_100_native" / mod.RESULTS_NAME)
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--pr-dim",
            "none",
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    outputs = mod.run(args)

    with np.load(outputs["results_npz"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["pr_dim"], [-1])
        np.testing.assert_array_equal(data["pr_projected"], [False])
        assert tuple(str(v) for v in data["pr_dim_label"].tolist()) == ("native",)
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["config"]["pr_projected"] is False
    assert summary["config"]["pr_dim"] is None
    assert summary["config"]["pr_dim_label"] == "native"
    assert "n100_repeat00_native" in summary["case_paths"]
    assert "n100_repeat00_native" in summary["cache_hits"]


def test_visualization_only_writes_flow_loss_outputs_from_fake_caches(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_None" / mod.RESULTS_NAME)
    _write_flow_loss_npz(tmp_path / "case_100_None" / "flow" / "symmetric_kl_flow_matching_skl_results.npz")
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--metric",
            "symmetric_kl",
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    outputs = mod.run(args)

    assert outputs["flow_loss_figure_svg"].is_file()
    assert outputs["flow_loss_figure_png"].is_file()
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["config"]["loss_yscale"] == "linear"
    assert Path(summary["outputs"]["flow_loss_figure_svg"]).is_file()
    assert Path(summary["outputs"]["flow_loss_figure_png"]).is_file()


def test_visualization_only_filters_cached_metric(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_None" / mod.RESULTS_NAME)
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--metric",
            "cosine",
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    outputs = mod.run(args)

    with np.load(outputs["results_npz"], allow_pickle=False) as data:
        assert tuple(str(v) for v in data["metric_names"].tolist()) == ("cosine",)
        assert data["n_sweep_classical_matrices"].shape == (1, 1, 3, 3)
    with outputs["errors_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["metric"] for row in rows} == {"cosine"}
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["config"]["metric"] == "cosine"
    assert summary["config"]["metrics"] == ["cosine"]


def test_visualization_only_requested_metric_missing_from_cache_fails(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _stub_baseline_ground_truth(monkeypatch, mod)
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=2, repeat_idx=0, n_repeats=1: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(
        tmp_path / "case_100_None" / mod.RESULTS_NAME,
        metrics=("squared_euclidean",),
    )
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--metric",
            "cosine",
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    with pytest.raises(ValueError, match="missing requested metric"):
        mod.run(args)
