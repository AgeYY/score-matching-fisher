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


def _write_native_mog5_npz(path: Path, *, n_total: int = 25) -> Path:
    k = 5
    labels = np.arange(n_total, dtype=np.int64) % k
    theta_all = np.eye(k, dtype=np.float64)[labels]
    means = np.asarray(
        [
            [-2.0, -1.0],
            [0.0, -1.4],
            [2.0, -0.8],
            [-1.0, 1.3],
            [1.4, 1.2],
        ],
        dtype=np.float64,
    )
    variances = np.asarray(
        [
            [0.20, 0.35],
            [0.30, 0.25],
            [0.28, 0.40],
            [0.35, 0.22],
            [0.24, 0.32],
        ],
        dtype=np.float64,
    )
    offsets = np.column_stack(
        [
            0.05 * np.sin(np.arange(n_total, dtype=np.float64)),
            0.05 * np.cos(np.arange(n_total, dtype=np.float64)),
        ]
    )
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
            "x_dim": 2,
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


def test_parser_defaults() -> None:
    mod = _load_cli_module()
    repo_root = Path(__file__).resolve().parent.parent
    args = mod.build_parser().parse_args([])

    assert args.n_list == [100, 550, 1000, 1550]
    assert args.pr_dim == 2
    assert not hasattr(args, "pr_dim_list")
    assert args.n_total == 1000
    assert args.device == "cuda"
    assert args.metric == "all"
    assert mod.resolve_metric_names(args) == ALL_METRICS
    assert args.early_ema_alpha == pytest.approx(0.05)
    assert args.yscale == "linear"
    assert args.loss_yscale == "linear"
    assert args.output_dir == repo_root / "data" / "mog5_pr_distance_sweeps"

    native_args = mod.build_parser().parse_args(["--pr-dim", "none"])
    assert native_args.pr_dim is None
    assert native_args.output_dir == repo_root / "data" / "mog5_native_distance_sweeps"


def test_aggregate_mean_pairwise_abs_errors(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100"])
    data = {
        (100, 2): mod._load_case_cache(_write_case_npz(tmp_path / "case_a.npz", offset=0.0)),
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


def test_native_aggregate_metadata_and_npz(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--pr-dim", "none", "--n-list", "100"])
    data = {
        (100, -1): mod._load_case_cache(_write_case_npz(tmp_path / "case_native.npz", offset=0.0)),
    }

    aggregate, rows = mod.aggregate_sweeps(args=args, case_data=data)
    out = mod.write_aggregate_npz(tmp_path / "sweep.npz", aggregate)

    assert aggregate["pr_projected"] is False
    assert aggregate["pr_dim"] is None
    assert aggregate["pr_dim_label"] == "native"
    assert rows[0]["pr_projected"] is False
    assert rows[0]["pr_dim"] is None
    assert rows[0]["pr_dim_label"] == "native"
    with np.load(out, allow_pickle=False) as npz:
        np.testing.assert_array_equal(npz["pr_dim"], [-1])
        np.testing.assert_array_equal(npz["pr_projected"], [False])
        assert tuple(str(v) for v in npz["pr_dim_label"].tolist()) == ("native",)


def test_relative_error_uses_denominator_floor_for_zero_ground_truth(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100", "--metric", "squared_euclidean"])
    data = {
        (100, 2): mod._load_case_cache(_write_zero_gt_case_npz(tmp_path / "case_a.npz")),
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
    aggregate, _ = mod.aggregate_sweeps(args=args, case_data={(100, 2): case_100, (200, 2): case_200})

    yvals = mod._mean_pair_error_curve(
        aggregate["n_sweep_flow_matching_matrices"],
        aggregate["n_sweep_ground_truth_matrices"],
        aggregate["pair_indices"],
        metric_idx=0,
        relative=False,
    )

    np.testing.assert_allclose(yvals, [11.0 / 3.0, 11.0 / 3.0])


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


def test_cache_hits_do_not_rerun_and_duplicate_case_is_deduped(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    calls: list[object] = []
    real_single = mod._load_single_case_module()

    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_1000_2" / mod.RESULTS_NAME)

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
            "--yscale",
            "linear",
        ]
    )
    mod.run(args)

    assert len(calls) == 1
    assert calls[0].n_total == 2000
    assert calls[0].pr_dim == 2


def test_visualization_only_missing_cache_fails(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    args = mod.build_parser().parse_args(["--visualization-only", "--n-list", "100", "--output-dir", str(tmp_path / "sweep")])

    with pytest.raises(FileNotFoundError, match="visualization-only"):
        mod.run(args)


def test_visualization_only_writes_outputs_from_fake_caches(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_2" / mod.RESULTS_NAME)
    _write_native_mog5_npz(tmp_path / "random_mog_categorical.npz")
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--n-list",
            "100",
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
    assert "dataset_figure_svg" in summary["outputs"]
    assert "dataset_figure_png" in summary["outputs"]
    assert "flow_loss_figure_svg" not in summary["outputs"]


def test_skip_dataset_viz_omits_sweep_dataset_figure(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_2" / mod.RESULTS_NAME)
    _write_native_mog5_npz(tmp_path / "random_mog_categorical.npz")
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--skip-dataset-viz",
            "--n-list",
            "100",
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
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{mod._pr_dim_label(pr_dim)}",
    )
    _write_case_npz(tmp_path / "case_100_native" / mod.RESULTS_NAME)
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--pr-dim",
            "none",
            "--n-list",
            "100",
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
    assert "n100_native" in summary["case_paths"]
    assert "n100_native" in summary["cache_hits"]


def test_visualization_only_writes_flow_loss_outputs_from_fake_caches(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_2" / mod.RESULTS_NAME)
    _write_flow_loss_npz(tmp_path / "case_100_2" / "flow" / "symmetric_kl_flow_matching_skl_results.npz")
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--metric",
            "symmetric_kl",
            "--n-list",
            "100",
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
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_2" / mod.RESULTS_NAME)
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--metric",
            "cosine",
            "--n-list",
            "100",
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
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(
        tmp_path / "case_100_2" / mod.RESULTS_NAME,
        metrics=("squared_euclidean",),
    )
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--metric",
            "cosine",
            "--n-list",
            "100",
            "--output-dir",
            str(tmp_path / "sweep"),
        ]
    )

    with pytest.raises(ValueError, match="missing requested metric"):
        mod.run(args)
