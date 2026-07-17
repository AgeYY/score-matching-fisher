from __future__ import annotations

import csv
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import (
    classical_linear_fisher,
    native_linear_fisher_curve,
    parse_pr_dim,
)
from fisher.data import ToyConditionalGaussianRandampSqrtdDataset
from fisher.shared_fisher_est import analytic_fisher_curve


def _load_parallel_module():
    path = _REPO_ROOT / "bin" / "compare_continuous_pr_fisher_sweeps_parallel.py"
    spec = importlib.util.spec_from_file_location("compare_continuous_pr_fisher_sweeps_parallel", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_pr_dim_accepts_none_tokens_and_ints() -> None:
    assert parse_pr_dim("none") is None
    assert parse_pr_dim("null") is None
    assert parse_pr_dim(30) == 30
    assert parse_pr_dim("6") == 6
    with pytest.raises(ValueError):
        parse_pr_dim("bad")


def test_native_linear_fisher_matches_analytic_mean_term() -> None:
    dataset = ToyConditionalGaussianRandampSqrtdDataset(
        theta_low=-2.0,
        theta_high=2.0,
        x_dim=4,
        seed=123,
        randamp_mu_amp_per_dim=np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
    )
    theta = np.linspace(-1.5, 1.5, 9, dtype=np.float64).reshape(-1, 1)
    got = native_linear_fisher_curve(theta, dataset)

    dmu = dataset.tuning_curve_derivative(theta)
    cov = dataset.covariance(theta)
    inv_cov = np.linalg.inv(cov)
    expected_mean = np.einsum("bi,bij,bj->b", dmu, inv_cov, dmu)
    full = analytic_fisher_curve(theta, dataset)

    np.testing.assert_allclose(got, expected_mean, rtol=1e-12, atol=1e-12)
    assert np.all(full >= got - 1e-12)


def test_classical_linear_fisher_endpoint_windows() -> None:
    theta_all = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64).reshape(-1, 1)
    x_all = np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 4.0]], dtype=np.float64)
    theta_grid = np.asarray([0.0, 1.0], dtype=np.float64).reshape(-1, 1)

    got = classical_linear_fisher(
        theta_all=theta_all,
        x_all=x_all,
        theta_grid=theta_grid,
        ridge=0.0,
        window_radius=1e-12,
        min_endpoint_samples=2,
    )

    np.testing.assert_allclose(got, np.asarray([9.0], dtype=np.float64), rtol=1e-12, atol=1e-12)


def test_parallel_parser_pr_dims_defaults_and_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_parallel_module()
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 12)
    args = mod.build_parser().parse_args(
        [
            "--n-list",
            "100,200",
            "--pr-dims",
            "none,30",
            "--n-repeats",
            "2",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--epochs",
            "7",
            "--batch-size",
            "32",
            "--gpu-ids",
            "2,3",
            "--jobs-per-gpu",
            "2",
        ]
    )

    assert args.n_list == [100, 200]
    assert args.pr_dims == [None, 30]
    default_args = mod.build_parser().parse_args([])
    assert default_args.native_x_dim == 4
    assert default_args.train_frac == pytest.approx(0.8)
    assert default_args.pr_dims == [None]
    assert default_args.gpu_ids == [0]
    assert default_args.device == "cuda:0"
    assert default_args.skip_dataset_viz is False
    assert default_args.composite_smoothing == "kernel"
    assert default_args.kernel_smooth_bandwidth_grid == pytest.approx(2.0)
    assert default_args.theta_embedding == "gaussian-rbf"
    assert default_args.theta_rbf_num_centers == 8
    assert default_args.theta_rbf_bandwidth is None
    skip_viz_args = mod.build_parser().parse_args(["--skip-dataset-viz"])
    assert skip_viz_args.skip_dataset_viz is True
    assert args.cpu_threads_per_job == 3
    tasks = mod.plan_cases(args)
    assert (tasks[0].n_total, tasks[0].pr_dim, tasks[0].repeat_idx, tasks[0].seed) == (100, None, 0, 7)
    assert (tasks[3].n_total, tasks[3].pr_dim, tasks[3].repeat_idx, tasks[3].seed) == (100, 30, 1, 8)

    command = mod.build_case_command(args, tasks[2])
    joined = " ".join(command)
    assert command[1].endswith("bin/compare_continuous_pr_fisher.py")
    assert "--dataset-dir" in command
    assert str(tasks[2].dataset_dir) in command
    assert "--output-dir" in command
    assert str(tasks[2].output_dir) in command
    assert "--pr-dim 30" in joined
    assert "--epochs 7" in joined
    assert "--batch-size 32" in joined
    assert "--theta-embedding gaussian-rbf" in joined
    assert "--theta-rbf-num-centers 8" in joined
    env = mod.build_case_env({"PATH": "/bin"}, gpu_id=3, cpu_threads_per_job=4)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OMP_NUM_THREADS"] == "4"


def test_parallel_repeat_seeds_are_base_seed_plus_repeat_idx(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        ["--n-list", "50", "--pr-dims", "none", "--n-repeats", "5", "--seed", "7", "--output-dir", str(tmp_path)]
    )

    tasks = mod.plan_cases(args)

    assert [task.repeat_idx for task in tasks] == [0, 1, 2, 3, 4]
    assert [task.seed for task in tasks] == [7, 8, 9, 10, 11]
    assert mod.repeat_seeds(args) == [7, 8, 9, 10, 11]


def test_parallel_one_repeat_paths_and_zero_errorbar(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        ["--n-list", "50", "--pr-dims", "none", "--n-repeats", "1", "--output-dir", str(tmp_path)]
    )
    task = mod.plan_cases(args)[0]

    assert task.dataset_dir == tmp_path / "n50" / "native"
    assert "repeat_00" not in task.dataset_dir.parts

    ns, means, sds = mod._errorbar_series(
        [
            {
                "n_total": 50,
                "pr_dim": "none",
                "repeat_idx": 0,
                "repeat_seed": 7,
                "method": "flow_full",
                "mae_abs_error": 0.25,
            }
        ],
        method="flow_full",
        pr_dim="none",
    )
    np.testing.assert_array_equal(ns, [50])
    np.testing.assert_allclose(means, [0.25])
    np.testing.assert_allclose(sds, [0.0])


def test_errorbar_series_uses_sample_sd_not_sem() -> None:
    mod = _load_parallel_module()
    rows = [
        {"n_total": 100, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 1.0},
        {"n_total": 100, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 3.0},
        {"n_total": 100, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 5.0},
    ]

    ns, means, sds = mod._errorbar_series(rows, method="flow_full", pr_dim="none")

    np.testing.assert_array_equal(ns, [100])
    np.testing.assert_allclose(means, [3.0])
    np.testing.assert_allclose(sds, [2.0])


def test_linear_fisher_family_includes_gkr_only() -> None:
    mod = _load_parallel_module()

    assert "gkr_linear" in mod._fisher_family_methods("linear")
    assert "gkr_linear" not in mod._fisher_family_methods("full")
    assert "gkr_full" in mod._fisher_family_methods("full")
    assert "gkr_full" not in mod._fisher_family_methods("linear")


def test_plot_sweep_errors_writes_errorbar_style_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_parallel_module()
    calls: list[dict[str, object]] = []
    original_errorbar = mod.plt.Axes.errorbar

    def spy_errorbar(self, x, y, *args, **kwargs):
        calls.append({"x": np.asarray(x), "y": np.asarray(y), "yerr": np.asarray(kwargs.get("yerr"))})
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(mod.plt.Axes, "errorbar", spy_errorbar)
    rows = [
        {"n_total": 100, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 1.0},
        {"n_total": 100, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 3.0},
        {"n_total": 200, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 2.0},
        {"n_total": 200, "pr_dim": "none", "method": "flow_full", "mae_abs_error": 6.0},
    ]

    svg, png = mod.plot_sweep_errors(rows, tmp_path)

    assert svg.is_file()
    assert png.is_file()
    assert calls
    np.testing.assert_allclose(calls[0]["y"], [2.0, 4.0])
    np.testing.assert_allclose(calls[0]["yerr"], [np.sqrt(2.0), 2.0 * np.sqrt(2.0)])
    assert "errorbars=mean_sd" in svg.read_text(encoding="utf-8")


def test_parallel_cache_hit_and_visualization_only(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(["--n-list", "100", "--pr-dims", "none", "--output-dir", str(tmp_path)])
    task = mod.plan_cases(args)[0]
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task.result_path,
        theta_midpoints=np.asarray([[0.5]], dtype=np.float64),
        flow_full_fisher=np.asarray([1.0], dtype=np.float64),
    )

    to_run, hits = mod.select_tasks_to_run([task], args)
    assert to_run == []
    assert hits[task.key] is True

    missing = mod.CaseTask(
        n_total=200,
        pr_dim=None,
        repeat_idx=0,
        seed=7,
        dataset_dir=tmp_path / "missing",
        output_dir=tmp_path / "missing" / "out",
        result_path=tmp_path / "missing" / "out" / "continuous_pr_fisher_results.npz",
    )
    viz_args = mod.build_parser().parse_args(
        ["--visualization-only", "--n-list", "100", "--pr-dims", "none", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(FileNotFoundError):
        mod.select_tasks_to_run([missing], viz_args)


def test_parallel_representative_native_npz_uses_largest_native_case(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        ["--n-list", "1000,5000", "--pr-dims", "none", "--output-dir", str(tmp_path / "sweep")]
    )
    tasks = mod.plan_cases(args)

    got = mod.representative_native_npz(tasks, args)

    assert got == (
        tmp_path
        / "sweep"
        / "n5000"
        / "native"
        / "randamp_gaussian_sqrtd_xdim4_native.npz"
    )


def test_parallel_composite_fisher_example_uses_cached_case_closest_to_5500(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        ["--n-list", "100,1500,5000", "--pr-dims", "none", "--output-dir", str(tmp_path / "sweep")]
    )
    tasks = mod.plan_cases(args)
    for task in tasks:
        if task.n_total in (100, 1500, 5000):
            task.result_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                task.result_path,
                theta_midpoints=np.asarray([[0.5]], dtype=np.float64),
                flow_full_fisher=np.asarray([1.0], dtype=np.float64),
            )

    got = mod.composite_fisher_example_task(tasks, args)

    assert got.n_total == 5000


def test_parallel_run_records_representative_dataset_figure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        ["--visualization-only", "--n-list", "100", "--pr-dims", "none", "--output-dir", str(tmp_path / "sweep")]
    )
    task = mod.plan_cases(args)[0]
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task.result_path,
        theta_midpoints=np.asarray([[0.5]], dtype=np.float64),
        flow_full_fisher=np.asarray([1.0], dtype=np.float64),
        flow_full_abs_error=np.asarray([0.25], dtype=np.float64),
    )
    native_npz = mod.representative_native_npz([task], args)
    native_npz.parent.mkdir(parents=True, exist_ok=True)
    native_npz.touch()

    calls: dict[str, object] = {}

    def fake_plot_representative_dataset(
        native_path: Path,
        *,
        output_dir: Path,
        scatter_max_points: int = 500,
    ):
        calls["native_path"] = native_path
        calls["output_dir"] = output_dir
        calls["scatter_max_points"] = int(scatter_max_points)
        return output_dir / mod.DATASET_VIZ_SVG_NAME, output_dir / mod.DATASET_VIZ_PNG_NAME

    monkeypatch.setattr(mod, "plot_representative_dataset", fake_plot_representative_dataset)

    def fake_plot_composite_figure(
        *,
        native_npz: Path,
        representative_result_path: Path,
        representative_n_total: int | None = None,
        rows: list[dict[str, object]],
        output_dir: Path,
        yscale: str = "linear",
    ):
        calls["composite_native_npz"] = native_npz
        calls["composite_result_path"] = representative_result_path
        calls["composite_n_total"] = representative_n_total
        calls["composite_rows"] = len(rows)
        calls["composite_output_dir"] = output_dir
        calls["composite_yscale"] = yscale
        return output_dir / mod.COMPOSITE_SVG_NAME, output_dir / mod.COMPOSITE_PNG_NAME

    monkeypatch.setattr(mod, "plot_composite_figure", fake_plot_composite_figure)

    def fake_plot_composite_kernel_smoothed_figure(
        *,
        native_npz: Path,
        representative_result_path: Path,
        representative_n_total: int | None = None,
        rows: list[dict[str, object]],
        output_dir: Path,
        yscale: str = "linear",
        bandwidth_grid: float = 2.0,
    ):
        calls["smoothed_composite_native_npz"] = native_npz
        calls["smoothed_composite_result_path"] = representative_result_path
        calls["smoothed_composite_n_total"] = representative_n_total
        calls["smoothed_composite_rows"] = len(rows)
        calls["smoothed_composite_output_dir"] = output_dir
        calls["smoothed_composite_yscale"] = yscale
        calls["smoothed_composite_bandwidth_grid"] = float(bandwidth_grid)
        return output_dir / mod.COMPOSITE_KERNEL_SMOOTHED_SVG_NAME, output_dir / mod.COMPOSITE_KERNEL_SMOOTHED_PNG_NAME

    monkeypatch.setattr(mod, "plot_composite_kernel_smoothed_figure", fake_plot_composite_kernel_smoothed_figure)
    monkeypatch.setattr(
        mod,
        "plot_composite_gp_smoothed_figure",
        lambda **kwargs: pytest.fail("GP-smoothed composite should not be written by default"),
    )

    outputs = mod.run(args)
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))

    assert calls["native_path"] == native_npz
    assert calls["output_dir"] == tmp_path / "sweep"
    assert calls["scatter_max_points"] == 500
    assert outputs["dataset_figure_svg"] == tmp_path / "sweep" / mod.DATASET_VIZ_SVG_NAME
    assert outputs["dataset_figure_png"] == tmp_path / "sweep" / mod.DATASET_VIZ_PNG_NAME
    assert outputs["composite_svg"] == tmp_path / "sweep" / mod.COMPOSITE_SVG_NAME
    assert outputs["composite_png"] == tmp_path / "sweep" / mod.COMPOSITE_PNG_NAME
    assert outputs["composite_kernel_smoothed_svg"] == tmp_path / "sweep" / mod.COMPOSITE_KERNEL_SMOOTHED_SVG_NAME
    assert outputs["composite_kernel_smoothed_png"] == tmp_path / "sweep" / mod.COMPOSITE_KERNEL_SMOOTHED_PNG_NAME
    assert "composite_gp_smoothed_svg" not in outputs
    assert "composite_gp_smoothed_png" not in outputs
    assert calls["composite_native_npz"] == native_npz
    assert calls["composite_result_path"] == task.result_path
    assert calls["composite_n_total"] == 100
    assert calls["composite_rows"] == 1
    assert calls["composite_output_dir"] == tmp_path / "sweep"
    assert calls["composite_yscale"] == "linear"
    assert calls["smoothed_composite_native_npz"] == native_npz
    assert calls["smoothed_composite_result_path"] == task.result_path
    assert calls["smoothed_composite_n_total"] == 100
    assert calls["smoothed_composite_rows"] == 0
    assert calls["smoothed_composite_output_dir"] == tmp_path / "sweep"
    assert calls["smoothed_composite_yscale"] == "linear"
    assert calls["smoothed_composite_bandwidth_grid"] == pytest.approx(2.0)
    assert summary["outputs"]["dataset_figure_svg"] == str(outputs["dataset_figure_svg"])
    assert summary["outputs"]["dataset_figure_png"] == str(outputs["dataset_figure_png"])
    assert summary["outputs"]["composite_svg"] == str(outputs["composite_svg"])
    assert summary["outputs"]["composite_png"] == str(outputs["composite_png"])
    assert summary["config"]["composite_smoothing"] == "kernel"
    assert summary["outputs"]["composite_kernel_smoothed_svg"] == str(outputs["composite_kernel_smoothed_svg"])
    assert summary["outputs"]["composite_kernel_smoothed_png"] == str(outputs["composite_kernel_smoothed_png"])
    assert "composite_gp_smoothed_svg" not in summary["outputs"]


def test_aggregate_repeat_seed_metadata_in_csv_npz_and_summary(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        [
            "--n-list",
            "100",
            "--pr-dims",
            "none",
            "--n-repeats",
            "2",
            "--seed",
            "31",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--skip-dataset-viz",
            "--composite-smoothing",
            "none",
        ]
    )
    tasks = mod.plan_cases(args)
    for offset, task in enumerate(tasks):
        task.result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            task.result_path,
            theta_midpoints=np.asarray([0.5], dtype=np.float64),
            flow_full_abs_error=np.asarray([1.0 + offset, 3.0 + offset], dtype=np.float64),
        )

    npz_path, csv_path, summary_path, svg_path, png_path, *_ = mod.aggregate_results(
        tasks,
        tmp_path / "sweep",
        args=args,
        dataset_paths=None,
    )

    assert svg_path.is_file()
    assert png_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {int(row["repeat_idx"]) for row in rows} == {0, 1}
    assert {int(row["repeat_seed"]) for row in rows} == {31, 32}
    with np.load(npz_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["repeat_idx"], [0, 1])
        np.testing.assert_array_equal(data["repeat_seed"], [31, 32])
        assert int(data["n_repeats"]) == 2
        np.testing.assert_array_equal(data["repeat_indices"], [0, 1])
        np.testing.assert_array_equal(data["repeat_seeds"], [31, 32])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["config"]["seed"] == 31
    assert summary["config"]["n_repeats"] == 2
    assert summary["config"]["repeat_seeds"] == [31, 32]


def test_gp_smooth_nonnegative_curve_returns_finite_same_length() -> None:
    mod = _load_parallel_module()
    theta_mid = np.linspace(-2.0, 2.0, 9, dtype=np.float64)
    noisy = np.asarray([0.2, 0.4, -0.1, 0.9, 1.5, 0.8, 0.4, np.nan, 0.3], dtype=np.float64)

    got = mod._gp_smooth_nonnegative_curve(theta_mid, noisy)

    assert got.shape == noisy.shape
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0.0)


def test_kernel_smooth_nonnegative_curve_returns_finite_same_length() -> None:
    mod = _load_parallel_module()
    theta_mid = np.linspace(-2.0, 2.0, 9, dtype=np.float64)
    noisy = np.asarray([0.2, 0.4, -0.1, 0.9, 1.5, 0.8, 0.4, np.nan, 0.3], dtype=np.float64)

    got = mod._kernel_smooth_nonnegative_curve(theta_mid, noisy, bandwidth_grid=2.0)

    assert got.shape == noisy.shape
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0.0)


def test_kernel_smooth_noisy_interior_moves_toward_weighted_neighbors() -> None:
    mod = _load_parallel_module()
    theta_mid = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    noisy = np.asarray([0.0, 10.0, 0.0], dtype=np.float64)

    got = mod._kernel_smooth_nonnegative_curve(theta_mid, noisy, bandwidth_grid=1.0)

    assert 0.0 < got[1] < noisy[1]
    assert got[1] == pytest.approx(10.0 / (1.0 + 2.0 * np.exp(-0.5)))


def test_smoothed_mae_uses_smoothed_curve_not_stored_abs_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_parallel_module()
    result_path = tmp_path / "continuous_pr_fisher_results.npz"
    theta_mid = np.asarray([0.0, 1.0], dtype=np.float64).reshape(-1, 1)
    np.savez_compressed(
        result_path,
        theta_midpoints=theta_mid,
        flow_full_fisher=np.asarray([10.0, 10.0], dtype=np.float64),
        ground_truth_native_full_fisher=np.asarray([1.0, 3.0], dtype=np.float64),
        flow_full_abs_error=np.asarray([999.0, 999.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        mod,
        "_gp_smooth_nonnegative_curve",
        lambda theta_midpoints, values: np.asarray([1.0, 2.0], dtype=np.float64),
    )

    got = mod._smoothed_mae_abs_error_from_npz(result_path, "flow_full")

    assert got == pytest.approx(0.5)


def test_kernel_smoothed_mae_uses_smoothed_curve_not_stored_abs_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_parallel_module()
    result_path = tmp_path / "continuous_pr_fisher_results.npz"
    theta_mid = np.asarray([0.0, 1.0], dtype=np.float64).reshape(-1, 1)
    np.savez_compressed(
        result_path,
        theta_midpoints=theta_mid,
        flow_full_fisher=np.asarray([10.0, 10.0], dtype=np.float64),
        ground_truth_native_full_fisher=np.asarray([1.0, 3.0], dtype=np.float64),
        flow_full_abs_error=np.asarray([999.0, 999.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        mod,
        "_kernel_smooth_nonnegative_curve",
        lambda theta_midpoints, values, *, bandwidth_grid=2.0: np.asarray([1.0, 2.0], dtype=np.float64),
    )

    got = mod._smoothed_mae_abs_error_from_npz(
        result_path,
        "flow_full",
        smoothing="kernel",
        kernel_bandwidth_grid=2.0,
    )

    assert got == pytest.approx(0.5)


def test_parallel_run_records_explicit_gp_composite_and_skips_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--n-list",
            "100",
            "--pr-dims",
            "none",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--composite-smoothing",
            "gp",
        ]
    )
    task = mod.plan_cases(args)[0]
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task.result_path,
        theta_midpoints=np.asarray([[0.5]], dtype=np.float64),
        flow_full_fisher=np.asarray([1.0], dtype=np.float64),
        flow_full_abs_error=np.asarray([0.25], dtype=np.float64),
    )
    native_npz = mod.representative_native_npz([task], args)
    native_npz.parent.mkdir(parents=True, exist_ok=True)
    native_npz.touch()

    monkeypatch.setattr(
        mod,
        "plot_representative_dataset",
        lambda native_path, *, output_dir, scatter_max_points=500: (
            output_dir / mod.DATASET_VIZ_SVG_NAME,
            output_dir / mod.DATASET_VIZ_PNG_NAME,
        ),
    )
    monkeypatch.setattr(
        mod,
        "plot_composite_figure",
        lambda *, native_npz, representative_result_path, representative_n_total=None, rows, output_dir, yscale="linear": (
            output_dir / mod.COMPOSITE_SVG_NAME,
            output_dir / mod.COMPOSITE_PNG_NAME,
        ),
    )
    monkeypatch.setattr(
        mod,
        "plot_composite_gp_smoothed_figure",
        lambda *, native_npz, representative_result_path, representative_n_total=None, rows, output_dir, yscale="linear": (
            output_dir / mod.COMPOSITE_GP_SMOOTHED_SVG_NAME,
            output_dir / mod.COMPOSITE_GP_SMOOTHED_PNG_NAME,
        ),
    )
    monkeypatch.setattr(
        mod,
        "plot_composite_kernel_smoothed_figure",
        lambda **kwargs: pytest.fail("Kernel-smoothed composite should not be written in GP mode"),
    )

    outputs = mod.run(args)
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))

    assert outputs["composite_gp_smoothed_svg"] == tmp_path / "sweep" / mod.COMPOSITE_GP_SMOOTHED_SVG_NAME
    assert outputs["composite_gp_smoothed_png"] == tmp_path / "sweep" / mod.COMPOSITE_GP_SMOOTHED_PNG_NAME
    assert "composite_kernel_smoothed_svg" not in outputs
    assert "composite_kernel_smoothed_png" not in outputs
    assert summary["config"]["composite_smoothing"] == "gp"
    assert "composite_gp_smoothed_svg" in summary["outputs"]
    assert "composite_kernel_smoothed_svg" not in summary["outputs"]


def test_parallel_run_records_kernel_composite_and_skips_gp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--n-list",
            "100",
            "--pr-dims",
            "none",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--composite-smoothing",
            "kernel",
            "--kernel-smooth-bandwidth-grid",
            "2.0",
        ]
    )
    task = mod.plan_cases(args)[0]
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task.result_path,
        theta_midpoints=np.asarray([[0.5]], dtype=np.float64),
        flow_full_fisher=np.asarray([1.0], dtype=np.float64),
        flow_full_abs_error=np.asarray([0.25], dtype=np.float64),
    )
    native_npz = mod.representative_native_npz([task], args)
    native_npz.parent.mkdir(parents=True, exist_ok=True)
    native_npz.touch()

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        mod,
        "plot_representative_dataset",
        lambda native_path, *, output_dir, scatter_max_points=500: (
            output_dir / mod.DATASET_VIZ_SVG_NAME,
            output_dir / mod.DATASET_VIZ_PNG_NAME,
        ),
    )
    monkeypatch.setattr(
        mod,
        "plot_composite_figure",
        lambda *, native_npz, representative_result_path, representative_n_total=None, rows, output_dir, yscale="linear": (
            output_dir / mod.COMPOSITE_SVG_NAME,
            output_dir / mod.COMPOSITE_PNG_NAME,
        ),
    )

    def fake_plot_composite_kernel_smoothed_figure(
        *,
        native_npz: Path,
        representative_result_path: Path,
        representative_n_total: int | None = None,
        rows: list[dict[str, object]],
        output_dir: Path,
        yscale: str = "linear",
        bandwidth_grid: float = 2.0,
    ):
        calls["kernel_rows"] = len(rows)
        calls["kernel_bandwidth_grid"] = float(bandwidth_grid)
        return output_dir / mod.COMPOSITE_KERNEL_SMOOTHED_SVG_NAME, output_dir / mod.COMPOSITE_KERNEL_SMOOTHED_PNG_NAME

    monkeypatch.setattr(mod, "plot_composite_kernel_smoothed_figure", fake_plot_composite_kernel_smoothed_figure)
    monkeypatch.setattr(
        mod,
        "plot_composite_gp_smoothed_figure",
        lambda **kwargs: pytest.fail("GP-smoothed composite should not be written in kernel mode"),
    )

    outputs = mod.run(args)
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))

    assert outputs["composite_kernel_smoothed_svg"] == tmp_path / "sweep" / mod.COMPOSITE_KERNEL_SMOOTHED_SVG_NAME
    assert outputs["composite_kernel_smoothed_png"] == tmp_path / "sweep" / mod.COMPOSITE_KERNEL_SMOOTHED_PNG_NAME
    assert "composite_gp_smoothed_svg" not in outputs
    assert "composite_gp_smoothed_png" not in outputs
    assert calls["kernel_rows"] == 0
    assert calls["kernel_bandwidth_grid"] == pytest.approx(2.0)
    assert "composite_kernel_smoothed_svg" in summary["outputs"]
    assert "composite_gp_smoothed_svg" not in summary["outputs"]


def test_plot_composite_figure_writes_svg_and_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_parallel_module()
    dataset = ToyConditionalGaussianRandampSqrtdDataset(theta_low=-2.0, theta_high=2.0, x_dim=3, seed=123)
    monkeypatch.setattr(mod, "load_shared_dataset_npz", lambda path: SimpleNamespace(meta={}))
    monkeypatch.setattr(mod, "build_dataset_from_meta", lambda meta: dataset)

    theta_mid = np.linspace(-1.9, 1.9, 8, dtype=np.float64).reshape(-1, 1)
    result_path = tmp_path / "continuous_pr_fisher_results.npz"
    np.savez_compressed(
        result_path,
        theta_midpoints=theta_mid,
        ground_truth_native_full_fisher=np.linspace(2.0, 3.0, 8, dtype=np.float64),
        ground_truth_native_linear_fisher=np.linspace(1.5, 2.5, 8, dtype=np.float64),
        classical_linear_fisher=np.linspace(1.4, 2.4, 8, dtype=np.float64),
        classical_full_fisher=np.linspace(1.8, 2.8, 8, dtype=np.float64),
        flow_linear_fisher=np.linspace(1.55, 2.55, 8, dtype=np.float64),
        flow_full_fisher=np.linspace(1.95, 2.95, 8, dtype=np.float64),
    )
    rows = [
        {"n_total": n, "pr_dim": "none", "repeat_idx": 0, "method": method, "mae_abs_error": err}
        for n, scale in [(100, 1.0), (500, 0.6)]
        for method, err in [
            ("classical_linear", 0.4 * scale),
            ("classical_full", 0.35 * scale),
            ("flow_linear", 0.2 * scale),
            ("flow_full", 0.15 * scale),
        ]
    ]

    svg, png = mod.plot_composite_figure(
        native_npz=tmp_path / "native.npz",
        representative_result_path=result_path,
        rows=rows,
        output_dir=tmp_path,
    )

    assert svg.exists()
    assert svg.stat().st_size > 0
    assert png.exists()
    assert png.stat().st_size > 0


def test_plot_composite_gp_smoothed_figure_writes_svg_and_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_parallel_module()
    dataset = ToyConditionalGaussianRandampSqrtdDataset(theta_low=-2.0, theta_high=2.0, x_dim=3, seed=123)
    monkeypatch.setattr(mod, "load_shared_dataset_npz", lambda path: SimpleNamespace(meta={}))
    monkeypatch.setattr(mod, "build_dataset_from_meta", lambda meta: dataset)

    theta_mid = np.linspace(-1.9, 1.9, 8, dtype=np.float64).reshape(-1, 1)
    result_path = tmp_path / "continuous_pr_fisher_results.npz"
    np.savez_compressed(
        result_path,
        theta_midpoints=theta_mid,
        ground_truth_native_full_fisher=np.linspace(2.0, 3.0, 8, dtype=np.float64),
        ground_truth_native_linear_fisher=np.linspace(1.5, 2.5, 8, dtype=np.float64),
        classical_linear_fisher=np.linspace(1.4, 2.4, 8, dtype=np.float64),
        classical_full_fisher=np.linspace(1.8, 2.8, 8, dtype=np.float64),
        flow_linear_fisher=np.linspace(1.55, 2.55, 8, dtype=np.float64),
        flow_full_fisher=np.linspace(1.95, 2.95, 8, dtype=np.float64),
    )
    rows = [
        {"n_total": n, "pr_dim": "none", "repeat_idx": 0, "method": method, "mae_abs_error": err}
        for n, scale in [(100, 1.0), (500, 0.6)]
        for method, err in [
            ("classical_linear", 0.4 * scale),
            ("classical_full", 0.35 * scale),
            ("flow_linear", 0.2 * scale),
            ("flow_full", 0.15 * scale),
        ]
    ]

    svg, png = mod.plot_composite_gp_smoothed_figure(
        native_npz=tmp_path / "native.npz",
        representative_result_path=result_path,
        rows=rows,
        output_dir=tmp_path,
    )

    assert svg == tmp_path / mod.COMPOSITE_GP_SMOOTHED_SVG_NAME
    assert png == tmp_path / mod.COMPOSITE_GP_SMOOTHED_PNG_NAME
    assert svg.exists()
    assert svg.stat().st_size > 0
    assert png.exists()
    assert png.stat().st_size > 0


def test_plot_composite_kernel_smoothed_figure_writes_svg_and_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_parallel_module()
    dataset = ToyConditionalGaussianRandampSqrtdDataset(theta_low=-2.0, theta_high=2.0, x_dim=3, seed=123)
    monkeypatch.setattr(mod, "load_shared_dataset_npz", lambda path: SimpleNamespace(meta={}))
    monkeypatch.setattr(mod, "build_dataset_from_meta", lambda meta: dataset)

    theta_mid = np.linspace(-1.9, 1.9, 8, dtype=np.float64).reshape(-1, 1)
    result_path = tmp_path / "continuous_pr_fisher_results.npz"
    np.savez_compressed(
        result_path,
        theta_midpoints=theta_mid,
        ground_truth_native_full_fisher=np.linspace(2.0, 3.0, 8, dtype=np.float64),
        ground_truth_native_linear_fisher=np.linspace(1.5, 2.5, 8, dtype=np.float64),
        classical_linear_fisher=np.linspace(1.4, 2.4, 8, dtype=np.float64),
        classical_full_fisher=np.linspace(1.8, 2.8, 8, dtype=np.float64),
        flow_linear_fisher=np.linspace(1.55, 2.55, 8, dtype=np.float64),
        flow_full_fisher=np.linspace(1.95, 2.95, 8, dtype=np.float64),
    )
    rows = [
        {"n_total": n, "pr_dim": "none", "repeat_idx": 0, "method": method, "mae_abs_error": err}
        for n, scale in [(100, 1.0), (500, 0.6)]
        for method, err in [
            ("classical_linear", 0.4 * scale),
            ("classical_full", 0.35 * scale),
            ("flow_linear", 0.2 * scale),
            ("flow_full", 0.15 * scale),
        ]
    ]

    svg, png = mod.plot_composite_kernel_smoothed_figure(
        native_npz=tmp_path / "native.npz",
        representative_result_path=result_path,
        rows=rows,
        output_dir=tmp_path,
        bandwidth_grid=2.0,
    )

    assert svg == tmp_path / mod.COMPOSITE_KERNEL_SMOOTHED_SVG_NAME
    assert png == tmp_path / mod.COMPOSITE_KERNEL_SMOOTHED_PNG_NAME
    assert svg.exists()
    assert svg.stat().st_size > 0
    assert png.exists()
    assert png.stat().st_size > 0
