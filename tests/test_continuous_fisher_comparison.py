from __future__ import annotations

import importlib.util
import json
import sys
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
    assert default_args.native_x_dim == 3
    assert default_args.train_frac == pytest.approx(0.9)
    assert default_args.pr_dims == [None]
    assert default_args.skip_dataset_viz is False
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
    env = mod.build_case_env({"PATH": "/bin"}, gpu_id=3, cpu_threads_per_job=4)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OMP_NUM_THREADS"] == "4"


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
        / "randamp_gaussian_sqrtd_xdim3_native.npz"
    )


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

    outputs = mod.run(args)
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))

    assert calls["native_path"] == native_npz
    assert calls["output_dir"] == tmp_path / "sweep"
    assert calls["scatter_max_points"] == 500
    assert outputs["dataset_figure_svg"] == tmp_path / "sweep" / mod.DATASET_VIZ_SVG_NAME
    assert outputs["dataset_figure_png"] == tmp_path / "sweep" / mod.DATASET_VIZ_PNG_NAME
    assert summary["outputs"]["dataset_figure_svg"] == str(outputs["dataset_figure_svg"])
    assert summary["outputs"]["dataset_figure_png"] == str(outputs["dataset_figure_png"])
