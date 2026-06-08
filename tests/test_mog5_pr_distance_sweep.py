from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "compare_mog5_pr_distance_sweeps.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distance_sweeps", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_case_npz(path: Path, *, offset: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = np.asarray(["squared_euclidean", "cosine"])
    labels = np.asarray(["category_0", "category_1", "category_2"])
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    classical = np.zeros((2, 3, 3), dtype=np.float64)
    flow = np.zeros((2, 3, 3), dtype=np.float64)
    gt = np.zeros((2, 3, 3), dtype=np.float64)
    for metric_idx in range(2):
        for i, j in pairs:
            gt_val = offset + 10.0 + metric_idx
            classical_val = gt_val + float(i + 1)
            flow_val = gt_val - float(j + 2)
            gt[metric_idx, i, j] = gt[metric_idx, j, i] = gt_val
            classical[metric_idx, i, j] = classical[metric_idx, j, i] = classical_val
            flow[metric_idx, i, j] = flow[metric_idx, j, i] = flow_val
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
    assert args.pr_dim == 5
    assert args.pr_dim_list == [3, 5, 8, 11]
    assert args.n_total == 1000
    assert args.device == "cuda"
    assert args.yscale == "log"
    assert args.output_dir == repo_root / "data" / "mog5_pr_distance_sweeps"


def test_aggregate_mean_pairwise_abs_errors(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100", "--pr-dim-list", "3"])
    data = {
        (100, 5): mod._load_case_cache(_write_case_npz(tmp_path / "case_a.npz", offset=0.0)),
        (1000, 3): mod._load_case_cache(_write_case_npz(tmp_path / "case_b.npz", offset=5.0)),
    }

    aggregate, rows = mod.aggregate_sweeps(args=args, case_data=data)

    assert aggregate["n_sweep_classical_matrices"].shape == (1, 2, 3, 3)
    n_rows = [r for r in rows if r["axis"] == "n_total" and r["metric"] == "squared_euclidean"]
    classical_errors = [r["abs_error"] for r in n_rows if r["estimator"] == "classical"]
    flow_errors = [r["abs_error"] for r in n_rows if r["estimator"] == "flow_matching"]
    classical_rel_errors = [r["rel_error"] for r in n_rows if r["estimator"] == "classical"]
    flow_rel_errors = [r["rel_error"] for r in n_rows if r["estimator"] == "flow_matching"]
    assert np.mean(classical_errors) == pytest.approx((1.0 + 1.0 + 2.0) / 3.0)
    assert np.mean(flow_errors) == pytest.approx((3.0 + 4.0 + 4.0) / 3.0)
    assert np.mean(classical_rel_errors) == pytest.approx(((1.0 / 10.0) + (1.0 / 10.0) + (2.0 / 10.0)) / 3.0)
    assert np.mean(flow_rel_errors) == pytest.approx(((3.0 / 10.0) + (4.0 / 10.0) + (4.0 / 10.0)) / 3.0)


def test_relative_error_uses_denominator_floor_for_zero_ground_truth(tmp_path: Path) -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--n-list", "100", "--pr-dim-list", "3"])
    data = {
        (100, 5): mod._load_case_cache(_write_zero_gt_case_npz(tmp_path / "case_a.npz")),
        (1000, 3): mod._load_case_cache(_write_zero_gt_case_npz(tmp_path / "case_b.npz")),
    }

    _, rows = mod.aggregate_sweeps(args=args, case_data=data)

    n_rows = [r for r in rows if r["axis"] == "n_total"]
    classical = next(r for r in n_rows if r["estimator"] == "classical")
    flow = next(r for r in n_rows if r["estimator"] == "flow_matching")
    assert classical["rel_error"] == pytest.approx(2.0 / mod.REL_ERROR_DENOM_FLOOR)
    assert flow["rel_error"] == pytest.approx(3.0 / mod.REL_ERROR_DENOM_FLOOR)


def test_cache_hits_do_not_rerun_and_duplicate_case_is_deduped(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    calls: list[object] = []
    real_single = mod._load_single_case_module()

    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_1000_5" / mod.RESULTS_NAME)

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
            "1000",
            "--pr-dim-list",
            "5,8",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--yscale",
            "linear",
        ]
    )
    mod.run(args)

    assert len(calls) == 1
    assert calls[0].n_total == 1000
    assert calls[0].pr_dim == 8


def test_visualization_only_missing_cache_fails(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    args = mod.build_parser().parse_args(
        ["--visualization-only", "--n-list", "100", "--pr-dim-list", "3", "--output-dir", str(tmp_path / "sweep")]
    )

    with pytest.raises(FileNotFoundError, match="visualization-only"):
        mod.run(args)


def test_visualization_only_writes_outputs_from_fake_caches(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(
        mod,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name: tmp_path / f"case_{n_total}_{pr_dim}",
    )
    _write_case_npz(tmp_path / "case_100_5" / mod.RESULTS_NAME)
    _write_case_npz(tmp_path / "case_1000_3" / mod.RESULTS_NAME, offset=2.0)
    args = mod.build_parser().parse_args(
        [
            "--visualization-only",
            "--n-list",
            "100",
            "--pr-dim-list",
            "3",
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
    assert outputs["summary_json"].is_file()
    with np.load(outputs["results_npz"], allow_pickle=False) as data:
        assert data["n_sweep_classical_matrices"].shape == (1, 2, 3, 3)
    with outputs["errors_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert "rel_error" in rows[0]
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["config"]["abs_error_yscale"] == "log"
    assert summary["config"]["rel_error_yscale"] == "linear"
    assert "abs_error_figure_svg" in summary["outputs"]
    assert "rel_error_figure_svg" in summary["outputs"]
