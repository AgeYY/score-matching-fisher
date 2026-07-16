from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "bin" / "visualize_mog5_dimension_sweep.py"
    spec = importlib.util.spec_from_file_location("visualize_mog5_dimension_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_payload(*, dimension: int) -> dict[str, np.ndarray]:
    metrics = np.asarray(["cosine", "fid", "symmetric_kl"])
    repeats = 2
    k = 3
    ground_truth = np.zeros((1, repeats, len(metrics), k, k), dtype=np.float64)
    for metric_idx in range(len(metrics)):
        matrix = np.asarray(
            [
                [0.0, 1.0 + metric_idx, 2.0 + metric_idx],
                [1.0 + metric_idx, 0.0, 3.0 + metric_idx],
                [2.0 + metric_idx, 3.0 + metric_idx, 0.0],
            ],
            dtype=np.float64,
        )
        ground_truth[0, :, metric_idx] = matrix
    classical = ground_truth + 0.1 * float(dimension)
    flow = ground_truth + 0.05 * float(dimension)
    flow_nll = ground_truth + 0.03 * float(dimension)
    tre = np.full_like(ground_truth, np.nan)
    tre[:, :, 2] = ground_truth[:, :, 2] + 0.02 * float(dimension)
    return {
        "metric_names": metrics,
        "condition_labels": np.asarray(["A", "B", "C"]),
        "pair_indices": np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64),
        "n_list": np.asarray([1000], dtype=np.int64),
        "native_x_dim": np.asarray([dimension], dtype=np.int64),
        "repeat_indices": np.asarray([0, 1], dtype=np.int64),
        "repeat_seeds": np.asarray([7, 8], dtype=np.int64),
        "n_repeat_classical_matrices": classical,
        "n_repeat_flow_matching_matrices": flow,
        "n_repeat_flow_matching_nll_finetuned_matrices": flow_nll,
        "n_repeat_ground_truth_matrices": ground_truth,
        "n_repeat_tre_matrices": tre,
    }


def test_collect_errors_limits_tre_to_jeffreys() -> None:
    module = _load_module()
    runs = {dimension: _run_payload(dimension=dimension) for dimension in (3, 10)}
    errors = module.collect_errors(runs)

    assert tuple(errors) == module.METRIC_ORDER
    assert tuple(errors["cosine"]) == (
        "classical",
        "flow_matching",
        "flow_matching_nll_finetuned",
    )
    assert tuple(errors["fid"]) == (
        "classical",
        "flow_matching",
        "flow_matching_nll_finetuned",
    )
    assert tuple(errors["symmetric_kl"]) == (
        "classical",
        "flow_matching",
        "flow_matching_nll_finetuned",
        "tre",
    )
    assert errors["symmetric_kl"]["tre"]["relative"].shape == (2, 2)
    assert np.all(np.isfinite(errors["symmetric_kl"]["tre"]["relative"]))


def test_plot_and_tables_are_written(tmp_path: Path) -> None:
    module = _load_module()
    dimensions = (3, 10)
    runs = {dimension: _run_payload(dimension=dimension) for dimension in dimensions}
    for dimension, payload in runs.items():
        module._verify_run(payload, dimension=dimension, n_total=1000)
    errors = module.collect_errors(runs)

    rel_png, rel_svg = module._plot(
        errors, dimensions=dimensions, relative=True, output_dir=tmp_path
    )
    abs_png, abs_svg = module._plot(
        errors, dimensions=dimensions, relative=False, output_dir=tmp_path
    )
    csv_path, npz_path, summary_path = module._write_outputs(
        errors,
        dimensions=dimensions,
        n_total=1000,
        input_paths={dimension: tmp_path / f"xdim{dimension}.npz" for dimension in dimensions},
        output_dir=tmp_path,
    )

    for path in (rel_png, rel_svg, abs_png, abs_svg, csv_path, npz_path, summary_path):
        assert path.is_file() and path.stat().st_size > 0
    assert "Native data dimension" in rel_svg.read_text(encoding="utf-8")


def test_verify_run_rejects_wrong_dimension() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Expected native_x_dim=10"):
        module._verify_run(_run_payload(dimension=3), dimension=10, n_total=1000)
