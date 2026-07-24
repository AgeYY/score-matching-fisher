from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "compare_toy_linear_fisher_density_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_toy_linear_fisher_density_sweep", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_rows_computes_repeat_mean_and_error() -> None:
    module = _load_script()
    rows = []
    for dataset in module.DATASETS:
        for method in module.METHODS:
            for seed, mae in ((7, 1.0), (8, 3.0)):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": module.DATASETS[dataset],
                        "seed": seed,
                        "n_total": 125,
                        "n_train": 100,
                        "train_density": 2.0,
                        "median_ole_endpoint_density": 0.2,
                        "method": method,
                        "mae": mae,
                    }
                )
    grouped = module._group_rows(rows)
    assert len(grouped) == 2 * len(module.METHODS)
    assert all(row["mae_mean"] == 2.0 for row in grouped)
    assert all(row["mae_std"] > 0.0 for row in grouped)


def test_density_plot_writes_png_and_svg(tmp_path: Path) -> None:
    module = _load_script()
    grouped = []
    for dataset in module.DATASETS:
        for method in module.METHODS:
            for density in (2.0, 4.0):
                grouped.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "train_density": density,
                        "mae_mean": 1.0 / density,
                        "mae_std": 0.1 / density,
                    }
                )
    png, svg = module._plot(grouped, tmp_path)
    assert png.is_file()
    assert svg.is_file()


def test_density_plot_supports_one_dataset(tmp_path: Path) -> None:
    module = _load_script()
    dataset = next(iter(module.DATASETS))
    grouped = []
    for method in module.METHODS:
        for density in (2.0, 4.0):
            grouped.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "train_density": density,
                    "mae_mean": 1.0 / density,
                    "mae_std": 0.1 / density,
                }
            )
    png, svg = module._plot(grouped, tmp_path)
    assert png.is_file()
    assert svg.is_file()
