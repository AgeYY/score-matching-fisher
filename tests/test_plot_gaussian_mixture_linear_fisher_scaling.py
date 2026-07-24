from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "plot_gaussian_mixture_linear_fisher_scaling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_gaussian_mixture_linear_fisher_scaling", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_case(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(-1.0, 1.0, 4)[:, None]
    truth = np.linspace(1.0, 2.0, 4)
    np.savez_compressed(
        path,
        theta_midpoints=theta,
        ground_truth=truth,
        flow_fisher=truth + offset,
        gkr_fisher=truth + 2.0 * offset,
        ole_fisher=truth + 3.0 * offset,
    )


def test_collect_and_plot_scaling_results(tmp_path: Path) -> None:
    module = _load_script()
    sample_root = tmp_path / "sample"
    dimension_root = tmp_path / "dimension"
    for n_total in (500, 5000):
        for seed in (7, 8):
            _write_case(
                module.case_result(
                    sample_root=sample_root,
                    dimension_root=dimension_root,
                    x_dim=50,
                    n_total=n_total,
                    seed=seed,
                ),
                offset=0.1 * seed / n_total,
            )
    for x_dim in (3, 50):
        for seed in (7, 8):
            _write_case(
                module.case_result(
                    sample_root=sample_root,
                    dimension_root=dimension_root,
                    x_dim=x_dim,
                    n_total=3000,
                    seed=seed,
                ),
                offset=0.01 * x_dim,
            )

    args = type(
        "Args",
        (),
        {
            "sample_root": sample_root,
            "dimension_root": dimension_root,
            "n_list": [500, 5000],
            "dimension_list": [3, 50],
            "seeds": [7, 8],
            "representative_n": 5000,
            "representative_seed": 7,
        },
    )()
    results = module.collect(args)
    png, svg = module.plot(results, tmp_path / "out")
    npz, summary = module.save_results(results, tmp_path / "out", png, svg)

    assert np.asarray(results["sample_mae"]["Flow Matching"]).shape == (2, 2)
    assert np.asarray(results["dimension_mae"]["GKR"]).shape == (2, 2)
    assert png.is_file()
    assert svg.is_file()
    assert npz.is_file()
    assert summary.is_file()
