from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "run_stringer_fisher_second_row.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_stringer_fisher_second_row", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binned_lw_recovers_finite_positive_direction() -> None:
    module = _load_script()
    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, np.pi, size=800)
    x = np.column_stack(
        [
            2.0 * np.cos(2.0 * theta) + 0.2 * rng.standard_normal(theta.size),
            np.sin(2.0 * theta) + 0.2 * rng.standard_normal(theta.size),
        ]
    )
    grid = np.linspace(0.0, np.pi, 9).reshape(-1, 1)
    result = module.fit_binned_lw_direction_estimator(
        theta_train=theta,
        x_train=x,
        theta_grid=grid,
    )
    assert result["linear_fisher"].shape == (8,)
    assert result["direction"].shape == (8, 2)
    assert np.all(np.isfinite(result["linear_fisher"]))
    assert np.all(result["linear_fisher"] > 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(result["direction"], axis=1), 1.0, atol=1e-8
    )


def test_exact_stratified_split_holds_out_half_without_overlap() -> None:
    module = _load_script()
    theta = np.linspace(0.0, np.pi, 4599, endpoint=False)
    train, validation, test = module.exact_stratified_fit_validation_test_indices(
        theta,
        train_fraction=0.4,
        validation_fraction=0.1,
        n_strata=16,
        seed=7,
    )
    assert test.size == theta.size // 2
    assert validation.size == round(0.2 * (theta.size - test.size))
    assert np.intersect1d(train, validation).size == 0
    assert np.intersect1d(train, test).size == 0
    assert np.intersect1d(validation, test).size == 0
    assert np.unique(np.concatenate([train, validation, test])).size == theta.size


def test_method_order_matches_requested_panel_order() -> None:
    module = _load_script()
    assert module.LINEAR_METHOD_KEYS == (
        "gkr",
        "bin_lw",
        "ole_crossfit",
        "affine_flow",
    )
    assert module.LINEAR_METHOD_TITLES == (
        "GKR",
        "Binning + LW",
        "OLE (crossfit)",
        "Affine Flow",
    )


def test_achieved_axis_limits_cover_sessions_without_forcing_zero() -> None:
    module = _load_script()
    achieved = np.asarray(
        [
            [161.8, 167.9, 167.5, 171.7],
            [144.0, 165.8, 166.0, 153.6],
            [170.0, 168.2, 170.1, 176.5],
        ]
    )
    sem = np.asarray([2.0, 1.5, 1.3, 2.2])
    lower, upper = module._achieved_axis_limits(achieved, sem)
    assert 0.0 < lower < np.min(achieved)
    assert upper > np.max(achieved)


def test_standalone_plots_use_readable_aspect_ratios(tmp_path: Path) -> None:
    module = _load_script()
    theta = np.linspace(0.0, np.pi, 16)
    cases = []
    for session_index in range(6):
        arrays = {"theta_midpoints": theta}
        for method_index, key in enumerate(module.LINEAR_METHOD_KEYS):
            arrays[f"{key}_achieved_raw"] = np.asarray(
                [140.0 + 4.0 * session_index + method_index]
            )
            arrays[f"{key}_linear_fisher"] = (
                10.0 + method_index + np.sin(theta) ** 2
            )
        for method_index, key in enumerate(module.FULL_METHOD_KEYS):
            arrays[f"{key}_full_fisher"] = (
                20.0 + method_index + np.cos(theta) ** 2
            )
        cases.append(
            {
                "summary": {
                    "session_index": session_index,
                    "session_label": f"S{session_index}",
                },
                "arrays": arrays,
            }
        )

    achieved_png, achieved_svg = module._plot_achieved_information(
        cases, output_dir=tmp_path
    )
    curves_png, curves_svg = module._plot_fisher_curves(
        cases, output_dir=tmp_path
    )
    assert achieved_svg.is_file()
    assert curves_svg.is_file()
    with Image.open(achieved_png) as achieved_image:
        assert 1.1 < achieved_image.width / achieved_image.height < 1.5
    with Image.open(curves_png) as curves_image:
        assert 5.5 < curves_image.width / curves_image.height < 6.5
