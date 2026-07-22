from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "bin" / "run_stringer_flow_all_sessions.py"
    spec = importlib.util.spec_from_file_location("run_stringer_flow_all_sessions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uncentered_projection_does_not_remove_feature_mean() -> None:
    module = _load_script()
    rng = np.random.default_rng(7)
    responses = rng.normal(size=(100, 90)) + 10.0
    projected, projector = module._uncentered_projection(
        responses, n_components=82, random_state=0
    )
    assert projected.shape == (100, 82)
    assert projector.components_.shape == (82, 90)
    assert abs(float(np.mean(projected[:, 0]))) > 1.0


def test_trial_and_validation_splits_have_exact_sizes() -> None:
    module = _load_script()
    theta = np.repeat(
        (np.arange(16, dtype=np.float64) + 0.5) * np.pi / 16.0,
        10,
    )
    selected = module._select_trial_indices(
        theta, n_trials=100, n_strata=16, seed=19
    )
    train, validation = module._train_validation_indices(
        theta[selected], validation_fraction=0.2, n_strata=16, seed=23
    )
    assert selected.size == 100
    assert train.size == 80
    assert validation.size == 20
    assert np.intersect1d(train, validation).size == 0
    assert np.union1d(train, validation).size == 100


def test_full_dataset_indices_preserve_every_trial() -> None:
    module = _load_script()
    theta = np.linspace(0.0, np.pi, 137, endpoint=False)
    selected = module._session_trial_indices(
        theta,
        use_full_dataset=True,
        n_trials=10,
        n_strata=16,
        seed=7,
    )
    assert np.array_equal(selected, np.arange(137, dtype=np.int64))


def test_all_session_flow_plot_writes_png_and_svg(tmp_path: Path) -> None:
    module = _load_script()
    theta = np.linspace(0.0, np.pi, 16, endpoint=False) + np.pi / 32.0
    results = [
        {
            "theta_midpoints": theta,
            "linear_fisher": np.full(theta.size, index + 1.0),
            "metadata": {"label": label},
        }
        for index, label in enumerate(("GT1", "GT2", "TX38"))
    ]
    png, svg = module._plot(results, tmp_path)
    assert png.is_file()
    assert svg.is_file()


def test_all_session_full_fisher_plot_writes_kind_specific_files(tmp_path: Path) -> None:
    module = _load_script()
    theta = np.linspace(0.0, np.pi, 16, endpoint=False) + np.pi / 32.0
    results = [
        {
            "theta_midpoints": theta,
            "full_fisher": np.linspace(1.0, 2.0, theta.size),
            "metadata": {"label": "GT1", "nll_finetuned": False},
        }
    ]
    png, svg = module._plot(results, tmp_path, "full")
    assert png.name == "stringer_flow_all_sessions_full_fisher.png"
    assert svg.name == "stringer_flow_all_sessions_full_fisher.svg"
    assert png.is_file()
    assert svg.is_file()
    assert "Full Fisher" in svg.read_text(encoding="utf-8")
