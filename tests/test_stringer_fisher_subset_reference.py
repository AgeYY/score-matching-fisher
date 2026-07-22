from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "bin" / "compare_stringer_fisher_subset_reference.py"
    spec = importlib.util.spec_from_file_location("compare_stringer_fisher_subset_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stringer_full_fisher_requires_five_disjoint_feasible_subsets() -> None:
    module = _load_script()
    args = SimpleNamespace(
        pca_dim=82,
        theta_grid_size=17,
        train_fraction=0.8,
        subset_sizes=[200, 400, 800],
    )
    module._validate_args(args, n_observations=4598, n_features=23_589)

    args.subset_sizes = [920]
    with pytest.raises(ValueError, match=r"K <= full dataset size // 5 = 919"):
        module._validate_args(args, n_observations=4598, n_features=23_589)


def test_stringer_full_fisher_rows_compare_both_methods_to_tre_reference() -> None:
    module = _load_script()
    reference = np.array([2.0, 4.0])
    rows = module._rows_for_case(
        k=200,
        repeat=3,
        subset_seed=91,
        subset_index=np.arange(200),
        reference=reference,
        flow={"flow_full_fisher": np.array([2.5, 3.5])},
        tre={"tre_full_fisher": np.array([1.0, 3.0])},
    )

    assert [row["method"] for row in rows] == [module.METHOD_FLOW, module.METHOD_TRE]
    assert all(row["K"] == 200 and row["repeat"] == 3 for row in rows)
    assert rows[0]["mean_full_fisher"] == pytest.approx(3.0)
    assert rows[0]["absolute_error_to_reference_mean"] == pytest.approx(0.0)
    assert rows[0]["curve_mae_to_reference"] == pytest.approx(0.5)
    assert rows[1]["absolute_error_to_reference_mean"] == pytest.approx(1.0)


def test_stringer_full_fisher_main_writes_five_disjoint_repeats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script()
    n_observations = 200
    theta = (np.arange(n_observations, dtype=np.float64) + 0.5) * np.pi / n_observations
    session = SimpleNamespace(
        grating_orientation=theta,
        neural_responses=np.zeros((n_observations, 100), dtype=np.float32),
        session_file=tmp_path / "session.npy",
    )
    monkeypatch.setattr(module, "require_device", lambda value: module.torch.device("cpu"))
    monkeypatch.setattr(module, "load_stringer_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(
        module,
        "_preprocess_full_session",
        lambda *args, **kwargs: (
            np.zeros((n_observations, 82), dtype=np.float32),
            {"artifact": str(tmp_path / "pca.npz")},
        ),
    )

    observed_subset_indices: list[np.ndarray] = []

    def fake_tre(args, *, theta, case_dir, **kwargs):
        del args, kwargs
        if "subsets" in case_dir.parts:
            observed_subset_indices.append(np.load(case_dir / "full_dataset_indices.npy"))
        return {"tre_full_fisher": np.full(16, float(theta.size), dtype=np.float64)}

    def fake_flow(args, *, theta, **kwargs):
        del args, kwargs
        return {"flow_full_fisher": np.full(16, float(theta.size) + 1.0, dtype=np.float64)}

    monkeypatch.setattr(module, "_fit_tre", fake_tre)
    monkeypatch.setattr(module, "_fit_flow", fake_flow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_stringer_fisher_subset_reference.py",
            "--device",
            "cuda:1",
            "--output-dir",
            str(tmp_path / "output"),
            "--subset-sizes",
            "32",
        ],
    )

    assert module.main() == 0
    assert len(observed_subset_indices) == 5
    assert all(index.size == 32 for index in observed_subset_indices)
    assert np.unique(np.concatenate(observed_subset_indices)).size == 160
    summary_path = tmp_path / "output" / "stringer_full_fisher_subset_reference_summary.json"
    summary = module.json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_subsets_per_K"] == 5
    assert len(summary["rows"]) == 10
    assert Path(summary["artifacts"]["rows_csv"]).is_file()
    assert Path(summary["artifacts"]["figure_png"]).is_file()
    assert Path(summary["artifacts"]["figure_svg"]).is_file()
