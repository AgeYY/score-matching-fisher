from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "plot_stringer_all_sessions_achieved_information.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_stringer_all_sessions_achieved_information", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, offset: float) -> None:
    module = _load_script()
    rows = []
    for method_index, method in enumerate(module.METHODS):
        for seed_index, seed in enumerate((7, 19)):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_fraction": 0.5,
                    "mean_achieved_fisher": offset + method_index + seed_index,
                }
            )
    path.mkdir(parents=True)
    (path / "train-test-allocation_stringer.json").write_text(
        json.dumps({"rows": rows}) + "\n", encoding="utf-8"
    )


def test_aggregate_and_plot_all_stringer_sessions(tmp_path: Path) -> None:
    module = _load_script()
    _write_summary(tmp_path / "session_00_GT1", 10.0)
    _write_summary(tmp_path / "session_01_GT2", 20.0)

    grouped = module._aggregate(tmp_path)
    assert len(grouped) == 2 * len(module.METHODS)
    assert {row["n_seeds"] for row in grouped} == {2}
    assert {tuple(row["seeds"]) for row in grouped} == {(7, 19)}
    fm_gt1 = next(
        row
        for row in grouped
        if row["session_label"] == "GT1" and row["method"] == "Flow matching"
    )
    assert fm_gt1["mean_achieved_fisher"] == 10.5
    assert fm_gt1["ci95_half_width"] > 0.0

    png, svg = module._plot(grouped, tmp_path / "figures")
    assert png.is_file()
    assert svg.is_file()
