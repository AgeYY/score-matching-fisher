from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "download_allen_drifting_gratings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "download_allen_drifting_gratings", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixed_group(
    parent: h5py.Group,
    name: str,
    columns: dict[str, np.ndarray],
) -> None:
    group = parent.create_group(name)
    names = list(columns)
    values = np.column_stack([columns[key] for key in names])
    group.create_dataset("axis0", data=np.asarray(names, dtype="S32"))
    group.create_dataset("axis1", data=np.arange(values.shape[0]))
    group.create_dataset("block0_items", data=np.asarray(names, dtype="S32"))
    group.create_dataset("block0_values", data=values)


def _write_analysis(path: Path) -> None:
    n_cells, n_trials = 3, 4
    with h5py.File(path, "w") as handle:
        analysis = handle.create_group("analysis")
        analysis.create_dataset(
            "response_dg",
            data=np.arange(8 * 6 * (n_cells + 1) * 3).reshape(
                8, 6, n_cells + 1, 3
            ),
        )
        _write_fixed_group(
            analysis,
            "peak",
            {
                "cell_specimen_id": np.asarray([101, 102, 103]),
                "ori_dg": np.asarray([0, 1, 2]),
            },
        )
        _write_fixed_group(
            analysis,
            "mean_sweep_response_dg",
            {
                "0": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "1": np.asarray([2.0, 3.0, 4.0, 5.0]),
                "2": np.asarray([3.0, 4.0, 5.0, 6.0]),
                "dx": np.asarray([0.0, 1.0, 2.0, 3.0]),
            },
        )
        _write_fixed_group(
            analysis,
            "stim_table_dg",
            {
                "temporal_frequency": np.asarray([0.0, 1.0, 2.0, 4.0]),
                "orientation": np.asarray([0.0, 0.0, 45.0, 90.0]),
                "blank_sweep": np.asarray([1, 0, 0, 0]),
                "start": np.asarray([0, 10, 20, 30]),
                "end": np.asarray([5, 15, 25, 35]),
            },
        )


def test_inspect_and_extract_analysis(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "analysis.h5"
    output = tmp_path / "response.npz"
    _write_analysis(source)

    inspected = module.inspect_analysis_file(source)
    assert inspected["n_cells"] == 3
    assert inspected["n_trials"] == 4
    assert inspected["response_shape"] == [8, 6, 4, 3]
    np.testing.assert_array_equal(
        inspected["cell_specimen_ids"], [101, 102, 103]
    )

    extracted = module.extract_analysis_npz(source, output)
    assert extracted["n_cells"] == 3
    with np.load(output) as saved:
        assert saved["condition_cell_response"].shape == (8, 6, 3, 3)
        assert saved["trial_cell_response"].shape == (4, 3)
        np.testing.assert_array_equal(
            saved["orientation_values"], [0.0, 45.0, 90.0]
        )
        np.testing.assert_array_equal(
            saved["temporal_frequency_values"], [0.0, 1.0, 2.0, 4.0]
        )


def test_catalog_session_extracts_metadata() -> None:
    module = _load_script()
    row = {
        "id": 123,
        "experiment_container_id": 456,
        "stimulus_name": "three_session_A",
        "date_of_acquisition": "2017-01-01T00:00:00Z",
        "imaging_depth": 275,
        "specimen_id": 789,
        "fail_eye_tracking": False,
        "targeted_structure": {"acronym": "VISp", "name": "Primary visual area"},
        "experiment_container": {"failed": False},
        "specimen": {
            "name": "specimen",
            "donor": {
                "external_donor_name": "mouse",
                "transgenic_lines": [
                    {
                        "name": "Driver-Cre",
                        "transgenic_line_type_name": "driver",
                    },
                    {
                        "name": "Reporter",
                        "transgenic_line_type_name": "reporter",
                    },
                ],
            },
        },
        "well_known_files": [
            {
                "id": 999,
                "download_link": "/download/999",
                "path": "/source/file.h5",
                "well_known_file_type": {
                    "name": "OphysExperimentCellRoiMetricsFile"
                },
            }
        ],
    }
    record = module._catalog_session(row)
    assert record["ophys_experiment_id"] == 123
    assert record["targeted_structure"] == "VISp"
    assert record["cre_line"] == "Driver-Cre"
    assert record["reporter_line"] == "Reporter"
    assert record["analysis_download_url"].endswith("/download/999")
