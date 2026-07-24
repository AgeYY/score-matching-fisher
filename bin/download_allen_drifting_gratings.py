#!/usr/bin/env python3
"""Download Allen Brain Observatory drifting-grating response analyses."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR


API_ROOT = "https://api.brain-map.org"
RMA_QUERY_URL = f"{API_ROOT}/api/v2/data/query.json"
SESSION_TYPE = "three_session_A"
ANALYSIS_FILE_TYPE = "OphysExperimentCellRoiMetricsFile"
DRIFTING_RESPONSE_KEY = "analysis/response_dg"
BLOCK_ITEMS_PATTERN = re.compile(r"^block(\d+)_items$")


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "allen_brain_observatory_drifting_gratings",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--experiment-ids", type=_csv_ints)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    return parser.parse_args()


def _query_catalog(*, timeout: float) -> list[dict[str, Any]]:
    criteria = (
        "model::OphysExperiment,"
        f"rma::criteria,[stimulus_name$eq'{SESSION_TYPE}'],"
        "rma::include,"
        "experiment_container,"
        "well_known_files(well_known_file_type),"
        "targeted_structure,"
        "specimen(donor(age,transgenic_lines)),"
        "rma::options[num_rows$eqall][count$eqfalse]"
    )
    response = requests.get(
        RMA_QUERY_URL,
        params={"criteria": criteria},
        timeout=(30.0, timeout),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Allen RMA query failed: {payload.get('msg')}")
    return list(payload["msg"])


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def _find_transgenic_line(
    specimen: dict[str, Any], *, line_type: str
) -> str | None:
    donor = specimen.get("donor") or {}
    for line in donor.get("transgenic_lines") or []:
        name = str(line.get("name"))
        if (
            line.get("transgenic_line_type_name") == line_type
            and (line_type != "driver" or "Cre" in name)
        ):
            return name
    return None


def _analysis_file(row: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in row.get("well_known_files", [])
        if (item.get("well_known_file_type") or {}).get("name")
        == ANALYSIS_FILE_TYPE
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one analysis file for experiment {row['id']}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _catalog_session(row: dict[str, Any]) -> dict[str, Any]:
    file_info = _analysis_file(row)
    specimen = row.get("specimen") or {}
    donor = specimen.get("donor") or {}
    structure = row.get("targeted_structure") or {}
    return {
        "ophys_experiment_id": int(row["id"]),
        "experiment_container_id": int(row["experiment_container_id"]),
        "session_type": str(row["stimulus_name"]),
        "date_of_acquisition": row.get("date_of_acquisition"),
        "targeted_structure": structure.get("acronym"),
        "targeted_structure_name": structure.get("name"),
        "imaging_depth_um": int(row["imaging_depth"]),
        "specimen_id": int(row["specimen_id"]),
        "specimen_name": specimen.get("name"),
        "donor_name": donor.get("external_donor_name"),
        "cre_line": _find_transgenic_line(specimen, line_type="driver"),
        "reporter_line": _find_transgenic_line(specimen, line_type="reporter"),
        "fail_eye_tracking": row.get("fail_eye_tracking"),
        "analysis_well_known_file_id": int(file_info["id"]),
        "analysis_download_url": f"{API_ROOT}{file_info['download_link']}",
        "analysis_source_path": file_info.get("path"),
    }


def query_drifting_grating_sessions(*, timeout: float) -> tuple[list[dict[str, Any]], int]:
    rows = _query_catalog(timeout=timeout)
    valid_rows = [
        row
        for row in rows
        if not (row.get("experiment_container") or {}).get("failed", False)
    ]
    sessions = sorted(
        (_catalog_session(row) for row in valid_rows),
        key=lambda item: int(item["ophys_experiment_id"]),
    )
    return sessions, len(rows) - len(valid_rows)


def _fixed_dataframe_columns(group: h5py.Group) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    n_rows = int(np.asarray(group["axis1"]).size)
    for key in sorted(group.keys()):
        match = BLOCK_ITEMS_PATTERN.match(key)
        if match is None:
            continue
        block_index = match.group(1)
        item_names = _decode_strings(group[key][()])
        values = np.asarray(group[f"block{block_index}_values"][()])
        if values.dtype.kind == "O":
            continue
        if values.ndim == 1 and len(item_names) == 1:
            values = values.reshape(n_rows, 1)
        if values.shape == (len(item_names), n_rows):
            values = values.T
        if values.shape != (n_rows, len(item_names)):
            raise ValueError(
                f"Unexpected HDF block shape {values.shape}; "
                f"expected {(n_rows, len(item_names))}."
            )
        for index, name in enumerate(item_names):
            columns[name] = np.asarray(values[:, index])
    return columns


def _peak_cell_specimen_ids(group: h5py.Group) -> np.ndarray:
    columns = _fixed_dataframe_columns(group)
    if "cell_specimen_id" not in columns:
        raise KeyError("The analysis peak table has no cell_specimen_id column.")
    return np.asarray(columns["cell_specimen_id"], dtype=np.int64)


def inspect_analysis_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if DRIFTING_RESPONSE_KEY not in handle:
            raise KeyError(f"{path} has no {DRIFTING_RESPONSE_KEY} dataset.")
        response = handle[DRIFTING_RESPONSE_KEY]
        if response.ndim != 4 or int(response.shape[-1]) != 3:
            raise ValueError(f"Unexpected response_dg shape: {response.shape}")
        n_cells = int(response.shape[2]) - 1
        if n_cells < 1:
            raise ValueError(f"Invalid response cell dimension: {response.shape}")
        cell_ids = _peak_cell_specimen_ids(handle["analysis/peak"])
        if int(cell_ids.size) != n_cells:
            raise ValueError(
                f"response_dg has {n_cells} cells but peak has "
                f"{cell_ids.size} cell IDs."
            )
        n_trials = int(handle["analysis/stim_table_dg/axis1"].shape[0])
        return {
            "n_cells": n_cells,
            "n_trials": n_trials,
            "response_shape": [int(value) for value in response.shape],
            "cell_specimen_ids": cell_ids,
        }


def extract_analysis_npz(source_path: Path, output_path: Path) -> dict[str, Any]:
    with h5py.File(source_path, "r") as handle:
        response = np.asarray(handle[DRIFTING_RESPONSE_KEY][()])
        n_cells = int(response.shape[2]) - 1
        cell_ids = _peak_cell_specimen_ids(handle["analysis/peak"])
        mean_response = _fixed_dataframe_columns(
            handle["analysis/mean_sweep_response_dg"]
        )
        stimulus = _fixed_dataframe_columns(handle["analysis/stim_table_dg"])

        trial_cell_response = np.column_stack(
            [np.asarray(mean_response[str(index)]) for index in range(n_cells)]
        )
        trial_running_speed = np.asarray(mean_response["dx"])
        blank_sweep = np.asarray(stimulus["blank_sweep"], dtype=np.int8)
        orientation = np.asarray(stimulus["orientation"], dtype=np.float32)
        temporal_frequency = np.asarray(
            stimulus["temporal_frequency"], dtype=np.float32
        )
        nonblank = blank_sweep == 0
        orientation_values = np.unique(orientation[nonblank])
        temporal_frequency_values = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float32),
                np.unique(temporal_frequency[nonblank]),
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            cell_specimen_ids=cell_ids,
            orientation_values=orientation_values,
            temporal_frequency_values=temporal_frequency_values,
            condition_cell_response=response[:, :, :n_cells, :],
            condition_running_response=response[:, :, n_cells, :],
            trial_cell_response=trial_cell_response,
            trial_running_speed=trial_running_speed,
            trial_orientation=orientation,
            trial_temporal_frequency=temporal_frequency,
            trial_blank_sweep=blank_sweep,
            trial_start_frame=np.asarray(stimulus["start"], dtype=np.int64),
            trial_end_frame=np.asarray(stimulus["end"], dtype=np.int64),
        )
    return {
        "n_cells": n_cells,
        "n_trials": int(trial_cell_response.shape[0]),
        "response_shape": [int(value) for value in response.shape],
        "cell_specimen_ids": cell_ids,
    }


def _remote_size(url: str, *, timeout: float) -> int:
    response = requests.head(url, timeout=(30.0, timeout))
    response.raise_for_status()
    return int(response.headers.get("Content-Length", 0))


def _download(
    *,
    url: str,
    output_path: Path,
    expected_size: int,
    timeout: float,
    force: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not force:
        if expected_size <= 0 or output_path.stat().st_size == expected_size:
            return

    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    offset = 0 if force or not partial_path.is_file() else partial_path.stat().st_size
    headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30.0, timeout),
    )
    response.raise_for_status()
    append = offset > 0 and response.status_code == 206
    mode = "ab" if append else "wb"
    with partial_path.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    if expected_size > 0 and partial_path.stat().st_size != expected_size:
        raise IOError(
            f"Incomplete download for {output_path.name}: "
            f"{partial_path.stat().st_size} != {expected_size} bytes."
        )
    partial_path.replace(output_path)


def _process_session(
    session: dict[str, Any],
    *,
    output_dir: Path,
    timeout: float,
    force: bool,
    metadata_only: bool,
    skip_extract: bool,
) -> dict[str, Any]:
    experiment_id = int(session["ophys_experiment_id"])
    remote_size = _remote_size(session["analysis_download_url"], timeout=timeout)
    record = dict(session)
    record["analysis_file_bytes"] = remote_size
    if metadata_only:
        return record

    h5_path = (
        output_dir
        / "analysis_h5"
        / f"{experiment_id}_{SESSION_TYPE}_analysis.h5"
    )
    _download(
        url=str(session["analysis_download_url"]),
        output_path=h5_path,
        expected_size=remote_size,
        timeout=timeout,
        force=force,
    )
    npz_path = output_dir / "responses_npz" / f"{experiment_id}_drifting_gratings.npz"
    if skip_extract:
        analysis = inspect_analysis_file(h5_path)
    elif npz_path.is_file() and not force:
        analysis = inspect_analysis_file(h5_path)
    else:
        analysis = extract_analysis_npz(h5_path, npz_path)
    record.update(
        {
            "n_cells": int(analysis["n_cells"]),
            "n_trials": int(analysis["n_trials"]),
            "response_shape": analysis["response_shape"],
            "analysis_h5_path": str(h5_path.resolve()),
            "response_npz_path": (
                str(npz_path.resolve()) if not skip_extract else None
            ),
            "cell_specimen_ids": [
                int(value) for value in analysis["cell_specimen_ids"]
            ],
        }
    )
    return record


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "ophys_experiment_id",
        "experiment_container_id",
        "date_of_acquisition",
        "targeted_structure",
        "imaging_depth_um",
        "cre_line",
        "reporter_line",
        "donor_name",
        "specimen_name",
        "fail_eye_tracking",
        "n_cells",
        "n_trials",
        "analysis_file_bytes",
        "analysis_h5_path",
        "response_npz_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _summary(
    records: list[dict[str, Any]],
    *,
    excluded_failed_containers: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    sizes = np.asarray(
        [record.get("analysis_file_bytes", 0) for record in records],
        dtype=np.int64,
    )
    cell_counts = np.asarray(
        [record["n_cells"] for record in records if "n_cells" in record],
        dtype=np.int64,
    )
    summary: dict[str, Any] = {
        "dataset": "Allen Brain Observatory Visual Coding",
        "stimulus": "drifting_gratings",
        "session_type": SESSION_TYPE,
        "n_sessions": len(records),
        "excluded_failed_experiment_containers": excluded_failed_containers,
        "analysis_download_bytes": int(sizes.sum()),
        "analysis_download_gib": float(sizes.sum() / (1024**3)),
        "targeted_structure_counts": dict(
            sorted(Counter(record["targeted_structure"] for record in records).items())
        ),
        "cre_line_counts": dict(
            sorted(
                Counter(
                    record.get("cre_line") or "unknown" for record in records
                ).items()
            )
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if cell_counts.size:
        summary.update(
            {
                "n_sessions_with_validated_responses": int(cell_counts.size),
                "total_session_neurons": int(cell_counts.sum()),
                "unique_cell_specimen_ids": len(
                    {
                        cell_id
                        for record in records
                        for cell_id in record.get("cell_specimen_ids", [])
                    }
                ),
                "neurons_per_session": {
                    "minimum": int(cell_counts.min()),
                    "median": float(np.median(cell_counts)),
                    "mean": float(cell_counts.mean()),
                    "maximum": int(cell_counts.max()),
                },
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    if int(args.workers) < 1:
        raise ValueError("--workers must be positive.")
    if args.max_sessions is not None and int(args.max_sessions) < 1:
        raise ValueError("--max-sessions must be positive.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sessions, excluded_failed = query_drifting_grating_sessions(
        timeout=float(args.request_timeout)
    )
    (output_dir / "catalog_sessions.json").write_text(
        json.dumps(sessions, indent=2) + "\n", encoding="utf-8"
    )

    if args.experiment_ids:
        selected = set(int(value) for value in args.experiment_ids)
        sessions = [
            session
            for session in sessions
            if int(session["ophys_experiment_id"]) in selected
        ]
        missing = selected - {
            int(session["ophys_experiment_id"]) for session in sessions
        }
        if missing:
            raise ValueError(f"Unknown or invalid experiment IDs: {sorted(missing)}")
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        futures = {
            executor.submit(
                _process_session,
                session,
                output_dir=output_dir,
                timeout=float(args.request_timeout),
                force=bool(args.force),
                metadata_only=bool(args.metadata_only),
                skip_extract=bool(args.skip_extract),
            ): session
            for session in sessions
        }
        with tqdm(total=len(futures), unit="session") as progress:
            for future in concurrent.futures.as_completed(futures):
                session = futures[future]
                try:
                    records.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            "ophys_experiment_id": int(
                                session["ophys_experiment_id"]
                            ),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                progress.update(1)
    records.sort(key=lambda item: int(item["ophys_experiment_id"]))
    elapsed_seconds = time.perf_counter() - started

    _write_csv(output_dir / "sessions.csv", records)
    (output_dir / "sessions.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    summary = _summary(
        records,
        excluded_failed_containers=excluded_failed,
        elapsed_seconds=elapsed_seconds,
    )
    summary["n_failures"] = len(failures)
    summary["output_dir"] = str(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"Failures: {output_dir / 'failures.json'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
