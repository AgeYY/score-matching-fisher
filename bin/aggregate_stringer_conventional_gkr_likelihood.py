#!/usr/bin/env python3
"""Replace GKR in the six-session likelihood figure with saved GKR fits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR

METHOD_KEYS = (
    "gkr_test_log_likelihood",
    "binned_test_log_likelihood",
    "affine_test_log_likelihood",
    "nonlinear_test_log_likelihood",
)
METHOD_LABELS = ("Binning + LW", "GKR", "Affine Flow", "Nonlinear Flow")


def _load_visualization_module():
    path = REPO_ROOT / "bin" / "visualize_stringer_pca_four_methods.py"
    spec = importlib.util.spec_from_file_location(
        "visualize_stringer_pca_four_methods_conventional_gkr", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_pca82_four_methods_all_sessions",
    )
    parser.add_argument(
        "--gkr-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_gkr_conventional_kernel_all_sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_pca82_four_methods_conventional_gkr_all_sessions",
    )
    parser.add_argument("--example-session-index", type=int, default=0)
    parser.add_argument("--display-generated-samples", type=int, default=600)
    parser.add_argument("--gkr-title", default="GKR")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _case_directories(root: Path) -> list[Path]:
    cases = sorted(path for path in root.glob("session_*") if path.is_dir())
    if len(cases) != 6:
        raise ValueError(f"Expected six session directories under {root}, found {len(cases)}.")
    return cases


def main() -> int:
    args = parse_args()
    original_root = args.original_root.expanduser().resolve()
    gkr_root = args.gkr_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    original_cases = _case_directories(original_root)
    gkr_cases = _case_directories(gkr_root)
    rows: list[dict[str, Any]] = []
    example: dict[str, np.ndarray] | None = None
    example_gkr: dict[str, np.ndarray] | None = None
    for original_case, gkr_case in zip(original_cases, gkr_cases, strict=True):
        original_summary = json.loads(
            (original_case / "summary.json").read_text(encoding="utf-8")
        )
        gkr_summary = json.loads(
            (gkr_case / "summary.json").read_text(encoding="utf-8")
        )
        session_index = int(original_summary["session_index"])
        if session_index != int(gkr_summary["protocol"]["session_index"]):
            raise ValueError("Original and conventional-GKR session ordering differs.")
        with np.load(
            original_case / "four_method_results.npz", allow_pickle=False
        ) as saved:
            original = {key: np.asarray(saved[key]) for key in saved.files}
        with np.load(
            gkr_case / "conventional_gkr_results.npz", allow_pickle=False
        ) as saved:
            conventional = {key: np.asarray(saved[key]) for key in saved.files}
        values = np.asarray(
            [
                np.mean(original["binned_test_log_likelihood"]),
                np.mean(conventional["conventional_test_log_likelihood"]),
                np.mean(original["affine_test_log_likelihood"]),
                np.mean(original["nonlinear_test_log_likelihood"]),
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "session_index": session_index,
                "session_label": str(original_summary["session_label"]),
                "method_means": values,
                "original_gkr_mean": float(
                    np.mean(original["gkr_test_log_likelihood"])
                ),
            }
        )
        if session_index == int(args.example_session_index):
            example = original
            example_gkr = conventional
    if example is None or example_gkr is None:
        raise ValueError("example-session-index did not match a completed session.")

    session_values = np.vstack([row["method_means"] for row in rows])
    labels = [str(row["session_label"]) for row in rows]
    module = _load_visualization_module()
    png, svg = module._plot(
        test_pc12=np.asarray(example["test_pc12"], dtype=np.float64),
        theta_test=np.asarray(example["theta_test"], dtype=np.float64),
        selected_theta=np.asarray(example["selected_theta"], dtype=np.float64),
        affine_mean=np.asarray(example["affine_mean"], dtype=np.float64),
        affine_covariance=np.asarray(example["affine_covariance"], dtype=np.float64),
        gkr_mean=np.asarray(example_gkr["conventional_mean"], dtype=np.float64),
        gkr_covariance=np.asarray(
            example_gkr["conventional_covariance"], dtype=np.float64
        ),
        binned_mean=np.asarray(example["binned_selected_mean"], dtype=np.float64),
        binned_covariance=np.asarray(
            example["binned_selected_covariance"], dtype=np.float64
        ),
        generated_pc12=np.asarray(
            example["nonlinear_generated_x"], dtype=np.float64
        )[:, :2],
        generated_theta=np.asarray(
            example["nonlinear_generated_theta"], dtype=np.float64
        ),
        likelihoods={
            "Bin +\nLW": np.asarray(
                example["binned_test_log_likelihood"], dtype=np.float64
            ),
            "GKR": np.asarray(
                example_gkr["conventional_test_log_likelihood"], dtype=np.float64
            ),
            "Affine\nFlow": np.asarray(
                example["affine_test_log_likelihood"], dtype=np.float64
            ),
            "Nonlinear\nFlow": np.asarray(
                example["nonlinear_test_log_likelihood"], dtype=np.float64
            ),
        },
        display_generated_samples=int(args.display_generated_samples),
        likelihood_session_values=session_values,
        session_labels=labels,
        gkr_title=str(args.gkr_title),
        binned_title="Bin + LW",
        moment_order=("binned", "gkr", "affine"),
        likelihood_short_labels=[
            "Bin+LW",
            "GKR",
            "Affine\nFlow",
            "Uncon.\nFlow",
        ],
        likelihood_reference_label="Bin +\nLW",
        output_dir=output_dir / "figures",
    )
    means = np.mean(session_values, axis=0)
    sem = np.std(session_values, axis=0, ddof=1) / np.sqrt(session_values.shape[0])
    original_gkr = np.asarray(
        [row["original_gkr_mean"] for row in rows], dtype=np.float64
    )
    summary = {
        "protocol": {
            "description": (
                "Original PCA-82 and 80% fit / 20% held-out test split. "
                "Only the saved GKR fit is replaced."
            ),
            "gkr_root": gkr_root,
            "gkr_kernel_parameterization": str(
                gkr_summary["protocol"]["kernel_parameterization"]
            ),
            "gkr_covariance_epochs": int(
                gkr_summary["protocol"]["covariance_epochs"]
            ),
        },
        "sessions": [
            {
                "session_index": int(row["session_index"]),
                "session_label": str(row["session_label"]),
                "original_gkr": float(row["original_gkr_mean"]),
                **{
                    method: float(value)
                    for method, value in zip(
                        METHOD_LABELS, row["method_means"], strict=True
                    )
                },
            }
            for row in rows
        ],
        "across_session": {
            method: {"mean": float(mean), "standard_error": float(error)}
            for method, mean, error in zip(
                METHOD_LABELS, means, sem, strict=True
            )
        },
        "original_gkr_across_session_mean": float(np.mean(original_gkr)),
        "replacement_gkr_minus_original_mean": float(
            np.mean(session_values[:, 1] - original_gkr)
        ),
        "replacement_gkr_minus_bin_lw_mean": float(
            np.mean(session_values[:, 1] - session_values[:, 0])
        ),
        "artifacts": {"png": png, "svg": svg},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary), indent=2), flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
