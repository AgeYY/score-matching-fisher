#!/usr/bin/env python3
"""Aggregate six Stringer four-method density comparisons into one figure."""

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
    "affine_test_log_likelihood",
    "gkr_test_log_likelihood",
    "binned_test_log_likelihood",
    "nonlinear_test_log_likelihood",
)
METHOD_LABELS = ("Affine Flow", "GKR", "Binning + LW", "Nonlinear Flow")


def _load_visualization_module():
    path = REPO_ROOT / "bin" / "visualize_stringer_pca_four_methods.py"
    spec = importlib.util.spec_from_file_location(
        "visualize_stringer_pca_four_methods", path
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
        "--input-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_pca82_four_methods_all_sessions",
    )
    parser.add_argument("--example-session-index", type=int, default=0)
    parser.add_argument("--display-generated-samples", type=int, default=600)
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


def _load_case(case_dir: Path) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    result_path = case_dir / "four_method_results.npz"
    if not summary_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Incomplete four-method session result: {case_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(result_path, allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    missing = [key for key in METHOD_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"{case_dir} is missing likelihood arrays: {missing}")
    return {"case_dir": case_dir, "summary": summary, "arrays": arrays}


def _session_mean_likelihoods(case: dict[str, Any]) -> np.ndarray:
    arrays = case["arrays"]
    return np.asarray([np.mean(arrays[key]) for key in METHOD_KEYS], dtype=np.float64)


def main() -> int:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    cases = [_load_case(path) for path in sorted(root.glob("session_*"))]
    if len(cases) != 6:
        raise ValueError(f"Expected exactly six session results under {root}, found {len(cases)}.")
    indices = [int(case["summary"]["session_index"]) for case in cases]
    if sorted(indices) != list(range(6)):
        raise ValueError(f"Expected session indices 0 through 5, found {indices}.")
    cases.sort(key=lambda case: int(case["summary"]["session_index"]))
    labels = [str(case["summary"]["session_label"]) for case in cases]
    session_values = np.vstack([_session_mean_likelihoods(case) for case in cases])
    plot_order = np.asarray([1, 2, 0, 3], dtype=np.int64)
    plot_session_values = session_values[:, plot_order]
    example_matches = [
        case
        for case in cases
        if int(case["summary"]["session_index"]) == int(args.example_session_index)
    ]
    if len(example_matches) != 1:
        raise ValueError("example-session-index must identify exactly one completed session.")
    example = example_matches[0]
    arrays = example["arrays"]
    module = _load_visualization_module()
    png, svg = module._plot(
        test_pc12=np.asarray(arrays["test_pc12"], dtype=np.float64),
        theta_test=np.asarray(arrays["theta_test"], dtype=np.float64),
        selected_theta=np.asarray(arrays["selected_theta"], dtype=np.float64),
        affine_mean=np.asarray(arrays["affine_mean"], dtype=np.float64),
        affine_covariance=np.asarray(arrays["affine_covariance"], dtype=np.float64),
        gkr_mean=np.asarray(arrays["gkr_mean"], dtype=np.float64),
        gkr_covariance=np.asarray(arrays["gkr_covariance"], dtype=np.float64),
        binned_mean=np.asarray(arrays["binned_selected_mean"], dtype=np.float64),
        binned_covariance=np.asarray(
            arrays["binned_selected_covariance"], dtype=np.float64
        ),
        generated_pc12=np.asarray(arrays["nonlinear_generated_x"], dtype=np.float64)[
            :, :2
        ],
        generated_theta=np.asarray(
            arrays["nonlinear_generated_theta"], dtype=np.float64
        ),
        likelihoods={
            "GKR": np.asarray(arrays[METHOD_KEYS[1]], dtype=np.float64),
            "Bin +\nLW": np.asarray(arrays[METHOD_KEYS[2]], dtype=np.float64),
            "Affine\nFlow": np.asarray(arrays[METHOD_KEYS[0]], dtype=np.float64),
            "Nonlinear\nFlow": np.asarray(arrays[METHOD_KEYS[3]], dtype=np.float64),
        },
        display_generated_samples=int(args.display_generated_samples),
        likelihood_session_values=plot_session_values,
        session_labels=labels,
        output_dir=root / "figures",
    )
    rows = []
    for case, label, values in zip(cases, labels, session_values, strict=True):
        row = {
            "session_index": int(case["summary"]["session_index"]),
            "session_label": label,
            "n_test": int(case["summary"]["n_test"]),
            "binning_test_overlap": int(
                case["summary"]["binning"]["held_out_test_overlap"]
            ),
        }
        row.update(
            {
                f"{key}_mean_test_log_likelihood": float(value)
                for key, value in zip(
                    ("affine_flow", "gkr", "binning_lw", "nonlinear_flow"),
                    values,
                    strict=True,
                )
            }
        )
        rows.append(row)
    across_mean = np.mean(session_values, axis=0)
    across_sem = np.std(session_values, axis=0, ddof=1) / np.sqrt(session_values.shape[0])
    summary = {
        "n_sessions": 6,
        "example_session_index": int(args.example_session_index),
        "display_generated_samples": int(args.display_generated_samples),
        "aggregation": "mean and SEM across six session-level mean held-out log likelihoods",
        "sessions": rows,
        "across_session": {
            label: {"mean": float(mean), "standard_error": float(sem)}
            for label, mean, sem in zip(
                METHOD_LABELS, across_mean, across_sem, strict=True
            )
        },
        "likelihood_note": (
            "Affine Flow, GKR, and binning use conditional Gaussian likelihoods; "
            "nonlinear Flow uses its conditional CNF likelihood."
        ),
        "artifacts": {"png": str(png), "svg": str(svg)},
    }
    summary_path = root / "all_sessions_summary.json"
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
