#!/usr/bin/env python3
"""Aggregate six Stringer PCA/Flow/GKR runs into the paired-likelihood figure."""

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


def _load_visualization_module():
    path = REPO_ROOT / "bin" / "visualize_stringer_pca_flow_gkr.py"
    spec = importlib.util.spec_from_file_location("visualize_stringer_pca_flow_gkr", path)
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
        default=Path(DATA_DIR) / "stringer_pca82_flow_gkr_all_sessions",
    )
    parser.add_argument("--example-session-index", type=int, default=0)
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
    moments_path = case_dir / "selected_theta_moments.npz"
    if not summary_path.is_file() or not moments_path.is_file():
        raise FileNotFoundError(f"Incomplete session result: {case_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(moments_path, allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    return {"case_dir": case_dir, "summary": summary, "arrays": arrays}


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
    flow_means = np.asarray(
        [np.mean(case["arrays"]["flow_test_log_likelihood"]) for case in cases],
        dtype=np.float64,
    )
    gkr_means = np.asarray(
        [np.mean(case["arrays"]["gkr_test_log_likelihood"]) for case in cases],
        dtype=np.float64,
    )
    example = cases[int(args.example_session_index)]
    arrays = example["arrays"]
    module = _load_visualization_module()
    png, svg = module._plot(
        test_pc12=np.asarray(arrays["test_pc12"], dtype=np.float64),
        theta_test=np.asarray(arrays["theta_test"], dtype=np.float64),
        selected_theta=np.asarray(arrays["selected_theta"], dtype=np.float64),
        flow_mean=np.asarray(arrays["flow_mean"], dtype=np.float64),
        flow_covariance=np.asarray(arrays["flow_covariance"], dtype=np.float64),
        gkr_mean=np.asarray(arrays["gkr_mean"], dtype=np.float64),
        gkr_covariance=np.asarray(arrays["gkr_covariance"], dtype=np.float64),
        flow_test_log_likelihood=np.asarray(
            arrays["flow_test_log_likelihood"], dtype=np.float64
        ),
        gkr_test_log_likelihood=np.asarray(
            arrays["gkr_test_log_likelihood"], dtype=np.float64
        ),
        session_labels=labels,
        flow_session_log_likelihood=flow_means,
        gkr_session_log_likelihood=gkr_means,
        output_dir=root / "figures",
    )
    rows = [
        {
            "session_index": int(case["summary"]["session_index"]),
            "session_label": label,
            "n_test": int(case["summary"]["n_test"]),
            "flow_mean_test_log_likelihood": float(flow),
            "gkr_mean_test_log_likelihood": float(gkr),
            "flow_minus_gkr": float(flow - gkr),
        }
        for case, label, flow, gkr in zip(
            cases, labels, flow_means, gkr_means, strict=True
        )
    ]
    summary = {
        "n_sessions": 6,
        "example_session_index": int(args.example_session_index),
        "likelihood_definition": "mean conditional 82D Gaussian joint log likelihood",
        "sessions": rows,
        "flow_across_session_mean": float(np.mean(flow_means)),
        "gkr_across_session_mean": float(np.mean(gkr_means)),
        "paired_flow_minus_gkr_mean": float(np.mean(flow_means - gkr_means)),
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
