#!/usr/bin/env python3
"""Export linear Fisher curves from fitted Stringer log-lambda GKR models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.gkr import estimate_gkr_linear_fisher, restore_gkr_checkpoint
from fisher.shared_fisher_est import require_device
from global_setting import DATA_DIR

PERIOD = float(np.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_gkr_log_lambda_lr001_cov100_all_sessions",
    )
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--solve-jitter", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true")
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


def main() -> int:
    args = parse_args()
    if int(args.theta_grid_size) < 2:
        raise ValueError("--theta-grid-size must be at least two.")
    if float(args.solve_jitter) < 0.0:
        raise ValueError("--solve-jitter must be nonnegative.")
    device = require_device(str(args.device))
    input_root = args.input_root.expanduser().resolve()
    cases = sorted(path for path in input_root.glob("session_*") if path.is_dir())
    if len(cases) != 6:
        raise ValueError(f"Expected six session directories under {input_root}.")

    theta_grid = np.linspace(0.0, PERIOD, int(args.theta_grid_size))
    spacing = np.diff(theta_grid)
    midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    rows: list[dict[str, Any]] = []
    for case_dir in cases:
        checkpoint_path = case_dir / "conventional_gkr_model.pt"
        summary_path = case_dir / "summary.json"
        output_path = case_dir / "log_lambda_linear_fisher.npz"
        if not checkpoint_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing fitted GKR artifacts under {case_dir}.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if output_path.is_file() and not args.force:
            with np.load(output_path, allow_pickle=False) as saved:
                linear_fisher = np.asarray(saved["linear_fisher"], dtype=np.float64)
        else:
            model = restore_gkr_checkpoint(
                checkpoint_path, device=device, dtype=torch.float64
            )
            estimate = estimate_gkr_linear_fisher(
                model,
                midpoints[:, None],
                finite_difference_step=spacing[:, None],
                solve_jitter=float(args.solve_jitter),
            )
            linear_fisher = np.asarray(estimate.linear_fisher, dtype=np.float64)
            np.savez_compressed(
                output_path,
                theta_grid=theta_grid,
                theta_midpoints=midpoints,
                finite_difference_step=spacing,
                linear_fisher=linear_fisher,
                mean_jacobian=np.asarray(estimate.mean_jacobian, dtype=np.float64),
                covariance=np.asarray(estimate.covariance, dtype=np.float64),
                solve_jitter=np.asarray(float(args.solve_jitter)),
                checkpoint_path=np.asarray(str(checkpoint_path)),
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(
            {
                "session_index": int(summary["protocol"]["session_index"]),
                "session_label": str(summary["protocol"]["session"]),
                "curve": output_path,
                "mean_linear_fisher": float(np.mean(linear_fisher)),
            }
        )

    rows.sort(key=lambda row: int(row["session_index"]))
    root_summary = {
        "theta_grid_size": int(args.theta_grid_size),
        "solve_jitter": float(args.solve_jitter),
        "source": "saved periodic log-lambda GKR checkpoints",
        "sessions": rows,
    }
    summary_path = input_root / "linear_fisher_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(root_summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(root_summary), indent=2), flush=True)
    print(f"Saved: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
