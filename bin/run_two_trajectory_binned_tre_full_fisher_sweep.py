#!/usr/bin/env python3
"""Run TRE-8 on completed repeated two-trajectory full-Fisher cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_NAME = "two_trajectory_binned_tre_full_fisher_results.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9, 10, 11])
    parser.add_argument("--sample-x-dim", type=int, default=50)
    parser.add_argument("--sample-n-list", type=int, nargs="+", default=[500, 1000, 3000, 5000, 10000])
    parser.add_argument("--dimension-n-total", type=int, default=3000)
    parser.add_argument("--dimension-list", type=int, nargs="+", default=[3, 10, 30, 50, 70, 90, 110])
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def unique_cases(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    cases = {
        (int(seed), int(args.sample_x_dim), int(n_total))
        for seed in args.seeds
        for n_total in args.sample_n_list
    }
    cases.update(
        (int(seed), int(x_dim), int(args.dimension_n_total))
        for seed in args.seeds
        for x_dim in args.dimension_list
    )
    return sorted(cases)


def run_worker(
    device: str, cases: list[tuple[int, int, int]], *, case_root: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed, x_dim, n_total in cases:
        case_dir = case_root / f"seed{seed}" / f"xdim{x_dim}_n{n_total}"
        result_path = case_dir / RESULTS_NAME
        if result_path.is_file() and result_path.stat().st_size > 0:
            records.append(
                {"seed": seed, "x_dim": x_dim, "n_total": n_total, "status": "reused"}
            )
            continue
        log_path = case_dir / "binned_tre_run.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "bin" / "run_two_trajectory_binned_tre_full_fisher.py"),
            "--device",
            device,
            "--case-dir",
            str(case_dir),
            "--seed",
            str(seed),
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"TRE case seed={seed}, x_dim={x_dim}, n_total={n_total} failed; "
                f"see {log_path}."
            )
        records.append(
            {"seed": seed, "x_dim": x_dim, "n_total": n_total, "status": "completed"}
        )
    return records


def main() -> int:
    args = parse_args()
    devices = list(dict.fromkeys(str(device) for device in args.devices))
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device.")
    case_root = args.case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cases = unique_cases(args)
    assignments = [cases[index :: len(devices)] for index in range(len(devices))]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            executor.submit(run_worker, device, assignment, case_root=case_root): device
            for device, assignment in zip(devices, assignments, strict=True)
        }
        for future in as_completed(futures):
            records.extend(future.result())

    plot_command = [
        sys.executable,
        str(REPO_ROOT / "bin" / "plot_two_trajectory_full_fisher_three_panel.py"),
        "--case-root",
        str(case_root),
        "--output-dir",
        str(output_dir),
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--sample-x-dim",
        str(args.sample_x_dim),
        "--sample-n-list",
        *(str(value) for value in args.sample_n_list),
        "--dimension-n-total",
        str(args.dimension_n_total),
        "--dimension-list",
        *(str(value) for value in args.dimension_list),
    ]
    subprocess.run(plot_command, cwd=REPO_ROOT, check=True)
    summary = {
        "devices": devices,
        "seeds": [int(seed) for seed in args.seeds],
        "n_cases": len(cases),
        "n_completed": sum(record["status"] == "completed" for record in records),
        "n_reused": sum(record["status"] == "reused" for record in records),
        "case_root": str(case_root),
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "binned_tre_sweep_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
