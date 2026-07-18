#!/usr/bin/env python3
"""Run repeated two-trajectory full-Fisher sample and dimension sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N_LIST = (500, 1_000, 3_000, 5_000, 10_000)
DEFAULT_DIMENSIONS = (3, 10, 30, 50, 70, 90, 110)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9, 10, 11])
    parser.add_argument("--dataset-seed", type=int, default=7)
    parser.add_argument("--sample-x-dim", type=int, default=50)
    parser.add_argument("--sample-n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    parser.add_argument("--dimension-n-total", type=int, default=3_000)
    parser.add_argument("--dimension-list", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
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
    device: str,
    cases: list[tuple[int, int, int]],
    *,
    dataset_seed: int,
    case_root: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed, x_dim, n_total in cases:
        output_dir = case_root / f"seed{seed}" / f"xdim{x_dim}_n{n_total}"
        result_path = output_dir / "two_trajectory_full_fisher_results.npz"
        if result_path.is_file() and result_path.stat().st_size > 0:
            records.append(
                {"seed": seed, "x_dim": x_dim, "n_total": n_total, "status": "reused"}
            )
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "bin" / "run_two_trajectory_full_fisher.py"),
            "--device",
            device,
            "--seed",
            str(seed),
            "--dataset-seed",
            str(dataset_seed),
            "--x-dim",
            str(x_dim),
            "--n-total",
            str(n_total),
            "--output-dir",
            str(output_dir),
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
                f"Case seed={seed}, x_dim={x_dim}, n_total={n_total} failed; see {log_path}."
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
    case_root.mkdir(parents=True, exist_ok=True)
    cases = unique_cases(args)
    assignments = [cases[index :: len(devices)] for index in range(len(devices))]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            executor.submit(
                run_worker,
                device,
                assignment,
                dataset_seed=int(args.dataset_seed),
                case_root=case_root,
            ): device
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
    run_summary = {
        "devices": devices,
        "dataset_seed": int(args.dataset_seed),
        "seeds": [int(seed) for seed in args.seeds],
        "n_cases": len(cases),
        "n_completed": sum(record["status"] == "completed" for record in records),
        "n_reused": sum(record["status"] == "reused" for record in records),
        "case_root": str(case_root),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "sweep_run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
