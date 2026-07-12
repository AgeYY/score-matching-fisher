#!/usr/bin/env python3
"""Run controlled MoG5 dataset-parameter screens for FM distance estimation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "data" / "mog5_dataset_hparam_screen" / "screen"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    seed: int = 7
    obs_noise_scale: float = 1.0
    cov_theta_amp_scale: float = 1.0
    mean_min_dist: float | None = None


SCREEN_CONFIGS = (
    DatasetConfig("baseline_seed7"),
    DatasetConfig("seed19", seed=19),
    DatasetConfig("seed31", seed=31),
    DatasetConfig("noise075", obs_noise_scale=0.75),
    DatasetConfig("noise075_seed19", seed=19, obs_noise_scale=0.75),
    DatasetConfig("noise075_seed31", seed=31, obs_noise_scale=0.75),
    DatasetConfig("noise125", obs_noise_scale=1.25),
    DatasetConfig("cov050", cov_theta_amp_scale=0.5),
    DatasetConfig("cov050_seed19", seed=19, cov_theta_amp_scale=0.5),
    DatasetConfig("cov050_seed31", seed=31, cov_theta_amp_scale=0.5),
    DatasetConfig("cov200", cov_theta_amp_scale=2.0),
    DatasetConfig("noise075_cov050", obs_noise_scale=0.75, cov_theta_amp_scale=0.5),
    DatasetConfig("sep060", mean_min_dist=0.6),
    DatasetConfig("sep110", mean_min_dist=1.1),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--n-list", type=str, default="1000")
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Comma-separated configuration names or 'all'.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip configurations whose aggregate summary JSON already exists.",
    )
    return parser


def select_configs(value: str) -> tuple[DatasetConfig, ...]:
    if str(value).strip().lower() == "all":
        return SCREEN_CONFIGS
    requested = [part.strip() for part in str(value).split(",") if part.strip()]
    by_name = {config.name: config for config in SCREEN_CONFIGS}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown screen configuration(s): {', '.join(missing)}")
    return tuple(by_name[name] for name in requested)


def build_command(args: argparse.Namespace, config: DatasetConfig) -> list[str]:
    output_dir = Path(args.output_root).expanduser() / config.name
    command = [
        sys.executable,
        str(_REPO_ROOT / "bin" / "compare_mog5_pr_distance_sweeps_parallel.py"),
        "--device",
        str(args.device),
        "--gpu-ids",
        str(args.gpu_ids),
        "--pr-dim",
        "none",
        "--native-x-dim",
        "3",
        "--n-list",
        str(args.n_list),
        "--n-repeats",
        str(int(args.n_repeats)),
        "--seed",
        str(int(config.seed)),
        "--dataset-obs-noise-scale",
        str(float(config.obs_noise_scale)),
        "--dataset-cov-theta-amp-scale",
        str(float(config.cov_theta_amp_scale)),
        "--case-output-name",
        f"distance_comparison_dataset_screen_{config.name}",
        "--output-dir",
        str(output_dir),
        "--force-dataset",
        "--force-comparison",
        "--skip-dataset-viz",
    ]
    if config.mean_min_dist is not None:
        command.extend(["--dataset-mog-mean-min-dist", str(float(config.mean_min_dist))])
    return command


def main() -> None:
    args = build_parser().parse_args()
    configs = select_configs(args.configs)
    for index, config in enumerate(configs, start=1):
        output_dir = Path(args.output_root).expanduser() / config.name
        summary_path = output_dir / "mog5_pr_distance_sweep_summary.json"
        if bool(args.reuse_existing) and summary_path.is_file():
            print(f"[dataset-screen] reuse {config.name}: {summary_path}", flush=True)
            continue
        print(f"[dataset-screen] {index}/{len(configs)} start {config.name}", flush=True)
        subprocess.run(build_command(args, config), cwd=_REPO_ROOT, check=True)
        print(f"[dataset-screen] {index}/{len(configs)} done {config.name}", flush=True)


if __name__ == "__main__":
    main()
