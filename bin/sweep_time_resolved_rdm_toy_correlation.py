#!/usr/bin/env python3
"""Sweep trial count and response dimension for the controlled-rotation RDM toy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("classical", "flow")
COLORS = {"classical": "#4477AA", "flow": "#CC6677"}
LABELS = {"classical": "Classical", "flow": "Flow"}


def _positive_int_list(text: str) -> list[int]:
    values = [int(piece.strip()) for piece in str(text).split(",") if piece.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sweep values must be unique.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/time_resolved_rdm_toy_correlation_scaling",
    )
    parser.add_argument("--trial-counts", type=_positive_int_list, default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--fixed-dim", type=int, default=100)
    parser.add_argument("--dimensions", type=_positive_int_list, default=[3, 10, 30, 50, 70, 100])
    parser.add_argument("--fixed-trials", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--population-seed", type=int, default=7)
    parser.add_argument("--repeat-seed-base", type=int, default=20_260_718)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--patience", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cases with a complete summary, results archive, and best checkpoint.",
    )
    return parser.parse_args()


def _case_dir(
    output_dir: Path,
    *,
    sweep: str,
    x_dim: int,
    n_trials: int,
    repeat_idx: int,
) -> Path:
    if sweep == "trials":
        return output_dir / "trial_sweep" / f"d{int(x_dim)}" / f"n{int(n_trials)}" / f"repeat_{repeat_idx:02d}"
    return output_dir / "dimension_sweep" / f"n{int(n_trials)}" / f"d{int(x_dim)}" / f"repeat_{repeat_idx:02d}"


def _case_complete(case_dir: Path) -> bool:
    flow_dir = case_dir / "correlation_classical_flow"
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            case_dir / "two_class_time_resolved_rdm_toy.npz",
            case_dir / "classical_correlation_bin0p5" / "binned_correlation_distance.npz",
            flow_dir / "correlation_classical_flow_results.npz",
            flow_dir / "correlation_flow_best.pt",
            flow_dir / "summary.json",
        )
    )


def _run_commands(commands: list[list[str]], *, case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                raise RuntimeError(
                    f"Case command failed with code {completed.returncode}: {' '.join(command)}\n"
                    + "\n".join(tail)
                )


def _run_case(
    args: argparse.Namespace,
    *,
    sweep: str,
    x_dim: int,
    n_trials: int,
    repeat_idx: int,
) -> Path:
    case_dir = _case_dir(
        args.output_dir,
        sweep=sweep,
        x_dim=x_dim,
        n_trials=n_trials,
        repeat_idx=repeat_idx,
    )
    if bool(args.resume) and _case_complete(case_dir):
        print(f"[sweep resume] {sweep} d={x_dim} n={n_trials} repeat={repeat_idx}", flush=True)
        return case_dir

    repeat_seed = int(args.repeat_seed_base) + int(repeat_idx)
    dataset_npz = case_dir / "two_class_time_resolved_rdm_toy.npz"
    classical_dir = case_dir / "classical_correlation_bin0p5"
    flow_dir = case_dir / "correlation_classical_flow"
    validation_fraction = 0.0 if int(n_trials) == 1 else 0.2
    python = sys.executable
    commands = [
        [
            python,
            str(ROOT / "bin/make_time_resolved_rdm_toy.py"),
            "--output-dir",
            str(case_dir),
            "--x-dim",
            str(int(x_dim)),
            "--n-trials-per-class",
            str(int(n_trials)),
            "--trajectory-mode",
            "controlled_rotation",
            "--seed",
            str(int(args.population_seed)),
            "--sample-seed",
            str(repeat_seed),
            "--correlation-only",
            "--skip-plot",
        ],
        [
            python,
            str(ROOT / "bin/estimate_time_resolved_rdm_toy_correlation.py"),
            "--dataset-npz",
            str(dataset_npz),
            "--output-dir",
            str(classical_dir),
            "--bin-width",
            str(float(args.bin_width)),
            "--skip-plot",
        ],
        [
            python,
            str(ROOT / "bin/fit_time_resolved_rdm_toy_correlation_flow.py"),
            "--dataset-npz",
            str(dataset_npz),
            "--classical-npz",
            str(classical_dir / "binned_correlation_distance.npz"),
            "--output-dir",
            str(flow_dir),
            "--device",
            str(args.device),
            "--seed",
            str(repeat_seed),
            "--validation-fraction",
            str(validation_fraction),
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--batch-size",
            str(int(args.batch_size)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--hidden-dim",
            str(int(args.hidden_dim)),
            "--depth",
            str(int(args.depth)),
            "--divergence-estimator",
            "hutchinson",
            "--hutchinson-probes",
            "4",
            "--skip-plots",
        ],
    ]
    print(f"[sweep start] {sweep} d={x_dim} n={n_trials} repeat={repeat_idx}", flush=True)
    started = time.perf_counter()
    _run_commands(commands, case_dir=case_dir)
    summary = json.loads((flow_dir / "summary.json").read_text(encoding="utf-8"))
    print(
        f"[sweep done] {sweep} d={x_dim} n={n_trials} repeat={repeat_idx} "
        f"classical={summary['classical_mean_absolute_error']:.6g} "
        f"flow={summary['flow_bin_matched_mean_absolute_error']:.6g} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return case_dir


def _read_case(case_dir: Path) -> dict[str, Any]:
    summary_path = case_dir / "correlation_classical_flow" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "classical": float(summary["classical_mean_absolute_error"]),
        "flow": float(summary["flow_bin_matched_mean_absolute_error"]),
        "best_epoch": int(summary["best_epoch"]),
        "stopped_epoch": int(summary["stopped_epoch"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "summary_path": str(summary_path.resolve()),
    }


def _mean_sd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=-1)
    if values.shape[-1] > 1:
        sd = np.std(values, axis=-1, ddof=1)
    else:
        sd = np.zeros_like(mean)
    return mean, sd


def _plot_errorbar(axis: plt.Axes, x: np.ndarray, values: np.ndarray, method: str) -> None:
    mean, sd = _mean_sd(values)
    lower_endpoint = np.maximum(mean - sd, np.finfo(np.float64).tiny)
    asymmetric = np.stack([mean - lower_endpoint, sd], axis=0)
    axis.errorbar(
        x,
        mean,
        yerr=asymmetric,
        color=COLORS[method],
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        elinewidth=1.2,
        capsize=3.0,
        label=LABELS[method],
    )


def _plot_sweeps(
    output_dir: Path,
    *,
    trial_counts: np.ndarray,
    trial_errors: np.ndarray,
    dimensions: np.ndarray,
    dimension_errors: np.ndarray,
    fixed_dim: int,
    fixed_trials: int,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.5), sharey=True, layout="constrained"
    )
    for method_index, method in enumerate(METHODS):
        _plot_errorbar(axes[0], trial_counts, trial_errors[method_index], method)
        _plot_errorbar(axes[1], dimensions, dimension_errors[method_index], method)
    axes[0].set_title(rf"$D={int(fixed_dim)}$")
    axes[0].set_xlabel("Trials per class")
    axes[0].set_ylabel("Mean absolute error")
    axes[1].set_title(rf"$n={int(fixed_trials)}$ trials/class")
    axes[1].set_xlabel("Response dimension")
    for axis, ticks in zip(axes, (trial_counts, dimensions), strict=True):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(ticks)
        axis.set_xlim(float(ticks[0]) / 1.25, float(ticks[-1]) * 1.25)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8, length=4)
    axes[0].legend(frameon=False, loc="best")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "correlation_rdm_error_vs_trials_and_dimension"
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(png_path, dpi=300, pad_inches=0.08)
    figure.savefig(svg_path, pad_inches=0.08)
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    args = parse_args()
    if str(args.device) != "cuda:0":
        raise ValueError("This project requires --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    if int(args.n_repeats) < 1:
        raise ValueError("n_repeats must be positive.")
    if int(args.fixed_dim) < 3 or any(int(value) < 3 for value in args.dimensions):
        raise ValueError("Controlled rotation requires every response dimension to be >= 3.")
    if int(args.fixed_trials) < 1:
        raise ValueError("fixed_trials must be positive.")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_counts = np.asarray(args.trial_counts, dtype=np.int64)
    dimensions = np.asarray(args.dimensions, dtype=np.int64)
    trial_cases: dict[tuple[int, int], Path] = {}
    dimension_cases: dict[tuple[int, int], Path] = {}

    for repeat_idx in range(int(args.n_repeats)):
        for n_trials in trial_counts:
            trial_cases[(int(n_trials), repeat_idx)] = _run_case(
                args,
                sweep="trials",
                x_dim=int(args.fixed_dim),
                n_trials=int(n_trials),
                repeat_idx=repeat_idx,
            )
        for x_dim in dimensions:
            if int(x_dim) == int(args.fixed_dim) and int(args.fixed_trials) in set(trial_counts.tolist()):
                dimension_cases[(int(x_dim), repeat_idx)] = trial_cases[(int(args.fixed_trials), repeat_idx)]
            else:
                dimension_cases[(int(x_dim), repeat_idx)] = _run_case(
                    args,
                    sweep="dimension",
                    x_dim=int(x_dim),
                    n_trials=int(args.fixed_trials),
                    repeat_idx=repeat_idx,
                )

    trial_errors = np.empty((len(METHODS), trial_counts.size, int(args.n_repeats)), dtype=np.float64)
    dimension_errors = np.empty((len(METHODS), dimensions.size, int(args.n_repeats)), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for trial_index, n_trials in enumerate(trial_counts):
        for repeat_idx in range(int(args.n_repeats)):
            case_dir = trial_cases[(int(n_trials), repeat_idx)]
            result = _read_case(case_dir)
            for method_index, method in enumerate(METHODS):
                error = float(result[method])
                trial_errors[method_index, trial_index, repeat_idx] = error
                records.append(
                    {
                        "sweep": "trials",
                        "x_dim": int(args.fixed_dim),
                        "n_trials_per_class": int(n_trials),
                        "repeat_idx": repeat_idx,
                        "repeat_seed": int(args.repeat_seed_base) + repeat_idx,
                        "method": method,
                        "mae": error,
                        "summary_path": result["summary_path"],
                    }
                )
    for dimension_index, x_dim in enumerate(dimensions):
        for repeat_idx in range(int(args.n_repeats)):
            case_dir = dimension_cases[(int(x_dim), repeat_idx)]
            result = _read_case(case_dir)
            for method_index, method in enumerate(METHODS):
                error = float(result[method])
                dimension_errors[method_index, dimension_index, repeat_idx] = error
                records.append(
                    {
                        "sweep": "dimension",
                        "x_dim": int(x_dim),
                        "n_trials_per_class": int(args.fixed_trials),
                        "repeat_idx": repeat_idx,
                        "repeat_seed": int(args.repeat_seed_base) + repeat_idx,
                        "method": method,
                        "mae": error,
                        "summary_path": result["summary_path"],
                    }
                )

    csv_path = args.output_dir / "correlation_rdm_scaling_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    npz_path = args.output_dir / "correlation_rdm_scaling_results.npz"
    np.savez_compressed(
        npz_path,
        methods=np.asarray(METHODS),
        trial_counts=trial_counts,
        trial_errors=trial_errors,
        dimensions=dimensions,
        dimension_errors=dimension_errors,
        fixed_dim=np.asarray(int(args.fixed_dim)),
        fixed_trials=np.asarray(int(args.fixed_trials)),
        repeat_seeds=np.arange(int(args.repeat_seed_base), int(args.repeat_seed_base) + int(args.n_repeats)),
    )
    png_path, svg_path = _plot_sweeps(
        args.output_dir,
        trial_counts=trial_counts,
        trial_errors=trial_errors,
        dimensions=dimensions,
        dimension_errors=dimension_errors,
        fixed_dim=int(args.fixed_dim),
        fixed_trials=int(args.fixed_trials),
    )
    trial_mean, trial_sd = _mean_sd(trial_errors)
    dimension_mean, dimension_sd = _mean_sd(dimension_errors)
    summary = {
        "error": "mean absolute error against native population correlation distance at the 24 classical bin centers",
        "error_bars": "one sample standard deviation across repeats",
        "methods": list(METHODS),
        "trial_sweep": {
            "fixed_dim": int(args.fixed_dim),
            "trial_counts": trial_counts.tolist(),
            "mean": trial_mean.tolist(),
            "sample_sd": trial_sd.tolist(),
        },
        "dimension_sweep": {
            "fixed_trials_per_class": int(args.fixed_trials),
            "dimensions": dimensions.tolist(),
            "mean": dimension_mean.tolist(),
            "sample_sd": dimension_sd.tolist(),
        },
        "n_repeats": int(args.n_repeats),
        "population_seed": int(args.population_seed),
        "repeat_seeds": [int(args.repeat_seed_base) + idx for idx in range(int(args.n_repeats))],
        "flow": {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "device": str(args.device),
            "divergence_estimator": "hutchinson",
            "hutchinson_probes": 4,
            "n_equals_one_selection": "same endpoints with independent fixed flow-path noise; no held-out trial",
        },
        "artifacts": {
            "rows_csv": str(csv_path.resolve()),
            "results_npz": str(npz_path.resolve()),
            "figure_png": str(png_path.resolve()),
            "figure_svg": str(svg_path.resolve()),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[sweep output] {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
