#!/usr/bin/env python3
"""Sweep trials and dimension for four time-resolved toy RDM metrics."""

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
METRICS = ("cosine", "euclidean", "mahalanobis_sq", "fid")
METHODS = ("classical", "flow")
METRIC_LABELS = {
    "cosine": "Cosine",
    "euclidean": "Euclidean",
    "mahalanobis_sq": r"Mahalanobis$^2$",
    "fid": "FID",
}
COLORS = {"classical": "#4477AA", "flow": "#CC6677"}
METHOD_LABELS = {"classical": "Classical", "flow": "Flow"}


def _positive_int_list(text: str) -> list[int]:
    values = [int(piece.strip()) for piece in str(text).split(",") if piece.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sweep values must be unique.")
    return values


def _metric_list(text: str) -> list[str]:
    values = [piece.strip().lower() for piece in str(text).split(",") if piece.strip()]
    if not values or any(value not in METRICS for value in values):
        raise argparse.ArgumentTypeError(f"Metrics must be selected from {METRICS}.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Metrics must be unique.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/time_resolved_rdm_toy_other_metrics_scaling_r2",
    )
    parser.add_argument("--metrics", type=_metric_list, default=list(METRICS))
    parser.add_argument(
        "--trial-counts", type=_positive_int_list, default=[1, 2, 5, 10, 20, 50, 100]
    )
    parser.add_argument("--fixed-dim", type=int, default=100)
    parser.add_argument(
        "--dimensions", type=_positive_int_list, default=[3, 10, 30, 50, 70, 100]
    )
    parser.add_argument("--fixed-trials", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--population-seed", type=int, default=7)
    parser.add_argument("--repeat-seed-base", type=int, default=20_260_718)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--patience", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--covariance-ode-steps", type=int, default=48)
    parser.add_argument("--covariance-ridge", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete datasets and metric runs.",
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
        return output_dir / "trial_sweep" / f"d{x_dim}" / f"n{n_trials}" / f"repeat_{repeat_idx:02d}"
    return output_dir / "dimension_sweep" / f"n{n_trials}" / f"d{x_dim}" / f"repeat_{repeat_idx:02d}"


def _metric_dir(case_dir: Path, metric: str) -> Path:
    return case_dir / f"{metric}_classical_flow"


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _dataset_complete(case_dir: Path) -> bool:
    return _nonempty(case_dir / "two_class_time_resolved_rdm_toy.npz")


def _metric_complete(case_dir: Path, metric: str) -> bool:
    output_dir = _metric_dir(case_dir, metric)
    return all(
        _nonempty(path)
        for path in (
            output_dir / f"{metric}_classical_flow_results.npz",
            output_dir / f"{metric}_flow_best.pt",
            output_dir / "summary.json",
        )
    )


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
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
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"Command failed with code {completed.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def _ensure_dataset(
    args: argparse.Namespace,
    case_dir: Path,
    *,
    x_dim: int,
    n_trials: int,
    repeat_idx: int,
) -> None:
    if bool(args.resume) and _dataset_complete(case_dir):
        return
    command = [
        sys.executable,
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
        str(int(args.repeat_seed_base) + int(repeat_idx)),
        "--skip-plot",
    ]
    _run_logged(command, case_dir / "dataset.log")


def _run_metric(
    args: argparse.Namespace,
    case_dir: Path,
    *,
    metric: str,
    x_dim: int,
    n_trials: int,
    repeat_idx: int,
    sweep: str,
) -> None:
    if bool(args.resume) and _metric_complete(case_dir, metric):
        print(
            f"[resume] {metric} {sweep} d={x_dim} n={n_trials} repeat={repeat_idx}",
            flush=True,
        )
        return
    validation_fraction = 0.0 if int(n_trials) == 1 else 0.2
    metric_output = _metric_dir(case_dir, metric)
    command = [
        sys.executable,
        str(ROOT / "bin/fit_time_resolved_rdm_toy_metric_flow.py"),
        "--dataset-npz",
        str(case_dir / "two_class_time_resolved_rdm_toy.npz"),
        "--output-dir",
        str(metric_output),
        "--metric",
        metric,
        "--bin-width",
        str(float(args.bin_width)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.repeat_seed_base) + int(repeat_idx)),
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
        "--covariance-ode-steps",
        str(int(args.covariance_ode_steps)),
        "--covariance-ridge",
        str(float(args.covariance_ridge)),
        "--divergence-estimator",
        "hutchinson",
        "--hutchinson-probes",
        "4",
    ]
    print(
        f"[start] {metric} {sweep} d={x_dim} n={n_trials} repeat={repeat_idx}",
        flush=True,
    )
    started = time.perf_counter()
    _run_logged(command, metric_output / "run.log")
    summary = json.loads((metric_output / "summary.json").read_text(encoding="utf-8"))
    print(
        f"[done] {metric} {sweep} d={x_dim} n={n_trials} repeat={repeat_idx} "
        f"classical={summary['classical_mean_absolute_error']:.6g} "
        f"flow={summary['flow_bin_matched_mean_absolute_error']:.6g} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )


def _read_metric(case_dir: Path, metric: str) -> dict[str, Any]:
    summary_path = _metric_dir(case_dir, metric) / "summary.json"
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
    sd = np.std(values, axis=-1, ddof=1) if values.shape[-1] > 1 else np.zeros_like(mean)
    return mean, sd


def _plot_errorbar(
    axis: plt.Axes, x: np.ndarray, values: np.ndarray, method: str
) -> None:
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
        label=METHOD_LABELS[method],
    )


def _plot_sweeps(
    output_dir: Path,
    *,
    metrics: list[str],
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
        len(metrics),
        2,
        figsize=(8.0, 3.5 * len(metrics)),
        squeeze=False,
        layout="constrained",
    )
    for metric_index, metric in enumerate(metrics):
        for method_index, method in enumerate(METHODS):
            _plot_errorbar(
                axes[metric_index, 0],
                trial_counts,
                trial_errors[metric_index, method_index],
                method,
            )
            _plot_errorbar(
                axes[metric_index, 1],
                dimensions,
                dimension_errors[metric_index, method_index],
                method,
            )
        axes[metric_index, 0].set_ylabel(f"{METRIC_LABELS[metric]} MAE")
        for column, ticks in enumerate((trial_counts, dimensions)):
            axis = axes[metric_index, column]
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
    axes[0, 0].set_title(rf"Trial sweep ($D={int(fixed_dim)}$)")
    axes[0, 1].set_title(rf"Dimension sweep ($n={int(fixed_trials)}$/class)")
    axes[-1, 0].set_xlabel("Trials per class")
    axes[-1, 1].set_xlabel("Response dimension")
    axes[0, 0].legend(frameon=False, loc="best")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "other_metrics_rdm_error_vs_trials_and_dimension"
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
    if int(args.fixed_dim) < 3 or any(value < 3 for value in args.dimensions):
        raise ValueError("Controlled rotation requires dimensions >= 3.")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(args.metrics)
    trial_counts = np.asarray(args.trial_counts, dtype=np.int64)
    dimensions = np.asarray(args.dimensions, dtype=np.int64)
    trial_cases: dict[tuple[int, int], Path] = {}
    dimension_cases: dict[tuple[int, int], Path] = {}

    for repeat_idx in range(int(args.n_repeats)):
        for n_trials in trial_counts:
            case_dir = _case_dir(
                args.output_dir,
                sweep="trials",
                x_dim=int(args.fixed_dim),
                n_trials=int(n_trials),
                repeat_idx=repeat_idx,
            )
            _ensure_dataset(
                args,
                case_dir,
                x_dim=int(args.fixed_dim),
                n_trials=int(n_trials),
                repeat_idx=repeat_idx,
            )
            trial_cases[(int(n_trials), repeat_idx)] = case_dir
        for x_dim in dimensions:
            if int(x_dim) == int(args.fixed_dim) and int(args.fixed_trials) in set(
                trial_counts.tolist()
            ):
                case_dir = trial_cases[(int(args.fixed_trials), repeat_idx)]
            else:
                case_dir = _case_dir(
                    args.output_dir,
                    sweep="dimension",
                    x_dim=int(x_dim),
                    n_trials=int(args.fixed_trials),
                    repeat_idx=repeat_idx,
                )
                _ensure_dataset(
                    args,
                    case_dir,
                    x_dim=int(x_dim),
                    n_trials=int(args.fixed_trials),
                    repeat_idx=repeat_idx,
                )
            dimension_cases[(int(x_dim), repeat_idx)] = case_dir

    for metric in metrics:
        for repeat_idx in range(int(args.n_repeats)):
            for n_trials in trial_counts:
                _run_metric(
                    args,
                    trial_cases[(int(n_trials), repeat_idx)],
                    metric=metric,
                    x_dim=int(args.fixed_dim),
                    n_trials=int(n_trials),
                    repeat_idx=repeat_idx,
                    sweep="trials",
                )
            for x_dim in dimensions:
                if int(x_dim) == int(args.fixed_dim) and int(args.fixed_trials) in set(
                    trial_counts.tolist()
                ):
                    continue
                _run_metric(
                    args,
                    dimension_cases[(int(x_dim), repeat_idx)],
                    metric=metric,
                    x_dim=int(x_dim),
                    n_trials=int(args.fixed_trials),
                    repeat_idx=repeat_idx,
                    sweep="dimension",
                )

    trial_errors = np.empty(
        (len(metrics), len(METHODS), trial_counts.size, int(args.n_repeats)),
        dtype=np.float64,
    )
    dimension_errors = np.empty(
        (len(metrics), len(METHODS), dimensions.size, int(args.n_repeats)),
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        for trial_index, n_trials in enumerate(trial_counts):
            for repeat_idx in range(int(args.n_repeats)):
                result = _read_metric(trial_cases[(int(n_trials), repeat_idx)], metric)
                for method_index, method in enumerate(METHODS):
                    error = float(result[method])
                    trial_errors[metric_index, method_index, trial_index, repeat_idx] = error
                    records.append(
                        {
                            "metric": metric,
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
                result = _read_metric(dimension_cases[(int(x_dim), repeat_idx)], metric)
                for method_index, method in enumerate(METHODS):
                    error = float(result[method])
                    dimension_errors[
                        metric_index, method_index, dimension_index, repeat_idx
                    ] = error
                    records.append(
                        {
                            "metric": metric,
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

    csv_path = args.output_dir / "other_metrics_rdm_scaling_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    npz_path = args.output_dir / "other_metrics_rdm_scaling_results.npz"
    np.savez_compressed(
        npz_path,
        metrics=np.asarray(metrics),
        methods=np.asarray(METHODS),
        trial_counts=trial_counts,
        trial_errors=trial_errors,
        dimensions=dimensions,
        dimension_errors=dimension_errors,
        fixed_dim=np.asarray(int(args.fixed_dim)),
        fixed_trials=np.asarray(int(args.fixed_trials)),
        repeat_seeds=np.arange(
            int(args.repeat_seed_base),
            int(args.repeat_seed_base) + int(args.n_repeats),
        ),
    )
    png_path, svg_path = _plot_sweeps(
        args.output_dir,
        metrics=metrics,
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
        "error": "MAE against the native population metric at 24 classical-bin centers",
        "error_bars": "one sample standard deviation across two repeats",
        "metrics": metrics,
        "methods": list(METHODS),
        "metric_conventions": {
            "cosine": "one minus cosine similarity between class means",
            "euclidean": "literal Euclidean norm between class means",
            "mahalanobis_sq": "squared Mahalanobis under one class-shared covariance",
            "fid": "squared Gaussian 2-Wasserstein distance with class-specific covariances",
        },
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
        "repeat_seeds": [
            int(args.repeat_seed_base) + index for index in range(int(args.n_repeats))
        ],
        "flow": {
            "velocity_families": {
                "cosine": "translation_fixed_norm",
                "euclidean": "translation",
                "mahalanobis_sq": "covariate_affine_diag with time-only covariance context",
                "fid": "condition_affine_diag",
            },
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "covariance_ode_steps": int(args.covariance_ode_steps),
            "covariance_ridge": float(args.covariance_ridge),
            "device": str(args.device),
            "divergence_estimator": "hutchinson",
            "hutchinson_probes": 4,
            "n_equals_one_selection": (
                "same endpoints with independent fixed flow-path noise; no held-out trial"
            ),
        },
        "artifacts": {
            "rows_csv": str(csv_path.resolve()),
            "results_npz": str(npz_path.resolve()),
            "figure_png": str(png_path.resolve()),
            "figure_svg": str(svg_path.resolve()),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[sweep output] {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
