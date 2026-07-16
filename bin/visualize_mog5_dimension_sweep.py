#!/usr/bin/env python3
"""Aggregate fixed-N MoG5 runs and plot distance error versus native x dimension."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR

METRIC_ORDER = ("cosine", "fid", "symmetric_kl")
METRIC_TITLES = {
    "cosine": "Cosine",
    "fid": "FID",
    "symmetric_kl": "Jeffreys divergence",
}
METHOD_ORDER = ("classical", "flow_matching", "flow_matching_nll_finetuned", "tre")
METHOD_STYLES = {
    "classical": {"label": "Classical", "color": "C2", "marker": "o", "linestyle": "-"},
    "flow_matching": {
        "label": "Flow matching",
        "color": "C0",
        "marker": "s",
        "linestyle": "-",
    },
    "flow_matching_nll_finetuned": {
        "label": "Flow matching + NLL",
        "color": "C1",
        "marker": "^",
        "linestyle": "--",
    },
    "tre": {"label": "TRE, 8 bridges", "color": "C3", "marker": "D", "linestyle": "-"},
}
RESULTS_NAME = "mog5_pr_distance_sweep_results.npz"
REL_FIGURE_STEM = "mog5_dimension_sweep_rel_error"
ABS_FIGURE_STEM = "mog5_dimension_sweep_abs_error"
CSV_NAME = "mog5_dimension_sweep_errors.csv"
SUMMARY_NAME = "mog5_dimension_sweep_summary.json"
ERRORS_NPZ_NAME = "mog5_dimension_sweep_errors.npz"


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("Expected comma-separated dimensions, each >= 2.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Dimensions must be unique.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(DATA_DIR) / "mog5_dimension_sweep_n1000_cosine_fid_jeffreys_r5",
        help="Root containing one xdim{D}/ sweep directory per dimension.",
    )
    parser.add_argument("--dimensions", type=_parse_int_list, default=(3, 10, 30, 100))
    parser.add_argument("--n-total", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Aggregate output directory; defaults to <input-root>/dimension_summary.",
    )
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _metric_index(data: dict[str, np.ndarray], metric: str) -> int:
    metric_names = tuple(str(value) for value in data["metric_names"])
    if metric not in metric_names:
        raise KeyError(f"Metric {metric!r} is absent from {metric_names}.")
    return metric_names.index(metric)


def _verify_run(data: dict[str, np.ndarray], *, dimension: int, n_total: int) -> None:
    native_x_dim = int(np.asarray(data["native_x_dim"]).reshape(-1)[0])
    if native_x_dim != int(dimension):
        raise ValueError(f"Expected native_x_dim={dimension}, got {native_x_dim}.")
    n_list = np.asarray(data["n_list"], dtype=np.int64).reshape(-1)
    if n_list.tolist() != [int(n_total)]:
        raise ValueError(f"Expected n_list=[{n_total}], got {n_list.tolist()} for d={dimension}.")
    metric_names = tuple(str(value) for value in data["metric_names"])
    missing = tuple(metric for metric in METRIC_ORDER if metric not in metric_names)
    if missing:
        raise ValueError(f"Run d={dimension} is missing metrics: {missing}.")
    for key in (
        "n_repeat_classical_matrices",
        "n_repeat_flow_matching_matrices",
        "n_repeat_flow_matching_nll_finetuned_matrices",
        "n_repeat_ground_truth_matrices",
        "n_repeat_tre_matrices",
    ):
        if key not in data:
            raise KeyError(f"Run d={dimension} is missing {key}.")


def _pair_errors(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    pair_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(estimate, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if estimate.shape != ground_truth.shape or estimate.ndim != 3:
        raise ValueError(f"Expected matched [repeats, K, K] matrices; got {estimate.shape} and {ground_truth.shape}.")
    pairs = np.asarray(pair_indices, dtype=np.int64)
    absolute = np.empty(estimate.shape[0], dtype=np.float64)
    relative = np.empty(estimate.shape[0], dtype=np.float64)
    for repeat_idx in range(estimate.shape[0]):
        est_values = np.asarray(
            [estimate[repeat_idx, int(i), int(j)] for i, j in pairs], dtype=np.float64
        )
        gt_values = np.asarray(
            [ground_truth[repeat_idx, int(i), int(j)] for i, j in pairs], dtype=np.float64
        )
        delta = np.abs(est_values - gt_values)
        absolute[repeat_idx] = float(np.mean(delta))
        relative[repeat_idx] = float(np.mean(delta / np.maximum(np.abs(gt_values), 1e-12)))
    return absolute, relative


def collect_errors(
    runs: dict[int, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    dimensions = tuple(sorted(int(value) for value in runs))
    errors: dict[str, dict[str, dict[str, np.ndarray]]] = {
        metric: {} for metric in METRIC_ORDER
    }
    key_by_method = {
        "classical": "n_repeat_classical_matrices",
        "flow_matching": "n_repeat_flow_matching_matrices",
        "flow_matching_nll_finetuned": "n_repeat_flow_matching_nll_finetuned_matrices",
        "tre": "n_repeat_tre_matrices",
    }
    for metric in METRIC_ORDER:
        methods = METHOD_ORDER if metric == "symmetric_kl" else METHOD_ORDER[:3]
        for method in methods:
            absolute_by_dimension = []
            relative_by_dimension = []
            for dimension in dimensions:
                data = runs[dimension]
                metric_idx = _metric_index(data, metric)
                ground_truth = np.asarray(
                    data["n_repeat_ground_truth_matrices"][0, :, metric_idx], dtype=np.float64
                )
                estimate = np.asarray(
                    data[key_by_method[method]][0, :, metric_idx], dtype=np.float64
                )
                absolute, relative = _pair_errors(estimate, ground_truth, data["pair_indices"])
                absolute_by_dimension.append(absolute)
                relative_by_dimension.append(relative)
            errors[metric][method] = {
                "absolute": np.stack(absolute_by_dimension, axis=0),
                "relative": np.stack(relative_by_dimension, axis=0),
            }
    return errors


def _plot(
    errors: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    dimensions: tuple[int, ...],
    relative: bool,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    error_key = "relative" if relative else "absolute"
    y_label = "Mean relative\nabsolute error" if relative else "Mean absolute error"
    x_values = np.asarray(dimensions, dtype=np.int64)
    fig, axes_obj = plt.subplots(1, len(METRIC_ORDER), figsize=(4.0 * len(METRIC_ORDER), 3.5), squeeze=False)
    axes = axes_obj[0]
    for panel_idx, metric in enumerate(METRIC_ORDER):
        ax = axes[panel_idx]
        for method in METHOD_ORDER:
            if method not in errors[metric]:
                continue
            values = np.asarray(errors[metric][method][error_key], dtype=np.float64)
            mean = np.mean(values, axis=1)
            std = np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros_like(mean)
            style = METHOD_STYLES[method]
            ax.errorbar(
                x_values,
                mean,
                yerr=std,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=6.5,
                capsize=3.0,
                label=style["label"],
            )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xscale("log")
        ax.set_xticks(x_values)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        if panel_idx == 0:
            ax.set_ylabel(y_label)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=1.8, length=5)
        ax.legend(frameon=False, loc="best")
    fig.supxlabel("Native data dimension", fontsize=16, y=0.02)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.87, wspace=0.32)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = REL_FIGURE_STEM if relative else ABS_FIGURE_STEM
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def _write_outputs(
    errors: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    dimensions: tuple[int, ...],
    n_total: int,
    input_paths: dict[int, Path],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    rows: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {"dimensions": np.asarray(dimensions, dtype=np.int64)}
    summary_metrics: dict[str, Any] = {}
    for metric in METRIC_ORDER:
        summary_metrics[metric] = {}
        for method, method_errors in errors[metric].items():
            summary_metrics[metric][method] = {}
            for error_kind, values in method_errors.items():
                values = np.asarray(values, dtype=np.float64)
                npz_payload[f"{metric}_{method}_{error_kind}_errors"] = values
                summary_metrics[metric][method][error_kind] = {
                    "mean_by_dimension": np.mean(values, axis=1).tolist(),
                    "std_by_dimension": (
                        np.std(values, axis=1, ddof=1).tolist()
                        if values.shape[1] > 1
                        else np.zeros(values.shape[0], dtype=np.float64).tolist()
                    ),
                    "repeat_errors": values.tolist(),
                }
            repeats = int(method_errors["relative"].shape[1])
            for dimension_idx, dimension in enumerate(dimensions):
                for repeat_idx in range(repeats):
                    rows.append(
                        {
                            "native_x_dim": int(dimension),
                            "n_total": int(n_total),
                            "repeat_idx": int(repeat_idx),
                            "metric": metric,
                            "estimator": method,
                            "mean_absolute_error": float(method_errors["absolute"][dimension_idx, repeat_idx]),
                            "mean_relative_absolute_error": float(
                                method_errors["relative"][dimension_idx, repeat_idx]
                            ),
                        }
                    )
    csv_path = output_dir / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    npz_path = output_dir / ERRORS_NPZ_NAME
    np.savez_compressed(npz_path, **npz_payload)
    summary = {
        "dimensions": list(dimensions),
        "n_total": int(n_total),
        "input_results": {str(dimension): str(input_paths[dimension]) for dimension in dimensions},
        "metrics": summary_metrics,
    }
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, npz_path, summary_path


def main() -> None:
    args = build_parser().parse_args()
    dimensions = tuple(sorted(int(value) for value in args.dimensions))
    input_root = args.input_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_root / "dimension_summary"
    )
    input_paths = {
        dimension: input_root / f"xdim{dimension}" / RESULTS_NAME for dimension in dimensions
    }
    runs = {dimension: _load(path) for dimension, path in input_paths.items()}
    for dimension, data in runs.items():
        _verify_run(data, dimension=dimension, n_total=int(args.n_total))
    reference = runs[dimensions[0]]
    for dimension in dimensions[1:]:
        for key in ("condition_labels", "pair_indices", "repeat_indices", "repeat_seeds"):
            np.testing.assert_array_equal(reference[key], runs[dimension][key])
    errors = collect_errors(runs)
    rel_png, rel_svg = _plot(
        errors, dimensions=dimensions, relative=True, output_dir=output_dir
    )
    abs_png, abs_svg = _plot(
        errors, dimensions=dimensions, relative=False, output_dir=output_dir
    )
    csv_path, npz_path, summary_path = _write_outputs(
        errors,
        dimensions=dimensions,
        n_total=int(args.n_total),
        input_paths=input_paths,
        output_dir=output_dir,
    )
    print(f"Saved relative-error figure: {rel_png}")
    print(f"Saved relative-error vector figure: {rel_svg}")
    print(f"Saved absolute-error figure: {abs_png}")
    print(f"Saved absolute-error vector figure: {abs_svg}")
    print(f"Saved per-repeat errors: {csv_path}")
    print(f"Saved error arrays: {npz_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
