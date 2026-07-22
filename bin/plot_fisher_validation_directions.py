#!/usr/bin/env python3
"""Plot three independent detailed evaluations of linear Fisher estimators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR

DATASETS = ["Gaussian toy", "Gaussian-mixture toy", "Stringer"]
DATASET_TOKENS = {
    "randamp_gaussian_sqrtd": "Gaussian toy",
    "cosine_gmm": "Gaussian-mixture toy",
    "stringer": "Stringer",
}
METHODS = ["Flow matching", "GKR", "OLE (cross-fit)", "Oracle"]
COLORS = {
    "Flow matching": "C0",
    "GKR": "C2",
    "OLE (cross-fit)": "C1",
    "Oracle": "0.25",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root = Path(DATA_DIR) / "fisher_validation_directions"
    parser.add_argument("--input-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    parser.add_argument(
        "--datasets",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=None,
        help="Optional comma-separated dataset tokens or display labels.",
    )
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)


def _load_rows(input_dir: Path, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob(f"{prefix}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def _selected(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    method: str,
) -> list[dict[str, object]]:
    return [row for row in rows if row["dataset"] == dataset and row["method"] == method]


def _mean_ci(rows: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([np.asarray(row[key], dtype=np.float64) for row in rows])
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot_condition_resolved(
    rows: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    metrics = (
        ("achieved_fisher", "Held-out achieved information"),
        ("balanced_error", "Linear decoding error"),
    )
    for column, dataset in enumerate(DATASETS):
        for row_index, (key, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            for method in METHODS:
                method_rows = _selected(rows, dataset=dataset, method=method)
                if not method_rows:
                    continue
                x = np.asarray(method_rows[0]["theta_midpoints"], dtype=np.float64)
                mean, ci = _mean_ci(method_rows, key)
                axis.plot(x, mean, color=COLORS[method], linewidth=2.2, label=method)
                axis.fill_between(x, mean - ci, mean + ci, color=COLORS[method], alpha=0.16, linewidth=0)
            if row_index == 0:
                axis.set_title(dataset)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 1:
                axis.set_xlabel(r"Condition $\theta$")
                axis.set_ylim(0.0, 0.5)
            if row_index == 0 and column == 0:
                axis.legend(frameon=False)
            _style_axis(axis)
    return _save(fig, output_dir, "condition_resolved_performance")


def _identity_limits(axis: plt.Axes, values: list[np.ndarray]) -> None:
    combined = np.concatenate(values)
    low = max(0.0, float(np.min(combined)))
    high = min(0.5, float(np.max(combined)))
    margin = max(0.02, 0.08 * (high - low))
    low = max(0.0, low - margin)
    high = min(0.5, high + margin)
    axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.8)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)


def plot_linear_decoder_calibration(
    rows: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for column, dataset in enumerate(DATASETS):
        scatter_axis = axes[0, column]
        curve_axis = axes[1, column]
        limits: list[np.ndarray] = []
        for method in METHODS[:-1]:
            method_rows = _selected(rows, dataset=dataset, method=method)
            if not method_rows:
                continue
            predicted = np.concatenate(
                [np.asarray(row["predicted_error"], dtype=np.float64) for row in method_rows]
            )
            observed = np.concatenate(
                [np.asarray(row["balanced_error"], dtype=np.float64) for row in method_rows]
            )
            limits.extend((predicted, observed))
            scatter_axis.scatter(
                predicted,
                observed,
                s=18,
                alpha=0.42,
                color=COLORS[method],
                edgecolors="none",
                label=method,
            )
            x = np.asarray(method_rows[0]["theta_midpoints"], dtype=np.float64)
            observed_mean, observed_ci = _mean_ci(method_rows, "balanced_error")
            predicted_mean, _ = _mean_ci(method_rows, "predicted_error")
            curve_axis.plot(x, observed_mean, color=COLORS[method], linewidth=2.2)
            curve_axis.fill_between(
                x,
                observed_mean - observed_ci,
                observed_mean + observed_ci,
                color=COLORS[method],
                alpha=0.16,
                linewidth=0,
            )
            curve_axis.plot(x, predicted_mean, color=COLORS[method], linewidth=2.0, linestyle="--")
        if limits:
            _identity_limits(scatter_axis, limits)
        scatter_axis.set_title(dataset)
        scatter_axis.set_xlabel("Fisher-predicted error")
        curve_axis.set_xlabel(r"Condition $\theta$")
        curve_axis.set_ylim(0.0, 0.5)
        if column == 0:
            scatter_axis.set_ylabel("Observed linear error")
            curve_axis.set_ylabel("Linear decoding error")
            scatter_axis.legend(frameon=False)
            curve_axis.legend(
                handles=(
                    Line2D([0], [0], color="black", linewidth=2.2, label="Observed"),
                    Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Predicted"),
                ),
                frameon=False,
            )
        _style_axis(scatter_axis)
        _style_axis(curve_axis)
    return _save(fig, output_dir, "linear_decoder_fisher_calibration")


def _reliability_curve(predicted: np.ndarray, observed: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.quantile(predicted, np.linspace(0.0, 1.0, n_bins + 1))
    boundaries = np.unique(boundaries)
    x_values = []
    y_values = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (predicted >= lower) & (predicted <= upper if upper == boundaries[-1] else predicted < upper)
        if np.any(mask):
            x_values.append(float(np.mean(predicted[mask])))
            y_values.append(float(np.mean(observed[mask])))
    return np.asarray(x_values), np.asarray(y_values)


def plot_reliability(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    for column, dataset in enumerate(DATASETS):
        axis = axes[column]
        values: list[np.ndarray] = []
        for method in METHODS[:-1]:
            method_rows = _selected(rows, dataset=dataset, method=method)
            if not method_rows:
                continue
            predicted = np.concatenate(
                [np.asarray(row["predicted_error"], dtype=np.float64) for row in method_rows]
            )
            observed = np.concatenate(
                [np.asarray(row["balanced_error"], dtype=np.float64) for row in method_rows]
            )
            x, y = _reliability_curve(predicted, observed)
            values.extend((x, y))
            axis.plot(x, y, marker="o", markersize=5, linewidth=2.2, color=COLORS[method], label=method)
        if values:
            _identity_limits(axis, values)
        axis.set_title(dataset)
        axis.set_xlabel("Mean predicted error")
        if column == 0:
            axis.set_ylabel("Mean observed error")
            axis.legend(frameon=False)
        _style_axis(axis)
    return _save(fig, output_dir, "linear_decoder_error_reliability")


def plot_validation_allocation(
    rows: list[dict[str, object]], output_dir: Path
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.5), constrained_layout=True)
    scalar_metrics = (
        ("mean_achieved_fisher", "Mean achieved information"),
        ("mean_balanced_error", "Mean linear decoding error"),
        ("mean_auc", "Mean orientation-invariant AUC"),
    )
    for column, dataset in enumerate(DATASETS):
        for row_index, (key, ylabel) in enumerate(scalar_metrics):
            axis = axes[row_index, column]
            for method in METHODS[:-1]:
                method_rows = _selected(rows, dataset=dataset, method=method)
                if not method_rows:
                    continue
                fractions = sorted({float(row["validation_fraction"]) for row in method_rows})
                for seed in sorted({int(row["seed"]) for row in method_rows}):
                    seed_rows = {
                        float(row["validation_fraction"]): row
                        for row in method_rows
                        if int(row["seed"]) == seed
                    }
                    if all(fraction in seed_rows for fraction in fractions):
                        axis.plot(
                            np.asarray(fractions) * 100.0,
                            [float(seed_rows[fraction][key]) for fraction in fractions],
                            color=COLORS[method],
                            alpha=0.18,
                            linewidth=1.1,
                        )
                means = []
                errors = []
                for fraction in fractions:
                    values = np.asarray(
                        [float(row[key]) for row in method_rows if float(row["validation_fraction"]) == fraction]
                    )
                    means.append(float(np.mean(values)))
                    errors.append(
                        0.0 if values.size < 2 else float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
                    )
                axis.errorbar(
                    np.asarray(fractions) * 100.0,
                    means,
                    yerr=errors,
                    color=COLORS[method],
                    marker="o",
                    markersize=5,
                    linewidth=2.2,
                    capsize=3,
                    label=method,
                )
            if row_index == 0:
                axis.set_title(dataset)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(scalar_metrics) - 1:
                axis.set_xlabel("Validation fraction (%)")
            if key == "mean_balanced_error":
                axis.set_ylim(0.0, 0.5)
            elif key == "mean_auc":
                axis.set_ylim(0.5, 1.0)
            if row_index == 0 and column == 0:
                axis.legend(frameon=False)
            _style_axis(axis)
    primary = _save(fig, output_dir, "validation_allocation_sensitivity")

    epoch_fig, epoch_axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for column, dataset in enumerate(DATASETS):
        method_rows = _selected(rows, dataset=dataset, method="Flow matching")
        fractions = sorted({float(row["validation_fraction"]) for row in method_rows})
        for row_index, (key, ylabel) in enumerate(
            (("selected_epoch", "Selected epoch"), ("best_validation_loss", "Selected validation loss"))
        ):
            axis = epoch_axes[row_index, column]
            for seed in sorted({int(row["seed"]) for row in method_rows}):
                seed_rows = {
                    float(row["validation_fraction"]): row
                    for row in method_rows
                    if int(row["seed"]) == seed
                }
                axis.plot(
                    np.asarray(fractions) * 100.0,
                    [float(seed_rows[fraction][key]) for fraction in fractions],
                    color=COLORS["Flow matching"],
                    alpha=0.28,
                    linewidth=1.3,
                )
            if fractions:
                means = [
                    np.mean(
                        [float(row[key]) for row in method_rows if float(row["validation_fraction"]) == fraction]
                    )
                    for fraction in fractions
                ]
                axis.plot(
                    np.asarray(fractions) * 100.0,
                    means,
                    color=COLORS["Flow matching"],
                    marker="o",
                    markersize=5,
                    linewidth=2.4,
                )
            if row_index == 0:
                axis.set_title(dataset)
            if row_index == 1:
                axis.set_xlabel("Validation fraction (%)")
            if column == 0:
                axis.set_ylabel(ylabel)
            _style_axis(axis)
    epochs = _save(epoch_fig, output_dir, "validation_allocation_selected_epochs")
    return primary, epochs


def plot_train_test_allocation(
    rows: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    available_datasets = [
        dataset for dataset in DATASETS if any(row["dataset"] == dataset for row in rows)
    ]
    legend_column = len(available_datasets) // 2
    fig, axes = plt.subplots(
        1,
        len(available_datasets),
        figsize=(
            (10.0, 3.5)
            if len(available_datasets) == 3
            else (3.5 * len(available_datasets) + 0.5, 3.5)
        ),
        constrained_layout=True,
        squeeze=False,
    )
    for column, dataset in enumerate(available_datasets):
        axis = axes[0, column]
        for method in METHODS:
            method_rows = _selected(rows, dataset=dataset, method=method)
            if not method_rows:
                continue
            fractions = sorted({float(row["test_fraction"]) for row in method_rows})
            means = []
            errors = []
            for fraction in fractions:
                values = np.asarray(
                    [
                        float(row["mean_achieved_fisher"])
                        for row in method_rows
                        if float(row["test_fraction"]) == fraction
                    ],
                    dtype=np.float64,
                )
                means.append(float(np.mean(values)))
                errors.append(
                    0.0
                    if values.size < 2
                    else float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
                )
            axis.errorbar(
                np.asarray(fractions) * 100.0,
                means,
                yerr=errors,
                color=COLORS[method],
                marker="o",
                markersize=5,
                linewidth=2.2,
                capsize=3,
                label=method,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Test fraction (%)")
        if column == 0:
            axis.set_ylabel("Held-out achieved information")
        if column == legend_column:
            legend_options = {"frameon": False}
            if dataset == "Gaussian-mixture toy":
                legend_options.update(
                    {
                        "loc": "center",
                        "bbox_to_anchor": (0.64, 0.66),
                        "fontsize": 14,
                        "handlelength": 1.5,
                        "handletextpad": 0.4,
                        "labelspacing": 0.2,
                    }
                )
            handles, labels = axis.get_legend_handles_labels()
            compact_labels = [
                "FM"
                if label == "Flow matching"
                else "OLE"
                if label == "OLE (cross-fit)"
                else label
                for label in labels
            ]
            axis.legend(handles, compact_labels, **legend_options)
        _style_axis(axis)
    return _save(fig, output_dir, "train_test_allocation_achieved_information")


def _metrics_summary(
    reference: list[dict[str, object]],
    allocation: list[dict[str, object]],
    train_test: list[dict[str, object]],
) -> dict[str, object]:
    calibration = []
    for dataset in DATASETS:
        for method in METHODS[:-1]:
            rows = _selected(reference, dataset=dataset, method=method)
            if not rows:
                continue
            predicted = np.concatenate([np.asarray(row["predicted_error"]) for row in rows])
            observed = np.concatenate([np.asarray(row["balanced_error"]) for row in rows])
            slope, intercept = np.polyfit(predicted, observed, deg=1)
            rng = np.random.default_rng(sum(ord(char) for char in f"{dataset}:{method}"))
            bootstrap = []
            for _ in range(1000):
                sampled_rows = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
                sampled_predicted = []
                sampled_observed = []
                for row in sampled_rows:
                    row_predicted = np.asarray(row["predicted_error"], dtype=np.float64)
                    row_observed = np.asarray(row["balanced_error"], dtype=np.float64)
                    pair_index = rng.integers(0, row_predicted.size, size=row_predicted.size)
                    sampled_predicted.append(row_predicted[pair_index])
                    sampled_observed.append(row_observed[pair_index])
                predicted_boot = np.concatenate(sampled_predicted)
                observed_boot = np.concatenate(sampled_observed)
                slope_boot, intercept_boot = np.polyfit(predicted_boot, observed_boot, deg=1)
                bootstrap.append(
                    (
                        slope_boot,
                        intercept_boot,
                        np.mean(np.abs(predicted_boot - observed_boot)),
                        spearmanr(predicted_boot, observed_boot).statistic,
                    )
                )
            bootstrap_array = np.asarray(bootstrap, dtype=np.float64)
            calibration_ci = np.nanquantile(bootstrap_array, [0.025, 0.975], axis=0)
            achieved = np.concatenate([np.asarray(row["achieved_fisher"]) for row in rows])
            calibration.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_seeds": len(rows),
                    "mean_achieved_fisher": float(np.mean([row["mean_achieved_fisher"] for row in rows])),
                    "mean_balanced_error": float(np.mean([row["mean_balanced_error"] for row in rows])),
                    "calibration_slope": float(slope),
                    "calibration_slope_95ci": calibration_ci[:, 0].tolist(),
                    "calibration_intercept": float(intercept),
                    "calibration_intercept_95ci": calibration_ci[:, 1].tolist(),
                    "calibration_mae": float(np.mean(np.abs(predicted - observed))),
                    "calibration_mae_95ci": calibration_ci[:, 2].tolist(),
                    "spearman": float(spearmanr(predicted, observed).statistic),
                    "spearman_95ci": calibration_ci[:, 3].tolist(),
                    "median_achieved_fisher": float(np.median(achieved)),
                    "achieved_fisher_iqr": np.quantile(achieved, [0.25, 0.75]).tolist(),
                    "worst_decile_achieved_fisher": float(np.quantile(achieved, 0.1)),
                }
            )
    allocation_summary = []
    for dataset in DATASETS:
        for method in METHODS[:-1]:
            rows = _selected(allocation, dataset=dataset, method=method)
            for fraction in sorted({float(row["validation_fraction"]) for row in rows}):
                selected = [row for row in rows if float(row["validation_fraction"]) == fraction]
                if selected:
                    summary_row = {
                        "dataset": dataset,
                        "method": method,
                        "validation_fraction": fraction,
                        "n_seeds": len(selected),
                        "mean_achieved_fisher": float(
                            np.mean([row["mean_achieved_fisher"] for row in selected])
                        ),
                        "mean_balanced_error": float(
                            np.mean([row["mean_balanced_error"] for row in selected])
                        ),
                        "mean_auc": float(np.mean([row["mean_auc"] for row in selected])),
                    }
                    if method == "Flow matching":
                        summary_row.update(
                            {
                                "mean_selected_epoch": float(
                                    np.mean([row["selected_epoch"] for row in selected])
                                ),
                                "mean_best_validation_loss": float(
                                    np.mean([row["best_validation_loss"] for row in selected])
                                ),
                            }
                        )
                    allocation_summary.append(summary_row)
    train_test_summary = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = _selected(train_test, dataset=dataset, method=method)
            for fraction in sorted({float(row["test_fraction"]) for row in rows}):
                selected = [row for row in rows if float(row["test_fraction"]) == fraction]
                if not selected:
                    continue
                values = np.asarray(
                    [float(row["mean_achieved_fisher"]) for row in selected],
                    dtype=np.float64,
                )
                train_test_summary.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "test_fraction": fraction,
                        "train_fraction": float(selected[0]["train_fraction"]),
                        "validation_fraction": float(selected[0]["validation_fraction"]),
                        "n_seeds": len(selected),
                        "mean_achieved_fisher": float(np.mean(values)),
                        "achieved_fisher_95ci": (
                            [float(np.mean(values)), float(np.mean(values))]
                            if values.size < 2
                            else (
                                np.mean(values)
                                + np.asarray([-1.0, 1.0])
                                * 1.96
                                * np.std(values, ddof=1)
                                / np.sqrt(values.size)
                            ).tolist()
                        ),
                        "mean_test_samples_per_endpoint": float(
                            np.mean([row["mean_test_samples_per_endpoint"] for row in selected])
                        ),
                    }
                )
    return {
        "reference_calibration": calibration,
        "validation_allocation": allocation_summary,
        "train_test_allocation": train_test_summary,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    reference = _load_rows(input_dir, "reference")
    allocation = _load_rows(input_dir, "validation-allocation")
    train_test = _load_rows(input_dir, "train-test-allocation")
    if args.datasets:
        selected_datasets = {
            DATASET_TOKENS.get(dataset, dataset) for dataset in args.datasets
        }
        unknown = selected_datasets.difference(DATASETS)
        if unknown:
            raise ValueError(f"Unknown dataset labels: {sorted(unknown)}")
        reference = [row for row in reference if row["dataset"] in selected_datasets]
        allocation = [row for row in allocation if row["dataset"] in selected_datasets]
        train_test = [row for row in train_test if row["dataset"] in selected_datasets]
    if not any((reference, allocation, train_test)):
        raise FileNotFoundError(f"No evaluation summaries found under {input_dir}.")
    artifacts: dict[str, object] = {}
    if reference:
        artifacts["condition_resolved"] = [
            str(path) for path in plot_condition_resolved(reference, output_dir)
        ]
        artifacts["linear_decoder_calibration"] = [
            str(path) for path in plot_linear_decoder_calibration(reference, output_dir)
        ]
        artifacts["linear_decoder_reliability"] = [
            str(path) for path in plot_reliability(reference, output_dir)
        ]
    if allocation:
        allocation_figure, epoch_figure = plot_validation_allocation(allocation, output_dir)
        artifacts["validation_allocation"] = [str(path) for path in allocation_figure]
        artifacts["validation_epochs"] = [str(path) for path in epoch_figure]
    if train_test:
        artifacts["train_test_allocation"] = [
            str(path) for path in plot_train_test_allocation(train_test, output_dir)
        ]
    metrics = _metrics_summary(reference, allocation, train_test)
    metrics_path = output_dir / "detailed_evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": artifacts, "metrics": str(metrics_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
