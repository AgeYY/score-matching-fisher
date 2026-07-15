#!/usr/bin/env python3
"""Plot MoG5 distance sweeps with TRE variants used for Jeffreys divergence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR
from fisher.distance_comparison import mahalanobis_sq_matrix_ledoit_wolf

METRIC_ORDER = (
    "correlation",
    "cosine",
    "squared_euclidean",
    "mahalanobis_sq",
    "fid",
    "symmetric_kl",
)
METRIC_TITLES = {
    "correlation": "Correlation",
    "cosine": "Cosine",
    "squared_euclidean": "Squared Euclidean",
    "mahalanobis_sq": "Squared Mahalanobis",
    "fid": "FID",
    "symmetric_kl": "Jeffreys divergence",
}
METHOD_STYLES = {
    "classical": {"label": "Classical", "color": "C2", "marker": "o", "linestyle": "-"},
    "classical_no_lw": {
        "label": "Classical (no LW)",
        "color": "C2",
        "marker": "o",
        "linestyle": "-",
    },
    "classical_lw": {
        "label": "Classical (LW)",
        "color": "C3",
        "marker": "D",
        "linestyle": "-",
    },
    "flow_matching": {
        "label": "Flow matching",
        "color": "C0",
        "marker": "s",
        "linestyle": "-",
    },
    "flow_matching_nll": {
        "label": "Flow matching + NLL",
        "color": "C1",
        "marker": "^",
        "linestyle": "--",
    },
    "tre_1": {"label": "TRE, 1 bridge", "color": "C2", "marker": "P", "linestyle": "-"},
    "tre_8": {"label": "TRE, 8 bridges", "color": "C3", "marker": "D", "linestyle": "-"},
    "tre_16": {"label": "TRE, 16 bridges", "color": "C4", "marker": "X", "linestyle": "-"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path(DATA_DIR) / "mog5_all_metrics_fm_nll" / "mog5_pr_distance_sweep_results.npz",
    )
    parser.add_argument("--fid-results", type=Path, required=True)
    parser.add_argument("--tre-one-results", type=Path, required=True)
    parser.add_argument("--tre-eight-results", type=Path, required=True)
    parser.add_argument("--tre-sixteen-results", type=Path, required=True)
    parser.add_argument(
        "--case-dataset-root",
        type=Path,
        default=Path(DATA_DIR) / "mog5_non_skl_constant_lr_val10paths_cases",
        help="Root containing the cached n{N}_native/repeat_{R} datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_all_metrics_fm_nll_tre_comparison",
    )
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _metric_index(data: dict[str, np.ndarray], metric: str) -> int:
    names = tuple(str(value) for value in data["metric_names"])
    if metric not in names:
        raise KeyError(f"Metric {metric!r} is absent from {names}.")
    return names.index(metric)


def _verify_inputs(
    baseline: dict[str, np.ndarray],
    fid: dict[str, np.ndarray],
    tre_runs: list[dict[str, np.ndarray]],
) -> None:
    baseline_names = tuple(str(value) for value in baseline["metric_names"])
    missing = [metric for metric in METRIC_ORDER if metric != "fid" and metric not in baseline_names]
    if missing:
        raise ValueError(f"Baseline aggregate is missing metrics: {missing}.")
    if "n_repeat_flow_matching_nll_finetuned_matrices" not in baseline:
        raise KeyError("Baseline aggregate has no FM+NLL matrices.")
    if "fid" not in tuple(str(value) for value in fid["metric_names"]):
        raise KeyError("FID aggregate has no fid metric.")
    if "n_repeat_flow_matching_nll_finetuned_matrices" not in fid:
        raise KeyError("FID aggregate has no FM+NLL matrices.")
    for key in ("n_list", "repeat_indices", "repeat_seeds", "condition_labels", "pair_indices"):
        np.testing.assert_array_equal(baseline[key], fid[key])
    baseline_skl_idx = _metric_index(baseline, "symmetric_kl")
    baseline_gt = baseline["n_repeat_ground_truth_matrices"][:, :, baseline_skl_idx]
    for tre in tre_runs:
        for key in ("n_list", "repeat_indices", "repeat_seeds", "condition_labels", "pair_indices"):
            np.testing.assert_array_equal(baseline[key], tre[key])
        if "n_repeat_tre_matrices" not in tre:
            raise KeyError("Each TRE aggregate must contain n_repeat_tre_matrices.")
        tre_skl_idx = _metric_index(tre, "symmetric_kl")
        tre_gt = tre["n_repeat_ground_truth_matrices"][:, :, tre_skl_idx]
        np.testing.assert_array_equal(baseline_gt, tre_gt)


def _repeat_pair_relative_error(
    matrices: np.ndarray,
    ground_truth: np.ndarray,
    pair_indices: np.ndarray,
) -> np.ndarray:
    estimate = np.asarray(matrices, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)
    if estimate.shape != truth.shape or estimate.ndim != 4:
        raise ValueError(f"Expected matched [N, repeats, K, K] matrices; got {estimate.shape}.")
    output = np.empty(estimate.shape[:2], dtype=np.float64)
    for n_idx in range(estimate.shape[0]):
        for repeat_idx in range(estimate.shape[1]):
            errors = []
            for i, j in np.asarray(pair_indices, dtype=np.int64):
                target = float(truth[n_idx, repeat_idx, int(i), int(j)])
                predicted = float(estimate[n_idx, repeat_idx, int(i), int(j)])
                errors.append(abs(predicted - target) / max(abs(target), 1e-12))
            output[n_idx, repeat_idx] = float(np.mean(np.asarray(errors, dtype=np.float64)))
    return output


def _collect_errors(
    baseline: dict[str, np.ndarray],
    fid: dict[str, np.ndarray],
    tre_runs: dict[str, dict[str, np.ndarray]],
    *,
    classical_lw_matrices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    pair_indices = baseline["pair_indices"]
    errors: dict[str, dict[str, np.ndarray]] = {}
    for metric in METRIC_ORDER:
        source = fid if metric == "fid" else baseline
        metric_idx = _metric_index(source, metric)
        ground_truth = source["n_repeat_ground_truth_matrices"][:, :, metric_idx]
        metric_errors = {
            "flow_matching": _repeat_pair_relative_error(
                source["n_repeat_flow_matching_matrices"][:, :, metric_idx],
                ground_truth,
                pair_indices,
            ),
            "flow_matching_nll": _repeat_pair_relative_error(
                source["n_repeat_flow_matching_nll_finetuned_matrices"][:, :, metric_idx],
                ground_truth,
                pair_indices,
            ),
        }
        if metric != "symmetric_kl":
            classical_method = "classical_no_lw" if metric == "mahalanobis_sq" else "classical"
            metric_errors[classical_method] = _repeat_pair_relative_error(
                source["n_repeat_classical_matrices"][:, :, metric_idx], ground_truth, pair_indices
            )
            if metric == "mahalanobis_sq":
                metric_errors["classical_lw"] = _repeat_pair_relative_error(
                    classical_lw_matrices,
                    ground_truth,
                    pair_indices,
                )
        else:
            for method, tre in tre_runs.items():
                tre_idx = _metric_index(tre, "symmetric_kl")
                metric_errors[method] = _repeat_pair_relative_error(
                    tre["n_repeat_tre_matrices"][:, :, tre_idx],
                    tre["n_repeat_ground_truth_matrices"][:, :, tre_idx],
                    pair_indices,
                )
        errors[metric] = metric_errors
    return errors


def _load_classical_lw_matrices(
    baseline: dict[str, np.ndarray],
    *,
    case_dataset_root: Path,
) -> np.ndarray:
    n_list = np.asarray(baseline["n_list"], dtype=np.int64)
    repeat_indices = np.asarray(baseline["repeat_indices"], dtype=np.int64)
    num_categories = int(np.asarray(baseline["condition_labels"]).size)
    matrices = np.empty(
        (n_list.size, repeat_indices.size, num_categories, num_categories),
        dtype=np.float64,
    )
    root = case_dataset_root.expanduser().resolve()
    for n_idx, n_total in enumerate(n_list):
        for repeat_pos, repeat_idx in enumerate(repeat_indices):
            dataset_path = (
                root
                / f"n{int(n_total)}_native"
                / f"repeat_{int(repeat_idx):02d}"
                / "random_mog_categorical.npz"
            )
            if not dataset_path.is_file():
                raise FileNotFoundError(dataset_path)
            with np.load(dataset_path, allow_pickle=False) as data:
                x = np.asarray(data["x_all"], dtype=np.float64)
                theta = np.asarray(data["theta_all"], dtype=np.float64)
            labels = np.argmax(theta, axis=1).astype(np.int64, copy=False)
            matrices[n_idx, repeat_pos] = mahalanobis_sq_matrix_ledoit_wolf(
                x,
                labels,
                num_categories=num_categories,
            )
    return matrices


def _plot(
    errors: dict[str, dict[str, np.ndarray]],
    *,
    n_list: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    x_values = np.asarray(n_list, dtype=np.int64)
    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(18.0, 3.5), squeeze=False)
    axes_row = axes[0]
    handles_by_method = {}
    for panel_idx, metric in enumerate(METRIC_ORDER):
        ax = axes_row[panel_idx]
        for method, values in errors[metric].items():
            style = METHOD_STYLES[method]
            mean = np.mean(values, axis=1)
            std = np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros_like(mean)
            handle = ax.errorbar(
                x_values,
                mean,
                yerr=std,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=6,
                linewidth=1.8,
                capsize=3,
                label=style["label"],
            )
            handles_by_method.setdefault(method, handle)
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xticks(x_values)
        ax.tick_params(axis="x", labelrotation=45)
        if panel_idx == 0:
            ax.set_ylabel("Mean relative\nabsolute error")
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.tick_params(width=1.8, length=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.25, top=0.86, wspace=0.28)
    fig.supxlabel("Total samples", fontsize=16, y=0.02)
    baseline_methods = ("classical", "flow_matching", "flow_matching_nll")
    axes_row[0].legend(
        [handles_by_method[method] for method in baseline_methods],
        [METHOD_STYLES[method]["label"] for method in baseline_methods],
        frameon=False,
        loc="upper right",
    )
    mahalanobis_methods = ("classical_no_lw", "classical_lw")
    axes_row[3].legend(
        [handles_by_method[method] for method in mahalanobis_methods],
        [METHOD_STYLES[method]["label"] for method in mahalanobis_methods],
        frameon=False,
        loc="upper right",
    )
    tre_methods = ("tre_1", "tre_8", "tre_16")
    axes_row[-1].legend(
        [handles_by_method[method] for method in tre_methods],
        [METHOD_STYLES[method]["label"] for method in tre_methods],
        frameon=False,
        loc="upper right",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mog5_all_metrics_relative_error.png"
    svg_path = output_dir / "mog5_all_metrics_relative_error.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    baseline = _load(args.baseline_results)
    fid = _load(args.fid_results)
    tre_runs = {
        "tre_1": _load(args.tre_one_results),
        "tre_8": _load(args.tre_eight_results),
        "tre_16": _load(args.tre_sixteen_results),
    }
    _verify_inputs(baseline, fid, list(tre_runs.values()))
    classical_lw_matrices = _load_classical_lw_matrices(
        baseline,
        case_dataset_root=args.case_dataset_root,
    )
    errors = _collect_errors(
        baseline,
        fid,
        tre_runs,
        classical_lw_matrices=classical_lw_matrices,
    )
    n_list = np.asarray(baseline["n_list"], dtype=np.int64)
    output_dir = args.output_dir.expanduser().resolve()
    png_path, svg_path = _plot(errors, n_list=n_list, output_dir=output_dir)

    rows = []
    for metric, method_values in errors.items():
        for method, values in method_values.items():
            for n_idx, repeat_values in enumerate(values):
                for repeat_idx, value in enumerate(repeat_values):
                    rows.append(
                        {
                            "metric": metric,
                            "estimator": method,
                            "n_total": int(n_list[n_idx]),
                            "repeat_idx": int(repeat_idx),
                            "mean_relative_absolute_error": float(value),
                        }
                    )
    csv_path = output_dir / "mog5_all_metrics_relative_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_list": n_list.tolist(),
        "metrics": {
            metric: {
                method: {
                    "mean_by_n": np.mean(values, axis=1).tolist(),
                    "std_by_n": np.std(values, axis=1, ddof=1).tolist(),
                    "repeat_errors": values.tolist(),
                }
                for method, values in method_values.items()
            }
            for metric, method_values in errors.items()
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved figure: {png_path}")
    print(f"Saved vector figure: {svg_path}")
    print(f"Saved per-repeat errors: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
