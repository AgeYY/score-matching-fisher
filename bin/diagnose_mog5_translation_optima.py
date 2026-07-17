#!/usr/bin/env python3
"""Diagnose finite-sample and optimization gaps for translation-flow metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)
from fisher.distance_comparison import (
    FLOW_VELOCITY_FAMILY_BY_METRIC,
    METRIC_CORRELATION,
    METRIC_COSINE,
    METRIC_SQUARED_EUCLIDEAN,
    FlowComparisonConfig,
    _build_and_train_flow_model,
    _seed_flow_rng,
    class_means,
    correlation_distance_matrix,
    cosine_distance_matrix,
    finetune_flow_skl_cnf_likelihood,
    labels_from_theta,
    squared_euclidean_mean_distance_matrix,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


_METRICS = (METRIC_SQUARED_EUCLIDEAN, METRIC_COSINE, METRIC_CORRELATION)
_METRIC_LABELS = {
    METRIC_SQUARED_EUCLIDEAN: "Euclidean$^2$",
    METRIC_COSINE: "Cosine",
    METRIC_CORRELATION: "Correlation",
}
_READOUTS = {
    METRIC_SQUARED_EUCLIDEAN: squared_euclidean_mean_distance_matrix,
    METRIC_COSINE: cosine_distance_matrix,
    METRIC_CORRELATION: correlation_distance_matrix,
}
_ESTIMATOR_ORDER = (
    "classical_all",
    "closed_form_train",
    "fm_exact",
    "fm_mc",
    "nll_exact",
    "nll_mc",
)
_ESTIMATOR_LABELS = {
    "classical_all": "Classical, all data",
    "closed_form_train": "Closed form, train",
    "fm_exact": "FM, exact readout",
    "fm_mc": "FM, MC readout",
    "nll_exact": "FM+NLL, exact readout",
    "nll_mc": "FM+NLL, MC readout",
}


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-list", type=_parse_int_list, default=[3000])
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--source-case-name",
        type=str,
        default="distance_comparison_seed19_r10_batch3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "mog5_translation_optimum_diagnostic",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--batch-size", type=int, default=3_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    parser.add_argument("--nll-epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--nll-batch-size", type=int, default=3_000)
    parser.add_argument("--nll-lr", type=float, default=3e-5)
    parser.add_argument("--nll-ode-steps", type=int, default=32)
    parser.add_argument(
        "--nll-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    return parser


def relative_pair_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate_arr = np.asarray(estimate, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    if estimate_arr.shape != reference_arr.shape or estimate_arr.ndim != 2:
        raise ValueError("estimate and reference must be square matrices with matching shapes.")
    pairs = np.triu_indices(int(reference_arr.shape[0]), 1)
    denominator = np.maximum(np.abs(reference_arr[pairs]), 1e-12)
    return float(np.mean(np.abs(estimate_arr[pairs] - reference_arr[pairs]) / denominator))


@torch.no_grad()
def endpoint_means(model: torch.nn.Module, theta_eval: np.ndarray, device: torch.device) -> np.ndarray:
    if not hasattr(model, "endpoint_mean"):
        raise TypeError("Translation diagnostic requires model.endpoint_mean().")
    theta = torch.from_numpy(np.asarray(theta_eval, dtype=np.float32)).to(device)
    model.eval()
    return model.endpoint_mean(theta).detach().cpu().numpy().astype(np.float64)


def _config(args: argparse.Namespace) -> FlowComparisonConfig:
    return FlowComparisonConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule=str(args.path_schedule),
        likelihood_finetune_epochs=int(args.nll_epochs),
        likelihood_finetune_batch_size=int(args.nll_batch_size),
        likelihood_finetune_lr=float(args.nll_lr),
        likelihood_finetune_ode_steps=int(args.nll_ode_steps),
        likelihood_finetune_patience=int(args.nll_patience),
    )


def _source_paths(args: argparse.Namespace, n_total: int, repeat_idx: int) -> tuple[Path, Path]:
    dataset_dir = _REPO_ROOT / "data" / f"mog_5native_xdim3_n{int(n_total)}" / f"repeat_{repeat_idx:02d}"
    dataset_path = dataset_dir / "random_mog_categorical.npz"
    result_path = dataset_dir / str(args.source_case_name) / "mog5_pr_distance_comparison_results.npz"
    if not dataset_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Missing source dataset or result: {dataset_path}, {result_path}")
    return dataset_path, result_path


def run(args: argparse.Namespace) -> dict[str, Path]:
    if int(args.n_repeats) < 1:
        raise ValueError("--n-repeats must be positive.")
    device = require_device(str(args.device))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    rows: list[dict[str, object]] = []
    endpoint_records: list[dict[str, object]] = []

    for n_total in args.n_list:
        for repeat_idx in range(int(args.n_repeats)):
            repeat_seed = int(args.seed) + int(repeat_idx)
            dataset_path, result_path = _source_paths(args, int(n_total), int(repeat_idx))
            bundle = load_shared_dataset_npz(dataset_path)
            k = int(bundle.meta["num_categories"])
            theta_eval = np.eye(k, dtype=np.float64)
            train_labels = labels_from_theta(bundle.theta_train, num_categories=k)
            all_labels = labels_from_theta(bundle.theta_all, num_categories=k)
            train_means = class_means(bundle.x_train, train_labels, num_categories=k)
            all_means = class_means(bundle.x_all, all_labels, num_categories=k)
            with np.load(result_path, allow_pickle=False) as source:
                metric_names = tuple(str(value) for value in source["metric_names"].tolist())
                ground_truth = np.asarray(source["ground_truth_matrices"], dtype=np.float64)
                fm_mc_source = np.asarray(source["flow_matching_matrices"], dtype=np.float64)
                nll_mc_source = np.asarray(source["flow_matching_nll_finetuned_matrices"], dtype=np.float64)

            for metric in _METRICS:
                print(
                    f"[translation-diagnostic] N={n_total} repeat={repeat_idx} seed={repeat_seed} metric={metric}",
                    flush=True,
                )
                readout = _READOUTS[metric]
                metric_idx = metric_names.index(metric)
                gt_matrix = ground_truth[metric_idx]
                classical_all = readout(all_means)
                closed_form_train = readout(train_means)
                family = FLOW_VELOCITY_FAMILY_BY_METRIC[metric]
                model, fm_meta = _build_and_train_flow_model(
                    theta_train=np.asarray(bundle.theta_train, dtype=np.float64),
                    x_train=np.asarray(bundle.x_train, dtype=np.float64),
                    theta_val=np.asarray(bundle.theta_validation, dtype=np.float64),
                    x_val=np.asarray(bundle.x_validation, dtype=np.float64),
                    velocity_family=family,
                    device=device,
                    seed=repeat_seed,
                    config=config,
                )
                fm_endpoint = endpoint_means(model, theta_eval, device)
                fm_exact = readout(fm_endpoint)

                _seed_flow_rng(repeat_seed + 200_000, device)
                nll_meta = finetune_flow_skl_cnf_likelihood(
                    model=model,
                    theta_train=np.asarray(bundle.theta_train, dtype=np.float64),
                    x_train=np.asarray(bundle.x_train, dtype=np.float64),
                    theta_val=np.asarray(bundle.theta_validation, dtype=np.float64),
                    x_val=np.asarray(bundle.x_validation, dtype=np.float64),
                    device=device,
                    epochs=int(config.likelihood_finetune_epochs),
                    batch_size=int(config.likelihood_finetune_batch_size),
                    lr=float(config.likelihood_finetune_lr),
                    weight_decay=float(config.likelihood_finetune_weight_decay),
                    ode_steps=int(config.likelihood_finetune_ode_steps),
                    ode_method=str(config.likelihood_finetune_ode_method),
                    patience=int(config.likelihood_finetune_patience),
                    min_delta=float(config.likelihood_finetune_min_delta),
                    ema_alpha=float(config.likelihood_finetune_ema_alpha),
                    max_grad_norm=float(config.max_grad_norm),
                    checkpoint_selection=str(config.likelihood_finetune_checkpoint_selection),
                    log_every=max(1, int(config.log_every)),
                )
                nll_endpoint = endpoint_means(model, theta_eval, device)
                nll_exact = readout(nll_endpoint)
                matrices = {
                    "classical_all": classical_all,
                    "closed_form_train": closed_form_train,
                    "fm_exact": fm_exact,
                    "fm_mc": fm_mc_source[metric_idx],
                    "nll_exact": nll_exact,
                    "nll_mc": nll_mc_source[metric_idx],
                }
                for estimator, matrix in matrices.items():
                    rows.append(
                        {
                            "n_total": int(n_total),
                            "repeat_idx": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "metric": metric,
                            "estimator": estimator,
                            "relative_error_to_ground_truth": relative_pair_error(matrix, gt_matrix),
                            "relative_gap_to_train_optimum": relative_pair_error(matrix, closed_form_train),
                        }
                    )
                endpoint_records.append(
                    {
                        "n_total": int(n_total),
                        "repeat_idx": int(repeat_idx),
                        "metric": metric,
                        "train_means": train_means,
                        "fm_endpoint_means": fm_endpoint,
                        "nll_endpoint_means": nll_endpoint,
                        "fm_best_epoch": int(fm_meta["best_epoch"]),
                        "fm_best_val_loss": float(fm_meta["best_val_loss"]),
                        "nll_best_epoch": int(nll_meta["best_epoch"]),
                        "nll_best_val": float(nll_meta["best_val_nll"]),
                    }
                )

    rows_frame = pd.DataFrame(rows)
    summary = (
        rows_frame.groupby(["n_total", "metric", "estimator"], as_index=False)
        .agg(
            mean_relative_error_to_ground_truth=("relative_error_to_ground_truth", "mean"),
            std_relative_error_to_ground_truth=("relative_error_to_ground_truth", "std"),
            mean_relative_gap_to_train_optimum=("relative_gap_to_train_optimum", "mean"),
            std_relative_gap_to_train_optimum=("relative_gap_to_train_optimum", "std"),
        )
    )
    rows_path = output_dir / "translation_optimum_diagnostic_rows.csv"
    summary_path = output_dir / "translation_optimum_diagnostic_summary.csv"
    rows_frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    endpoint_path = output_dir / "translation_optimum_endpoint_means.npz"
    np.savez_compressed(endpoint_path, records=np.asarray(endpoint_records, dtype=object))
    metadata_path = output_dir / "translation_optimum_diagnostic_config.json"
    metadata_path.write_text(
        json.dumps(
            {
                "n_list": [int(value) for value in args.n_list],
                "n_repeats": int(args.n_repeats),
                "repeat_seeds": [int(args.seed) + idx for idx in range(int(args.n_repeats))],
                "source_case_name": str(args.source_case_name),
                "device": str(args.device),
                "config": config.__dict__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    png_path, svg_path = plot_summary(summary, output_dir)
    return {
        "rows_csv": rows_path,
        "summary_csv": summary_path,
        "endpoint_npz": endpoint_path,
        "config_json": metadata_path,
        "figure_png": png_path,
        "figure_svg": svg_path,
    }


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    n_total = int(summary["n_total"].max())
    selected = summary[summary["n_total"] == n_total]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    colors = ("#4C4C4C", "#9A9A9A", "#4C72B0", "#8AB0D7", "#55A868", "#9CCB9F")
    x = np.arange(len(_METRICS), dtype=np.float64)
    width = 0.12
    for index, estimator in enumerate(_ESTIMATOR_ORDER):
        estimator_rows = selected[selected["estimator"] == estimator].set_index("metric")
        values = [float(estimator_rows.loc[metric, "mean_relative_error_to_ground_truth"]) for metric in _METRICS]
        axes[0].bar(x + (index - 2.5) * width, values, width=width, color=colors[index], label=_ESTIMATOR_LABELS[estimator])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_METRIC_LABELS[metric] for metric in _METRICS], rotation=25, ha="right")
    axes[0].set_ylabel("Relative error to GT")
    axes[0].set_title(f"N = {n_total}")

    gap_estimators = ("fm_exact", "fm_mc", "nll_exact", "nll_mc")
    for index, estimator in enumerate(gap_estimators):
        estimator_rows = selected[selected["estimator"] == estimator].set_index("metric")
        values = [float(estimator_rows.loc[metric, "mean_relative_gap_to_train_optimum"]) for metric in _METRICS]
        axes[1].bar(x + (index - 1.5) * 0.18, values, width=0.17, color=colors[_ESTIMATOR_ORDER.index(estimator)], label=_ESTIMATOR_LABELS[estimator])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_METRIC_LABELS[metric] for metric in _METRICS], rotation=25, ha="right")
    axes[1].set_ylabel("Gap to train optimum")
    axes[1].set_title("Optimization and readout gap")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3)
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
    fig.tight_layout(w_pad=1.2)
    png_path = output_dir / "translation_optimum_diagnostic.png"
    svg_path = output_dir / "translation_optimum_diagnostic.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
