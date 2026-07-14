#!/usr/bin/env python3
"""Compare full-data and train-split classical MoG5 distances with saved flow estimates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import classical_metric_matrices, labels_from_theta
from fisher.shared_dataset_io import load_shared_dataset_npz


DEFAULT_SOURCE_DIR = (
    _REPO_ROOT
    / "data"
    / "mog5_non_skl_n50_1000_2000_3000_r5_constant_lr_e20000_pat1000_fmval10paths_nll500"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "classical_train_split_diagnostic"
METRIC_TITLES = {
    "correlation": "Correlation",
    "cosine": "Cosine",
    "squared_euclidean": "Squared Euclidean",
    "mahalanobis_sq": "Squared Mahalanobis",
}
ESTIMATOR_STYLES = {
    "classical_all": {"label": "classical, all data", "color": "C1", "linestyle": "-"},
    "classical_train": {"label": "classical, train 80%", "color": "C3", "linestyle": "--"},
    "flow_matching": {"label": "flow matching", "color": "C0", "linestyle": "-"},
    "flow_matching_nll_finetuned": {
        "label": "flow matching + NLL",
        "color": "C2",
        "linestyle": "--",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mahalanobis-ridge", type=float, default=1e-6)
    return parser


def _repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else _REPO_ROOT / value


def _load_case(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        required = (
            "metric_names",
            "condition_labels",
            "pair_indices",
            "classical_matrices",
            "flow_matching_matrices",
            "flow_matching_nll_finetuned_matrices",
            "ground_truth_matrices",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing fields in {path}: {', '.join(missing)}")
        return {
            "metric_names": tuple(str(value) for value in data["metric_names"].tolist()),
            "condition_labels": tuple(str(value) for value in data["condition_labels"].tolist()),
            "pair_indices": np.asarray(data["pair_indices"], dtype=np.int64),
            "classical_all": np.asarray(data["classical_matrices"], dtype=np.float64),
            "flow_matching": np.asarray(data["flow_matching_matrices"], dtype=np.float64),
            "flow_matching_nll_finetuned": np.asarray(
                data["flow_matching_nll_finetuned_matrices"], dtype=np.float64
            ),
            "ground_truth": np.asarray(data["ground_truth_matrices"], dtype=np.float64),
        }


def _case_rows(
    *,
    result_path: Path,
    n_total: int,
    repeat_idx: int,
    repeat_seed: int,
    mahalanobis_ridge: float,
) -> list[dict[str, object]]:
    case = _load_case(result_path)
    metric_names = tuple(case["metric_names"])
    dataset_path = result_path.parent.parent / "random_mog_categorical.npz"
    bundle = load_shared_dataset_npz(dataset_path)
    num_categories = int(bundle.meta.get("num_categories", bundle.theta_train.shape[1]))
    train_labels = labels_from_theta(bundle.theta_train, num_categories=num_categories)
    train_matrices = classical_metric_matrices(
        bundle.x_train,
        train_labels,
        num_categories=num_categories,
        metrics=metric_names,
        mahalanobis_ridge=float(mahalanobis_ridge),
    )
    classical_train = np.stack([train_matrices[metric] for metric in metric_names], axis=0)
    matrices = {
        "classical_all": np.asarray(case["classical_all"], dtype=np.float64),
        "classical_train": classical_train,
        "flow_matching": np.asarray(case["flow_matching"], dtype=np.float64),
        "flow_matching_nll_finetuned": np.asarray(case["flow_matching_nll_finetuned"], dtype=np.float64),
    }
    ground_truth = np.asarray(case["ground_truth"], dtype=np.float64)
    condition_labels = tuple(case["condition_labels"])
    rows: list[dict[str, object]] = []
    for metric_idx, metric in enumerate(metric_names):
        for i, j in np.asarray(case["pair_indices"], dtype=np.int64):
            ci, cj = int(i), int(j)
            truth = float(ground_truth[metric_idx, ci, cj])
            for estimator, values in matrices.items():
                estimate = float(values[metric_idx, ci, cj])
                abs_error = abs(estimate - truth)
                rows.append(
                    {
                        "n_total": int(n_total),
                        "n_train": int(bundle.x_train.shape[0]),
                        "repeat_idx": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "metric": str(metric),
                        "estimator": str(estimator),
                        "condition_i": condition_labels[ci],
                        "condition_j": condition_labels[cj],
                        "estimate": estimate,
                        "ground_truth": truth,
                        "abs_error": abs_error,
                        "rel_error": abs_error / max(abs(truth), 1e-12),
                    }
                )
    return rows


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    per_repeat = (
        rows.groupby(["metric", "estimator", "n_total", "n_train", "repeat_idx"], as_index=False)
        .agg(mae=("abs_error", "mean"), mrae=("rel_error", "mean"))
    )
    return (
        per_repeat.groupby(["metric", "estimator", "n_total", "n_train"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            mrae_mean=("mrae", "mean"),
            mrae_sd=("mrae", "std"),
            n_repeats=("repeat_idx", "size"),
        )
        .sort_values(["metric", "estimator", "n_total"])
        .reset_index(drop=True)
    )


def plot_summary(summary: pd.DataFrame, *, png_path: Path, svg_path: Path) -> None:
    metrics = [metric for metric in METRIC_TITLES if metric in set(summary["metric"])]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.0 * len(metrics), 7.0), squeeze=False)
    handles: dict[str, object] = {}
    for col, metric in enumerate(metrics):
        metric_frame = summary[summary["metric"].eq(metric)]
        axes[0, col].set_title(METRIC_TITLES[metric], fontsize=16)
        for row, (mean_col, sd_col, ylabel) in enumerate(
            (("mae_mean", "mae_sd", "MAE"), ("mrae_mean", "mrae_sd", "MRAE"))
        ):
            ax = axes[row, col]
            for estimator, style in ESTIMATOR_STYLES.items():
                frame = metric_frame[metric_frame["estimator"].eq(estimator)].sort_values("n_total")
                errorbar = ax.errorbar(
                    frame["n_total"],
                    frame[mean_col],
                    yerr=frame[sd_col],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    elinewidth=1.3,
                    capsize=3.0,
                    label=style["label"],
                )
                handles[str(style["label"])] = errorbar
            ax.set_xticks(sorted(int(value) for value in metric_frame["n_total"].unique()))
            ax.set_xlabel("Total sample size $N$", fontsize=16)
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=16)
            ax.tick_params(axis="both", labelsize=14, width=1.8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.15, wspace=0.30, hspace=0.35)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Path]:
    source_dir = _repo_path(args.source_dir)
    output_dir = _repo_path(args.output_dir)
    summary_path = source_dir / "mog5_pr_distance_sweep_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    config = payload["config"]
    if bool(config.get("pr_projected")):
        raise ValueError("This diagnostic currently supports native-coordinate MoG5 sweeps only.")
    all_rows: list[dict[str, object]] = []
    for n_total in config["n_list"]:
        for repeat_idx, repeat_seed in enumerate(config["repeat_seeds"]):
            key = f"n{int(n_total)}_repeat{int(repeat_idx):02d}_native"
            result_path = _repo_path(payload["case_paths"][key])
            all_rows.extend(
                _case_rows(
                    result_path=result_path,
                    n_total=int(n_total),
                    repeat_idx=int(repeat_idx),
                    repeat_seed=int(repeat_seed),
                    mahalanobis_ridge=float(args.mahalanobis_ridge),
                )
            )
    rows = pd.DataFrame(all_rows)
    summary = summarize_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "classical_train_split_rows.csv"
    table_path = output_dir / "classical_train_split_summary.csv"
    png_path = output_dir / "classical_train_split_comparison.png"
    svg_path = output_dir / "classical_train_split_comparison.svg"
    json_path = output_dir / "classical_train_split_config.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(table_path, index=False)
    plot_summary(summary, png_path=png_path, svg_path=svg_path)
    json_path.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "source_summary": str(summary_path),
                "output_dir": str(output_dir),
                "mahalanobis_ridge": float(args.mahalanobis_ridge),
                "n_list": [int(value) for value in config["n_list"]],
                "repeat_seeds": [int(value) for value in config["repeat_seeds"]],
                "estimators": list(ESTIMATOR_STYLES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = {
        "rows_csv": rows_path,
        "summary_csv": table_path,
        "figure_png": png_path,
        "figure_svg": svg_path,
        "config_json": json_path,
    }
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if float(args.mahalanobis_ridge) < 0.0:
        raise ValueError("--mahalanobis-ridge must be nonnegative.")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
