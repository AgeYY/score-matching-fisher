#!/usr/bin/env python3
"""Run MoG5 PR distance comparisons across sample-size sweeps."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import METRIC_NAMES

RESULTS_NAME = "mog5_pr_distance_comparison_results.npz"
SWEEP_NPZ_NAME = "mog5_pr_distance_sweep_results.npz"
SWEEP_CSV_NAME = "mog5_pr_distance_sweep_errors.csv"
SWEEP_SUMMARY_NAME = "mog5_pr_distance_sweep_summary.json"
SWEEP_SVG_NAME = "mog5_pr_distance_sweep_abs_error.svg"
SWEEP_PNG_NAME = "mog5_pr_distance_sweep_abs_error.png"
SWEEP_REL_SVG_NAME = "mog5_pr_distance_sweep_rel_error.svg"
SWEEP_REL_PNG_NAME = "mog5_pr_distance_sweep_rel_error.png"
SWEEP_FLOW_LOSS_SVG_NAME = "mog5_pr_distance_sweep_flow_loss_vs_epoch.svg"
SWEEP_FLOW_LOSS_PNG_NAME = "mog5_pr_distance_sweep_flow_loss_vs_epoch.png"
REL_ERROR_DENOM_FLOOR = 1e-12


def _load_single_case_module() -> Any:
    path = _REPO_ROOT / "bin" / "compare_mog5_pr_distances.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distances", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return vals


def _repo_data_dir() -> Path:
    return _REPO_ROOT / "data"


def default_output_dir() -> Path:
    return _repo_data_dir() / "mog5_pr_distance_sweeps"


def build_parser() -> argparse.ArgumentParser:
    single = _load_single_case_module()
    p = single.build_parser()
    p.description = __doc__
    p.set_defaults(n_total=1_000, pr_dim=5, output_dir=default_output_dir())
    for action in p._actions:
        if action.dest == "output_dir":
            action.help = "Aggregate sweep output directory."
    p.add_argument(
        "--n-list",
        type=_parse_int_list,
        default=[100, 550, 1000, 1550],
        help="Comma-separated sample-size sweep values.",
    )
    p.add_argument(
        "--case-output-name",
        type=str,
        default="distance_comparison_flow_skl",
        help="Per-case output directory name under each MoG5 PR dataset directory.",
    )
    p.add_argument(
        "--force-comparison",
        action="store_true",
        help="Rerun single-case comparisons even when the per-case result NPZ exists.",
    )
    p.add_argument(
        "--visualization-only",
        action="store_true",
        help="Only rebuild aggregate tables and figures from cached per-case result NPZs.",
    )
    p.add_argument("--yscale", choices=("log", "linear"), default="log", help="Y-axis scale for the error figure.")
    p.add_argument(
        "--loss-yscale",
        choices=("log", "linear"),
        default="linear",
        help="Y-axis scale for the aggregate flow loss-vs-epoch figure.",
    )
    return p


def case_output_dir(*, n_total: int, pr_dim: int, case_output_name: str) -> Path:
    single = _load_single_case_module()
    return single.default_dataset_dir(n_total=int(n_total), pr_dim=int(pr_dim)) / str(case_output_name)


def case_results_npz(*, n_total: int, pr_dim: int, case_output_name: str) -> Path:
    return case_output_dir(n_total=n_total, pr_dim=pr_dim, case_output_name=case_output_name) / RESULTS_NAME


def case_flow_loss_npz(*, n_total: int, pr_dim: int, case_output_name: str, metric: str) -> Path:
    return (
        case_output_dir(n_total=n_total, pr_dim=pr_dim, case_output_name=case_output_name)
        / "flow"
        / f"{metric}_flow_matching_skl_results.npz"
    )


def _unique_cases(args: argparse.Namespace) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for n_total in args.n_list:
        case = (int(n_total), int(args.pr_dim))
        if case not in seen:
            cases.append(case)
            seen.add(case)
    return cases


def resolve_metric_names(args: argparse.Namespace) -> tuple[str, ...]:
    single = _load_single_case_module()
    if hasattr(single, "resolve_metric_names"):
        return tuple(str(m) for m in single.resolve_metric_names(args))
    metric = str(getattr(args, "metric", "all"))
    if metric == "all":
        return tuple(METRIC_NAMES)
    return (metric,)


def _single_case_args(args: argparse.Namespace, *, n_total: int, pr_dim: int, output_dir: Path) -> argparse.Namespace:
    single = _load_single_case_module()
    case_args = single.build_parser().parse_args([])
    for key, value in vars(args).items():
        if hasattr(case_args, key):
            setattr(case_args, key, value)
    case_args.n_total = int(n_total)
    case_args.pr_dim = int(pr_dim)
    case_args.dataset_dir = None
    case_args.output_dir = Path(output_dir)
    return case_args


def _load_case_cache(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing cached comparison results: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "metric_names": tuple(str(v) for v in data["metric_names"].tolist()),
            "condition_labels": tuple(str(v) for v in data["condition_labels"].tolist()),
            "pair_indices": np.asarray(data["pair_indices"], dtype=np.int64),
            "classical_matrices": np.asarray(data["classical_matrices"], dtype=np.float64),
            "flow_matching_matrices": np.asarray(data["flow_matching_matrices"], dtype=np.float64),
            "ground_truth_matrices": np.asarray(data["ground_truth_matrices"], dtype=np.float64),
        }


def _load_flow_loss_cache(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing cached flow loss results: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ("train_losses", "val_losses")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Cached flow loss results {path} are missing: {', '.join(missing)}")
        out: dict[str, Any] = {
            "train_losses": np.asarray(data["train_losses"], dtype=np.float64),
            "val_losses": np.asarray(data["val_losses"], dtype=np.float64),
        }
        if "val_monitor_losses" in data.files:
            out["val_monitor_losses"] = np.asarray(data["val_monitor_losses"], dtype=np.float64)
        for key in ("best_epoch", "stopped_epoch", "stopped_early"):
            if key in data.files:
                value = np.asarray(data[key]).reshape(-1)
                if value.size:
                    out[key] = value[0].item()
        return out


def _filter_case_metrics(data: dict[str, Any], metrics: tuple[str, ...], *, path: Path | None = None) -> dict[str, Any]:
    available = tuple(str(v) for v in data["metric_names"])
    missing = [metric for metric in metrics if metric not in available]
    if missing:
        where = "" if path is None else f" in {path}"
        raise ValueError(f"Cached comparison results{where} are missing requested metric(s): {', '.join(missing)}")
    indices = [available.index(metric) for metric in metrics]
    return {
        "metric_names": tuple(metrics),
        "condition_labels": tuple(data["condition_labels"]),
        "pair_indices": np.asarray(data["pair_indices"], dtype=np.int64),
        "classical_matrices": np.asarray(data["classical_matrices"], dtype=np.float64)[indices],
        "flow_matching_matrices": np.asarray(data["flow_matching_matrices"], dtype=np.float64)[indices],
        "ground_truth_matrices": np.asarray(data["ground_truth_matrices"], dtype=np.float64)[indices],
    }


def ensure_case_results(args: argparse.Namespace, *, n_total: int, pr_dim: int) -> tuple[Path, bool]:
    output_dir = case_output_dir(n_total=n_total, pr_dim=pr_dim, case_output_name=str(args.case_output_name))
    result_path = output_dir / RESULTS_NAME
    if result_path.is_file() and not bool(args.force_comparison):
        requested_metrics = resolve_metric_names(args)
        try:
            _filter_case_metrics(_load_case_cache(result_path), requested_metrics, path=result_path)
            print(f"[sweep] cache hit n_total={n_total} pr_dim={pr_dim}: {result_path}", flush=True)
            return result_path, True
        except ValueError:
            if bool(args.visualization_only):
                raise
            print(f"[sweep] cache missing requested metrics; rerunning n_total={n_total} pr_dim={pr_dim}", flush=True)
    if bool(args.visualization_only):
        raise FileNotFoundError(f"--visualization-only requires cached results: {result_path}")

    single = _load_single_case_module()
    print(f"[sweep] running comparison n_total={n_total} pr_dim={pr_dim}", flush=True)
    paths = single.run(_single_case_args(args, n_total=n_total, pr_dim=pr_dim, output_dir=output_dir))
    return Path(paths["results_npz"]), False


def _mean_pair_abs_error(est: np.ndarray, gt: np.ndarray, pairs: np.ndarray) -> float:
    vals = [abs(float(est[int(i), int(j)]) - float(gt[int(i), int(j)])) for i, j in pairs]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _relative_error(abs_error: float, ground_truth: float) -> float:
    return float(abs_error) / max(abs(float(ground_truth)), REL_ERROR_DENOM_FLOOR)


def _mean_pair_error(est: np.ndarray, gt: np.ndarray, pairs: np.ndarray, *, relative: bool) -> float:
    vals = []
    for i, j in pairs:
        ci, cj = int(i), int(j)
        truth = float(gt[ci, cj])
        abs_error = abs(float(est[ci, cj]) - truth)
        vals.append(_relative_error(abs_error, truth) if relative else abs_error)
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def aggregate_sweeps(
    *,
    args: argparse.Namespace,
    case_data: dict[tuple[int, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first = case_data[_unique_cases(args)[0]]
    metric_names = tuple(first["metric_names"])
    condition_labels = tuple(first["condition_labels"])
    pair_indices = np.asarray(first["pair_indices"], dtype=np.int64)

    for case, data in case_data.items():
        if tuple(data["metric_names"]) != metric_names:
            raise ValueError(f"Metric names differ for case {case}.")
        if tuple(data["condition_labels"]) != condition_labels:
            raise ValueError(f"Condition labels differ for case {case}.")
        np.testing.assert_array_equal(np.asarray(data["pair_indices"], dtype=np.int64), pair_indices)

    n_cases = [(int(n_total), int(args.pr_dim)) for n_total in args.n_list]
    def stack(cases: list[tuple[int, int]], key: str) -> np.ndarray:
        return np.stack([np.asarray(case_data[case][key], dtype=np.float64) for case in cases], axis=0)

    aggregate = {
        "metric_names": metric_names,
        "condition_labels": condition_labels,
        "pair_indices": pair_indices,
        "n_list": np.asarray(args.n_list, dtype=np.int64),
        "pr_dim": int(args.pr_dim),
        "n_total": int(args.n_total),
        "n_sweep_classical_matrices": stack(n_cases, "classical_matrices"),
        "n_sweep_flow_matching_matrices": stack(n_cases, "flow_matching_matrices"),
        "n_sweep_ground_truth_matrices": stack(n_cases, "ground_truth_matrices"),
    }

    rows: list[dict[str, Any]] = []
    for n_total, pr_dim in n_cases:
        data = case_data[(n_total, pr_dim)]
        for metric_idx, metric in enumerate(metric_names):
            gt = np.asarray(data["ground_truth_matrices"][metric_idx], dtype=np.float64)
            for i, j in pair_indices:
                ci, cj = int(i), int(j)
                for estimator, matrix_key in (
                    ("classical", "classical_matrices"),
                    ("flow_matching", "flow_matching_matrices"),
                ):
                    est = float(np.asarray(data[matrix_key][metric_idx], dtype=np.float64)[ci, cj])
                    truth = float(gt[ci, cj])
                    abs_error = abs(est - truth)
                    rows.append(
                        {
                            "axis": "n_total",
                            "n_total": int(n_total),
                            "pr_dim": int(pr_dim),
                            "metric": str(metric),
                            "estimator": estimator,
                            "condition_i": condition_labels[ci],
                            "condition_j": condition_labels[cj],
                            "estimate": est,
                            "ground_truth": truth,
                            "abs_error": abs_error,
                            "rel_error": _relative_error(abs_error, truth),
                        }
                    )
    return aggregate, rows


def write_aggregate_npz(path: Path, aggregate: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metric_names=np.asarray(aggregate["metric_names"]),
        condition_labels=np.asarray(aggregate["condition_labels"]),
        pair_indices=np.asarray(aggregate["pair_indices"], dtype=np.int64),
        n_list=np.asarray(aggregate["n_list"], dtype=np.int64),
        pr_dim=np.asarray([int(aggregate["pr_dim"])], dtype=np.int64),
        n_total=np.asarray([int(aggregate["n_total"])], dtype=np.int64),
        n_sweep_classical_matrices=np.asarray(aggregate["n_sweep_classical_matrices"], dtype=np.float64),
        n_sweep_flow_matching_matrices=np.asarray(aggregate["n_sweep_flow_matching_matrices"], dtype=np.float64),
        n_sweep_ground_truth_matrices=np.asarray(aggregate["n_sweep_ground_truth_matrices"], dtype=np.float64),
    )
    return path


def write_errors_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "axis",
        "n_total",
        "pr_dim",
        "metric",
        "estimator",
        "condition_i",
        "condition_j",
        "estimate",
        "ground_truth",
        "abs_error",
        "rel_error",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_sweep_error(
    aggregate: dict[str, Any],
    *,
    svg_path: Path,
    png_path: Path,
    yscale: str,
    relative: bool,
) -> tuple[Path, Path]:
    metric_names = tuple(str(v) for v in aggregate["metric_names"])
    pair_indices = np.asarray(aggregate["pair_indices"], dtype=np.int64)
    colors = plt.get_cmap("tab10")
    estimator_styles = {
        "classical": {"marker": "o", "linestyle": "-", "label": "classical"},
        "flow_matching": {"marker": "s", "linestyle": "--", "label": "flow matching"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    panels = (
        (
            axes[0],
            np.asarray(aggregate["n_list"], dtype=np.int64),
            "n_total",
            aggregate["n_sweep_classical_matrices"],
            aggregate["n_sweep_flow_matching_matrices"],
            aggregate["n_sweep_ground_truth_matrices"],
            f"Sample-size sweep (PR dim={int(aggregate['pr_dim'])})",
            ("classical", "flow_matching"),
        ),
        (
            axes[1],
            np.asarray(aggregate["n_list"], dtype=np.int64),
            "n_total",
            aggregate["n_sweep_classical_matrices"],
            aggregate["n_sweep_flow_matching_matrices"],
            aggregate["n_sweep_ground_truth_matrices"],
            f"Flow estimation only (PR dim={int(aggregate['pr_dim'])})",
            ("flow_matching",),
        ),
    )

    handles_by_label: dict[str, Any] = {}
    for ax, xvals, xlabel, classical, flow, gt, title, estimators in panels:
        for metric_idx, metric in enumerate(metric_names):
            color = colors(metric_idx % 10)
            matrices_by_estimator = {
                "classical": classical,
                "flow_matching": flow,
            }
            for estimator in estimators:
                matrices = matrices_by_estimator[estimator]
                yvals = [
                    _mean_pair_error(
                        np.asarray(matrices[row_idx, metric_idx], dtype=np.float64),
                        np.asarray(gt[row_idx, metric_idx], dtype=np.float64),
                        pair_indices,
                        relative=bool(relative),
                    )
                    for row_idx in range(len(xvals))
                ]
                style = estimator_styles[estimator]
                (line,) = ax.plot(
                    xvals,
                    np.asarray(yvals, dtype=np.float64),
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    markersize=4.5,
                    label=f"{metric} · {style['label']}",
                )
                handles_by_label[line.get_label()] = line
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean relative absolute error" if relative else "Mean absolute error")
        ax.set_xticks(xvals)
        ax.grid(True, which="both", alpha=0.25)
        if str(yscale) == "log":
            ax.set_yscale("log")
            ymin = min((line.get_ydata()[line.get_ydata() > 0].min() for line in ax.lines if np.any(line.get_ydata() > 0)), default=1e-12)
            ax.set_ylim(bottom=max(float(ymin) * 0.5, 1e-12))

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    title_kind = "relative absolute error" if relative else "absolute error"
    fig.suptitle(f"MoG5 PR distance comparison {title_kind}", fontsize=13)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return svg_path, png_path


def plot_abs_error(aggregate: dict[str, Any], *, svg_path: Path, png_path: Path, yscale: str) -> tuple[Path, Path]:
    return plot_sweep_error(aggregate, svg_path=svg_path, png_path=png_path, yscale=yscale, relative=False)


def plot_flow_loss_sweep(
    *,
    loss_data: dict[tuple[int, int, str], dict[str, Any]],
    n_list: list[int],
    pr_dim: int,
    metrics: tuple[str, ...],
    svg_path: Path,
    png_path: Path,
    yscale: str,
) -> tuple[Path, Path] | None:
    if not loss_data:
        return None

    metric_tuple = tuple(
        str(metric)
        for metric in metrics
        if any((int(n_total), int(pr_dim), str(metric)) in loss_data for n_total in n_list)
    )
    if not metric_tuple:
        return None

    fig_width = max(6.5, 5.2 * len(metric_tuple))
    fig, axes_obj = plt.subplots(1, len(metric_tuple), figsize=(fig_width, 4.9), squeeze=False, constrained_layout=True)
    axes = axes_obj[0]
    loss_styles = {
        "train": {"color": "tab:blue", "linestyle": "-", "label": "train"},
        "val": {"color": "tab:orange", "linestyle": "--", "label": "val"},
        "val_monitor": {"color": "tab:green", "linestyle": ":", "label": "val EMA"},
    }
    handles_by_label: dict[str, Any] = {}

    for ax, metric in zip(axes, metric_tuple):
        for n_total in n_list:
            case_key = (int(n_total), int(pr_dim), str(metric))
            item = loss_data.get(case_key)
            if item is None:
                continue
            train_losses = np.asarray(item["train_losses"], dtype=np.float64).reshape(-1)
            val_losses = np.asarray(item["val_losses"], dtype=np.float64).reshape(-1)
            val_monitor_losses = np.asarray(item.get("val_monitor_losses", []), dtype=np.float64).reshape(-1)
            if train_losses.size == 0 and val_losses.size == 0 and val_monitor_losses.size == 0:
                continue
            if train_losses.size:
                style = loss_styles["train"]
                epochs = np.arange(1, train_losses.size + 1, dtype=np.int64)
                (line,) = ax.plot(
                    epochs,
                    train_losses,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    label=f"n={int(n_total)} {style['label']}",
                )
                handles_by_label[line.get_label()] = line
            if val_losses.size:
                style = loss_styles["val"]
                epochs = np.arange(1, val_losses.size + 1, dtype=np.int64)
                (line,) = ax.plot(
                    epochs,
                    val_losses,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    label=f"n={int(n_total)} {style['label']}",
                )
                handles_by_label[line.get_label()] = line

                best_epoch = item.get("best_epoch")
                if best_epoch is not None:
                    best_idx = int(best_epoch) - 1
                    if 0 <= best_idx < val_losses.size:
                        ax.scatter(
                            [best_idx + 1],
                            [float(val_losses[best_idx])],
                            color=style["color"],
                            marker="o",
                            s=28,
                            edgecolors="black",
                            linewidths=0.45,
                            zorder=4,
                        )

                stopped_epoch = item.get("stopped_epoch")
                if stopped_epoch is not None:
                    stopped_idx = int(stopped_epoch) - 1
                    if 0 <= stopped_idx < val_losses.size:
                        stopped_early = bool(item.get("stopped_early", False))
                        ax.scatter(
                            [stopped_idx + 1],
                            [float(val_losses[stopped_idx])],
                            color=style["color"],
                            marker="x" if stopped_early else "|",
                            s=38,
                            linewidths=1.3,
                            zorder=5,
                        )
            if val_monitor_losses.size:
                style = loss_styles["val_monitor"]
                epochs = np.arange(1, val_monitor_losses.size + 1, dtype=np.int64)
                (line,) = ax.plot(
                    epochs,
                    val_monitor_losses,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    label=f"n={int(n_total)} {style['label']}",
                )
                handles_by_label[line.get_label()] = line
        ax.set_title(str(metric))
        ax.set_xlabel("epoch")
        ax.set_ylabel("flow matching loss")
        ax.grid(True, which="both", alpha=0.25)
        if str(yscale) == "log":
            ax.set_yscale("log")
            positive = [
                float(value)
                for line in ax.lines
                for value in np.asarray(line.get_ydata(), dtype=np.float64)
                if np.isfinite(value) and value > 0.0
            ]
            if positive:
                ax.set_ylim(bottom=max(min(positive) * 0.5, 1e-12))

    if not handles_by_label:
        plt.close(fig)
        return None

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=min(4, max(1, len(handles_by_label))),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(f"MoG5 PR flow training loss sweep (PR dim={int(pr_dim)})", fontsize=13)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return svg_path, png_path


def write_summary(path: Path, *, args: argparse.Namespace, case_paths: dict[tuple[int, int], Path], cache_hits: dict[tuple[int, int], bool], outputs: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = resolve_metric_names(args)
    payload = {
        "script": "bin/compare_mog5_pr_distance_sweeps.py",
        "config": {
            "n_list": [int(v) for v in args.n_list],
            "pr_dim": int(args.pr_dim),
            "n_total": int(args.n_total),
            "seed": int(args.seed),
            "device": str(args.device),
            "case_output_name": str(args.case_output_name),
            "force_comparison": bool(args.force_comparison),
            "visualization_only": bool(args.visualization_only),
            "metric": str(args.metric),
            "metrics": list(metrics),
            "yscale": str(args.yscale),
            "abs_error_yscale": str(args.yscale),
            "rel_error_yscale": "linear",
            "loss_yscale": str(args.loss_yscale),
        },
        "case_paths": {
            f"n{int(n_total)}_pr{int(pr_dim)}": str(path)
            for (n_total, pr_dim), path in sorted(case_paths.items())
        },
        "cache_hits": {
            f"n{int(n_total)}_pr{int(pr_dim)}": bool(hit)
            for (n_total, pr_dim), hit in sorted(cache_hits.items())
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = resolve_metric_names(args)

    case_paths: dict[tuple[int, int], Path] = {}
    cache_hits: dict[tuple[int, int], bool] = {}
    case_data: dict[tuple[int, int], dict[str, Any]] = {}
    for n_total, pr_dim in _unique_cases(args):
        path, cache_hit = ensure_case_results(args, n_total=n_total, pr_dim=pr_dim)
        case = (int(n_total), int(pr_dim))
        case_paths[case] = Path(path)
        cache_hits[case] = bool(cache_hit)
        case_data[case] = _filter_case_metrics(_load_case_cache(Path(path)), metrics, path=Path(path))

    flow_loss_data: dict[tuple[int, int, str], dict[str, Any]] = {}
    flow_loss_warnings: list[str] = []
    for n_total, pr_dim in _unique_cases(args):
        for metric in metrics:
            loss_path = case_flow_loss_npz(
                n_total=int(n_total),
                pr_dim=int(pr_dim),
                case_output_name=str(args.case_output_name),
                metric=str(metric),
            )
            try:
                flow_loss_data[(int(n_total), int(pr_dim), str(metric))] = _load_flow_loss_cache(loss_path)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                flow_loss_warnings.append(str(exc))

    aggregate, rows = aggregate_sweeps(args=args, case_data=case_data)
    npz_path = write_aggregate_npz(output_dir / SWEEP_NPZ_NAME, aggregate)
    csv_path = write_errors_csv(output_dir / SWEEP_CSV_NAME, rows)
    svg_path, png_path = plot_abs_error(
        aggregate,
        svg_path=output_dir / SWEEP_SVG_NAME,
        png_path=output_dir / SWEEP_PNG_NAME,
        yscale=str(args.yscale),
    )
    rel_svg_path, rel_png_path = plot_sweep_error(
        aggregate,
        svg_path=output_dir / SWEEP_REL_SVG_NAME,
        png_path=output_dir / SWEEP_REL_PNG_NAME,
        yscale="linear",
        relative=True,
    )
    loss_paths = plot_flow_loss_sweep(
        loss_data=flow_loss_data,
        n_list=[int(v) for v in args.n_list],
        pr_dim=int(args.pr_dim),
        metrics=metrics,
        svg_path=output_dir / SWEEP_FLOW_LOSS_SVG_NAME,
        png_path=output_dir / SWEEP_FLOW_LOSS_PNG_NAME,
        yscale=str(args.loss_yscale),
    )
    outputs = {
        "results_npz": npz_path,
        "errors_csv": csv_path,
        "figure_svg": svg_path,
        "figure_png": png_path,
        "abs_error_figure_svg": svg_path,
        "abs_error_figure_png": png_path,
        "rel_error_figure_svg": rel_svg_path,
        "rel_error_figure_png": rel_png_path,
    }
    if loss_paths is not None:
        outputs["flow_loss_figure_svg"] = loss_paths[0]
        outputs["flow_loss_figure_png"] = loss_paths[1]
    summary_path = write_summary(
        output_dir / SWEEP_SUMMARY_NAME,
        args=args,
        case_paths=case_paths,
        cache_hits=cache_hits,
        outputs=outputs,
    )
    outputs["summary_json"] = summary_path
    print(f"results_npz: {npz_path}", flush=True)
    print(f"errors_csv: {csv_path}", flush=True)
    print(f"figure_svg: {svg_path}", flush=True)
    print(f"figure_png: {png_path}", flush=True)
    print(f"rel_error_figure_svg: {rel_svg_path}", flush=True)
    print(f"rel_error_figure_png: {rel_png_path}", flush=True)
    if loss_paths is None:
        if flow_loss_warnings:
            print(
                f"[sweep] warning: no usable flow loss histories found; skipped flow loss figure. First issue: {flow_loss_warnings[0]}",
                flush=True,
            )
        else:
            print("[sweep] warning: no usable flow loss histories found; skipped flow loss figure.", flush=True)
    else:
        print(f"flow_loss_figure_svg: {loss_paths[0]}", flush=True)
        print(f"flow_loss_figure_png: {loss_paths[1]}", flush=True)
        if flow_loss_warnings:
            print(
                f"[sweep] warning: skipped {len(flow_loss_warnings)} missing or incomplete flow loss cache(s). First issue: {flow_loss_warnings[0]}",
                flush=True,
            )
    print(f"summary_json: {summary_path}", flush=True)
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
