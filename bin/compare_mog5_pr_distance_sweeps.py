#!/usr/bin/env python3
"""Run MoG5 PR distance comparisons across sample-size and PR-dimension sweeps."""

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

RESULTS_NAME = "mog5_pr_distance_comparison_results.npz"
SWEEP_NPZ_NAME = "mog5_pr_distance_sweep_results.npz"
SWEEP_CSV_NAME = "mog5_pr_distance_sweep_errors.csv"
SWEEP_SUMMARY_NAME = "mog5_pr_distance_sweep_summary.json"
SWEEP_SVG_NAME = "mog5_pr_distance_sweep_abs_error.svg"
SWEEP_PNG_NAME = "mog5_pr_distance_sweep_abs_error.png"
SWEEP_REL_SVG_NAME = "mog5_pr_distance_sweep_rel_error.svg"
SWEEP_REL_PNG_NAME = "mog5_pr_distance_sweep_rel_error.png"
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
        "--pr-dim-list",
        type=_parse_int_list,
        default=[3, 5, 8, 11],
        help="Comma-separated PR-dimension sweep values.",
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
    return p


def case_output_dir(*, n_total: int, pr_dim: int, case_output_name: str) -> Path:
    single = _load_single_case_module()
    return single.default_dataset_dir(n_total=int(n_total), pr_dim=int(pr_dim)) / str(case_output_name)


def case_results_npz(*, n_total: int, pr_dim: int, case_output_name: str) -> Path:
    return case_output_dir(n_total=n_total, pr_dim=pr_dim, case_output_name=case_output_name) / RESULTS_NAME


def _unique_cases(args: argparse.Namespace) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for n_total in args.n_list:
        case = (int(n_total), int(args.pr_dim))
        if case not in seen:
            cases.append(case)
            seen.add(case)
    for pr_dim in args.pr_dim_list:
        case = (int(args.n_total), int(pr_dim))
        if case not in seen:
            cases.append(case)
            seen.add(case)
    return cases


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


def ensure_case_results(args: argparse.Namespace, *, n_total: int, pr_dim: int) -> tuple[Path, bool]:
    output_dir = case_output_dir(n_total=n_total, pr_dim=pr_dim, case_output_name=str(args.case_output_name))
    result_path = output_dir / RESULTS_NAME
    if result_path.is_file() and not bool(args.force_comparison):
        print(f"[sweep] cache hit n_total={n_total} pr_dim={pr_dim}: {result_path}", flush=True)
        return result_path, True
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
    pr_cases = [(int(args.n_total), int(pr_dim)) for pr_dim in args.pr_dim_list]

    def stack(cases: list[tuple[int, int]], key: str) -> np.ndarray:
        return np.stack([np.asarray(case_data[case][key], dtype=np.float64) for case in cases], axis=0)

    aggregate = {
        "metric_names": metric_names,
        "condition_labels": condition_labels,
        "pair_indices": pair_indices,
        "n_list": np.asarray(args.n_list, dtype=np.int64),
        "pr_dim": int(args.pr_dim),
        "pr_dim_list": np.asarray(args.pr_dim_list, dtype=np.int64),
        "n_total": int(args.n_total),
        "n_sweep_classical_matrices": stack(n_cases, "classical_matrices"),
        "n_sweep_flow_matching_matrices": stack(n_cases, "flow_matching_matrices"),
        "n_sweep_ground_truth_matrices": stack(n_cases, "ground_truth_matrices"),
        "pr_dim_sweep_classical_matrices": stack(pr_cases, "classical_matrices"),
        "pr_dim_sweep_flow_matching_matrices": stack(pr_cases, "flow_matching_matrices"),
        "pr_dim_sweep_ground_truth_matrices": stack(pr_cases, "ground_truth_matrices"),
    }

    rows: list[dict[str, Any]] = []
    for axis, cases in (("n_total", n_cases), ("pr_dim", pr_cases)):
        for n_total, pr_dim in cases:
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
                                "axis": axis,
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
        pr_dim_list=np.asarray(aggregate["pr_dim_list"], dtype=np.int64),
        n_total=np.asarray([int(aggregate["n_total"])], dtype=np.int64),
        n_sweep_classical_matrices=np.asarray(aggregate["n_sweep_classical_matrices"], dtype=np.float64),
        n_sweep_flow_matching_matrices=np.asarray(aggregate["n_sweep_flow_matching_matrices"], dtype=np.float64),
        n_sweep_ground_truth_matrices=np.asarray(aggregate["n_sweep_ground_truth_matrices"], dtype=np.float64),
        pr_dim_sweep_classical_matrices=np.asarray(aggregate["pr_dim_sweep_classical_matrices"], dtype=np.float64),
        pr_dim_sweep_flow_matching_matrices=np.asarray(aggregate["pr_dim_sweep_flow_matching_matrices"], dtype=np.float64),
        pr_dim_sweep_ground_truth_matrices=np.asarray(aggregate["pr_dim_sweep_ground_truth_matrices"], dtype=np.float64),
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
            int(aggregate["pr_dim"]),
            aggregate["n_sweep_classical_matrices"],
            aggregate["n_sweep_flow_matching_matrices"],
            aggregate["n_sweep_ground_truth_matrices"],
            f"Sample-size sweep (PR dim={int(aggregate['pr_dim'])})",
        ),
        (
            axes[1],
            np.asarray(aggregate["pr_dim_list"], dtype=np.int64),
            "pr_dim",
            int(aggregate["n_total"]),
            aggregate["pr_dim_sweep_classical_matrices"],
            aggregate["pr_dim_sweep_flow_matching_matrices"],
            aggregate["pr_dim_sweep_ground_truth_matrices"],
            f"PR-dimension sweep (n={int(aggregate['n_total'])})",
        ),
    )

    handles_by_label: dict[str, Any] = {}
    for ax, xvals, xlabel, _, classical, flow, gt, title in panels:
        for metric_idx, metric in enumerate(metric_names):
            color = colors(metric_idx % 10)
            for estimator, matrices in (("classical", classical), ("flow_matching", flow)):
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


def write_summary(path: Path, *, args: argparse.Namespace, case_paths: dict[tuple[int, int], Path], cache_hits: dict[tuple[int, int], bool], outputs: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "bin/compare_mog5_pr_distance_sweeps.py",
        "config": {
            "n_list": [int(v) for v in args.n_list],
            "pr_dim": int(args.pr_dim),
            "pr_dim_list": [int(v) for v in args.pr_dim_list],
            "n_total": int(args.n_total),
            "seed": int(args.seed),
            "device": str(args.device),
            "case_output_name": str(args.case_output_name),
            "force_comparison": bool(args.force_comparison),
            "visualization_only": bool(args.visualization_only),
            "yscale": str(args.yscale),
            "abs_error_yscale": str(args.yscale),
            "rel_error_yscale": "linear",
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

    case_paths: dict[tuple[int, int], Path] = {}
    cache_hits: dict[tuple[int, int], bool] = {}
    case_data: dict[tuple[int, int], dict[str, Any]] = {}
    for n_total, pr_dim in _unique_cases(args):
        path, cache_hit = ensure_case_results(args, n_total=n_total, pr_dim=pr_dim)
        case = (int(n_total), int(pr_dim))
        case_paths[case] = Path(path)
        cache_hits[case] = bool(cache_hit)
        case_data[case] = _load_case_cache(Path(path))

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
    print(f"summary_json: {summary_path}", flush=True)
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
