#!/usr/bin/env python3
"""Compare one- and eight-bridge TRE with cached MoG5 Jeffreys baselines."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-bridge-results",
        type=Path,
        default=Path(DATA_DIR) / "mog5_jeffreys_n2000_r2_tre_m1" / "mog5_pr_distance_sweep_results.npz",
    )
    parser.add_argument(
        "--eight-bridge-results",
        type=Path,
        default=Path(DATA_DIR) / "mog5_jeffreys_n2000_r2_tre" / "mog5_pr_distance_sweep_results.npz",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="Optional matched aggregate supplying classical, FM, FM+NLL, and CTSM-v.",
    )
    parser.add_argument(
        "--sixteen-bridge-results",
        type=Path,
        default=None,
        help="Optional matched aggregate containing TRE with 16 bridges.",
    )
    parser.add_argument("--exclude-classical", action="store_true")
    parser.add_argument("--exclude-ctsm-v", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_jeffreys_n2000_r2_tre_bridge_comparison",
    )
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _verify_shared(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
    exact_keys = (
        "metric_names",
        "condition_labels",
        "pair_indices",
        "n_list",
        "repeat_indices",
        "repeat_seeds",
        "n_repeat_ground_truth_matrices",
        "n_repeat_classical_matrices",
        "n_repeat_flow_matching_matrices",
    )
    optional_keys = ("n_repeat_flow_matching_nll_finetuned_matrices",)
    for key in exact_keys:
        if key not in reference or key not in candidate:
            raise KeyError(f"Required matched field {key!r} is missing.")
        np.testing.assert_array_equal(reference[key], candidate[key])
    for key in optional_keys:
        if (key in reference) != (key in candidate):
            raise ValueError(f"Baseline availability differs for {key!r}.")
        if key in reference:
            np.testing.assert_array_equal(reference[key], candidate[key])


def _verify_matched(one: dict[str, np.ndarray], eight: dict[str, np.ndarray]) -> None:
    _verify_shared(one, eight)
    if "n_repeat_tre_matrices" not in one or "n_repeat_tre_matrices" not in eight:
        raise KeyError("Both inputs must contain n_repeat_tre_matrices.")
    if tuple(str(value) for value in eight["metric_names"]) != ("symmetric_kl",):
        raise ValueError("This comparison expects a Jeffreys-only sweep.")
    if np.asarray(eight["n_list"], dtype=np.int64).size < 1:
        raise ValueError("The comparison requires at least one sample size.")


def _repeat_mean_relative_errors(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    pair_indices: np.ndarray,
) -> np.ndarray:
    est = np.asarray(estimate, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if est.shape != gt.shape or est.ndim != 5 or est.shape[2] != 1:
        raise ValueError(f"Expected matched [N, repeats, 1, K, K] arrays; got {est.shape} and {gt.shape}.")
    values = np.empty((est.shape[0], est.shape[1]), dtype=np.float64)
    for n_idx in range(est.shape[0]):
        for repeat_idx in range(est.shape[1]):
            pair_errors = []
            for i, j in np.asarray(pair_indices, dtype=np.int64):
                truth = float(gt[n_idx, repeat_idx, 0, int(i), int(j)])
                prediction = float(est[n_idx, repeat_idx, 0, int(i), int(j)])
                pair_errors.append(abs(prediction - truth) / max(abs(truth), 1e-12))
            values[n_idx, repeat_idx] = float(
                np.mean(np.asarray(pair_errors, dtype=np.float64))
            )
    return values


def _collect_methods(
    baseline: dict[str, np.ndarray],
    one: dict[str, np.ndarray],
    eight: dict[str, np.ndarray],
    sixteen: dict[str, np.ndarray] | None,
    *,
    include_classical: bool,
    include_ctsm_v: bool,
) -> list[tuple[str, str, np.ndarray]]:
    methods = []
    if include_classical:
        methods.append(("classical", "Classical", baseline["n_repeat_classical_matrices"]))
    methods.append(("flow_matching", "Flow matching", baseline["n_repeat_flow_matching_matrices"]))
    if "n_repeat_flow_matching_nll_finetuned_matrices" in baseline:
        methods.append(
            (
                "flow_matching_nll_finetuned",
                "Flow matching + NLL",
                baseline["n_repeat_flow_matching_nll_finetuned_matrices"],
            )
        )
    if include_ctsm_v and "n_repeat_ctsm_v_matrices" in baseline:
        methods.append(("ctsm_v", "CTSM-v", baseline["n_repeat_ctsm_v_matrices"]))
    methods.extend(
        [
            ("tre_1_bridge", "TRE, 1 bridge", one["n_repeat_tre_matrices"]),
            ("tre_8_bridges", "TRE, 8 bridges", eight["n_repeat_tre_matrices"]),
        ]
    )
    if sixteen is not None:
        methods.append(("tre_16_bridges", "TRE, 16 bridges", sixteen["n_repeat_tre_matrices"]))
    return methods


def _plot(
    method_errors: list[tuple[str, str, np.ndarray]],
    *,
    n_list: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    colors = [f"C{index}" for index in range(len(method_errors))]
    markers = ("o", "s", "^", "P", "X", "D", "v")
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    linestyles = ("-", "-", "--", ":", "-.", "-", "--")
    x_values = np.asarray(n_list, dtype=np.int64)
    for color, marker, linestyle, (_, label, values) in zip(
        colors,
        markers[: len(method_errors)],
        linestyles[: len(method_errors)],
        method_errors,
        strict=True,
    ):
        mean = np.mean(values, axis=1)
        error = (
            np.std(values, axis=1, ddof=1)
            if values.shape[1] > 1
            else np.zeros(values.shape[0], dtype=np.float64)
        )
        ax.errorbar(
            x_values,
            mean,
            yerr=error,
            color=color,
            marker=marker,
            markersize=7,
            linestyle=linestyle,
            linewidth=2.0,
            capsize=4,
            label=label,
        )
    ax.set_xticks(x_values)
    ax.set_xlabel("Total samples")
    ax.set_ylabel("Mean relative\nabsolute error")
    ax.legend(frameon=False, loc="best")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(width=1.8, length=5)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mog5_tre_bridge_comparison_rel_error.png"
    svg_path = output_dir / "mog5_tre_bridge_comparison_rel_error.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    one = _load(args.one_bridge_results)
    eight = _load(args.eight_bridge_results)
    _verify_matched(one, eight)
    baseline = eight if args.baseline_results is None else _load(args.baseline_results)
    _verify_shared(eight, baseline)
    sixteen = None if args.sixteen_bridge_results is None else _load(args.sixteen_bridge_results)
    if sixteen is not None:
        _verify_shared(eight, sixteen)
        if "n_repeat_tre_matrices" not in sixteen:
            raise KeyError("The 16-bridge input must contain n_repeat_tre_matrices.")
    ground_truth = baseline["n_repeat_ground_truth_matrices"]
    pair_indices = baseline["pair_indices"]
    n_list = np.asarray(baseline["n_list"], dtype=np.int64)
    method_errors = [
        (key, label, _repeat_mean_relative_errors(matrix, ground_truth, pair_indices))
        for key, label, matrix in _collect_methods(
            baseline,
            one,
            eight,
            sixteen,
            include_classical=not bool(args.exclude_classical),
            include_ctsm_v=not bool(args.exclude_ctsm_v),
        )
    ]

    output_dir = args.output_dir.expanduser().resolve()
    png_path, svg_path = _plot(method_errors, n_list=n_list, output_dir=output_dir)
    rows = [
        {
            "estimator": key,
            "n_total": int(n_list[n_idx]),
            "repeat_idx": repeat_idx,
            "mean_relative_absolute_error": float(value),
        }
        for key, _, values in method_errors
        for n_idx, repeat_values in enumerate(values)
        for repeat_idx, value in enumerate(repeat_values)
    ]
    csv_path = output_dir / "mog5_tre_bridge_comparison_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: {
            "label": label,
            "repeat_errors": values.tolist(),
            "mean_relative_absolute_error_by_n": np.mean(values, axis=1).tolist(),
            "std_relative_absolute_error_by_n": (
                np.std(values, axis=1, ddof=1).tolist()
                if values.shape[1] > 1
                else np.zeros(values.shape[0], dtype=np.float64).tolist()
            ),
        }
        for key, label, values in method_errors
    }
    summary["n_list"] = n_list.tolist()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved figure: {png_path}")
    print(f"Saved vector figure: {svg_path}")
    print(f"Saved per-repeat errors: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
