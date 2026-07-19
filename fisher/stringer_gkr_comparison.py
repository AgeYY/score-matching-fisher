"""GKR extension for the Stringer half-session Fisher benchmark.

This module intentionally leaves the legacy classical/flow cache format alone.
It reconstructs the same deterministic half-session tasks, fits GKR curves in a
separate cache, and combines their identification matrices with an existing
classical/flow subsample result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher.continuous_fisher_comparison import (
    ContinuousFlowConfig,
    METHOD_CLASSICAL_LINEAR,
    METHOD_FLOW_LINEAR,
)
from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.stringer_dataset import StringerSessionInfo, load_stringer_session
from fisher.stringer_session_identification import (
    ASUBSAMPLE_TASK_ENDPOINT_FULL_A,
    ASUBSAMPLE_TASK_REFERENCE_B,
    ASUBSAMPLE_TASK_SUBSET_A,
    DIRECTION_A_TO_B,
    DISTANCES,
    DISTANCE_PRIMARY,
    ASubsampleCurveTask,
    curve_distance,
    fit_half_pca,
    json_ready,
    plan_a_subsample_curve_tasks,
    resolve_a_subsample_curve_task,
    split_train_validation,
    theta_midpoints,
)

METHOD_GKR_LINEAR = "gkr_linear"
COMPARISON_METHODS = (
    METHOD_CLASSICAL_LINEAR,
    METHOD_FLOW_LINEAR,
    METHOD_GKR_LINEAR,
)

RESULTS_NPZ_NAME = "stringer_gkr_linear_fisher_comparison_results.npz"
SUMMARY_JSON_NAME = "stringer_gkr_linear_fisher_comparison_summary.json"
TOPK_SVG_NAME = "stringer_gkr_linear_fisher_topk_comparison.svg"
TOPK_PNG_NAME = "stringer_gkr_linear_fisher_topk_comparison.png"


@dataclass(frozen=True)
class GKRCurve:
    task: ASubsampleCurveTask
    session_key: str
    half_label: str
    n_trials: int
    n_train: int
    theta_midpoints: np.ndarray
    fisher: np.ndarray
    mean_loss: np.ndarray
    covariance_loss: np.ndarray
    cache_path: Path


@dataclass(frozen=True)
class StringerGKRComparisonResult:
    session_keys: list[str]
    theta_grid: np.ndarray
    theta_midpoints: np.ndarray
    n_values: list[int]
    repeats: int
    endpoint_matrices: dict[str, dict[str, np.ndarray]]
    subset_matrices: dict[str, dict[str, np.ndarray]]
    summary: dict[str, Any]


def _signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gkr_cache_path(output_dir: Path, task: ASubsampleCurveTask) -> Path:
    return Path(output_dir) / "gkr_curve_cache" / f"{task.label}.npz"


def _load_gkr_cache(path: Path, *, signature: str, task: ASubsampleCurveTask) -> GKRCurve | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["signature"]).reshape(-1)[0]) != signature:
            return None
        return GKRCurve(
            task=task,
            session_key=str(np.asarray(data["session_key"]).reshape(-1)[0]),
            half_label=str(np.asarray(data["half_label"]).reshape(-1)[0]),
            n_trials=int(np.asarray(data["n_trials"]).reshape(-1)[0]),
            n_train=int(np.asarray(data["n_train"]).reshape(-1)[0]),
            theta_midpoints=np.asarray(data["theta_midpoints"], dtype=np.float64),
            fisher=np.asarray(data["gkr_linear_fisher"], dtype=np.float64),
            mean_loss=np.asarray(data["mean_loss"], dtype=np.float64),
            covariance_loss=np.asarray(data["covariance_loss"], dtype=np.float64),
            cache_path=path,
        )


def fit_gkr_curve_task(
    task: ASubsampleCurveTask,
    *,
    sessions: list[StringerSessionInfo],
    theta_grid: np.ndarray,
    period: float,
    pca_dim: int,
    pca_random_state: int,
    pca_whiten: bool,
    train_frac: float,
    seed: int,
    n_values: list[int],
    sampling: str,
    replace: bool,
    flow_config: ContinuousFlowConfig,
    gkr_config: GKRConfig,
    gkr_solve_jitter: float,
    device: torch.device,
    output_dir: Path,
    force: bool = False,
) -> GKRCurve:
    """Fit one GKR curve using the exact split and PCA seed of the baseline task."""

    resolved = resolve_a_subsample_curve_task(
        task,
        sessions=sessions,
        theta_grid=theta_grid,
        period=float(period),
        pca_dim=int(pca_dim),
        pca_random_state=int(pca_random_state),
        pca_whiten=bool(pca_whiten),
        train_frac=float(train_frac),
        seed=int(seed),
        flow_config=flow_config,
        output_dir=Path(output_dir) / "resolved_tasks",
        n_values=n_values,
        sampling=str(sampling),
        replace=bool(replace),
    )
    cache_path = _gkr_cache_path(output_dir, task)
    signature = _signature(
        {
            "task": task.to_json_dict(),
            "session_file": str(resolved.session_info.session_file),
            "half_indices": np.asarray(resolved.half_indices, dtype=np.int64),
            "theta_grid": np.asarray(theta_grid, dtype=np.float64),
            "period": float(period),
            "pca_dim": int(pca_dim),
            "pca_random_state": int(resolved.pca_random_state),
            "pca_whiten": bool(pca_whiten),
            "train_frac": float(train_frac),
            "split_seed": int(resolved.seed),
            "gkr_data_protocol": "matched_flow_training_split_v1",
            "gkr_config": asdict(gkr_config),
            "gkr_solve_jitter": float(gkr_solve_jitter),
        }
    )
    if not force:
        cached = _load_gkr_cache(cache_path, signature=signature, task=task)
        if cached is not None:
            print(f"[stringer-gkr] cache hit task={task.label}", flush=True)
            return cached

    session = load_stringer_session(resolved.session_info, orientation_period=float(period))
    indices = np.asarray(resolved.half_indices, dtype=np.int64)
    theta = np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1)[indices]
    responses = np.asarray(session.neural_responses)[indices]
    pca = fit_half_pca(
        responses,
        n_components=int(pca_dim),
        random_state=int(resolved.pca_random_state),
        whiten=bool(pca_whiten),
        session_key=task.session_key,
        half_label=resolved.half_label,
    )
    train_indices, _validation_indices = split_train_validation(
        int(indices.size),
        train_frac=float(train_frac),
        seed=int(resolved.seed),
    )
    print(
        f"[stringer-gkr] fitting task={task.label} n={indices.size} "
        f"n_train={train_indices.size} pca_dim={pca_dim}",
        flush=True,
    )
    model = TorchGKR(
        n_input=1,
        n_output=int(pca_dim),
        circular_period=float(period),
        config=gkr_config,
        dtype=torch.float64,
        device=device,
        seed=int(resolved.seed),
    )
    model.fit(pca.x_all[train_indices], theta[train_indices].reshape(-1, 1))
    mids = theta_midpoints(theta_grid)
    estimate = estimate_gkr_linear_fisher(
        model,
        mids,
        finite_difference_step=np.diff(np.asarray(theta_grid, dtype=np.float64), axis=0),
        solve_jitter=float(gkr_solve_jitter),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=np.asarray([signature]),
        session_key=np.asarray([task.session_key]),
        half_label=np.asarray([resolved.half_label]),
        n_trials=np.asarray([indices.size], dtype=np.int64),
        n_train=np.asarray([train_indices.size], dtype=np.int64),
        theta_midpoints=mids,
        gkr_linear_fisher=np.asarray(estimate.linear_fisher, dtype=np.float64),
        mean_loss=np.asarray(estimate.mean_loss, dtype=np.float64),
        covariance_loss=np.asarray(estimate.covariance_loss, dtype=np.float64),
    )
    return GKRCurve(
        task=task,
        session_key=task.session_key,
        half_label=resolved.half_label,
        n_trials=int(indices.size),
        n_train=int(train_indices.size),
        theta_midpoints=mids,
        fisher=np.asarray(estimate.linear_fisher, dtype=np.float64),
        mean_loss=np.asarray(estimate.mean_loss, dtype=np.float64),
        covariance_loss=np.asarray(estimate.covariance_loss, dtype=np.float64),
        cache_path=cache_path,
    )


def identification_matrix(
    query_curves: list[GKRCurve],
    reference_curves: list[GKRCurve],
    theta_mid: np.ndarray,
    *,
    distance: str,
    session_keys: list[str],
) -> np.ndarray:
    queries = {curve.session_key: curve for curve in query_curves}
    references = {curve.session_key: curve for curve in reference_curves}
    if set(queries) != set(session_keys) or set(references) != set(session_keys):
        raise ValueError("Need exactly one query and reference GKR curve per session.")
    return np.asarray(
        [
            [
                curve_distance(
                    queries[query_key].fisher,
                    references[reference_key].fisher,
                    theta_mid,
                    distance=distance,
                )
                for reference_key in session_keys
            ]
            for query_key in session_keys
        ],
        dtype=np.float64,
    )


def summarize_identification_matrix(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Identification matrix must be square.")
    ranks = []
    tie_counts = []
    for query_index, row in enumerate(matrix):
        order = np.argsort(row, kind="mergesort")
        ranks.append(int(np.flatnonzero(order == query_index)[0]) + 1)
        tie_counts.append(int(np.sum(np.isclose(row, row[order[0]], rtol=1e-12, atol=1e-12))))
    rank_array = np.asarray(ranks, dtype=np.int64)
    n = int(matrix.shape[0])
    return {
        "top1_accuracy": float(np.mean(rank_array == 1)),
        "top2_accuracy": float(np.mean(rank_array <= min(2, n))),
        "top3_accuracy": float(np.mean(rank_array <= min(3, n))),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
        "ranks": ranks,
        "tie_counts": tie_counts,
    }


def _convergence_summary(
    *,
    endpoint_matrices: dict[str, dict[str, np.ndarray]],
    subset_matrices: dict[str, dict[str, np.ndarray]],
    n_values: list[int],
    repeats: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for method in COMPARISON_METHODS:
        output[method] = {}
        for distance in DISTANCES:
            metrics = {
                name: np.empty((len(n_values), repeats), dtype=np.float64)
                for name in ("top1", "top2", "top3", "mrr")
            }
            for ni in range(len(n_values)):
                for repeat in range(repeats):
                    one = summarize_identification_matrix(subset_matrices[method][distance][ni, repeat])
                    metrics["top1"][ni, repeat] = one["top1_accuracy"]
                    metrics["top2"][ni, repeat] = one["top2_accuracy"]
                    metrics["top3"][ni, repeat] = one["top3_accuracy"]
                    metrics["mrr"][ni, repeat] = one["mean_reciprocal_rank"]
            output[method][distance] = {
                "n_values": list(map(int, n_values)),
                **{f"{name}_by_repeat": values.tolist() for name, values in metrics.items()},
                **{f"{name}_mean": np.mean(values, axis=1).tolist() for name, values in metrics.items()},
                "full_a": summarize_identification_matrix(endpoint_matrices[method][distance]),
            }
    return output


def load_baseline_arrays(
    baseline_dir: Path,
    *,
    n_values: list[int] | None,
    repeats: int | None,
    max_sessions: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_dir = Path(baseline_dir)
    summary_path = baseline_dir / "stringer_session_identification_a_subsample_convergence_summary.json"
    results_path = baseline_dir / "stringer_session_identification_a_subsample_convergence_results.npz"
    if not summary_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"Missing baseline summary/results in {baseline_dir}")
    summary = json.loads(summary_path.read_text())
    available_n = [int(value) for value in summary["n_values"]]
    selected_n = available_n if n_values is None else [int(value) for value in n_values]
    missing = sorted(set(selected_n) - set(available_n))
    if missing:
        raise ValueError(f"Requested n_values absent from baseline: {missing}")
    selected_repeats = int(summary["repeats"]) if repeats is None else int(repeats)
    if selected_repeats < 1 or selected_repeats > int(summary["repeats"]):
        raise ValueError("repeats must be between 1 and the baseline repeat count.")
    session_keys = [str(value) for value in summary["session_keys"]]
    if max_sessions is not None:
        session_keys = session_keys[: int(max_sessions)]
    if len(session_keys) < 2:
        raise ValueError("At least two sessions are required for identification.")
    n_indices = [available_n.index(value) for value in selected_n]
    session_count = len(session_keys)
    arrays: dict[str, Any] = {
        "session_keys": session_keys,
        "n_values": selected_n,
        "repeats": selected_repeats,
    }
    with np.load(results_path, allow_pickle=False) as data:
        arrays["theta_grid"] = np.asarray(data["theta_grid"], dtype=np.float64)
        arrays["theta_midpoints"] = np.asarray(data["theta_midpoints"], dtype=np.float64)
        arrays["endpoint"] = {method: {} for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)}
        arrays["subset"] = {method: {} for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)}
        for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR):
            for distance in DISTANCES:
                endpoint_key = f"endpoint_{method}_{distance}_{DIRECTION_A_TO_B}"
                subset_key = f"subset_{method}_{distance}_{DIRECTION_A_TO_B}"
                endpoint = np.asarray(data[endpoint_key], dtype=np.float64)
                arrays["endpoint"][method][distance] = endpoint[
                    :session_count, :session_count
                ]
                subset = np.asarray(data[subset_key], dtype=np.float64)
                arrays["subset"][method][distance] = subset[
                    n_indices, :selected_repeats, :session_count, :session_count
                ]
    return summary, arrays


def run_gkr_comparison(
    *,
    sessions: list[StringerSessionInfo],
    baseline_dir: Path,
    output_dir: Path,
    device: torch.device,
    gkr_config: GKRConfig,
    gkr_solve_jitter: float = 1e-6,
    n_values: list[int] | None = None,
    repeats: int | None = None,
    max_sessions: int | None = None,
    force: bool = False,
) -> StringerGKRComparisonResult:
    baseline_summary, baseline = load_baseline_arrays(
        baseline_dir,
        n_values=n_values,
        repeats=repeats,
        max_sessions=max_sessions,
    )
    session_by_key = {Path(info.session_file).stem: info for info in sessions}
    selected_sessions = [session_by_key[key] for key in baseline["session_keys"]]
    theta_grid = baseline["theta_grid"]
    mids = baseline["theta_midpoints"]
    selected_n = baseline["n_values"]
    selected_repeats = int(baseline["repeats"])
    flow_config = ContinuousFlowConfig(**baseline_summary["flow_config"])
    tasks = plan_a_subsample_curve_tasks(
        sessions=selected_sessions,
        n_values=selected_n,
        repeats=selected_repeats,
    )
    curves = [
        fit_gkr_curve_task(
            task,
            sessions=selected_sessions,
            theta_grid=theta_grid,
            period=float(baseline_summary["orientation_period"]),
            pca_dim=int(baseline_summary["pca_dim"]),
            pca_random_state=int(baseline_summary["pca_random_state"]),
            pca_whiten=bool(baseline_summary["pca_whiten"]),
            train_frac=float(baseline_summary["train_frac"]),
            seed=int(baseline_summary["seed"]),
            n_values=selected_n,
            sampling=str(baseline_summary["sampling"]),
            replace=bool(baseline_summary["subsample_replace"]),
            flow_config=flow_config,
            gkr_config=gkr_config,
            gkr_solve_jitter=float(gkr_solve_jitter),
            device=device,
            output_dir=output_dir,
            force=bool(force),
        )
        for task in tasks
    ]
    full_a = [curve for curve in curves if curve.task.kind == ASUBSAMPLE_TASK_ENDPOINT_FULL_A]
    reference_b = [curve for curve in curves if curve.task.kind == ASUBSAMPLE_TASK_REFERENCE_B]
    endpoint_matrices = {
        method: {distance: np.asarray(baseline["endpoint"][method][distance]) for distance in DISTANCES}
        for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)
    }
    endpoint_matrices[METHOD_GKR_LINEAR] = {
        distance: identification_matrix(
            full_a,
            reference_b,
            mids,
            distance=distance,
            session_keys=baseline["session_keys"],
        )
        for distance in DISTANCES
    }
    subset_matrices = {
        method: {distance: np.asarray(baseline["subset"][method][distance]) for distance in DISTANCES}
        for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)
    }
    subset_matrices[METHOD_GKR_LINEAR] = {
        distance: np.full(
            (len(selected_n), selected_repeats, len(selected_sessions), len(selected_sessions)),
            np.nan,
            dtype=np.float64,
        )
        for distance in DISTANCES
    }
    for ni, n_subset in enumerate(selected_n):
        for repeat_index in range(selected_repeats):
            queries = [
                curve
                for curve in curves
                if curve.task.kind == ASUBSAMPLE_TASK_SUBSET_A
                and curve.task.subset_n == n_subset
                and curve.task.repeat == repeat_index
            ]
            for distance in DISTANCES:
                subset_matrices[METHOD_GKR_LINEAR][distance][ni, repeat_index] = identification_matrix(
                    queries,
                    reference_b,
                    mids,
                    distance=distance,
                    session_keys=baseline["session_keys"],
                )
    convergence = _convergence_summary(
        endpoint_matrices=endpoint_matrices,
        subset_matrices=subset_matrices,
        n_values=selected_n,
        repeats=selected_repeats,
    )
    summary = {
        "experiment": "stringer_gkr_linear_fisher_session_identification",
        "methods": list(COMPARISON_METHODS),
        "distances": list(DISTANCES),
        "primary_distance": DISTANCE_PRIMARY,
        "session_keys": baseline["session_keys"],
        "n_values": selected_n,
        "repeats": selected_repeats,
        "baseline_dir": str(Path(baseline_dir).resolve()),
        "sampling": baseline_summary["sampling"],
        "subsample_replace": bool(baseline_summary["subsample_replace"]),
        "orientation_period": float(baseline_summary["orientation_period"]),
        "theta_grid_size": int(theta_grid.shape[0]),
        "pca_dim": int(baseline_summary["pca_dim"]),
        "pca_whiten": bool(baseline_summary["pca_whiten"]),
        "pca_fit_scope": "each A subset or B half independently",
        "gkr_training_fraction": float(baseline_summary["train_frac"]),
        "gkr_data_protocol": "same deterministic training indices as flow matching",
        "gkr_condition": "scalar orientation with periodic kernel",
        "gkr_finite_difference_step": np.diff(theta_grid, axis=0).reshape(-1).tolist(),
        "gkr_config": asdict(gkr_config),
        "gkr_solve_jitter": float(gkr_solve_jitter),
        "subsample_convergence": convergence,
    }
    return StringerGKRComparisonResult(
        session_keys=baseline["session_keys"],
        theta_grid=theta_grid,
        theta_midpoints=mids,
        n_values=selected_n,
        repeats=selected_repeats,
        endpoint_matrices=endpoint_matrices,
        subset_matrices=subset_matrices,
        summary=summary,
    )


def save_comparison(path: Path, result: StringerGKRComparisonResult) -> Path:
    fields: dict[str, Any] = {
        "session_keys": np.asarray(result.session_keys),
        "theta_grid": result.theta_grid,
        "theta_midpoints": result.theta_midpoints,
        "n_values": np.asarray(result.n_values, dtype=np.int64),
        "repeats": np.asarray([result.repeats], dtype=np.int64),
    }
    for method in COMPARISON_METHODS:
        for distance in DISTANCES:
            fields[f"endpoint_{method}_{distance}"] = result.endpoint_matrices[method][distance]
            fields[f"subset_{method}_{distance}"] = result.subset_matrices[method][distance]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)
    return path


def save_summary(path: Path, result: StringerGKRComparisonResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(result.summary), indent=2, sort_keys=True) + "\n")
    return path


def plot_topk_comparison(
    path_svg: Path,
    path_png: Path,
    result: StringerGKRComparisonResult,
) -> tuple[Path, Path]:
    method_labels = {
        METHOD_CLASSICAL_LINEAR: "Classical",
        METHOD_FLOW_LINEAR: "Flow matching",
        METHOD_GKR_LINEAR: "GKR",
    }
    method_colors = {
        METHOD_CLASSICAL_LINEAR: "C1",
        METHOD_FLOW_LINEAR: "C0",
        METHOD_GKR_LINEAR: "C2",
    }
    fig, axes = plt.subplots(2, len(DISTANCES), figsize=(12.0, 7.0), sharex=True, sharey=True, layout="constrained")
    x = np.asarray(result.n_values, dtype=np.float64)
    for row, (metric, ylabel) in enumerate((("top1", "Top-1 accuracy"), ("top3", "Top-3 accuracy"))):
        for col, distance in enumerate(DISTANCES):
            ax = axes[row, col]
            for method in COMPARISON_METHODS:
                values = np.asarray(
                    result.summary["subsample_convergence"][method][distance][
                        f"{metric}_by_repeat"
                    ],
                    dtype=np.float64,
                )
                mean = np.mean(values, axis=1)
                error = np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros_like(mean)
                ax.errorbar(
                    x,
                    mean,
                    yerr=error,
                    color=method_colors[method],
                    marker="o",
                    markersize=5.5,
                    linewidth=2.0,
                    capsize=3.0,
                    label=method_labels[method],
                )
            ax.axhline(
                (1.0 if metric == "top1" else min(3, len(result.session_keys))) / len(result.session_keys),
                color="0.35",
                linestyle=":",
                linewidth=1.4,
                label="Chance" if row == 0 and col == 0 else None,
            )
            if row == 0:
                ax.set_title(distance.replace("_", " "))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("A-half samples")
            ax.set_ylim(0.0, 1.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.8)
            ax.spines["bottom"].set_linewidth(1.8)
            ax.tick_params(labelsize=13, width=1.8)
            ax.grid(False)
    axes[0, 0].legend(frameon=False, fontsize=13, loc="lower right")
    for ax in axes.reshape(-1):
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(15)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_svg, path_png
