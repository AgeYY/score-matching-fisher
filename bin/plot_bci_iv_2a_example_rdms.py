#!/usr/bin/env python3
"""Audit and plot example BCI IV-2a RDMs from the production caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import CLASS_NAMES, load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    QUERY_RUNS,
    REFERENCE_RUNS,
    classical_mahalanobis_rdms,
    load_rdm_cache,
    select_half,
)


METHODS = ("classical_mahalanobis", "time_varying_shared_affine_flow")
METHOD_LABELS = {
    "classical_mahalanobis": "Classical Mahalanobis",
    "time_varying_shared_affine_flow": r"Flow with $\Sigma(u)$",
}
CLASS_ABBREVIATIONS = ("LH", "RH", "Feet", "Tongue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/log_bandpower",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/session_identification_time_varying_covariance",
    )
    parser.add_argument("--recording", default="A01T")
    parser.add_argument("--n-label", default="12")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--times", type=float, nargs="+", default=(-1.0, 0.0, 2.0, 3.5))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-loss-curves", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _cache_paths(
    cache_dir: Path,
    *,
    recording: str,
    n_label: str,
    repeat: int,
    method: str,
) -> tuple[Path, Path]:
    reference = cache_dir / f"reference_{recording}_{method}.npz"
    query = cache_dir / f"query_{recording}_n{n_label}_rep{repeat:02d}_{method}.npz"
    return reference, query


def _validate_rdms(rdms: np.ndarray, *, name: str) -> None:
    values = np.asarray(rdms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"{name} has invalid shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.min(values) < -1e-10:
        raise ValueError(f"{name} contains negative distances.")
    np.testing.assert_allclose(values, values.transpose(0, 2, 1), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        np.diagonal(values, axis1=1, axis2=2),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )


def _rdm_correlation(reference: np.ndarray, query: np.ndarray) -> float:
    upper = np.triu_indices(len(CLASS_NAMES), k=1)
    return float(np.corrcoef(reference[upper], query[upper])[0, 1])


def _annotate_upper_triangle(ax: plt.Axes, matrix: np.ndarray, *, vmax: float) -> None:
    threshold = 0.58 * float(vmax)
    for row in range(matrix.shape[0]):
        for col in range(row + 1, matrix.shape[1]):
            value = float(matrix[row, col])
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value >= threshold else "black",
            )


def _plot(
    *,
    selected_rdms: dict[tuple[str, str], np.ndarray],
    correlations: dict[str, list[float]],
    times: np.ndarray,
    output_dir: Path,
    stem: str,
    loss_history: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    loss_axes: dict[str, plt.Axes] = {}
    if loss_history is None:
        fig, axes = plt.subplots(
            4,
            len(times),
            figsize=(14.0, 11.0),
            constrained_layout=True,
        )
        if len(times) == 1:
            axes = np.asarray(axes).reshape(4, 1)
    else:
        fig = plt.figure(figsize=(18.0, 11.0), constrained_layout=True)
        grid = fig.add_gridspec(
            4,
            len(times) + 1,
            width_ratios=[1.0] * len(times) + [1.35],
        )
        axes = np.empty((4, len(times)), dtype=object)
        for row in range(4):
            for col in range(len(times)):
                axes[row, col] = fig.add_subplot(grid[row, col])
        loss_axes["reference"] = fig.add_subplot(grid[0:2, -1])
        loss_axes["subsample"] = fig.add_subplot(grid[2:4, -1])
    row_specs = (
        (METHODS[0], "reference"),
        (METHODS[0], "subsample"),
        (METHODS[1], "reference"),
        (METHODS[1], "subsample"),
    )
    images: dict[str, Any] = {}
    method_limits: dict[str, float] = {}
    for method in METHODS:
        method_limits[method] = max(
            float(np.max(selected_rdms[(method, split)]))
            for split in ("reference", "subsample")
        )

    for row, (method, split) in enumerate(row_specs):
        vmax = method_limits[method]
        for col, time in enumerate(times):
            ax = axes[row, col]
            matrix = selected_rdms[(method, split)][col]
            image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax)
            images[method] = image
            _annotate_upper_triangle(ax, matrix, vmax=vmax)
            ax.set_xticks(np.arange(len(CLASS_NAMES)))
            ax.set_yticks(np.arange(len(CLASS_NAMES)))
            if row == len(row_specs) - 1:
                ax.set_xticklabels(CLASS_ABBREVIATIONS, rotation=35, ha="right")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels(CLASS_ABBREVIATIONS)
                ax.set_ylabel(f"{METHOD_LABELS[method]}\n{split}")
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"EEG time {time:+.1f} s")
            if split == "subsample":
                ax.text(
                    0.5,
                    1.02,
                    f"split RDM $r={correlations[method][col]:.2f}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
            ax.tick_params(width=1.8, length=4)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)

    classical_bar = fig.colorbar(
        images[METHODS[0]],
        ax=axes[0:2, :].ravel().tolist(),
        fraction=0.018,
        pad=0.015,
    )
    classical_bar.set_label("Classical RDM distance")
    flow_bar = fig.colorbar(
        images[METHODS[1]],
        ax=axes[2:4, :].ravel().tolist(),
        fraction=0.018,
        pad=0.015,
    )
    flow_bar.set_label(r"Flow RDM distance")

    if loss_history is not None:
        colors = {
            "train_losses": "#4C78A8",
            "validation_losses": "#E45756",
            "monitored_validation_losses": "#54A24B",
        }
        labels = {
            "train_losses": "train",
            "validation_losses": "validation",
            "monitored_validation_losses": "validation EMA",
        }
        for split, ax in loss_axes.items():
            history = loss_history[split]
            for key in ("train_losses", "validation_losses", "monitored_validation_losses"):
                values = np.asarray(history[key], dtype=np.float64)
                epochs = np.arange(1, values.size + 1)
                ax.plot(epochs, values, color=colors[key], linewidth=2.0, label=labels[key])
            best_epoch = int(history["best_epoch"])
            ax.axvline(
                best_epoch,
                color="0.25",
                linewidth=1.8,
                linestyle="--",
                label=f"selected epoch {best_epoch}",
            )
            ax.set_title(f"Flow loss: {split}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Flow-matching MSE")
            ax.tick_params(width=1.8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
            ax.legend(frameon=False, fontsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    feature_path = args.feature_dir / f"{args.recording}.npz"
    features = load_features_npz(feature_path)
    cache_dir = args.run_dir / "rdm_cache"
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.run_dir / "diagnostics/example_rdms"
    )

    times = np.asarray(args.times, dtype=np.float64)
    time_indices = []
    for time in times:
        matches = np.flatnonzero(np.isclose(features.time_centers, time, atol=1e-12, rtol=0.0))
        if matches.size != 1:
            raise ValueError(f"Requested time {time} is not a unique feature-window center.")
        time_indices.append(int(matches[0]))
    time_indices_array = np.asarray(time_indices, dtype=np.int64)

    all_rdms: dict[tuple[str, str], np.ndarray] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    cache_paths: dict[tuple[str, str], Path] = {}
    for method in METHODS:
        reference_path, query_path = _cache_paths(
            cache_dir,
            recording=args.recording,
            n_label=args.n_label,
            repeat=args.repeat,
            method=method,
        )
        for split, path in (("reference", reference_path), ("subsample", query_path)):
            rdms, meta = load_rdm_cache(path)
            _validate_rdms(rdms, name=f"{method} {split}")
            all_rdms[(method, split)] = rdms
            metadata[(method, split)] = meta
            cache_paths[(method, split)] = path

    classical_indices = metadata[(METHODS[0], "subsample")]["selected_trial_indices"]
    flow_indices = metadata[(METHODS[1], "subsample")]["selected_trial_indices"]
    if classical_indices != flow_indices:
        raise AssertionError("Classical and flow query caches do not use the same trial indices.")

    x_reference, y_reference, _ = select_half(features, REFERENCE_RUNS)
    recomputed_reference = classical_mahalanobis_rdms(x_reference, y_reference)
    x_query, y_query, _ = select_half(features, QUERY_RUNS)
    selected = np.asarray(classical_indices, dtype=np.int64)
    recomputed_query = classical_mahalanobis_rdms(x_query[selected], y_query[selected])
    reference_error = float(
        np.max(np.abs(recomputed_reference - all_rdms[(METHODS[0], "reference")]))
    )
    query_error = float(
        np.max(np.abs(recomputed_query - all_rdms[(METHODS[0], "subsample")]))
    )
    np.testing.assert_allclose(
        recomputed_reference,
        all_rdms[(METHODS[0], "reference")],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        recomputed_query,
        all_rdms[(METHODS[0], "subsample")],
        atol=1e-12,
        rtol=1e-12,
    )

    loss_history: dict[str, dict[str, Any]] | None = None
    if args.include_loss_curves:
        loss_history = {}
        for split in ("reference", "subsample"):
            fit = metadata[(METHODS[1], split)]["fit"]
            required = (
                "train_losses",
                "validation_losses",
                "monitored_validation_losses",
            )
            if any(key not in fit for key in required):
                raise ValueError(
                    f"The {split} cache predates saved loss histories; rerun the flow fit."
                )
            loss_history[split] = {
                "best_epoch": int(fit["selected_epoch"]),
                "best_val_loss": float(fit["best_val_loss"]),
                "train_losses": fit["train_losses"],
                "validation_losses": fit["validation_losses"],
                "monitored_validation_losses": fit["monitored_validation_losses"],
            }

    selected_rdms = {
        key: values[time_indices_array]
        for key, values in all_rdms.items()
    }
    correlations = {
        method: [
            _rdm_correlation(
                selected_rdms[(method, "reference")][index],
                selected_rdms[(method, "subsample")][index],
            )
            for index in range(times.size)
        ]
        for method in METHODS
    }
    stem = f"{args.recording}_n{args.n_label}_rep{args.repeat:02d}_rdms_by_time"
    png, svg = _plot(
        selected_rdms=selected_rdms,
        correlations=correlations,
        times=times,
        output_dir=output_dir,
        stem=stem,
        loss_history=loss_history,
    )

    diagnostics = {
        "recording": args.recording,
        "query_n_label": args.n_label,
        "repeat": int(args.repeat),
        "time_centers_seconds_cue_relative": times.tolist(),
        "time_indices": time_indices_array.tolist(),
        "class_names": list(CLASS_NAMES),
        "query_selected_trial_indices": classical_indices,
        "query_selected_trial_count": len(classical_indices),
        "classical_cache_recomputation_max_abs_error": {
            "reference": reference_error,
            "subsample": query_error,
        },
        "flow_loss_history": loss_history,
        "split_rdm_correlations_by_method": correlations,
        "color_scale": "raw distances; shared across splits/times within method, separate between methods",
        "cache_paths": {
            f"{method}_{split}": str(path.resolve())
            for (method, split), path in cache_paths.items()
        },
        "rdms": {
            f"{method}_{split}": selected_rdms[(method, split)].tolist()
            for method in METHODS
            for split in ("reference", "subsample")
        },
        "figure_png": str(png.resolve()),
        "figure_svg": str(svg.resolve()),
    }
    diagnostics_path = output_dir / f"{stem}.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(f"[audit] classical reference max abs cache error: {reference_error:.3e}")
    print(f"[audit] classical subsample max abs cache error: {query_error:.3e}")
    print(f"[audit] shared query trial indices: {len(classical_indices)}")
    for method in METHODS:
        values = ", ".join(f"{value:.3f}" for value in correlations[method])
        print(f"[audit] {method} split RDM correlations: {values}")
    print(f"[audit] Saved figure: {png.resolve()}")
    print(f"[audit] Saved figure: {svg.resolve()}")
    print(f"[audit] Saved diagnostics: {diagnostics_path.resolve()}")


if __name__ == "__main__":
    main()
