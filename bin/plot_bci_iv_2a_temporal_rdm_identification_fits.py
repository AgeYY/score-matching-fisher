#!/usr/bin/env python3
"""Plot every fitted temporal RDM from a session-identification run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np


ROW_SPECS = (
    ("classical", "query", "Classical — query"),
    ("classical", "reference", "Classical — reference"),
    ("flow", "query", "Flow — query"),
    ("flow", "reference", "Flow — reference"),
)
BOUNDARIES_SECONDS = (0.0, 1.25)
LOSS_COLORS = {"query": "#2166ac", "reference": "#b2182b"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Completed temporal-RDM identification output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: input directory).",
    )
    return parser


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def _load_summary(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("recordings"):
        raise ValueError(f"No recordings listed in {path}")
    times = np.asarray(summary.get("evaluation_times_seconds"), dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError(f"Invalid evaluation_times_seconds in {path}")
    return summary


def _load_metric_rdms(
    input_dir: Path,
    recordings: list[str],
    metric: str,
) -> dict[tuple[str, str, str], np.ndarray]:
    rdms: dict[tuple[str, str, str], np.ndarray] = {}
    for method, split, _ in ROW_SPECS:
        for recording in recordings:
            path = input_dir / "rdm_cache" / f"{recording}_{split}_{method}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fitted temporal RDM: {path}")
            with np.load(path) as saved:
                key = f"{metric}_rdm"
                if key not in saved:
                    raise KeyError(f"{key} not found in {path}")
                matrix = np.asarray(saved[key], dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Expected a square RDM in {path}; got {matrix.shape}")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"Non-finite RDM values in {path}")
            if metric == "fid":
                # The cached FID convention is squared Gaussian W2. Plot W2 itself.
                matrix = np.sqrt(np.maximum(matrix, 0.0))
            rdms[(method, split, recording)] = matrix
    return rdms


def _strict_upper_values(rdms: dict[tuple[str, str, str], np.ndarray]) -> np.ndarray:
    values = []
    for matrix in rdms.values():
        values.append(matrix[np.triu_indices_from(matrix, k=1)])
    return np.concatenate(values)


def _panel_normalized(rdms: dict[tuple[str, str, str], np.ndarray]) -> dict[tuple[str, str, str], np.ndarray]:
    normalized: dict[tuple[str, str, str], np.ndarray] = {}
    for key, matrix in rdms.items():
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        scale = float(np.percentile(upper, 95.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid 95th-percentile scale for {key}: {scale}")
        normalized[key] = matrix / scale
    return normalized


def _load_loss_histories(
    summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    histories: dict[tuple[str, str], dict[str, Any]] = {}
    for recording in summary["recordings"]:
        for split in ("query", "reference"):
            training = summary["fit_summaries"][recording][split]["flow"]["flow_training"]
            train = np.asarray(training["train_losses"], dtype=np.float64)
            validation = np.asarray(training["val_monitor_losses"], dtype=np.float64)
            if train.ndim != 1 or train.size == 0 or train.shape != validation.shape:
                raise ValueError(f"Invalid loss histories for {recording} {split}.")
            if not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)):
                raise ValueError(f"Non-finite loss histories for {recording} {split}.")
            best_epoch = int(training["best_epoch"])
            if not 1 <= best_epoch <= train.size:
                raise ValueError(f"Invalid best epoch for {recording} {split}: {best_epoch}")
            histories[(recording, split)] = {
                "train": train,
                "validation": validation,
                "best_epoch": best_epoch,
                "best_validation": float(validation[best_epoch - 1]),
            }
    return histories


def _plot_loss_axis(
    ax: plt.Axes,
    *,
    recording: str,
    histories: dict[tuple[str, str], dict[str, Any]],
    y_limits: tuple[float, float],
    show_legend: bool,
) -> None:
    for split in ("query", "reference"):
        history = histories[(recording, split)]
        epochs = np.arange(1, history["train"].size + 1)
        color = LOSS_COLORS[split]
        label = "Query" if split == "query" else "Reference"
        ax.plot(
            epochs,
            history["train"],
            color=color,
            alpha=0.28,
            linestyle="--",
            linewidth=1.0,
            label=f"{label} train",
        )
        ax.plot(
            epochs,
            history["validation"],
            color=color,
            linewidth=1.8,
            label=f"{label} validation EMA",
        )
        ax.scatter(
            history["best_epoch"],
            history["best_validation"],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=36,
            zorder=3,
        )
    query_best = histories[(recording, "query")]["best_epoch"]
    reference_best = histories[(recording, "reference")]["best_epoch"]
    ax.text(
        0.03,
        0.04,
        f"best epoch: Q {query_best}, R {reference_best}",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="bottom",
    )
    ax.set_xlim(1, max(history["train"].size for history in histories.values()))
    ax.set_ylim(*y_limits)
    ax.set_title(recording)
    ax.set_xlabel("Epoch")
    ax.tick_params(width=1.8, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    if show_legend:
        ax.legend(frameon=False, fontsize=10, loc="upper right", handlelength=1.5)


def _plot_loss_only(
    *,
    histories: dict[tuple[str, str], dict[str, Any]],
    recordings: list[str],
    output_stem: Path,
) -> tuple[Path, Path]:
    _set_plot_style()
    all_losses = np.concatenate(
        [np.concatenate((history["train"], history["validation"])) for history in histories.values()]
    )
    y_limits = (float(np.min(all_losses) - 0.04), float(np.max(all_losses) + 0.04))
    fig, axes = plt.subplots(
        1,
        len(recordings),
        figsize=(4.0 * len(recordings), 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for col, recording in enumerate(recordings):
        ax = axes[0, col]
        _plot_loss_axis(
            ax,
            recording=recording,
            histories=histories,
            y_limits=y_limits,
            show_legend=col == 0,
        )
        if col == 0:
            ax.set_ylabel("Flow-matching loss")
    fig.suptitle("Flow training histories — shuffled query/reference halves", fontsize=20)
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _plot_loss_and_rdm_grid(
    *,
    histories: dict[tuple[str, str], dict[str, Any]],
    rdms: dict[tuple[str, str, str], np.ndarray],
    recordings: list[str],
    times: np.ndarray,
    title: str,
    colorbar_label: str,
    norm: Normalize,
    cmap: str,
    output_stem: Path,
    colorbar_extend: str = "neither",
) -> tuple[Path, Path]:
    _set_plot_style()
    ncols = len(recordings)
    nrows = 1 + len(ROW_SPECS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.5 * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    all_losses = np.concatenate(
        [np.concatenate((history["train"], history["validation"])) for history in histories.values()]
    )
    y_limits = (float(np.min(all_losses) - 0.04), float(np.max(all_losses) + 0.04))
    for col, recording in enumerate(recordings):
        _plot_loss_axis(
            axes[0, col],
            recording=recording,
            histories=histories,
            y_limits=y_limits,
            show_legend=col == 0,
        )
        if col == 0:
            axes[0, col].set_ylabel("Flow-matching loss")

    step = float(np.median(np.diff(times)))
    extent = (
        float(times[0] - step / 2.0),
        float(times[-1] + step / 2.0),
        float(times[0] - step / 2.0),
        float(times[-1] + step / 2.0),
    )
    image = None
    for rdm_row, (method, split, row_label) in enumerate(ROW_SPECS, start=1):
        for col, recording in enumerate(recordings):
            ax = axes[rdm_row, col]
            image = ax.imshow(
                rdms[(method, split, recording)],
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            for boundary in BOUNDARIES_SECONDS:
                ax.axvline(boundary, color="white", linestyle="--", linewidth=1.25, alpha=0.95)
                ax.axhline(boundary, color="white", linestyle="--", linewidth=1.25, alpha=0.95)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nTime from cue (s)")
            if rdm_row == nrows - 1:
                ax.set_xlabel("Time from cue (s)")
            ax.set_xticks((-2.0, 0.0, 1.25, 4.0), ("−2", "0", "1.25", "4"))
            ax.set_yticks((-2.0, 0.0, 1.25, 4.0), ("−2", "0", "1.25", "4"))
            if col != 0:
                ax.set_yticklabels([])
            if rdm_row != nrows - 1:
                ax.set_xticklabels([])
            ax.tick_params(width=1.8, length=5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
    if image is None:
        raise RuntimeError("No temporal RDM panels were plotted.")
    colorbar = fig.colorbar(
        image,
        ax=axes[1:, :],
        shrink=0.82,
        pad=0.015,
        extend=colorbar_extend,
    )
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(width=1.8)
    colorbar.outline.set_linewidth(1.8)
    fig.suptitle(title, fontsize=20)
    fig.text(
        0.5,
        -0.005,
        "RDM dashed lines: cue onset (0 s) and cue disappearance (1.25 s); dots mark selected epochs",
        ha="center",
        va="top",
        fontsize=14,
    )
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _plot_grid(
    *,
    rdms: dict[tuple[str, str, str], np.ndarray],
    recordings: list[str],
    times: np.ndarray,
    title: str,
    colorbar_label: str,
    norm: Normalize,
    cmap: str,
    output_stem: Path,
    colorbar_extend: str = "neither",
) -> tuple[Path, Path]:
    _set_plot_style()
    nrows = len(ROW_SPECS)
    ncols = len(recordings)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    step = float(np.median(np.diff(times)))
    extent = (
        float(times[0] - step / 2.0),
        float(times[-1] + step / 2.0),
        float(times[0] - step / 2.0),
        float(times[-1] + step / 2.0),
    )
    image = None
    for row, (method, split, row_label) in enumerate(ROW_SPECS):
        for col, recording in enumerate(recordings):
            ax = axes[row, col]
            image = ax.imshow(
                rdms[(method, split, recording)],
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            for boundary in BOUNDARIES_SECONDS:
                ax.axvline(boundary, color="white", linestyle="--", linewidth=1.25, alpha=0.95)
                ax.axhline(boundary, color="white", linestyle="--", linewidth=1.25, alpha=0.95)
            if row == 0:
                ax.set_title(recording)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nTime from cue (s)")
            if row == nrows - 1:
                ax.set_xlabel("Time from cue (s)")
            ax.set_xticks((-2.0, 0.0, 1.25, 4.0), ("−2", "0", "1.25", "4"))
            ax.set_yticks((-2.0, 0.0, 1.25, 4.0), ("−2", "0", "1.25", "4"))
            ax.tick_params(width=1.8, length=5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
    if image is None:
        raise RuntimeError("No temporal RDM panels were plotted.")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015, extend=colorbar_extend)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(width=1.8)
    colorbar.outline.set_linewidth(1.8)
    fig.suptitle(title, fontsize=20)
    fig.text(
        0.5,
        -0.01,
        "White dashed lines: cue onset (0 s) and cue disappearance (1.25 s)",
        ha="center",
        va="top",
        fontsize=14,
    )
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(input_dir)
    recordings = [str(value) for value in summary["recordings"]]
    times = np.asarray(summary["evaluation_times_seconds"], dtype=np.float64)
    histories = _load_loss_histories(summary)

    generated = list(
        _plot_loss_only(
            histories=histories,
            recordings=recordings,
            output_stem=output_dir / "all_sessions_flow_loss_vs_epoch",
        )
    )

    cosine_rdms = _load_metric_rdms(input_dir, recordings, "cosine")
    generated.extend(
        _plot_grid(
            rdms=cosine_rdms,
            recordings=recordings,
            times=times,
            title="Cosine temporal RDMs — all fitted sessions",
            colorbar_label="Cosine distance",
            norm=Normalize(vmin=0.0, vmax=2.0),
            cmap="viridis",
            output_stem=output_dir / "all_sessions_fitted_cosine_temporal_rdms",
        )
    )
    generated.extend(
        _plot_loss_and_rdm_grid(
            histories=histories,
            rdms=cosine_rdms,
            recordings=recordings,
            times=times,
            title="Flow training histories and cosine temporal RDMs",
            colorbar_label="Cosine distance",
            norm=Normalize(vmin=0.0, vmax=2.0),
            cmap="viridis",
            output_stem=output_dir / "all_sessions_flow_loss_and_cosine_temporal_rdms",
        )
    )

    w2_rdms = _load_metric_rdms(input_dir, recordings, "fid")
    w2_values = _strict_upper_values(w2_rdms)
    w2_vmax = float(np.percentile(w2_values, 99.5))
    generated.extend(
        _plot_grid(
            rdms=w2_rdms,
            recordings=recordings,
            times=times,
            title=r"Gaussian $W_2$ temporal RDMs — shared absolute scale",
            colorbar_label=r"Gaussian $W_2$ distance",
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=w2_vmax),
            cmap="magma",
            output_stem=output_dir / "all_sessions_fitted_gaussian_w2_temporal_rdms_absolute",
            colorbar_extend="max",
        )
    )
    generated.extend(
        _plot_loss_and_rdm_grid(
            histories=histories,
            rdms=w2_rdms,
            recordings=recordings,
            times=times,
            title=r"Flow training histories and Gaussian $W_2$ temporal RDMs",
            colorbar_label=r"Gaussian $W_2$ distance",
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=w2_vmax),
            cmap="magma",
            output_stem=output_dir / "all_sessions_flow_loss_and_gaussian_w2_temporal_rdms",
            colorbar_extend="max",
        )
    )

    generated.extend(
        _plot_grid(
            rdms=_panel_normalized(w2_rdms),
            recordings=recordings,
            times=times,
            title=r"Gaussian $W_2$ temporal RDM patterns — panel-normalized",
            colorbar_label=r"Gaussian $W_2$ / panel 95th percentile",
            norm=Normalize(vmin=0.0, vmax=1.5),
            cmap="magma",
            output_stem=output_dir / "all_sessions_fitted_gaussian_w2_temporal_rdms_pattern_normalized",
            colorbar_extend="max",
        )
    )
    for path in generated:
        print(f"[all-session temporal RDMs] Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
