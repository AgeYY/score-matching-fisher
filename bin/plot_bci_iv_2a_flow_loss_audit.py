#!/usr/bin/env python3
"""Plot and summarize flow-training histories from a BCI IV-2a run cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOW_SUFFIX = "time_varying_shared_affine_flow.npz"


@dataclass(frozen=True)
class FitHistory:
    path: Path
    recording: str
    role: str
    n_label: str
    repeat: int | None
    best_epoch: int
    maximum_epochs: int
    train: np.ndarray
    validation: np.ndarray
    validation_ema: np.ndarray

    @property
    def stop_epoch(self) -> int:
        return int(self.train.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/session_identification_time_varying_covariance_20k_9recordings_r5",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_history(path: Path) -> FitHistory:
    with np.load(path, allow_pickle=True) as cache:
        metadata = json.loads(str(np.asarray(cache["metadata_json"]).reshape(-1)[0]))
    fit: dict[str, Any] = metadata["fit"]
    train = np.asarray(fit["train_losses"], dtype=np.float64)
    validation = np.asarray(fit["validation_losses"], dtype=np.float64)
    validation_ema = np.asarray(fit["monitored_validation_losses"], dtype=np.float64)
    if not (train.shape == validation.shape == validation_ema.shape):
        raise ValueError(f"Inconsistent loss-history shapes in {path}.")
    if train.ndim != 1 or train.size == 0:
        raise ValueError(f"Invalid loss history in {path}.")
    if not all(np.all(np.isfinite(values)) for values in (train, validation, validation_ema)):
        raise ValueError(f"Non-finite loss history in {path}.")
    return FitHistory(
        path=path,
        recording=str(metadata["recording"]),
        role=str(metadata["role"]),
        n_label=str(metadata.get("n_label", "reference")),
        repeat=None if metadata.get("repeat") is None else int(metadata["repeat"]),
        best_epoch=int(fit["best_epoch"]),
        maximum_epochs=int(fit["config"]["epochs"]),
        train=train,
        validation=validation,
        validation_ema=validation_ema,
    )


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(width=1.8)
    ax.grid(False)


def _plot_group(ax: plt.Axes, histories: list[FitHistory], *, title: str) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(histories)))
    for color, history in zip(colors, histories, strict=True):
        epochs = np.arange(1, history.stop_epoch + 1)
        ax.plot(
            epochs,
            history.validation_ema,
            color=color,
            linewidth=1.35,
            label=history.recording,
        )
        ax.scatter(
            history.best_epoch,
            history.validation_ema[history.best_epoch - 1],
            color=color,
            s=18,
            zorder=3,
        )
    ax.set_title(title)
    ax.set_ylabel("Validation EMA loss")
    ax.legend(frameon=False, fontsize=9, ncol=3, handlelength=1.2, columnspacing=0.8)
    _style_axis(ax)


def _make_plot(histories: list[FitHistory], output_dir: Path) -> tuple[Path, Path]:
    references = sorted(
        (history for history in histories if history.role == "reference"),
        key=lambda history: history.recording,
    )
    all_queries = sorted(
        (
            history
            for history in histories
            if history.role == "query" and history.n_label == "all"
        ),
        key=lambda history: history.recording,
    )
    finite_queries = [
        history
        for history in histories
        if history.role == "query" and history.n_label != "all"
    ]
    hardest = max(histories, key=lambda history: history.best_epoch)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)

    _plot_group(axes[0, 0], references, title="Reference fits (9)")
    _plot_group(axes[0, 1], all_queries, title="All-data query fits (9)")

    finite_ax = axes[1, 0]
    for history in finite_queries:
        epochs = np.arange(1, history.stop_epoch + 1)
        finite_ax.plot(epochs, history.validation_ema, color="0.55", alpha=0.12, linewidth=0.7)
    hardest_epochs = np.arange(1, hardest.stop_epoch + 1)
    finite_ax.plot(
        hardest_epochs,
        hardest.validation_ema,
        color="#b2182b",
        linewidth=1.8,
        label=f"{hardest.recording}, n={hardest.n_label}, repeat {hardest.repeat}",
    )
    finite_ax.scatter(
        hardest.best_epoch,
        hardest.validation_ema[hardest.best_epoch - 1],
        color="#b2182b",
        s=30,
        zorder=3,
    )
    finite_ax.set_title(f"Finite-query fits ({len(finite_queries)})")
    finite_ax.set_xlabel("Epoch")
    finite_ax.set_ylabel("Validation EMA loss")
    finite_ax.legend(frameon=False, fontsize=9, loc="best")
    _style_axis(finite_ax)

    detail_ax = axes[1, 1]
    detail_ax.plot(hardest_epochs, hardest.train, color="#2166ac", alpha=0.65, linewidth=0.8, label="Train")
    detail_ax.plot(
        hardest_epochs,
        hardest.validation,
        color="0.55",
        alpha=0.65,
        linewidth=0.8,
        label="Validation",
    )
    detail_ax.plot(
        hardest_epochs,
        hardest.validation_ema,
        color="#b2182b",
        linewidth=1.8,
        label="Validation EMA",
    )
    detail_ax.axvline(hardest.best_epoch, color="black", linestyle="--", linewidth=1.3)
    detail_ax.set_title(f"Longest selection: epoch {hardest.best_epoch}")
    detail_ax.set_xlabel("Epoch")
    detail_ax.set_ylabel("Flow-matching loss")
    detail_ax.legend(frameon=False, fontsize=10, loc="best")
    _style_axis(detail_ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "flow_loss_curve_audit.png"
    svg_path = output_dir / "flow_loss_curve_audit.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def _write_summary(histories: list[FitHistory], output_dir: Path) -> Path:
    best_epochs = np.asarray([history.best_epoch for history in histories], dtype=np.int64)
    stop_epochs = np.asarray([history.stop_epoch for history in histories], dtype=np.int64)
    relative_improvements = np.asarray(
        [
            (history.validation_ema[0] - history.validation_ema[history.best_epoch - 1])
            / history.validation_ema[0]
            for history in histories
        ],
        dtype=np.float64,
    )
    categories = ("reference", "4", "8", "12", "18", "24", "all")
    category_summaries: dict[str, Any] = {}
    for category in categories:
        selected = [history for history in histories if history.n_label == category]
        values = np.asarray([history.best_epoch for history in selected], dtype=np.int64)
        category_summaries[category] = {
            "n_fits": len(selected),
            "best_epoch_min": int(values.min()),
            "best_epoch_median": float(np.median(values)),
            "best_epoch_max": int(values.max()),
        }
    hardest = max(histories, key=lambda history: history.best_epoch)
    summary = {
        "n_flow_fits": len(histories),
        "n_epoch_cap_hits": int(
            sum(history.stop_epoch >= history.maximum_epochs for history in histories)
        ),
        "stop_minus_best_epoch_unique": sorted(
            {history.stop_epoch - history.best_epoch for history in histories}
        ),
        "best_epoch_min": int(best_epochs.min()),
        "best_epoch_median": float(np.median(best_epochs)),
        "best_epoch_max": int(best_epochs.max()),
        "stop_epoch_max": int(stop_epochs.max()),
        "relative_validation_ema_improvement": {
            "min": float(relative_improvements.min()),
            "median": float(np.median(relative_improvements)),
            "max": float(relative_improvements.max()),
        },
        "best_epoch_by_category": category_summaries,
        "longest_selected_fit": {
            "cache_file": hardest.path.name,
            "recording": hardest.recording,
            "role": hardest.role,
            "n_label": hardest.n_label,
            "repeat": hardest.repeat,
            "best_epoch": hardest.best_epoch,
            "stop_epoch": hardest.stop_epoch,
        },
    }
    path = output_dir / "flow_loss_curve_audit.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    cache_dir = args.run_dir / "rdm_cache"
    paths = sorted(cache_dir.glob(f"*{FLOW_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No flow cache files found in {cache_dir}.")
    histories = [_load_history(path) for path in paths]
    output_dir = args.run_dir if args.output_dir is None else args.output_dir
    png_path, svg_path = _make_plot(histories, output_dir)
    summary_path = _write_summary(histories, output_dir)
    print(f"[loss-audit] fits={len(histories)}")
    print(f"[loss-audit] PNG: {png_path}")
    print(f"[loss-audit] SVG: {svg_path}")
    print(f"[loss-audit] summary: {summary_path}")


if __name__ == "__main__":
    main()
