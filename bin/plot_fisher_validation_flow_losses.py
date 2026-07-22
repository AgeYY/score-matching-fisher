#!/usr/bin/env python3
"""Plot Flow Matching loss histories for the Fisher-validation pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR

DATASETS = (
    ("Gaussian toy", Path("toy/randamp_gaussian_sqrtd")),
    ("Gaussian-mixture toy", Path("toy/cosine_gmm")),
    ("Stringer", Path("stringer/session0")),
)
PHASES = (
    ("Baseline", "baseline"),
    ("Control", "probe_control"),
    (r"Probe $J_{\max}=0.25$", "probe_peak_0.25_phase_0"),
    (r"Probe $J_{\max}=1$", "probe_peak_1_phase_0"),
    (r"Probe $J_{\max}=4$", "probe_peak_4_phase_0"),
)
SEED_STYLES = {7: "-", 19: "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir", type=Path, default=Path(DATA_DIR) / "fisher_validation_pilot"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(DATA_DIR) / "fisher_validation_pilot" / "figures"
    )
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    return parser.parse_args()


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for index in range(1, values.size):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(args.ema_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1].")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(3, 5, figsize=(20.0, 10.5), constrained_layout=True)
    selected_records: list[dict[str, object]] = []
    for row, (dataset_label, relative_dir) in enumerate(DATASETS):
        for column, (phase_label, phase_dir) in enumerate(PHASES):
            axis = axes[row, column]
            for seed, linestyle in SEED_STYLES.items():
                fit_dir = input_dir / relative_dir / f"seed{seed}" / phase_dir
                with np.load(fit_dir / "estimates.npz") as result:
                    train = np.asarray(result["flow_train_loss"], dtype=np.float64)
                    validation = np.asarray(result["flow_validation_loss"], dtype=np.float64)
                metadata = json.loads((fit_dir / "metadata.json").read_text(encoding="utf-8"))
                flow_metadata = metadata.get("flow_training", metadata)
                selected_epoch = int(
                    flow_metadata["selected_epoch"]
                    if "selected_epoch" in flow_metadata
                    else flow_metadata["flow_selected_epoch"]
                )
                validation_ema = _ema(validation, alpha)
                epoch = np.arange(1, train.size + 1)
                axis.plot(epoch, train, color="0.55", linewidth=0.7, alpha=0.28, linestyle=linestyle)
                axis.plot(epoch, validation_ema, color="C0", linewidth=2.0, linestyle=linestyle)
                selected_value = validation_ema[selected_epoch - 1]
                axis.scatter(
                    [selected_epoch],
                    [selected_value],
                    color="C3",
                    edgecolor="white",
                    linewidth=0.5,
                    s=30,
                    zorder=4,
                )
                selected_records.append(
                    {
                        "dataset": dataset_label,
                        "phase": phase_label,
                        "seed": seed,
                        "selected_epoch": selected_epoch,
                        "stopped_epoch": int(
                            flow_metadata.get("stopped_epoch", flow_metadata.get("flow_stopped_epoch"))
                        ),
                        "selected_validation_ema": float(selected_value),
                    }
                )
            if row == 0:
                axis.set_title(phase_label)
            if column == 0:
                axis.set_ylabel(f"{dataset_label}\nFM loss")
            if row == len(DATASETS) - 1:
                axis.set_xlabel("Epoch")
            _style_axis(axis)

    legend = [
        Line2D([0], [0], color="0.55", linewidth=1.2, label="Training loss (raw)"),
        Line2D([0], [0], color="C0", linewidth=2.0, label="Validation loss (EMA)"),
        Line2D([0], [0], color="C3", marker="o", linestyle="none", markersize=6, label="Selected checkpoint"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="-", label="Seed 7"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label="Seed 19"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
    png = output_dir / "flow_training_validation_losses.png"
    svg = output_dir / "flow_training_validation_losses.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    metadata_path = output_dir / "flow_training_validation_losses.json"
    metadata_path.write_text(json.dumps(selected_records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    print(f"Saved: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
