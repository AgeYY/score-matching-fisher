#!/usr/bin/env python3
"""Plot fresh A03T RBF20 Mahalanobis fits for two trial-count/seed settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VISIBLE_CUE_INTERVAL = (0.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_rbf20_seed40260715_n67_retrain",
    )
    parser.add_argument(
        "--right-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_rbf20_seed20460717_n136_retrain",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_rbf20_trial_seed_comparison_retrain",
    )
    return parser.parse_args()


def _load_run(run_dir: Path, expected_trials: int) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    result_path = run_dir / "a03t_reference_mahalanobis_rdms.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(result_path, allow_pickle=False) as archive:
        result = {
            "times": np.asarray(
                archive["time_seconds_cue_relative"], dtype=np.float64
            ),
            "classical": np.asarray(
                archive["classical_mean_distance"], dtype=np.float64
            ),
            "flow": np.asarray(archive["flow_mean_distance"], dtype=np.float64),
            "train_losses": np.asarray(archive["train_losses"], dtype=np.float64),
            "validation_losses": np.asarray(
                archive["validation_losses"], dtype=np.float64
            ),
            "monitored_validation_losses": np.asarray(
                archive["monitored_validation_losses"], dtype=np.float64
            ),
        }
    if int(summary["n_reference_trials"]) != expected_trials:
        raise ValueError(
            f"Expected {expected_trials} trials in {summary_path}, got "
            f"{summary['n_reference_trials']}."
        )
    if int(summary["time_rbf_num_centers"]) != 20:
        raise ValueError(f"Expected RBF20 in {summary_path}.")
    for name, values in result.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} values in {result_path}.")
    result["summary"] = summary
    return result


def main() -> None:
    args = parse_args()
    left = _load_run(args.left_dir, expected_trials=67)
    right = _load_run(args.right_dir, expected_trials=136)
    np.testing.assert_allclose(left["times"], right["times"], atol=0.0, rtol=0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(left["times"], dtype=np.float64)
    common_upper = 1.04 * max(
        float(np.max(left["classical"])),
        float(np.max(left["flow"])),
        float(np.max(right["classical"])),
        float(np.max(right["flow"])),
    )

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    panels = (
        (left, "67 trials — seed 40260715"),
        (right, "136 trials — seed 20460717"),
    )
    for axis, (run, title) in zip(axes, panels, strict=True):
        axis.axvspan(*VISIBLE_CUE_INTERVAL, color="0.93", linewidth=0.0, zorder=0)
        axis.axvline(0.0, color="0.35", linestyle=":", linewidth=1.5, zorder=1)
        axis.plot(
            times,
            run["classical"],
            color="#4477AA",
            linewidth=1.5,
            label="Classical",
            zorder=2,
        )
        axis.plot(
            times,
            run["flow"],
            color="#CC6677",
            linewidth=2.0,
            label="Flow: RBF20",
            zorder=3,
        )
        axis.set_title(title)
        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.set_ylim(0.0, common_upper)
        axis.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)

    figure.supxlabel("Time from cue onset (s)")
    figure.supylabel(r"Mean Mahalanobis$^2$ distance")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="outside upper center",
        ncol=2,
    )
    stem = "a03t_rbf20_n67_seed40260715_vs_n136_seed20460717"
    figure.savefig(args.output_dir / f"{stem}.png", dpi=300)
    figure.savefig(args.output_dir / f"{stem}.svg")
    plt.close(figure)

    loss_figure, loss_axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 7.0),
        sharex="col",
        sharey="row",
        layout="constrained",
    )
    loss_colors = {
        "train_losses": "#4477AA",
        "validation_losses": "#CC6677",
        "monitored_validation_losses": "#228833",
    }
    loss_labels = {
        "train_losses": "Training",
        "validation_losses": "Validation",
        "monitored_validation_losses": "EMA validation",
    }
    loss_panels = (
        (left, "67 trials — seed 40260715"),
        (right, "136 trials — seed 20460717"),
    )
    all_losses = np.concatenate(
        [
            np.asarray(run[key], dtype=np.float64)
            for run, _ in loss_panels
            for key in loss_colors
        ]
    )
    full_lower = float(np.min(all_losses)) - 0.02
    full_upper = float(np.max(all_losses)) + 0.03
    zoom_lower = max(0.0, float(np.min(all_losses)) - 0.005)
    zoom_upper = 0.66
    for column, (run, title) in enumerate(loss_panels):
        summary = run["summary"]
        best_epoch = int(summary["best_epoch"])
        stopped_epoch = int(summary["stopped_epoch"])
        for row in range(2):
            axis = loss_axes[row, column]
            for key, color in loss_colors.items():
                losses = np.asarray(run[key], dtype=np.float64)
                epochs = np.arange(1, losses.size + 1, dtype=np.int64)
                axis.plot(
                    epochs,
                    losses,
                    color=color,
                    linewidth=1.1 if key != "monitored_validation_losses" else 1.8,
                    alpha=0.55 if key != "monitored_validation_losses" else 1.0,
                    label=loss_labels[key],
                )
            axis.axvline(
                best_epoch,
                color="0.15",
                linestyle=":",
                linewidth=1.6,
                label="Selected epoch",
            )
            axis.axvline(
                stopped_epoch,
                color="0.45",
                linestyle="--",
                linewidth=1.6,
                label="Early stop",
            )
            axis.set_xlim(1, stopped_epoch)
            axis.set_ylim(
                (full_lower, full_upper) if row == 0 else (zoom_lower, zoom_upper)
            )
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(1.8)
            axis.spines["bottom"].set_linewidth(1.8)
            axis.tick_params(width=1.8)
        loss_axes[0, column].set_title(title)
        loss_axes[1, column].text(
            0.03,
            0.92,
            "Plateau zoom",
            transform=loss_axes[1, column].transAxes,
            ha="left",
            va="top",
            fontsize=14,
        )

    loss_figure.supxlabel("Epoch")
    loss_figure.supylabel("Flow-matching loss")
    loss_handles, loss_legend_labels = loss_axes[0, 0].get_legend_handles_labels()
    loss_figure.legend(
        loss_handles,
        loss_legend_labels,
        frameon=False,
        loc="outside upper center",
        ncol=5,
        fontsize=13,
    )
    loss_stem = f"{stem}_loss_vs_epoch"
    loss_figure.savefig(args.output_dir / f"{loss_stem}.png", dpi=300)
    loss_figure.savefig(args.output_dir / f"{loss_stem}.svg")
    plt.close(loss_figure)

    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        time_seconds_cue_relative=times,
        n67_classical=np.asarray(left["classical"], dtype=np.float64),
        n67_flow_rbf20=np.asarray(left["flow"], dtype=np.float64),
        n136_classical=np.asarray(right["classical"], dtype=np.float64),
        n136_flow_rbf20=np.asarray(right["flow"], dtype=np.float64),
    )
    comparison = {
        "left": {
            "n_trials": 67,
            "seed": 40260715,
            "best_epoch": int(left["summary"]["best_epoch"]),
            "stopped_epoch": int(left["summary"]["stopped_epoch"]),
            "best_validation_loss": float(
                left["summary"]["best_validation_loss"]
            ),
            "maximum_flow_distance": float(np.max(left["flow"])),
        },
        "right": {
            "n_trials": 136,
            "seed": 20460717,
            "best_epoch": int(right["summary"]["best_epoch"]),
            "stopped_epoch": int(right["summary"]["stopped_epoch"]),
            "best_validation_loss": float(
                right["summary"]["best_validation_loss"]
            ),
            "maximum_flow_distance": float(np.max(right["flow"])),
        },
        "posthoc_curve_smoothing": False,
        "time_rbf_num_centers": 20,
    }
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    print(f"[comparison] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
