#!/usr/bin/env python3
"""Plot ground-truth and fitted continuous-Fisher curves for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate_continuous_fisher_lw_repeats import fit_ledoit_wolf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-lw", action="store_true")
    parser.add_argument("--hide-classical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lw_path = output_dir / "classical_ledoit_wolf_fisher.npz"
    if args.reuse_lw and lw_path.is_file():
        with np.load(lw_path, allow_pickle=False) as saved:
            lw = {key: saved[key] for key in saved.files}
    else:
        lw = fit_ledoit_wolf(
            args.dataset_npz.resolve(), args.result_npz.resolve(), lw_path
        )

    with np.load(args.result_npz.resolve(), allow_pickle=False) as result:
        theta = np.asarray(result["theta_midpoints"], dtype=float).reshape(-1)
        curves = {
            "linear": {
                "Ground truth": np.asarray(
                    result["ground_truth_native_linear_fisher"], dtype=float
                ),
                "Classical + LW": np.asarray(
                    lw["classical_lw_linear_fisher"], dtype=float
                ),
                "Flow matching": np.asarray(result["flow_linear_fisher"], dtype=float),
                "Flow matching + NLL": np.asarray(
                    result["flow_linear_nll_fisher"], dtype=float
                ),
                "GKR": np.asarray(result["gkr_linear_fisher"], dtype=float),
            },
            "full": {
                "Ground truth": np.asarray(
                    result["ground_truth_native_full_fisher"], dtype=float
                ),
                "Classical + LW": np.asarray(
                    lw["classical_lw_full_fisher"], dtype=float
                ),
                "Flow matching": np.asarray(result["flow_full_fisher"], dtype=float),
                "Flow matching + NLL": np.asarray(
                    result["flow_full_nll_fisher"], dtype=float
                ),
                "GKR": np.asarray(result["gkr_full_fisher"], dtype=float),
            },
        }

    styles = {
        "Ground truth": {"color": "black", "linestyle": "--", "marker": None},
        "Classical + LW": {"color": "C1", "linestyle": "-", "marker": "s"},
        "Flow matching": {"color": "C0", "linestyle": "-", "marker": "o"},
        "Flow matching + NLL": {
            "color": "C1",
            "linestyle": "--",
            "marker": None,
        },
        "GKR": {"color": "C2", "linestyle": "-", "marker": "^"},
    }
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    plotted_methods = [
        "Ground truth",
        "Flow matching",
        "Flow matching + NLL",
        "GKR",
    ]
    if not args.hide_classical:
        plotted_methods.insert(1, "Classical + LW")
    for axis, family, title in zip(
        axes[0], ("linear", "full"), ("Linear Fisher", "Full Fisher"), strict=True
    ):
        for method in plotted_methods:
            axis.plot(
                theta,
                curves[family][method],
                label=method,
                linewidth=2.2 if method == "Ground truth" else 1.8,
                markersize=5,
                markevery=5,
                **styles[method],
            )
        axis.set_xlabel(r"$\theta$")
        axis.set_ylabel("Fisher information")
        axis.set_title(title)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)

    flow_dir = output_dir / "flow"
    loss_archives = {
        "Linear": flow_dir / "flow_linear_nll_flow_matching_skl_results.npz",
        "Full": flow_dir / "flow_full_nll_flow_matching_skl_results.npz",
    }
    losses: dict[str, dict[str, np.ndarray]] = {}
    for family, path in loss_archives.items():
        with np.load(path, allow_pickle=False) as saved:
            losses[family] = {
                "fm_train": np.asarray(saved["train_losses"], dtype=float),
                "fm_val": np.asarray(saved["val_losses"], dtype=float),
                "nll_train": np.asarray(saved["nll_train_losses"], dtype=float),
                "nll_val": np.asarray(saved["nll_val_losses"], dtype=float),
            }

    for family, color in (("Linear", "C0"), ("Full", "C1")):
        fm_epochs = np.arange(1, losses[family]["fm_train"].size + 1)
        axes[1, 0].plot(
            fm_epochs,
            losses[family]["fm_train"],
            color=color,
            linewidth=1.8,
            label=f"{family} train",
        )
        axes[1, 0].plot(
            fm_epochs,
            losses[family]["fm_val"],
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=f"{family} validation",
        )
        nll_epochs = np.arange(1, losses[family]["nll_train"].size + 1)
        axes[1, 1].plot(
            nll_epochs,
            losses[family]["nll_train"],
            color=color,
            linewidth=1.8,
            label=f"{family} train",
        )
        axes[1, 1].plot(
            nll_epochs,
            losses[family]["nll_val"],
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=f"{family} validation",
        )
    for axis, title, ylabel in (
        (axes[1, 0], "Flow-matching optimization", "FM loss"),
        (axes[1, 1], "NLL fine-tuning", "Negative log likelihood"),
    ):
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=9, ncol=2)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=1.8, w_pad=2.0)
    stem = output_dir / "continuous_fisher_method_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    summary: dict[str, dict[str, float]] = {}
    for family in ("linear", "full"):
        truth = curves[family]["Ground truth"]
        summary[family] = {
            method: float(np.mean(np.abs(values - truth)))
            for method, values in curves[family].items()
            if method != "Ground truth"
        }
    summary_path = output_dir / "continuous_fisher_method_curves_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"figure_png: {stem.with_suffix('.png')}")
    print(f"figure_svg: {stem.with_suffix('.svg')}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
