#!/usr/bin/env python3
"""Plot the seed-7 Jeffreys sweep after excluding one declared repeat."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    ("classical", "Classical", "#DD8452", "-"),
    ("flow_matching", "Flow matching", "#4C72B0", "-"),
    ("flow_matching_nll_finetuned", "Flow matching + NLL", "#55A868", "--"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/mog5_seed7_jeffreys_unified_n100-3000_r5/mog5_pr_distance_sweep_errors.csv"),
    )
    parser.add_argument("--exclude-n", type=int, default=3000)
    parser.add_argument("--exclude-seed", type=int, default=7)
    parser.add_argument("--error-kind", choices=("relative", "absolute"), default="relative")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "data/mog5_seed7_jeffreys_unified_n100-3000_r5/"
            "mog5_jeffreys_rel_error_without_n3000_seed7_outlier"
        ),
    )
    return parser


def summarize(csv_path: Path, *, exclude_n: int, exclude_seed: int, error_kind: str) -> pd.DataFrame:
    rows = pd.read_csv(csv_path)
    selected = rows[rows["metric"].eq("symmetric_kl")].copy()
    excluded = selected["n_total"].eq(int(exclude_n)) & selected["repeat_seed"].eq(int(exclude_seed))
    if not bool(excluded.any()):
        raise ValueError(f"No rows match excluded N={exclude_n}, seed={exclude_seed}.")
    selected = selected[~excluded]
    value_column = "rel_error" if error_kind == "relative" else "abs_error"
    per_repeat = selected.groupby(["n_total", "repeat_seed", "estimator"], as_index=False)[value_column].mean()
    return (
        per_repeat.groupby(["n_total", "estimator"], as_index=False)[value_column]
        .agg(mean_error="mean", std_error="std", n_repeats="size")
        .sort_values(["n_total", "estimator"])
    )


def plot(
    summary: pd.DataFrame,
    *,
    output_prefix: Path,
    exclude_n: int,
    exclude_seed: int,
    error_kind: str,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.5, 3.5))
    for key, label, color, linestyle in METHODS:
        values = summary[summary["estimator"].eq(key)].sort_values("n_total")
        axis.errorbar(
            values["n_total"],
            values["mean_error"],
            yerr=values["std_error"],
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=4,
            linewidth=1.8,
            elinewidth=1.2,
            capsize=3,
            label=label,
        )
    axis.set_xlabel("N data points")
    axis.set_ylabel("Mean relative error" if error_kind == "relative" else "Mean absolute error")
    axis.set_title(f"Jeffreys divergence\nN={exclude_n}, seed {exclude_seed} excluded")
    axis.set_xticks(np.asarray(sorted(summary["n_total"].unique()), dtype=np.int64))
    axis.legend(frameon=False, loc="upper right")
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(
        args.input_csv,
        exclude_n=args.exclude_n,
        exclude_seed=args.exclude_seed,
        error_kind=args.error_kind,
    )
    png_path, svg_path = plot(
        summary,
        output_prefix=args.output_prefix,
        exclude_n=args.exclude_n,
        exclude_seed=args.exclude_seed,
        error_kind=args.error_kind,
    )
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"summary_csv: {summary_path.resolve()}", flush=True)
    print(f"figure_png: {png_path.resolve()}", flush=True)
    print(f"figure_svg: {svg_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
