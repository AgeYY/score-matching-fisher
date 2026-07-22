#!/usr/bin/env python3
"""Aggregate held-out achieved information across Stringer sessions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Flow matching", "GKR", "OLE (cross-fit)")
LABELS = {"Flow matching": "FM", "GKR": "GKR", "OLE (cross-fit)": "OLE"}
COLORS = {"Flow matching": "C0", "GKR": "C2", "OLE (cross-fit)": "C1"}
MARKERS = {"Flow matching": "o", "GKR": "^", "OLE (cross-fit)": "s"}
SESSION_PATTERN = re.compile(r"session_(?P<index>\d+)_(?P<label>.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _session_identity(path: Path) -> tuple[int, str]:
    match = SESSION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected session directory name: {path.name}")
    return int(match.group("index")), match.group("label")


def _aggregate(input_dir: Path) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    session_dirs = sorted(
        (path for path in input_dir.glob("session_*_*") if path.is_dir()),
        key=lambda path: _session_identity(path)[0],
    )
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found under {input_dir}.")
    for session_dir in session_dirs:
        session_index, session_label = _session_identity(session_dir)
        summary_path = session_dir / "train-test-allocation_stringer.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = list(summary.get("rows", []))
        for method in METHODS:
            selected = [row for row in rows if row.get("method") == method]
            if not selected:
                raise ValueError(f"Missing {method} rows in {summary_path}.")
            values = np.asarray(
                [float(row["mean_achieved_fisher"]) for row in selected],
                dtype=np.float64,
            )
            test_fractions = {round(float(row["test_fraction"]), 12) for row in selected}
            if test_fractions != {0.5}:
                raise ValueError(f"Expected only test_fraction=0.5 in {summary_path}.")
            seeds = sorted({int(row["seed"]) for row in selected})
            standard_error = (
                0.0
                if values.size < 2
                else float(np.std(values, ddof=1) / np.sqrt(values.size))
            )
            grouped.append(
                {
                    "session_index": int(session_index),
                    "session_label": session_label,
                    "method": method,
                    "n_seeds": int(values.size),
                    "seeds": seeds,
                    "mean_achieved_fisher": float(np.mean(values)),
                    "ci95_half_width": float(1.96 * standard_error),
                }
            )
    return grouped


def _plot(grouped: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    sessions = sorted(
        {(int(row["session_index"]), str(row["session_label"])) for row in grouped}
    )
    positions = np.arange(len(sessions), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(6.0, 3.5), constrained_layout=True)
    for method in METHODS:
        rows = sorted(
            (row for row in grouped if row["method"] == method),
            key=lambda row: int(row["session_index"]),
        )
        axis.errorbar(
            positions,
            [float(row["mean_achieved_fisher"]) for row in rows],
            yerr=[float(row["ci95_half_width"]) for row in rows],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=5.5,
            capsize=2.5,
            label=LABELS[method],
        )
    axis.set_xticks(positions, [label for _, label in sessions])
    axis.set_xlabel("Session")
    axis.set_ylabel("Achieved information")
    axis.legend(frameon=False, ncol=3, loc="best")
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "stringer_all_sessions_achieved_information"
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        input_dir / "figures"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    grouped = _aggregate(input_dir)
    png, svg = _plot(grouped, output_dir)
    csv_path = output_dir / "stringer_all_sessions_achieved_information.csv"
    fieldnames = list(grouped[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped)
    summary_path = output_dir / "stringer_all_sessions_achieved_information.json"
    summary_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "rows": grouped,
                "artifacts": {"png": str(png), "svg": str(svg), "csv": str(csv_path)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for path in (summary_path, csv_path, png, svg):
        print(f"Saved: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
