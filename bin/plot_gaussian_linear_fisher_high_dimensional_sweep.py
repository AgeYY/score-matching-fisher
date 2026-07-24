#!/usr/bin/env python3
"""Plot Gaussian linear Fisher error versus response dimensionality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR


METHOD_STYLE = {
    "Flow Matching": {"color": "C0", "marker": "o"},
    "GKR": {"color": "C2", "marker": "^"},
}


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DATA_DIR) / "gaussian_linear_fisher_dimension_n1000_r1_flow_gkr",
    )
    parser.add_argument("--dimensions", type=_csv_ints, default=[50, 200, 500, 700])
    parser.add_argument("--n-total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    dimensions = sorted(dict.fromkeys(int(value) for value in args.dimensions))
    rows: list[dict[str, float | int]] = []
    for dimension in dimensions:
        metadata_path = (
            input_dir
            / f"xdim{dimension}"
            / "randamp_gaussian_sqrtd"
            / f"seed{int(args.seed)}"
            / f"n{int(args.n_total)}"
            / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "response_dimension": dimension,
                "flow_matching_mae": float(metadata["mae"]["Flow Matching"]),
                "gkr_mae": float(metadata["mae"]["GKR"]),
                "flow_selected_epoch": int(metadata["flow_selected_epoch"]),
                "flow_stopped_epoch": int(metadata["flow_stopped_epoch"]),
            }
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
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.0, 3.5))
    x = np.asarray(dimensions, dtype=np.int64)
    for method, key in (
        ("Flow Matching", "flow_matching_mae"),
        ("GKR", "gkr_mae"),
    ):
        style = METHOD_STYLE[method]
        axis.plot(
            x,
            [float(row[key]) for row in rows],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6.5,
            label=method,
        )
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xlabel("Response dimension")
    axis.set_ylabel("Linear Fisher MAE")
    axis.set_title(r"Gaussian, $N=1{,}000$")
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.legend(frameon=False, loc="upper left")

    stem = input_dir / "gaussian_linear_fisher_error_vs_dimension_n1000_r1"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)

    summary = {
        "dataset": "randamp_gaussian_sqrtd",
        "n_total": int(args.n_total),
        "seed": int(args.seed),
        "rows": rows,
        "artifacts": {"png": str(png_path), "svg": str(svg_path)},
    }
    summary_path = stem.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
