#!/usr/bin/env python3
"""Plot linear_x_flow_low_rank_t warmup losses vs epoch from a twofig training_losses NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--twofig-output-dir",
        type=str,
        required=True,
        help="Directory containing training_losses/linear_x_flow_low_rank_t/n_*.npz",
    )
    p.add_argument(
        "--out-svg",
        type=str,
        default="",
        help="Output SVG path (default: <twofig-output-dir>/lxf_low_rank_t_warmup_loss_vs_epoch.svg)",
    )
    p.add_argument(
        "--n-list",
        type=str,
        default="80,200",
        help="Comma-separated n values matching saved n_XXXXXX.npz files.",
    )
    args = p.parse_args()
    out_dir = Path(args.twofig_output_dir).resolve()
    loss_dir = out_dir / "training_losses" / "linear_x_flow_low_rank_t"
    ns = [int(x.strip()) for x in str(args.n_list).split(",") if x.strip()]
    out_svg = Path(args.out_svg).resolve() if str(args.out_svg).strip() else out_dir / "lxf_low_rank_t_warmup_loss_vs_epoch.svg"

    fig, axes = plt.subplots(len(ns), 1, figsize=(8.0, 2.8 * max(1, len(ns))), sharex=True, squeeze=False)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, n in zip(axes_arr, ns):
        src = loss_dir / f"n_{n:06d}.npz"
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")
        z = np.load(src, allow_pickle=True)
        if not bool(np.asarray(z["lxf_low_rank_t_warmup_enabled"]).reshape(-1)[0]):
            ax.text(0.5, 0.5, "warmup disabled", ha="center", va="center", transform=ax.transAxes)
            continue
        tr = np.asarray(z["lxf_low_rank_t_warmup_train_losses"], dtype=np.float64).ravel()
        va = np.asarray(z["lxf_low_rank_t_warmup_val_losses"], dtype=np.float64).ravel()
        mo = np.asarray(z["lxf_low_rank_t_warmup_val_monitor_losses"], dtype=np.float64).ravel()
        ep = np.arange(1, len(tr) + 1, dtype=np.float64)
        ax.plot(ep, tr, label="train FM", lw=1.0, alpha=0.9)
        ax.plot(ep, va, label="val FM", lw=1.0, alpha=0.9)
        ax.plot(ep, mo, label="val smooth (monitor)", lw=1.0, alpha=0.9)
        ax.set_ylabel("loss")
        ax.set_title(f"linear_x_flow_low_rank_t warmup | nested n={n}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes_arr[-1].set_xlabel("warmup epoch")
    fig.suptitle("Low-rank-t b-only warmup: loss vs epoch", fontsize=11, y=1.02)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(str(out_svg))


if __name__ == "__main__":
    main()
