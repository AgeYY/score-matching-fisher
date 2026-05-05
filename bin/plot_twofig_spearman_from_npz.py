#!/usr/bin/env python3
"""Plot Spearman rank correlation vs n from an existing ``h_decoding_twofig_results.npz``.

Uses the same off-diagonal mask and NaN handling as ``visualize_h_matrix_binned.matrix_corr_offdiag``
(Spearman on finite off-diagonal pairs). Does not retrain models.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import visualize_h_matrix_binned as vhb


def _save_figure_svg(fig: plt.Figure, path_svg: str) -> str:
    fig.savefig(path_svg)
    return path_svg


def _row_labels_from_npz(z: np.lib.npyio.NpzFile) -> list[str]:
    if "theta_field_rows" not in z.files:
        raise KeyError("results NPZ missing theta_field_rows")
    raw = np.asarray(z["theta_field_rows"])
    if raw.dtype == object:
        return [str(x) for x in raw.ravel().tolist()]
    return [str(x) for x in np.asarray(raw, dtype=str).ravel().tolist()]


def compute_spearman_vs_gt(h_sweep: np.ndarray, h_gt_sqrt: np.ndarray) -> np.ndarray:
    """Return shape (n_methods, n_n) Spearman rho vs GT for each method row and n column."""
    h_sw = np.asarray(h_sweep, dtype=np.float64)
    if h_sw.ndim != 4:
        raise ValueError(f"h_binned_sweep must be 4D (method, n, bins, bins); got {h_sw.shape}")
    h_gt_imp = vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64))
    n_m, n_n = int(h_sw.shape[0]), int(h_sw.shape[1])
    out = np.full((n_m, n_n), np.nan, dtype=np.float64)
    for i in range(n_m):
        for j in range(n_n):
            out[i, j] = vhb.matrix_corr_offdiag(
                vhb.impute_offdiag_nan_mean(np.asarray(h_sw[i, j], dtype=np.float64)),
                h_gt_imp,
            )
    return out


def compute_spearman_decode(decode_sweep: np.ndarray, decode_ref: np.ndarray) -> np.ndarray:
    d_sw = np.asarray(decode_sweep, dtype=np.float64)
    if d_sw.ndim != 3:
        raise ValueError(f"decode_sweep must be 3D (n, bins, bins); got {d_sw.shape}")
    ref_imp = vhb.impute_offdiag_nan_mean(np.asarray(decode_ref, dtype=np.float64))
    n_n = int(d_sw.shape[0])
    out = np.full((n_n,), np.nan, dtype=np.float64)
    for j in range(n_n):
        out[j] = vhb.matrix_corr_offdiag(
            vhb.impute_offdiag_nan_mean(np.asarray(d_sw[j], dtype=np.float64)),
            ref_imp,
        )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "results_npz",
        type=str,
        help="Path to h_decoding_twofig_results.npz from a prior twofig run.",
    )
    p.add_argument(
        "--output-svg",
        type=str,
        default=None,
        help="Output SVG path (default: alongside results as h_decoding_twofig_corr_vs_n_spearman.svg).",
    )
    p.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="Optional path to save rank_corr_h_binned_vs_gt_mc (+ decode if present) as NPZ.",
    )
    p.add_argument(
        "--no-decode",
        action="store_true",
        help="Do not plot the shared decoding curve even if decode_sweep is present.",
    )
    args = p.parse_args(argv)

    path = os.path.abspath(args.results_npz)
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")

    z = np.load(path, allow_pickle=True)
    required = ("h_binned_sweep", "h_gt_sqrt", "n")
    for k in required:
        if k not in z.files:
            raise SystemExit(f"{path} missing required array {k!r}")

    row_labels = _row_labels_from_npz(z)
    n_list = np.asarray(z["n"], dtype=np.int64).ravel().tolist()
    h_sweep = np.asarray(z["h_binned_sweep"], dtype=np.float64)
    h_gt = np.asarray(z["h_gt_sqrt"], dtype=np.float64)

    corr_h = compute_spearman_vs_gt(h_sweep, h_gt)
    if corr_h.shape != (len(row_labels), len(n_list)):
        raise SystemExit(
            f"shape mismatch: corr_h {corr_h.shape} vs rows={len(row_labels)} n={len(n_list)}"
        )

    corr_dec: np.ndarray | None = None
    if not args.no_decode and "decode_sweep" in z.files and "decode_ref" in z.files:
        corr_dec = compute_spearman_decode(
            np.asarray(z["decode_sweep"], dtype=np.float64),
            np.asarray(z["decode_ref"], dtype=np.float64),
        )

    out_svg = args.output_svg
    if out_svg is None:
        out_svg = os.path.join(os.path.dirname(path), "h_decoding_twofig_corr_vs_n_spearman.svg")
    out_svg = os.path.abspath(out_svg)
    os.makedirs(os.path.dirname(out_svg) or ".", exist_ok=True)

    n_arr = np.asarray(n_list, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    for i, label in enumerate(row_labels):
        ax.plot(
            n_arr,
            corr_h[i],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=f"{label} (H vs GT)",
        )
    if corr_dec is not None:
        ax.plot(
            n_arr,
            corr_dec,
            color="black",
            linestyle="--",
            marker="s",
            linewidth=1.6,
            markersize=3.5,
            label="decoding (shared)",
        )
    ax.set_xlabel("dataset size n", fontsize=10)
    ax.set_ylabel("Spearman rho (off-diagonal)", fontsize=10)
    ax.set_title("Rank correlation vs n", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_figure_svg(fig, out_svg)
    plt.close(fig)
    print(f"[spearman] Saved: {out_svg}", flush=True)

    if args.save_npz is not None:
        payload: dict[str, object] = {
            "n": np.asarray(n_list, dtype=np.int64),
            "theta_field_rows": np.asarray(row_labels, dtype=object),
            "rank_corr_h_binned_vs_gt_mc": np.asarray(corr_h, dtype=np.float64),
            "source_results_npz": np.asarray([path], dtype=object),
        }
        if corr_dec is not None:
            payload["rank_corr_decode_vs_ref_shared"] = np.asarray(corr_dec, dtype=np.float64)
        save_path = os.path.abspath(args.save_npz)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez_compressed(save_path, **payload)
        print(f"[spearman] Saved: {save_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
