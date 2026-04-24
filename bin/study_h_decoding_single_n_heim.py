#!/usr/bin/env python3
"""Single-n HeIM-only study with per-iteration Hellinger visualizations.

This script reuses the HeIM/x_flow machinery from ``study_h_decoding_convergence.py``
but runs exactly one subset size ``--n`` and writes iteration-by-iteration matrix
figures, including initialization (D^(0)-derived H^2).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import matplotlib.pyplot as plt
import numpy as np

import study_h_decoding_convergence as shc


def build_parser() -> argparse.ArgumentParser:
    p = shc.build_parser()
    p.description = (
        "Single-n HeIM-only study. Runs one subset size --n with x_flow and writes "
        "per-iteration Hellinger matrix figures (including initialization)."
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Single subset size to run (must satisfy 1 <= n <= n_total).",
    )
    return p


def _write_iteration_matrix_panel(
    *,
    out_png: str,
    left_title: str,
    h_sqrt: np.ndarray,
    h_gt_sqrt: np.ndarray,
    corr_h: float,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    vmax = float(np.nanmax(np.asarray([h_sqrt, h_gt_sqrt], dtype=np.float64)))
    if (not np.isfinite(vmax)) or vmax <= 0.0:
        vmax = 1.0
    im0 = axes[0].imshow(h_sqrt, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
    axes[0].set_title(left_title, fontsize=10)
    axes[0].set_xlabel(r"$\theta$ bin")
    axes[0].set_ylabel(r"$\theta$ bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    im1 = axes[1].imshow(h_gt_sqrt, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
    axes[1].set_title(rf"GT MC $\sqrt{{H^2}}$ (r={float(corr_h):.4f})", fontsize=10)
    axes[1].set_xlabel(r"$\theta$ bin")
    axes[1].set_ylabel(r"$\theta$ bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    out_svg = str(Path(out_png).with_suffix(".svg"))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_svg


def _iter_stem(*, n: int, label: str) -> str:
    return f"h_x_flow_heim_flow_{label}_n_{int(n):06d}"


def _h_sqrt_from_iter_run(
    *,
    iter_dir: str,
    dataset_family: str,
    bin_all: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    p = os.path.join(iter_dir, shc._h_matrix_results_npz_basename(dataset_family=dataset_family))
    if not os.path.isfile(p):
        alt = os.path.join(iter_dir, "h_matrix_results_theta_cov.npz")
        if os.path.isfile(alt):
            p = alt
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing iteration H-matrix npz: {p}")
    z = np.load(p, allow_pickle=True)
    if "h_sym" not in z.files:
        raise KeyError(f"{p} missing h_sym.")
    h_sym = np.asarray(z["h_sym"], dtype=np.float64)
    h_binned, _ = shc.vhb.average_matrix_by_bins(h_sym, np.asarray(bin_all, dtype=np.int64), int(n_bins))
    return shc._sqrt_h_like(h_binned)


def _render_iteration_visualizations(
    *,
    output_dir: str,
    n: int,
    dataset_family: str,
    n_bins: int,
    bin_all: np.ndarray,
    h_gt_sqrt: np.ndarray,
) -> list[dict[str, str]]:
    heim_npz = os.path.join(output_dir, "heim_flow_iterations.npz")
    if not os.path.isfile(heim_npz):
        raise FileNotFoundError(f"Expected HeIM iteration archive not found: {heim_npz}")
    z = np.load(heim_npz, allow_pickle=True)
    init_h2 = np.asarray(z["init_h2"], dtype=np.float64)
    rel_hist = np.asarray(z["rel_change_history"], dtype=np.float64).reshape(-1)
    iters_done = int(np.asarray(z["heim_iters_completed"]).reshape(-1)[0]) if "heim_iters_completed" in z.files else int(rel_hist.size)

    viz_dir = os.path.join(output_dir, "heim_flow_iter_viz")
    os.makedirs(viz_dir, exist_ok=True)
    rows: list[dict[str, str]] = []

    init_sqrt = shc._sqrt_h_like(init_h2)
    corr_init = float(shc.vhb.matrix_corr_offdiag_pearson(init_sqrt, h_gt_sqrt))
    init_stem = _iter_stem(n=n, label="iter_init")
    init_png = os.path.join(viz_dir, f"{init_stem}.png")
    init_svg = _write_iteration_matrix_panel(
        out_png=init_png,
        left_title=rf"HeIM init $\sqrt{{H^2}}$ (n={int(n)})",
        h_sqrt=init_sqrt,
        h_gt_sqrt=h_gt_sqrt,
        corr_h=corr_init,
    )
    rows.append(
        {
            "iter_label": "init",
            "iter_index": "-1",
            "corr_h_vs_gt_offdiag_pearson": f"{corr_init:.12g}",
            "fro_rel_change": "",
            "png": os.path.abspath(init_png),
            "svg": os.path.abspath(init_svg),
        }
    )

    heim_root = os.path.join(output_dir, "heim_flow")
    for k in range(int(iters_done)):
        iter_dir = os.path.join(heim_root, f"iter_{k:03d}")
        h_k_sqrt = _h_sqrt_from_iter_run(
            iter_dir=iter_dir,
            dataset_family=dataset_family,
            bin_all=bin_all,
            n_bins=n_bins,
        )
        corr_k = float(shc.vhb.matrix_corr_offdiag_pearson(h_k_sqrt, h_gt_sqrt))
        stem = _iter_stem(n=n, label=f"iter_{k:03d}")
        out_png = os.path.join(viz_dir, f"{stem}.png")
        out_svg = _write_iteration_matrix_panel(
            out_png=out_png,
            left_title=rf"HeIM iter {int(k)} $\sqrt{{H^2}}$ (n={int(n)})",
            h_sqrt=h_k_sqrt,
            h_gt_sqrt=h_gt_sqrt,
            corr_h=corr_k,
        )
        rel = float(rel_hist[k]) if k < rel_hist.size else float("nan")
        rows.append(
            {
                "iter_label": f"iter_{k:03d}",
                "iter_index": str(int(k)),
                "corr_h_vs_gt_offdiag_pearson": f"{corr_k:.12g}",
                "fro_rel_change": "" if not np.isfinite(rel) else f"{rel:.12g}",
                "png": os.path.abspath(out_png),
                "svg": os.path.abspath(out_svg),
            }
        )

    csv_path = os.path.join(output_dir, "heim_flow_iteration_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["iter_label", "iter_index", "corr_h_vs_gt_offdiag_pearson", "fro_rel_change", "png", "svg"],
        )
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir = os.path.abspath(str(args.output_dir))
    args.dataset_npz = os.path.abspath(str(args.dataset_npz))
    n = int(args.n)

    # Enforce single-n HeIM-only x_flow behavior.
    args.theta_field_method = "x_flow"
    args.heim_flow_enable = True
    args.theta_flow_onehot_state = False
    args.theta_flow_fourier_state = False
    args.theta_flow_segmented = False
    args.theta_flow_acc_mds_state = False
    args.visualization_only = False

    shc._validate_cli(args)
    os.makedirs(args.output_dir, exist_ok=True)

    bundle = shc.load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
    meta_family = str(meta.get("dataset_family", ""))
    if meta_family != str(args.dataset_family):
        raise ValueError(
            f"NPZ meta dataset_family={meta_family!r} does not match --dataset-family={str(args.dataset_family)!r}."
        )
    n_pool = int(bundle.theta_all.shape[0])
    if n < 1 or n > n_pool:
        raise ValueError(f"--n must satisfy 1 <= n <= n_total={n_pool}; got n={n}.")

    n_bins = int(args.num_theta_bins)
    base_seed = int(args.run_seed) if args.run_seed is not None else int(meta["seed"])
    perm_seed = base_seed + int(args.subset_seed_offset)
    rng_perm = np.random.default_rng(perm_seed)
    perm = rng_perm.permutation(n_pool)

    theta_raw_all = np.asarray(bundle.theta_all, dtype=np.float64)
    if theta_raw_all.ndim == 2 and int(theta_raw_all.shape[1]) != 1:
        raise ValueError(f"Single-n binning requires scalar theta; got theta_all shape={theta_raw_all.shape}.")
    theta_scalar_all = theta_raw_all.reshape(-1)
    theta_ref = np.asarray(theta_scalar_all[perm[: int(n)]], dtype=np.float64).reshape(-1)
    edges, edge_lo, edge_hi, theta_bin_source = shc.vhb.canonical_theta_bin_edges(
        meta=meta,
        theta_ref=theta_ref,
        n_bins=n_bins,
    )
    print(
        f"[single-n-heim] theta-bin edges source={theta_bin_source} range=[{edge_lo:.6g}, {edge_hi:.6g}] n_bins={n_bins}",
        flush=True,
    )
    bin_idx_all = shc.vhb.theta_to_bin_index(theta_scalar_all, edges, n_bins)
    subset = shc._subset_bundle(bundle, perm, n, meta, bin_idx_all=bin_idx_all, theta_state_all=None)

    dataset_for_gt = shc.build_dataset_from_meta(meta)
    gt_seed = base_seed if int(args.gt_hellinger_seed) < 0 else int(args.gt_hellinger_seed)
    if hasattr(dataset_for_gt, "rng"):
        dataset_for_gt.rng = np.random.default_rng(gt_seed)
    centers = shc.bin_centers_from_edges(edges)
    gt_n_mc = int(n) // n_bins
    if gt_n_mc < 1:
        raise ValueError(f"Need n//num_theta_bins >= 1 for GT MC; got n={n} n_bins={n_bins}.")
    h_gt_mc = shc.estimate_hellinger_sq_one_sided_mc(
        dataset_for_gt,
        centers,
        n_mc=gt_n_mc,
        symmetrize=bool(args.gt_hellinger_symmetrize),
    )
    h_gt_sqrt = shc._sqrt_h_like(h_gt_mc)

    h2_final, _clf_n, _loaded_final, _x_aligned = shc._run_heim_flow_for_subset(
        args=args,
        meta=meta,
        subset=subset,
        output_dir=args.output_dir,
        n_bins=n_bins,
        dataset_family=str(args.dataset_family),
    )
    h_final_sqrt = shc._sqrt_h_like(np.asarray(h2_final, dtype=np.float64))
    corr_final = float(shc.vhb.matrix_corr_offdiag_pearson(h_final_sqrt, h_gt_sqrt))
    fig_png, fig_svg = shc._write_heim_flow_h_matrix_figure(
        out_dir=args.output_dir,
        method_tag="x_flow",
        n=int(n),
        h_sqrt=h_final_sqrt,
        h_gt_sqrt=h_gt_sqrt,
        corr_h=corr_final,
    )
    iter_rows = _render_iteration_visualizations(
        output_dir=args.output_dir,
        n=int(n),
        dataset_family=str(args.dataset_family),
        n_bins=n_bins,
        bin_all=np.asarray(subset.bin_all, dtype=np.int64),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
    )

    heim_npz = os.path.join(args.output_dir, "heim_flow_iterations.npz")
    z_heim = np.load(heim_npz, allow_pickle=True)
    out_npz = os.path.join(args.output_dir, "single_n_heim_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.int64(int(n)),
        theta_bin_edges=np.asarray(edges, dtype=np.float64),
        theta_bin_centers=np.asarray(centers, dtype=np.float64),
        theta_bin_source=np.asarray([str(theta_bin_source)], dtype=object),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        h_final_sqrt=np.asarray(h_final_sqrt, dtype=np.float64),
        corr_h_vs_gt_offdiag=np.float64(corr_final),
        rel_change_history=np.asarray(z_heim["rel_change_history"], dtype=np.float64),
        heim_iters_requested=np.asarray(z_heim["heim_iters_requested"]).reshape(-1)[0],
        heim_iters_effective_budget=np.asarray(z_heim["heim_iters_effective_budget"]).reshape(-1)[0],
        heim_iters_completed=np.asarray(z_heim["heim_iters_completed"]).reshape(-1)[0],
    )

    summary = os.path.join(args.output_dir, "single_n_heim_summary.txt")
    with open(summary, "w", encoding="utf-8") as sf:
        sf.write("study_h_decoding_single_n_heim\n")
        sf.write(f"dataset_npz: {args.dataset_npz}\n")
        sf.write(f"dataset_family: {args.dataset_family}\n")
        sf.write(f"output_dir: {args.output_dir}\n")
        sf.write(f"n: {int(n)}\n")
        sf.write(f"num_theta_bins: {int(n_bins)}\n")
        sf.write(f"theta_bin_source: {theta_bin_source}\n")
        sf.write(f"gt_hellinger_n_mc: {int(gt_n_mc)}\n")
        sf.write(f"gt_hellinger_seed: {int(gt_seed)}\n")
        sf.write(f"corr_h_vs_gt_offdiag_pearson_final: {float(corr_final):.6g}\n")
        sf.write(f"final_figure_png: {fig_png}\n")
        sf.write(f"final_figure_svg: {fig_svg}\n")
        sf.write(f"results_npz: {out_npz}\n")
        sf.write(f"iteration_metrics_csv: {os.path.join(args.output_dir, 'heim_flow_iteration_metrics.csv')}\n")
        sf.write(f"iteration_viz_count: {len(iter_rows)}\n")

    print("[single-n-heim] Saved:")
    print(f"  - {fig_png}")
    print(f"  - {fig_svg}")
    print(f"  - {os.path.join(args.output_dir, 'heim_flow_iter_viz')}/")
    print(f"  - {os.path.join(args.output_dir, 'heim_flow_iteration_metrics.csv')}")
    print(f"  - {out_npz}")
    print(f"  - {summary}")


if __name__ == "__main__":
    main()
