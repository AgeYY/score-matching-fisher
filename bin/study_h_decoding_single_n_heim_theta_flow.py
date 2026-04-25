#!/usr/bin/env python3
"""Single-n posterior/prior theta-flow HeIM study with per-iteration diagnostics."""

from __future__ import annotations

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
import study_h_decoding_single_n_heim as single_x_heim

FORCED_TRAIN_FRAC = single_x_heim.FORCED_TRAIN_FRAC


def build_parser():
    p = single_x_heim.build_parser()
    p.description = (
        "Single-n HeIM study using posterior/prior theta-flow. Runs one subset size "
        "--n and writes per-iteration H, Bayes-ratio, and Delta-L diagnostics."
    )
    return p


def _iter_stem(*, n: int, label: str, kind: str) -> str:
    return f"{kind}_theta_flow_heim_flow_{label}_n_{int(n):06d}"


def _h_sqrt_from_iter_run(
    *,
    iter_dir: str,
    dataset_family: str,
    n_train: int,
    bin_validation: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    return single_x_heim._h_sqrt_from_iter_run(
        iter_dir=iter_dir,
        dataset_family=dataset_family,
        n_train=n_train,
        bin_validation=bin_validation,
        n_bins=n_bins,
    )


def _load_iter_npz(iter_dir: str, *, dataset_family: str) -> tuple[str, dict[str, np.ndarray]]:
    p = os.path.join(iter_dir, shc._h_matrix_results_npz_basename(dataset_family=dataset_family))
    if not os.path.isfile(p):
        alt = os.path.join(iter_dir, "h_matrix_results_theta_cov.npz")
        if os.path.isfile(alt):
            p = alt
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing iteration H-matrix npz: {p}")
    z = np.load(p, allow_pickle=True)
    return p, {k: np.asarray(z[k]) for k in z.files}


def _validation_binned_matrix(
    mat: np.ndarray,
    *,
    n_train: int,
    bin_validation: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    m_val = np.asarray(m[int(n_train) :, int(n_train) :], dtype=np.float64)
    bins = np.asarray(bin_validation, dtype=np.int64).reshape(-1)
    if m_val.shape[0] != bins.size:
        raise ValueError(f"Validation matrix/bin mismatch: {m_val.shape[0]} vs {bins.size}.")
    out, _ = shc.vhb.average_matrix_by_bins(m_val, bins, int(n_bins))
    return np.asarray(out, dtype=np.float64)


def _write_matrix_panel(
    *,
    out_png: str,
    left_title: str,
    left: np.ndarray,
    right_title: str,
    right: np.ndarray,
    colorbar_label: str,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    vals = np.asarray([np.nanmin(left), np.nanmax(left), np.nanmin(right), np.nanmax(right)], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    vmax_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax_abs <= 0.0:
        vmax_abs = 1.0
    for ax, title, mat in zip(axes, [left_title, right_title], [left, right], strict=True):
        im = ax.imshow(
            mat,
            origin="lower",
            vmin=-vmax_abs,
            vmax=vmax_abs,
            cmap="coolwarm",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\theta$ bin")
        ax.set_ylabel(r"$\theta$ bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=colorbar_label)
    out_svg = str(Path(out_png).with_suffix(".svg"))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_svg


def _render_h_iteration_visualizations(
    *,
    output_dir: str,
    n: int,
    dataset_family: str,
    n_bins: int,
    n_train: int,
    bin_validation: np.ndarray,
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
    init_stem = _iter_stem(n=n, label="iter_init", kind="h")
    init_png = os.path.join(viz_dir, f"{init_stem}.png")
    init_svg = single_x_heim._write_iteration_matrix_panel(
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
            n_train=int(n_train),
            bin_validation=np.asarray(bin_validation, dtype=np.int64),
            n_bins=n_bins,
        )
        corr_k = float(shc.vhb.matrix_corr_offdiag_pearson(h_k_sqrt, h_gt_sqrt))
        stem = _iter_stem(n=n, label=f"iter_{k:03d}", kind="h")
        out_png = os.path.join(viz_dir, f"{stem}.png")
        out_svg = single_x_heim._write_iteration_matrix_panel(
            out_png=out_png,
            left_title=rf"Theta-flow HeIM iter {int(k)} $\sqrt{{H^2}}$ (n={int(n)})",
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


def _render_bayes_iteration_visualizations(
    *,
    output_dir: str,
    n: int,
    dataset_family: str,
    n_bins: int,
    n_train: int,
    bin_validation: np.ndarray,
) -> list[dict[str, str]]:
    heim_npz = os.path.join(output_dir, "heim_flow_iterations.npz")
    z = np.load(heim_npz, allow_pickle=True)
    rel_hist = np.asarray(z["rel_change_history"], dtype=np.float64).reshape(-1)
    iters_done = int(np.asarray(z["heim_iters_completed"]).reshape(-1)[0]) if "heim_iters_completed" in z.files else int(rel_hist.size)
    viz_dir = os.path.join(output_dir, "heim_flow_iter_viz")
    os.makedirs(viz_dir, exist_ok=True)

    rows: list[dict[str, str]] = []
    heim_root = os.path.join(output_dir, "heim_flow")
    for k in range(int(iters_done)):
        iter_dir = os.path.join(heim_root, f"iter_{k:03d}")
        h_npz, data = _load_iter_npz(iter_dir, dataset_family=dataset_family)
        ratio_key = "c_matrix_ratio" if "c_matrix_ratio" in data else "c_matrix"
        if ratio_key not in data or "delta_l_matrix" not in data:
            raise KeyError(f"{h_npz} must contain {ratio_key!r} and 'delta_l_matrix'.")
        ratio_binned = _validation_binned_matrix(
            data[ratio_key],
            n_train=n_train,
            bin_validation=bin_validation,
            n_bins=n_bins,
        )
        delta_l_binned = _validation_binned_matrix(
            data["delta_l_matrix"],
            n_train=n_train,
            bin_validation=bin_validation,
            n_bins=n_bins,
        )
        stem = _iter_stem(n=n, label=f"iter_{k:03d}", kind="bayes")
        out_png = os.path.join(viz_dir, f"{stem}.png")
        out_svg = _write_matrix_panel(
            out_png=out_png,
            left_title=rf"$R_x(\theta)$ bins, iter {int(k)}",
            left=ratio_binned,
            right_title=rf"$\Delta L$ bins, iter {int(k)}",
            right=delta_l_binned,
            colorbar_label="log ratio",
        )
        rel = float(rel_hist[k]) if k < rel_hist.size else float("nan")
        rows.append(
            {
                "iter_label": f"iter_{k:03d}",
                "iter_index": str(int(k)),
                "fro_rel_change": "" if not np.isfinite(rel) else f"{rel:.12g}",
                "h_npz_path": os.path.abspath(h_npz),
                "ratio_key": ratio_key,
                "png": os.path.abspath(out_png),
                "svg": os.path.abspath(out_svg),
            }
        )

    csv_path = os.path.join(output_dir, "heim_flow_bayes_iteration_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["iter_label", "iter_index", "fro_rel_change", "h_npz_path", "ratio_key", "png", "svg"],
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

    args.theta_field_method = "theta_flow"
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
        f"[single-n-heim-theta-flow] theta-bin edges source={theta_bin_source} "
        f"range=[{edge_lo:.6g}, {edge_hi:.6g}] n_bins={n_bins}",
        flush=True,
    )
    bin_idx_all = shc.vhb.theta_to_bin_index(theta_scalar_all, edges, n_bins)
    meta_for_split = dict(meta)
    meta_for_split["train_frac"] = float(FORCED_TRAIN_FRAC)
    subset = shc._subset_bundle(bundle, perm, n, meta_for_split, bin_idx_all=bin_idx_all, theta_state_all=None)
    n_train = int(np.asarray(subset.bundle.x_train).shape[0])
    n_validation = int(np.asarray(subset.bundle.x_validation).shape[0])
    if n_validation < 1:
        raise ValueError("Validation split is empty; cannot run validation-only Hellinger evaluation.")

    dataset_for_gt = shc.build_dataset_from_meta(meta)
    gt_seed = base_seed if int(args.gt_hellinger_seed) < 0 else int(args.gt_hellinger_seed)
    if hasattr(dataset_for_gt, "rng"):
        dataset_for_gt.rng = np.random.default_rng(gt_seed)
    centers = shc.bin_centers_from_edges(edges)
    gt_n_mc = int(n_validation) // n_bins
    if gt_n_mc < 1:
        raise ValueError(
            f"Need n_validation//num_theta_bins >= 1 for GT MC; got n_validation={n_validation} n_bins={n_bins}."
        )
    h_gt_mc = shc.estimate_hellinger_sq_one_sided_mc(
        dataset_for_gt,
        centers,
        n_mc=gt_n_mc,
        symmetrize=bool(args.gt_hellinger_symmetrize),
    )
    h_gt_sqrt = shc._sqrt_h_like(h_gt_mc)

    _h2_final, _clf_n, _loaded_final, _x_aligned = shc._run_heim_flow_for_subset(
        args=args,
        meta=meta,
        subset=subset,
        output_dir=args.output_dir,
        n_bins=n_bins,
        dataset_family=str(args.dataset_family),
    )
    heim_npz = os.path.join(args.output_dir, "heim_flow_iterations.npz")
    z_heim = np.load(heim_npz, allow_pickle=True)
    iters_done = int(np.asarray(z_heim["heim_iters_completed"]).reshape(-1)[0])
    if iters_done < 1:
        raise ValueError("HeIM run completed zero iterations; cannot compute final validation-only H estimate.")
    final_iter_dir = os.path.join(args.output_dir, "heim_flow", f"iter_{iters_done - 1:03d}")
    h_final_sqrt = _h_sqrt_from_iter_run(
        iter_dir=final_iter_dir,
        dataset_family=str(args.dataset_family),
        n_train=int(n_train),
        bin_validation=np.asarray(subset.bin_validation, dtype=np.int64),
        n_bins=n_bins,
    )
    corr_final = float(shc.vhb.matrix_corr_offdiag_pearson(h_final_sqrt, h_gt_sqrt))
    fig_png, fig_svg = shc._write_heim_flow_h_matrix_figure(
        out_dir=args.output_dir,
        method_tag="theta_flow",
        n=int(n),
        h_sqrt=h_final_sqrt,
        h_gt_sqrt=h_gt_sqrt,
        corr_h=corr_final,
    )
    iter_rows = _render_h_iteration_visualizations(
        output_dir=args.output_dir,
        n=int(n),
        dataset_family=str(args.dataset_family),
        n_bins=n_bins,
        n_train=int(n_train),
        bin_validation=np.asarray(subset.bin_validation, dtype=np.int64),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
    )
    bayes_rows = _render_bayes_iteration_visualizations(
        output_dir=args.output_dir,
        n=int(n),
        dataset_family=str(args.dataset_family),
        n_bins=n_bins,
        n_train=int(n_train),
        bin_validation=np.asarray(subset.bin_validation, dtype=np.int64),
    )

    out_npz = os.path.join(args.output_dir, "single_n_heim_theta_flow_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.int64(int(n)),
        theta_bin_edges=np.asarray(edges, dtype=np.float64),
        theta_bin_centers=np.asarray(centers, dtype=np.float64),
        theta_bin_source=np.asarray([str(theta_bin_source)], dtype=object),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        h_final_sqrt=np.asarray(h_final_sqrt, dtype=np.float64),
        corr_h_vs_gt_offdiag=np.float64(corr_final),
        train_frac_used=np.float64(float(FORCED_TRAIN_FRAC)),
        n_train=np.int64(int(n_train)),
        n_validation=np.int64(int(n_validation)),
        hellinger_eval_pool=np.asarray(["validation_only"], dtype=object),
        log_likelihood_semantics=np.asarray(["log_p_x_given_theta_minus_log_p_x"], dtype=object),
        rel_change_history=np.asarray(z_heim["rel_change_history"], dtype=np.float64),
        heim_iters_requested=np.asarray(z_heim["heim_iters_requested"]).reshape(-1)[0],
        heim_iters_effective_budget=np.asarray(z_heim["heim_iters_effective_budget"]).reshape(-1)[0],
        heim_iters_completed=np.asarray(z_heim["heim_iters_completed"]).reshape(-1)[0],
    )

    summary = os.path.join(args.output_dir, "single_n_heim_theta_flow_summary.txt")
    with open(summary, "w", encoding="utf-8") as sf:
        sf.write("study_h_decoding_single_n_heim_theta_flow\n")
        sf.write(f"dataset_npz: {args.dataset_npz}\n")
        sf.write(f"dataset_family: {args.dataset_family}\n")
        sf.write(f"output_dir: {args.output_dir}\n")
        sf.write(f"n: {int(n)}\n")
        sf.write(f"train_frac_used: {float(FORCED_TRAIN_FRAC):.6g}\n")
        sf.write(f"n_train: {int(n_train)}\n")
        sf.write(f"n_validation: {int(n_validation)}\n")
        sf.write(f"num_theta_bins: {int(n_bins)}\n")
        sf.write("hellinger_eval_pool: validation_only\n")
        sf.write(f"theta_bin_source: {theta_bin_source}\n")
        sf.write(f"gt_hellinger_n_mc: {int(gt_n_mc)}\n")
        sf.write(f"gt_hellinger_seed: {int(gt_seed)}\n")
        sf.write(f"corr_h_vs_gt_offdiag_pearson_final: {float(corr_final):.6g}\n")
        sf.write("log_likelihood_semantics: log p(theta|x) - log p(theta) = log p(x|theta) - log p(x)\n")
        sf.write("absolute_log_likelihood_calibrated: false\n")
        sf.write(f"final_figure_png: {fig_png}\n")
        sf.write(f"final_figure_svg: {fig_svg}\n")
        sf.write(f"results_npz: {out_npz}\n")
        sf.write(f"iteration_metrics_csv: {os.path.join(args.output_dir, 'heim_flow_iteration_metrics.csv')}\n")
        sf.write(f"bayes_iteration_metrics_csv: {os.path.join(args.output_dir, 'heim_flow_bayes_iteration_metrics.csv')}\n")
        sf.write(f"iteration_viz_count: {len(iter_rows)}\n")
        sf.write(f"bayes_iteration_viz_count: {len(bayes_rows)}\n")

    print("[single-n-heim-theta-flow] Saved:")
    print(f"  - {fig_png}")
    print(f"  - {fig_svg}")
    print(f"  - {os.path.join(args.output_dir, 'heim_flow_iter_viz')}/")
    print(f"  - {os.path.join(args.output_dir, 'heim_flow_iteration_metrics.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'heim_flow_bayes_iteration_metrics.csv')}")
    print(f"  - {out_npz}")
    print(f"  - {summary}")


if __name__ == "__main__":
    main()
