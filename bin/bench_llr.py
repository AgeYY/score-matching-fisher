#!/usr/bin/env python3
"""Single-size categorical raw-LLR benchmark."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch

from global_setting import DATA_DIR

from fisher import h_decoding_convergence as conv
from fisher.h_decoding_categorical_twofig import (
    _abs_without_resolving_symlinks,
    _as_2d,
    _binned_gaussian_delta_l,
    _default_dataset_npz,
    _default_pr_output_npz,
    _ensure_dataset,
    _ensure_pr_projected_npz,
    _llr_comparison_metrics,
    _pairwise_delta_l_to_raw_llr,
    _pairwise_raw_llr_metrics,
    _save_method_training_loss_npz,
    _save_pairwise_raw_llr_est_vs_true_figure,
    _selected_eval_split,
    _train_one_method,
    _validation_only_work_sweep_subset,
    build_parser as _build_twofig_parser,
    compute_true_conditional_loglik_matrix,
    parse_methods,
)
from fisher.h_decoding_convergence_methods import prepare_categorical_binning_for_convergence
from fisher.h_decoding_twofig import _render_row_n_training_losses_panel, _sanitize_row_label
from fisher.h_matrix import HMatrixEstimator
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from fisher.svg_utils import (
    concatenate_pngs_horizontally,
    concatenate_svgs_horizontally,
    concatenate_svgs_horizontally_to_png,
    svg_viewbox_size,
)


def build_parser():
    p = _build_twofig_parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == "n_list":
            action.help = "Ignored by bench_llr.py; use --n-eval for the single benchmark size."
    p.add_argument("--n-eval", type=int, default=600, help="Single benchmark subset size.")
    return p


def _write_combined_llr_loss_outputs(
    *,
    llr_svg: Path,
    llr_png: Path,
    loss_panel_svg: Path,
    combined_svg: Path,
    combined_png: Path,
) -> tuple[Path, Path | None]:
    """Write horizontal LLR/loss SVG and best-effort PNG."""
    _, loss_height = svg_viewbox_size(loss_panel_svg)
    concatenate_svgs_horizontally(
        [llr_svg, loss_panel_svg],
        combined_svg,
        target_height=loss_height,
        valign="center",
    )
    combined_png_out: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="bench_llr_png_") as td:
            loss_png = Path(td) / "loss_panel.png"
            concatenate_svgs_horizontally_to_png([loss_panel_svg], loss_png, dpi=300)
            from PIL import Image

            with Image.open(loss_png) as im:
                loss_png_height = int(im.height)
            png = concatenate_pngs_horizontally(
                [llr_png, loss_png],
                combined_png,
                spacing=int(round(24.0 / 72.0 * 300.0)),
                target_height=loss_png_height,
                valign="center",
            )
        combined_png_out = Path(png).resolve()
    except Exception as exc:
        print(
            f"[bench-llr] WARNING: combined PNG failed ({type(exc).__name__}: {exc}); "
            "combined SVG, LLR diagnostics, loss panel, and results NPZ are still saved.",
            flush=True,
        )
    return combined_svg.resolve(), combined_png_out


def _write_summary(
    path: Path,
    *,
    args: Any,
    methods: list[str],
    eval_split: str,
    paths: dict[str, Path | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("bench_llr\n")
        f.write(f"output_dir: {Path(args.output_dir).resolve()}\n")
        f.write(f"dataset_npz: {Path(args.dataset_npz).resolve()}\n")
        f.write(f"num_categories: {int(args.num_categories)}\n")
        f.write(f"n_eval: {int(args.n_eval)}\n")
        f.write(f"methods: {','.join(methods)}\n")
        f.write(f"pr_project: {bool(args.pr_project)}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"eval_split: {eval_split}\n")
        for key, value in paths.items():
            if value is not None:
                f.write(f"{key}: {Path(value).resolve()}\n")


def _metric_table(metrics_by_method: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    method_names = np.asarray(list(metrics_by_method.keys()), dtype=object)
    metric_names = sorted({k for metrics in metrics_by_method.values() for k in metrics.keys()})
    values = np.full((len(method_names), len(metric_names)), np.nan, dtype=np.float64)
    for i, name in enumerate(method_names):
        metrics = metrics_by_method[str(name)]
        for j, metric_name in enumerate(metric_names):
            if metric_name in metrics:
                values[i, j] = float(metrics[metric_name])
    return method_names, np.asarray(metric_names, dtype=object), values


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = parse_methods(str(args.methods))
    if int(args.num_categories) < 2:
        raise ValueError("--num-categories must be >= 2.")
    if int(args.n_eval) < 1:
        raise ValueError("--n-eval must be >= 1.")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; bench_llr.py defaults to CUDA and will not switch to CPU.")
    if bool(args.visualization_only):
        raise ValueError("bench_llr.py does not support --visualization-only.")

    if args.dataset_npz is None:
        args.dataset_npz = _default_dataset_npz(int(args.num_categories))
    args.dataset_npz = _abs_without_resolving_symlinks(Path(args.dataset_npz))
    if args.output_dir is None:
        args.output_dir = Path(DATA_DIR) / "bench_llr"
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_dataset(args)
    native_npz = Path(args.dataset_npz).resolve()
    native_bundle = load_shared_dataset_npz(native_npz)
    native_meta = dict(native_bundle.meta)
    if str(native_meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical NPZ, got {native_meta.get('dataset_family')!r}.")
    if str(native_meta.get("theta_type", "")) != "categorical":
        raise ValueError(f"Expected categorical theta_type, got {native_meta.get('theta_type')!r}.")
    native_x_dim = int(native_bundle.x_all.shape[1])
    if native_x_dim != 2:
        raise ValueError(f"Native x_dim must be 2; got {native_x_dim}.")

    pr_project = bool(args.pr_project)
    pr_out_resolved: Path | None = None
    meta_work = dict(native_bundle.meta)
    if pr_project:
        if int(args.pr_dim) < native_x_dim:
            raise ValueError(f"--pr-dim must be >= native x_dim={native_x_dim}; got {args.pr_dim}.")
        pr_out = args.pr_output_npz
        if pr_out is None:
            pr_out = _default_pr_output_npz(native_npz, int(args.pr_dim))
        else:
            pr_out = _abs_without_resolving_symlinks(Path(pr_out))
        pr_out_resolved = Path(pr_out)
        _ensure_pr_projected_npz(args, native_npz=native_npz, pr_out=pr_out_resolved)
        work_bundle = load_shared_dataset_npz(pr_out_resolved)
        meta_work = dict(work_bundle.meta)
    else:
        work_bundle = native_bundle

    n_pool = int(native_bundle.theta_all.shape[0])
    if int(args.n_eval) > n_pool:
        raise ValueError(f"--n-eval={args.n_eval} exceeds n_total={n_pool}.")
    eval_split = _selected_eval_split(args)
    if eval_split == "validation" and float(meta_work.get("train_frac", 0.7)) >= 1.0:
        raise ValueError("--eval-split validation requires train_frac < 1 in dataset meta.")

    k_cat = int(native_meta.get("num_categories", int(args.num_categories)))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(native_bundle.theta_all, k_cat)
    rng = np.random.default_rng(int(args.run_seed))
    perm = rng.permutation(n_pool)

    subset_w = conv._subset_bundle(work_bundle, perm, int(args.n_eval), meta_work, bin_idx_all=bin_idx_all)
    subset_n = conv._subset_bundle(native_bundle, perm, int(args.n_eval), meta_work, bin_idx_all=bin_idx_all)
    train_theta = _as_2d(subset_w.bundle.theta_train)
    val_theta = _as_2d(subset_w.bundle.theta_validation)
    x_train = np.asarray(subset_w.bundle.x_train, dtype=np.float64)
    x_val = np.asarray(subset_w.bundle.x_validation, dtype=np.float64)
    bins_train = np.asarray(subset_w.bin_train, dtype=np.int64).reshape(-1)
    bins_val = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)

    if eval_split == "validation":
        if int(subset_w.bundle.x_validation.shape[0]) < 1:
            raise ValueError("Empty validation split; cannot use --eval-split validation.")
        theta_eval = val_theta
        x_eval = x_val
        x_native_eval = np.asarray(subset_n.bundle.x_validation, dtype=np.float64)
        bins_eval = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)
    else:
        theta_eval = _as_2d(subset_w.bundle.theta_all)
        x_eval = np.asarray(subset_w.bundle.x_all, dtype=np.float64)
        x_native_eval = np.asarray(subset_n.bundle.x_all, dtype=np.float64)
        bins_eval = np.asarray(subset_w.bin_all, dtype=np.int64).reshape(-1)

    true_c = compute_true_conditional_loglik_matrix(x_native_eval, theta_eval, native_meta)
    true_delta_l = HMatrixEstimator.compute_delta_l(true_c)
    pair_labels, true_pairwise_llr = _pairwise_delta_l_to_raw_llr(true_delta_l, bins_eval, k_cat=k_cat)

    dev = require_device(str(args.device))
    torch.manual_seed(int(args.run_seed))
    np.random.seed(int(args.run_seed))
    loss_root = args.output_dir / "training_losses"
    pairwise_llrs: dict[str, np.ndarray] = {}
    metrics_by_method: dict[str, dict[str, float]] = {}
    wall_s = np.full((len(methods),), np.nan, dtype=np.float64)

    for i, method_name in enumerate(methods):
        t0 = time.time()
        if method_name == "bin_gaussian":
            vf = float(
                getattr(
                    args,
                    "flow_theta_reg_variance_floor",
                    getattr(args, "flow_x_reg_variance_floor", 1e-6),
                )
            )
            subset_bg = _validation_only_work_sweep_subset(subset_w) if eval_split == "validation" else subset_w
            delta_l = _binned_gaussian_delta_l(subset_bg, k_cat, variance_floor=vf)
            result = {"train_out": None}
        else:
            result = _train_one_method(
                args,
                dev=dev,
                method_name=method_name,
                theta_train=train_theta,
                x_train=x_train,
                theta_val=val_theta,
                x_val=x_val,
                theta_all=theta_eval,
                x_all=x_eval,
                bins_train=bins_train,
                bins_val=bins_val,
                bins_all=bins_eval,
                k_cat=k_cat,
            )
            if "delta_l" not in result:
                raise ValueError(
                    f"Method {method_name!r} did not return delta_l; bench_llr.py only supports LLR methods."
                )
            delta_l = np.asarray(result["delta_l"], dtype=np.float64)
        wall_s[i] = time.time() - t0
        loss_npz = loss_root / _sanitize_row_label(method_name) / f"n_{int(args.n_eval):06d}.npz"
        _save_method_training_loss_npz(loss_npz, method_name=method_name, result=result)
        est_pair_labels, est_pairwise_llr = _pairwise_delta_l_to_raw_llr(delta_l, bins_eval, k_cat=k_cat)
        if not np.array_equal(est_pair_labels, pair_labels):
            raise ValueError(f"Method {method_name!r} pair labels do not match true pair labels.")
        pairwise_llrs[method_name] = est_pairwise_llr
        metrics = _pairwise_raw_llr_metrics(est_pairwise_llr, true_pairwise_llr)
        metrics.update(_llr_comparison_metrics(delta_l, true_delta_l))
        metrics_by_method[method_name] = metrics

    loss_panel_svg = Path(
        _render_row_n_training_losses_panel(
            row_labels=methods,
            n_list=[int(args.n_eval)],
            loss_root=str(loss_root),
            out_svg_path=str(args.output_dir / "bench_llr_training_losses_panel.svg"),
        )
    ).resolve()
    llr_base = args.output_dir / "llr_est_vs_true_all"
    _save_pairwise_raw_llr_est_vs_true_figure(
        pairwise_llrs,
        true_pairwise_llr,
        pair_labels,
        bins_eval,
        out_base=llr_base,
        metrics_by_method=metrics_by_method,
    )
    llr_svg = llr_base.with_suffix(".svg").resolve()
    llr_png = llr_base.with_suffix(".png").resolve()
    combined_svg, combined_png = _write_combined_llr_loss_outputs(
        llr_svg=llr_svg,
        llr_png=llr_png,
        loss_panel_svg=loss_panel_svg,
        combined_svg=args.output_dir / "bench_llr.svg",
        combined_png=args.output_dir / "bench_llr.png",
    )

    metric_method_names, metric_names, metric_values = _metric_table(metrics_by_method)
    out_npz = (args.output_dir / "bench_llr_results.npz").resolve()
    np.savez_compressed(
        out_npz,
        n_eval=np.int64(int(args.n_eval)),
        num_categories=np.int64(k_cat),
        method_names=np.asarray(methods, dtype=object),
        pair_labels=np.asarray(pair_labels, dtype=np.int64),
        true_llr_pairwise=np.asarray(true_pairwise_llr, dtype=np.float64),
        llr_pairwise_est=np.stack([pairwise_llrs[name] for name in methods], axis=0).astype(np.float64, copy=False),
        metric_method_names=metric_method_names,
        metric_names=metric_names,
        metric_values=metric_values,
        wall_seconds=wall_s,
        source_indices=np.asarray(perm[: int(args.n_eval)], dtype=np.int64),
        native_dataset_npz=np.asarray([str(native_npz)], dtype=object),
        work_dataset_npz=np.asarray(
            [str(pr_out_resolved) if pr_out_resolved is not None else str(native_npz)], dtype=object
        ),
        pr_projected=np.bool_(pr_project),
        pr_dim=np.int64(int(args.pr_dim) if pr_project else native_x_dim),
        training_losses_root=np.asarray([str(loss_root.resolve())], dtype=object),
        eval_split=np.asarray([eval_split], dtype=object),
    )

    summary_path = (args.output_dir / "bench_llr_summary.txt").resolve()
    _write_summary(
        summary_path,
        args=args,
        methods=methods,
        eval_split=eval_split,
        paths={
            "results_npz": out_npz,
            "bench_llr_training_losses_panel.svg": loss_panel_svg,
            "llr_est_vs_true_all.svg": llr_svg,
            "llr_est_vs_true_all.png": llr_png,
            "bench_llr.svg": combined_svg,
            "bench_llr.png": combined_png,
            "training_losses_root": loss_root.resolve(),
        },
    )

    print("[bench-llr] Saved:", flush=True)
    for p in (out_npz, loss_panel_svg, llr_svg, llr_png, combined_svg, combined_png, summary_path):
        if p is not None:
            print(f"  - {Path(p).resolve()}", flush=True)


if __name__ == "__main__":
    main()
