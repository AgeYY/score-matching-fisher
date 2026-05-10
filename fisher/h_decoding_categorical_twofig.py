#!/usr/bin/env python3
"""Nested-n sweep for categorical random MoG: LLR-based category Hellinger vs analytic GT.

Mirrors :mod:`fisher.h_decoding_twofig` layout (sweep / GT / corr / NMSE / **training-loss panel** SVGs
+ NPZ + summary) while using the categorical estimators and directed-then-symmetrized category aggregation from
``bin/debug_categorical_xflow_llr.py``. Specialized to ``random_mog_categorical`` only.

Use ``--hellinger-eval-split validation`` to evaluate learned H (and matching GT LLR scatter metrics) on the
held-out validation rows of each nested ``n`` subset only; decoding rows still use the full nested subset.
The convenience wrapper ``bin/study_h_decoding_categorical_twofig_val_hellinger.py`` defaults to that split.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = _repo_root / "bin"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import global_setting  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from global_setting import DATA_DIR

from fisher import h_binned_visualization as vhb
from fisher import h_decoding_convergence as conv
from fisher.decoding_native_x import decoding_x_train_all_from_native, load_native_bundle_for_pr_gt_decoding
from fisher.hellinger_gt import hellinger_sq_gaussian_diag
from fisher.h_decoding_convergence_methods import SweepSubset, prepare_categorical_binning_for_convergence
from fisher.shared_dataset_io import SharedDatasetBundle
from fisher.h_decoding_twofig import (
    _decode_accuracy_color_limits,
    _matrix_nmse_offdiag,
    _render_corr_vs_n_panel,
    _render_method_sweep_panel,
    _render_nmse_vs_n_panel,
    _render_row_n_training_losses_panel,
    _sanitize_row_label,
    _save_figure_svg,
    _theta_axis_tick_labels,
)
from fisher.h_matrix import HMatrixEstimator
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device

_METHOD_ALIASES: dict[str, str] = {
    "x-flow": "x_flow",
    "x_flow": "x_flow",
    "xflow": "x_flow",
    "binary-classification": "binary_classifier",
    "binary_classification": "binary_classifier",
    "binary-classifier": "binary_classifier",
    "binary_classifier": "binary_classifier",
    "classifier": "binary_classifier",
    "linear-x-flow-t": "linear_x_flow_t",
    "linear_x_flow_t": "linear_x_flow_t",
    "xflow-sir-lrank": "xflow_sir_lrank",
    "xflow_sir_lrank": "xflow_sir_lrank",
    "bin_gaussian": "bin_gaussian",
    "bin-gaussian": "bin_gaussian",
    "binned_gaussian": "bin_gaussian",
    "binned-gaussian": "bin_gaussian",
}
_DEFAULT_METHODS = ("x_flow", "binary_classifier", "linear_x_flow_t", "xflow_sir_lrank")
_SUPPORTED_METHODS_HELP = ", ".join(_DEFAULT_METHODS + ("bin_gaussian",))


@functools.lru_cache(maxsize=1)
def _debug_categorical_module() -> ModuleType:
    """Load ``bin/debug_categorical_xflow_llr.py`` for shared training / figure helpers."""
    path = _repo_root / "bin" / "debug_categorical_xflow_llr.py"
    spec = importlib.util.spec_from_file_location("_dbg_cat_xflow_llr", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_methods(methods: str) -> list[str]:
    toks = [t.strip() for t in str(methods).split(",") if t.strip()]
    if not toks:
        raise ValueError("--methods must contain at least one method.")
    out: list[str] = []
    for tok in toks:
        key = tok.strip().lower()
        norm = _METHOD_ALIASES.get(key)
        if norm is None:
            raise ValueError(f"Unknown method {tok!r}; valid methods: {_SUPPORTED_METHODS_HELP}")
        if norm not in out:
            out.append(norm)
    return out


def _default_dataset_npz(num_categories: int) -> Path:
    return _repo_root / "data" / f"random_mog_categorical_xdim2_k{int(num_categories)}" / "random_mog_categorical.npz"


def _abs_without_resolving_symlinks(path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root / p


def _default_pr_output_npz(native_npz: Path, pr_dim: int) -> Path:
    return _abs_without_resolving_symlinks(native_npz).parent / f"pr_xdim{int(pr_dim)}" / "random_mog_categorical_pr.npz"


def _run(cmd: list[str]) -> None:
    print("[cat-twofig] running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root), check=True)


def _ensure_dataset(args: argparse.Namespace) -> None:
    ds = Path(args.dataset_npz)
    ds.parent.mkdir(parents=True, exist_ok=True)
    if args.force_regenerate or not ds.exists():
        _run(
            [
                sys.executable,
                "bin/make_dataset.py",
                "--dataset-family",
                "random_mog_categorical",
                "--num-categories",
                str(int(args.num_categories)),
                "--n-total",
                str(int(args.n_total)),
                "--output-npz",
                str(ds),
            ]
        )
    else:
        print(f"[cat-twofig] using existing 2D dataset NPZ: {ds}", flush=True)


def _ensure_pr_projected_npz(args: argparse.Namespace, *, native_npz: Path, pr_out: Path) -> None:
    pr_out = Path(pr_out)
    native_npz = Path(native_npz).resolve()
    pr_out.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.force_regenerate) and pr_out.is_file():
        pr_out.unlink()
    if pr_out.is_file():
        print(f"[cat-twofig] using existing PR-projected NPZ: {pr_out}", flush=True)
        return
    cmd: list[str] = [
        sys.executable,
        str(_repo_root / "bin" / "project_dataset_pr_autoencoder.py"),
        "--input-npz",
        str(native_npz),
        "--output-npz",
        str(pr_out.resolve()),
        "--h-dim",
        str(int(args.pr_dim)),
        "--device",
        str(args.device),
        "--allow-non-randamp-sqrtd",
        "--cache-dir",
        str(args.pr_cache_dir),
    ]
    if bool(args.pr_use_cache):
        cmd.append("--use-cache")
    if bool(args.pr_skip_viz):
        cmd.append("--skip-viz")
    if args.pr_hidden1 is not None:
        cmd.extend(["--pr-hidden1", str(int(args.pr_hidden1))])
    if args.pr_hidden2 is not None:
        cmd.extend(["--pr-hidden2", str(int(args.pr_hidden2))])
    if args.pr_train_samples is not None:
        cmd.extend(["--pr-train-samples", str(int(args.pr_train_samples))])
    if args.pr_train_epochs is not None:
        cmd.extend(["--pr-train-epochs", str(int(args.pr_train_epochs))])
    if args.pr_train_batch_size is not None:
        cmd.extend(["--pr-train-batch-size", str(int(args.pr_train_batch_size))])
    if args.pr_train_lr is not None:
        cmd.extend(["--pr-train-lr", str(float(args.pr_train_lr))])
    if args.pr_lambda_pr is not None:
        cmd.extend(["--pr-lambda-pr", str(float(args.pr_lambda_pr))])
    if args.pr_eps is not None:
        cmd.extend(["--pr-eps", str(float(args.pr_eps))])
    print("[cat-twofig] running PR projection: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root), check=True)


def _validation_only_work_sweep_subset(subset_w: SweepSubset) -> SweepSubset:
    """Build a :class:`SweepSubset` whose ``bundle.x_all`` / ``bin_all`` are validation rows only.

    Used for ``--hellinger-eval-split validation`` with :func:`conv._binned_gaussian_hellinger_sq`,
    which reads ``subset.bundle.x_all`` and ``subset.bin_all``.
    """
    b = subset_w.bundle
    tv = np.asarray(b.theta_validation, dtype=np.float64)
    xv = np.asarray(b.x_validation, dtype=np.float64)
    binv = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)
    if tv.ndim == 1:
        tv = tv.reshape(-1, 1)
    elif tv.ndim != 2:
        raise ValueError("theta_validation must be 1D or 2D.")
    n = int(xv.shape[0])
    if n < 1:
        raise ValueError("validation split is empty.")
    if int(binv.shape[0]) != n:
        raise ValueError("bin_validation length must match x_validation.")
    idx = np.arange(n, dtype=np.int64)
    empty = np.arange(0, dtype=np.int64)
    meta = dict(b.meta)
    new_b = SharedDatasetBundle(
        meta=meta,
        theta_all=tv,
        x_all=xv,
        train_idx=empty,
        validation_idx=idx,
        theta_train=np.empty((0, tv.shape[1]), dtype=np.float64),
        x_train=np.empty((0, xv.shape[1]), dtype=np.float64),
        theta_validation=tv,
        x_validation=xv,
    )
    return SweepSubset(bundle=new_b, bin_all=binv, bin_train=binv, bin_validation=binv)


def compute_true_conditional_loglik_matrix(x_all: np.ndarray, theta_all: np.ndarray, meta: dict) -> np.ndarray:
    gen_ds = build_dataset_from_meta(dict(meta))
    n = int(np.asarray(x_all).shape[0])
    x_all = np.asarray(x_all, dtype=np.float64)
    theta_all = np.asarray(theta_all, dtype=np.float64)
    true_c = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        th_col = np.tile(theta_all[j : j + 1], (n, 1))
        true_c[:, j] = gen_ds.log_p_x_given_theta(x_all, th_col)
    return true_c


def hellinger_gt_sq_category_matrix(gen_ds: object) -> np.ndarray:
    means = np.asarray(getattr(gen_ds, "_mog_means"), dtype=np.float64)
    variances = np.asarray(getattr(gen_ds, "_mog_variances"), dtype=np.float64)
    k = int(means.shape[0])
    if int(variances.shape[0]) != k or int(means.shape[1]) != int(variances.shape[1]):
        raise ValueError("Inconsistent _mog_means / _mog_variances on dataset.")
    h2 = np.zeros((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            h2[a, b] = hellinger_sq_gaussian_diag(means[a], variances[a], means[b], variances[b])
    np.fill_diagonal(h2, 0.0)
    np.clip(h2, 0.0, 1.0, out=h2)
    return h2


def h_sq_directed_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    return HMatrixEstimator.compute_h_directed(np.asarray(delta_l, dtype=np.float64))


def h_sq_sym_sample_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    h_dir = h_sq_directed_from_delta_l(delta_l)
    return HMatrixEstimator.symmetrize(h_dir)


def h_sq_category_from_sample_directed(
    h_directed: np.ndarray,
    category_labels: np.ndarray,
    *,
    k_cat: int,
) -> np.ndarray:
    """Aggregate directed $H^2$ to categories, symmetrize, zero diagonal (debug script recipe)."""
    dbg = _debug_categorical_module()
    return np.asarray(
        dbg._h_sq_category_from_sample_directed(h_directed, category_labels, k_cat=int(k_cat)),
        dtype=np.float64,
    )


def binary_classifier_hellinger_sqrt(
    args: argparse.Namespace,
    *,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_all: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> np.ndarray:
    """Category-level ``sqrt(H^2)`` from pairwise binary-classifier LLR estimates."""
    dbg = _debug_categorical_module()
    result = dbg._train_binary_classifier_delta(
        args,
        x_train=np.asarray(x_train, dtype=np.float64),
        bins_train=np.asarray(bins_train, dtype=np.int64),
        x_all=np.asarray(x_all, dtype=np.float64),
        bins_all=np.asarray(bins_all, dtype=np.int64),
        k_cat=int(k_cat),
    )
    delta = np.asarray(result["delta_l"], dtype=np.float64)
    h_dir = h_sq_directed_from_delta_l(delta)
    h_cat_sq = h_sq_category_from_sample_directed(h_dir, np.asarray(bins_all, dtype=np.int64), k_cat=int(k_cat))
    h_sqrt = np.sqrt(np.clip(h_cat_sq, 0.0, 1.0))
    np.fill_diagonal(h_sqrt, 0.0)
    return h_sqrt


def _as_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}.")
    return arr


def _nmse_h_vs_gt(h_sweep: np.ndarray, h_gt_sqrt: np.ndarray) -> np.ndarray:
    h_sw = np.asarray(h_sweep, dtype=np.float64)
    h_gt = np.asarray(h_gt_sqrt, dtype=np.float64)
    if h_sw.ndim != 4:
        raise ValueError(f"h_sweep must be 4D; got {h_sw.shape}.")
    n_m, n_cols = int(h_sw.shape[0]), int(h_sw.shape[1])
    out = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    h_gt_imp = vhb.impute_offdiag_nan_mean(h_gt)
    for i in range(n_m):
        for j in range(n_cols):
            out[i, j] = _matrix_nmse_offdiag(
                vhb.impute_offdiag_nan_mean(np.asarray(h_sw[i, j], dtype=np.float64)),
                h_gt_imp,
            )
    return out


def _nmse_decode_vs_ref(decode_sweep: np.ndarray, decode_ref: np.ndarray) -> np.ndarray:
    d_sw = np.asarray(decode_sweep, dtype=np.float64)
    d_ref = np.asarray(decode_ref, dtype=np.float64)
    if d_sw.ndim != 3:
        raise ValueError(f"decode_sweep must be 3D; got {d_sw.shape}.")
    n_cols = int(d_sw.shape[0])
    out = np.full((n_cols,), np.nan, dtype=np.float64)
    d_ref_imp = vhb.impute_offdiag_nan_mean(d_ref)
    for j in range(n_cols):
        out[j] = _matrix_nmse_offdiag(
            vhb.impute_offdiag_nan_mean(np.asarray(d_sw[j], dtype=np.float64)),
            d_ref_imp,
        )
    return out


def _render_cat_gt_panel(
    *,
    h_gt_sqrt: np.ndarray,
    decode_ref: np.ndarray,
    decode_hellinger_ref_sqrt: np.ndarray | None,
    n_ref: int,
    k_cat: int,
    theta_centers: np.ndarray,
    out_svg_path: str,
    decode_sweep_for_decode_limits: np.ndarray | None = None,
) -> str:
    n_cols = 3 if decode_hellinger_ref_sqrt is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(3.1 * n_cols, 3.2), squeeze=False)
    tick_pos, tick_labs, axis_label = _theta_axis_tick_labels(theta_centers, int(k_cat))
    x_rot = 45 if len(tick_pos) > 6 else 0

    def _one(ax: Any, mat: np.ndarray, title: str, vmin: float, vmax: float) -> None:
        im = ax.imshow(
            np.asarray(mat, dtype=np.float64),
            vmin=float(vmin),
            vmax=float(vmax),
            cmap="viridis",
            aspect="equal",
            origin="lower",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=11)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labs, fontsize=11)
        ax.set_xlabel(axis_label, fontsize=11)
        conv._matrix_axes_show_top_right_spines(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=11)

    ax_h = axes[0, 0]
    _one(ax_h, h_gt_sqrt, r"Analytic GT $\sqrt{H^2}$ (category)", 0.0, 1.0)
    ax_h.set_ylabel("category", fontsize=11)
    if decode_hellinger_ref_sqrt is not None:
        ax_ch = axes[0, 1]
        _one(ax_ch, decode_hellinger_ref_sqrt, r"Classifier LLR $\sqrt{H^2}$", 0.0, 1.0)
    ax_c = axes[0, n_cols - 1]
    lim_arrays: list[np.ndarray] = [np.asarray(decode_ref, dtype=np.float64)]
    if decode_sweep_for_decode_limits is not None:
        dsw = np.asarray(decode_sweep_for_decode_limits, dtype=np.float64)
        for j in range(int(dsw.shape[0])):
            lim_arrays.append(np.asarray(dsw[j], dtype=np.float64))
    vmin_c, vmax_c = _decode_accuracy_color_limits(*lim_arrays)
    _one(ax_c, decode_ref, f"Pairwise decoding (n_ref={int(n_ref)})", vmin_c, vmax_c)
    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _load_cached_results(output_dir: str) -> dict[str, Any]:
    zpath = os.path.join(output_dir, "h_decoding_categorical_twofig_results.npz")
    if not os.path.isfile(zpath):
        raise FileNotFoundError(
            f"Missing {zpath}. Run without --visualization-only first, or pass --output-dir from a prior run."
        )
    with np.load(zpath, allow_pickle=True) as z:
        bundle = {k: z[k] for k in z.files}
    need = (
        "n",
        "n_ref",
        "method_names",
        "num_categories",
        "theta_bin_centers",
        "h_gt_sqrt",
        "decode_ref",
        "h_sqrt_sweep",
        "decode_sweep",
        "corr_h_vs_gt",
        "nmse_h_vs_gt",
        "corr_decode_vs_ref",
        "nmse_decode_vs_ref",
    )
    for k in need:
        if k not in bundle:
            raise KeyError(f"Cached NPZ missing key {k!r}: {zpath}")
    return bundle


def _validate_cached_cli(
    args: argparse.Namespace,
    cached: dict[str, Any],
    methods: list[str],
    ns: list[int],
) -> None:
    n_arr = np.asarray(cached["n"], dtype=np.int64).ravel()
    if n_arr.size != len(ns) or not np.array_equal(n_arr, np.asarray(ns, dtype=np.int64)):
        raise ValueError(
            f"--n-list {ns} does not match cached n={n_arr.tolist()}. "
            "Use the same --n-list as the run that produced the NPZ."
        )
    if int(np.asarray(cached["n_ref"]).reshape(-1)[0]) != int(args.n_ref):
        raise ValueError(
            f"Cached n_ref does not match --n-ref={args.n_ref}."
        )
    if int(np.asarray(cached["num_categories"]).reshape(-1)[0]) != int(args.num_categories):
        raise ValueError("Cached num_categories does not match --num-categories.")
    names = [str(x) for x in np.asarray(cached["method_names"], dtype=object).tolist()]
    if names != methods:
        raise ValueError(f"Cached method_names {names} do not match CLI methods {methods}.")
    if "native_dataset_npz" in cached:
        raw = cached["native_dataset_npz"]
        ds_cached = str(np.asarray(raw, dtype=object).reshape(-1)[0]) if np.size(raw) else None
        if ds_cached:
            want = str(Path(args.dataset_npz).resolve())
            got = str(Path(ds_cached).resolve())
            same = False
            if os.path.isfile(got) and os.path.isfile(want):
                try:
                    same = os.path.samefile(got, want)
                except OSError:
                    same = False
            if not same and got != want:
                raise ValueError(
                    f"--dataset-npz {args.dataset_npz!r} does not match cached native_dataset_npz={ds_cached!r}."
                )
    want_h = str(getattr(args, "hellinger_eval_split", "all")).strip().lower()
    if "hellinger_eval_split" in cached:
        got_h = str(np.asarray(cached["hellinger_eval_split"], dtype=object).reshape(-1)[0])
        if got_h != want_h:
            raise ValueError(
                f"--hellinger-eval-split {want_h!r} does not match cached hellinger_eval_split={got_h!r}."
            )
    elif want_h != "all":
        raise ValueError(
            "Cached results NPZ has no hellinger_eval_split (older run). "
            "Use --hellinger-eval-split all with --visualization-only, or re-run the study."
        )


def _run_visualization_only(args: argparse.Namespace, methods: list[str], ns: list[int]) -> None:
    cached = _load_cached_results(args.output_dir)
    _validate_cached_cli(args, cached, methods, ns)
    k_cat = int(np.asarray(cached["num_categories"]).reshape(-1)[0])
    theta_centers = np.asarray(cached["theta_bin_centers"], dtype=np.float64)
    h_gt_sqrt = np.asarray(cached["h_gt_sqrt"], dtype=np.float64)
    decode_ref = np.asarray(cached["decode_ref"], dtype=np.float64)
    decode_h_ref = (
        np.asarray(cached["decode_hellinger_ref_sqrt"], dtype=np.float64)
        if "decode_hellinger_ref_sqrt" in cached
        else None
    )
    h_sw = np.asarray(cached["h_sqrt_sweep"], dtype=np.float64)
    dec_sw = np.asarray(cached["decode_sweep"], dtype=np.float64)
    dec_h_sw = (
        np.asarray(cached["decode_hellinger_sweep_sqrt"], dtype=np.float64)
        if "decode_hellinger_sweep_sqrt" in cached
        else None
    )
    corr_h = np.asarray(cached["corr_h_vs_gt"], dtype=np.float64)
    corr_d = np.asarray(cached["corr_decode_vs_ref"], dtype=np.float64).ravel()
    corr_dh = (
        np.asarray(cached["corr_decode_hellinger_vs_gt"], dtype=np.float64).ravel()
        if "corr_decode_hellinger_vs_gt" in cached
        else None
    )
    nmse_h = np.asarray(cached["nmse_h_vs_gt"], dtype=np.float64)
    nmse_d = np.asarray(cached["nmse_decode_vs_ref"], dtype=np.float64).ravel()
    nmse_dh = (
        np.asarray(cached["nmse_decode_hellinger_vs_gt"], dtype=np.float64).ravel()
        if "nmse_decode_hellinger_vs_gt" in cached
        else None
    )
    zdummy = np.zeros_like(corr_h)
    sweep_svg = _render_method_sweep_panel(
        row_labels=methods,
        h_sweep=h_sw,
        clf_sweep_shared=dec_sw,
        clf_hellinger_sweep_shared=dec_h_sw,
        n_list=ns,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_sweep.svg"),
        n_bins=k_cat,
        theta_centers=theta_centers,
        clf_ref_decode_limits=np.asarray(decode_ref, dtype=np.float64),
    )
    gt_svg = _render_cat_gt_panel(
        h_gt_sqrt=h_gt_sqrt,
        decode_ref=decode_ref,
        decode_hellinger_ref_sqrt=decode_h_ref,
        n_ref=int(args.n_ref),
        k_cat=k_cat,
        theta_centers=theta_centers,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_gt.svg"),
        decode_sweep_for_decode_limits=dec_sw,
    )
    corr_svg = _render_corr_vs_n_panel(
        row_labels=methods,
        n_list=ns,
        corr_h=corr_h,
        corr_decode_shared=corr_d,
        corr_decode_hellinger_shared=corr_dh,
        corr_hellinger_lb=zdummy,
        corr_hellinger_ub=zdummy,
        show_hellinger_bounds=False,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_corr_vs_n.svg"),
    )
    nmse_svg = _render_nmse_vs_n_panel(
        row_labels=methods,
        n_list=ns,
        nmse_h=nmse_h,
        nmse_decode_shared=nmse_d,
        nmse_decode_hellinger_shared=nmse_dh,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_nmse_vs_n.svg"),
    )
    loss_root = os.path.join(str(args.output_dir), "training_losses")
    if not os.path.isdir(loss_root):
        raise FileNotFoundError(
            "visualization-only requires per-(method, n) curves under training_losses/ "
            f"(missing {os.path.abspath(loss_root)}). Re-run without --visualization-only first."
        )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=methods,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_training_losses_panel.svg"),
    )
    out_npz = os.path.join(args.output_dir, "h_decoding_categorical_twofig_results.npz")
    summary_path = os.path.join(args.output_dir, "h_decoding_categorical_twofig_summary.txt")
    _write_summary(
        summary_path,
        args=args,
        out_npz=os.path.abspath(out_npz),
        sweep_svg=os.path.abspath(sweep_svg),
        gt_svg=os.path.abspath(gt_svg),
        corr_svg=os.path.abspath(corr_svg),
        nmse_svg=os.path.abspath(nmse_svg),
        visualization_only=True,
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
    )
    print("[cat-twofig] Saved (visualization-only):", flush=True)
    for p in (sweep_svg, gt_svg, corr_svg, nmse_svg, loss_panel_svg, summary_path):
        print(f"  - {os.path.abspath(p)}", flush=True)


def _write_summary(
    path: str,
    *,
    args: argparse.Namespace,
    out_npz: str,
    sweep_svg: str,
    gt_svg: str,
    corr_svg: str,
    nmse_svg: str,
    visualization_only: bool,
    loss_panel_svg: str | None = None,
    training_losses_root: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_categorical_twofig\n")
        f.write(f"visualization_only: {bool(visualization_only)}\n")
        f.write(f"output_dir: {os.path.abspath(args.output_dir)}\n")
        f.write(f"dataset_npz: {os.path.abspath(str(args.dataset_npz))}\n")
        f.write(f"num_categories: {int(args.num_categories)}\n")
        f.write(f"n_list: {args.n_list}\n")
        f.write(f"n_ref: {int(args.n_ref)}\n")
        f.write(f"methods: {args.methods}\n")
        f.write(f"pr_project: {bool(args.pr_project)}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"results_npz: {out_npz}\n")
        f.write(f"h_decoding_categorical_twofig_sweep.svg: {sweep_svg}\n")
        f.write(f"h_decoding_categorical_twofig_gt.svg: {gt_svg}\n")
        f.write(f"h_decoding_categorical_twofig_corr_vs_n.svg: {corr_svg}\n")
        f.write(f"h_decoding_categorical_twofig_nmse_vs_n.svg: {nmse_svg}\n")
        if loss_panel_svg:
            f.write(f"h_decoding_categorical_twofig_training_losses_panel.svg: {loss_panel_svg}\n")
        if training_losses_root:
            f.write(f"training_losses_root: {training_losses_root}\n")
        f.write(f"hellinger_eval_split: {str(getattr(args, 'hellinger_eval_split', 'all'))}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-categories", type=int, default=2)
    p.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Native 2D NPZ. Default: data/random_mog_categorical_xdim2_kK/random_mog_categorical.npz",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--n-total", type=int, default=50000)
    p.add_argument("--n-list", type=str, default="80,200,400,600")
    p.add_argument("--n-ref", type=int, default=600)
    p.add_argument("--run-seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--methods", type=str, default=",".join(_DEFAULT_METHODS))
    p.add_argument("--visualization-only", action="store_true")
    p.add_argument(
        "--hellinger-eval-split",
        type=str,
        default="all",
        choices=("all", "validation"),
        help="For learned H and GT-LLR diagnostics: use all nested-n rows (default) or validation split only.",
    )
    p.add_argument(
        "--no-scatter-diagnostics",
        action="store_true",
        help="Skip LLR and H^2 scatter figures (llr_est_vs_true_all, hellinger_est_vs_gt_all).",
    )

    p.add_argument("--flow-arch", type=str, default="mlp", choices=["mlp", "film", "film_fourier"])
    p.add_argument("--flow-epochs", type=int, default=10000)
    p.add_argument("--flow-batch-size", type=int, default=256)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument("--flow-scheduler", type=str, default="cosine")
    p.add_argument("--flow-fm-t-eps", type=float, default=0.05)
    p.add_argument("--flow-early-patience", type=int, default=1000)
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--no-flow-restore-best", dest="flow_restore_best", action="store_false")
    p.set_defaults(flow_restore_best=True)
    p.add_argument("--flow-ode-steps", type=int, default=64)
    p.add_argument("--flow-likelihood-exact-divergence", action="store_true")
    p.add_argument("--h-batch-size", type=int, default=65536)
    p.add_argument("--clf-min-class-count", type=int, default=5)
    p.add_argument("--clf-random-state", type=int, default=-1)
    p.add_argument("--clf-max-iter", type=int, default=1000)
    p.add_argument("--lxfs-epochs", type=int, default=2000)
    p.add_argument("--lxfs-batch-size", type=int, default=1024)
    p.add_argument("--lxfs-lr", type=float, default=1e-3)
    p.add_argument("--lxfs-weight-decay", type=float, default=0.0)
    p.add_argument("--lxfs-hidden-dim", type=int, default=128)
    p.add_argument("--lxfs-depth", type=int, default=3)
    p.add_argument("--lxfs-path-schedule", type=str, default="cosine")
    p.add_argument("--lxfs-t-eps", type=float, default=0.05)
    p.add_argument("--lxfs-early-patience", type=int, default=1000)
    p.add_argument("--lxfs-early-min-delta", type=float, default=1e-4)
    p.add_argument("--lxfs-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--lxfs-weight-ema-decay", type=float, default=0.9)
    p.add_argument("--lxfs-max-grad-norm", type=float, default=10.0)
    p.add_argument("--lxfs-quadrature-steps", type=int, default=64)
    p.add_argument("--lxfs-pair-batch-size", type=int, default=65536)
    p.add_argument("--lxfs-solve-jitter", type=float, default=1e-6)
    p.add_argument("--lxf-low-rank-dim", type=int, default=1)
    p.add_argument("--lxf-low-rank-divergence-estimator", type=str, default="hutchinson")
    p.add_argument("--lxf-hutchinson-probes", type=int, default=1)
    p.add_argument("--lxf-nlpca-ode-steps", type=int, default=32)
    p.add_argument("--sir-num-bins", type=int, default=10)
    p.add_argument("--sir-ridge", type=float, default=1e-6)
    p.add_argument("--no-lxf-restore-best", dest="lxf_restore_best", action="store_false")
    p.set_defaults(lxf_restore_best=True)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--pr-project", action="store_true")
    p.add_argument("--pr-dim", type=int, default=10)
    p.add_argument("--pr-output-npz", type=Path, default=None)
    p.add_argument("--pr-use-cache", action="store_true")
    p.add_argument("--pr-cache-dir", type=str, default="data/pr_autoencoder_cache")
    p.add_argument("--pr-train-epochs", type=int, default=None)
    p.add_argument("--pr-train-samples", type=int, default=None)
    p.add_argument("--pr-train-batch-size", type=int, default=None)
    p.add_argument("--pr-train-lr", type=float, default=None)
    p.add_argument("--pr-lambda-pr", type=float, default=None)
    p.add_argument("--pr-eps", type=float, default=None)
    p.add_argument("--pr-hidden1", type=int, default=None)
    p.add_argument("--pr-hidden2", type=int, default=None)
    p.add_argument("--pr-skip-viz", action="store_true")
    p.add_argument(
        "--decode-source-npz",
        type=str,
        default="",
        help="Optional override for native NPZ when PR-embedded (GT decoding on native 2D x).",
    )
    return p


def _save_method_training_loss_npz(out_path: str | Path, *, method_name: str, result: dict[str, Any]) -> None:
    """Write per-(method, n) curves compatible with :func:`conv._load_per_n_training_loss_npz`."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    train_out = result.get("train_out")
    if train_out:
        tr = np.asarray(train_out.get("train_losses", []), dtype=np.float64).ravel()
        va = np.asarray(train_out.get("val_losses", []), dtype=np.float64).ravel()
        em = np.asarray(train_out.get("val_monitor_losses", []), dtype=np.float64).ravel()
    else:
        tr = np.asarray([], dtype=np.float64)
        va = np.asarray([], dtype=np.float64)
        em = np.asarray([], dtype=np.float64)
    np.savez_compressed(
        str(p),
        theta_field_method=np.asarray([str(method_name)], dtype=object),
        prior_enable=np.bool_(False),
        score_train_losses=tr,
        score_val_losses=va,
        score_val_monitor_losses=em,
    )


def _train_one_method(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    method_name: str,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    bins_train: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> dict[str, Any]:
    dbg = _debug_categorical_module()
    if method_name == "x_flow":
        return dbg._train_x_flow_delta(
            args,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
        )
    if method_name == "binary_classifier":
        return dbg._train_binary_classifier_delta(
            args,
            x_train=x_train,
            bins_train=bins_train,
            x_all=x_all,
            bins_all=bins_all,
            k_cat=k_cat,
        )
    if method_name in ("linear_x_flow_t", "xflow_sir_lrank"):
        return dbg._train_linear_x_flow_delta(
            args,
            method_name=method_name,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
        )
    raise RuntimeError(f"Unhandled method {method_name!r}")


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    methods = parse_methods(str(args.methods))
    if int(args.num_categories) < 2:
        raise ValueError("--num-categories must be >= 2.")
    if (not bool(args.visualization_only)) and str(args.device).strip().lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA machine or pass --device cuda when available.")

    ns = conv._parse_n_list(str(args.n_list))
    if not ns:
        raise ValueError("--n-list produced an empty list.")
    if max(ns) > int(args.n_ref):
        raise ValueError(f"Require max(n-list) <= n-ref; got max(n_list)={max(ns)} n_ref={args.n_ref}.")

    if args.dataset_npz is None:
        args.dataset_npz = _default_dataset_npz(int(args.num_categories))
    args.dataset_npz = _abs_without_resolving_symlinks(Path(args.dataset_npz))
    if args.output_dir is None:
        args.output_dir = Path(DATA_DIR) / "h_decoding_categorical_twofig"
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.visualization_only):
        _run_visualization_only(args, methods, ns)
        return

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
        if int(args.pr_dim) <= native_x_dim:
            raise ValueError(f"--pr-dim must exceed native x_dim={native_x_dim}; got {args.pr_dim}.")
        pr_out = args.pr_output_npz
        if pr_out is None:
            pr_out = _default_pr_output_npz(native_npz, int(args.pr_dim))
        else:
            pr_out = _abs_without_resolving_symlinks(Path(pr_out))
        pr_out_resolved = Path(pr_out)
        _ensure_pr_projected_npz(args, native_npz=native_npz, pr_out=pr_out_resolved)
        work_bundle = load_shared_dataset_npz(pr_out_resolved)
        meta_work = dict(work_bundle.meta)
        if int(work_bundle.x_all.shape[0]) != int(native_bundle.x_all.shape[0]):
            raise ValueError("Native and projected NPZ row counts disagree.")
        nt = np.asarray(native_bundle.theta_all, dtype=np.float64)
        wt = np.asarray(work_bundle.theta_all, dtype=np.float64)
        if nt.shape != wt.shape or float(np.max(np.abs(nt - wt))) > 1e-5:
            raise ValueError("Native vs projected theta_all mismatch.")
    else:
        work_bundle = native_bundle

    n_pool = int(native_bundle.theta_all.shape[0])
    if int(args.n_ref) > n_pool:
        raise ValueError(f"--n-ref={args.n_ref} exceeds n_total={n_pool}.")
    need = max(int(args.n_ref), max(ns))
    if n_pool < need:
        raise ValueError(f"Dataset n_total={n_pool} < max(n_ref, max(n_list))={need}.")

    he_split = str(getattr(args, "hellinger_eval_split", "all")).strip().lower()
    if he_split not in ("all", "validation"):
        raise ValueError("--hellinger-eval-split must be 'all' or 'validation'.")
    if he_split == "validation" and float(meta_work.get("train_frac", 0.7)) >= 1.0:
        raise ValueError(
            "--hellinger-eval-split validation requires train_frac < 1 in dataset meta; "
            f"got train_frac={meta_work.get('train_frac')!r}."
        )

    k_cat = int(native_meta.get("num_categories", int(args.num_categories)))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(native_bundle.theta_all, k_cat)
    theta_centers = np.arange(k_cat, dtype=np.float64).reshape(-1, 1)

    rng = np.random.default_rng(int(args.run_seed))
    perm = rng.permutation(n_pool)

    gen_ds_h = build_dataset_from_meta(dict(native_meta))
    hellinger_gt_sq = hellinger_gt_sq_category_matrix(gen_ds_h)
    h_gt_sqrt = np.sqrt(np.clip(hellinger_gt_sq, 0.0, 1.0))
    np.fill_diagonal(h_gt_sqrt, 0.0)

    clf_rs = int(args.run_seed) if int(args.clf_random_state) < 0 else int(args.clf_random_state)
    subset_ref = conv._subset_bundle(work_bundle, perm, int(args.n_ref), meta_work, bin_idx_all=bin_idx_all)
    decode_gt_x_train: np.ndarray | None = None
    decode_gt_x_all: np.ndarray | None = None
    if bool(meta_work.get("pr_autoencoder_embedded")):
        override_npz = str(getattr(args, "decode_source_npz", "") or "").strip() or None
        native_gt, tried_gt = load_native_bundle_for_pr_gt_decoding(
            work_bundle,
            meta_work,
            str(pr_out_resolved if pr_out_resolved is not None else native_npz),
            override_npz,
        )
        if native_gt is None:
            msg = "\n  ".join(tried_gt) if tried_gt else "(no candidates)"
            raise FileNotFoundError(
                "PR-embedded dataset: could not load native NPZ for GT decoding. "
                f"Pass --decode-source-npz. Tried:\n  {msg}"
            )
        decode_gt_x_train, decode_gt_x_all = decoding_x_train_all_from_native(
            native_gt,
            perm,
            int(args.n_ref),
            meta_work,
        )
    decode_ref = conv._pairwise_clf_from_bundle(
        args=args,
        meta=meta_work,
        subset=subset_ref,
        output_dir=str(args.output_dir / "reference"),
        n_bins=k_cat,
        clf_min_class_count=int(args.clf_min_class_count),
        clf_random_state=clf_rs,
        decode_x_train=decode_gt_x_train,
        decode_x_all=decode_gt_x_all,
    )
    decode_h_x_train = subset_ref.bundle.x_train if decode_gt_x_train is None else decode_gt_x_train
    decode_h_x_all = subset_ref.bundle.x_all if decode_gt_x_all is None else decode_gt_x_all
    decode_hellinger_ref_sqrt = binary_classifier_hellinger_sqrt(
        args,
        x_train=decode_h_x_train,
        bins_train=subset_ref.bin_train,
        x_all=decode_h_x_all,
        bins_all=subset_ref.bin_all,
        k_cat=k_cat,
    )

    n_m = len(methods)
    n_cols = len(ns)
    h_sqrt_sweep = np.full((n_m, n_cols, k_cat, k_cat), np.nan, dtype=np.float64)
    wall_s = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    corr_h = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    nmse_h = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    llr_pearson_offdiag = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    hellinger_pearson_offdiag_cat = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    decode_sweep_list: list[np.ndarray] = []
    decode_hellinger_sweep_list: list[np.ndarray] = []

    dev = require_device(str(args.device))
    torch.manual_seed(int(args.run_seed))
    np.random.seed(int(args.run_seed))

    dbg = _debug_categorical_module()
    scatter_deltas: dict[str, np.ndarray] = {}
    scatter_cat_h2: dict[str, np.ndarray] = {}
    scatter_true_delta: np.ndarray | None = None
    n_scatter = max(ns)
    out_dir = str(args.output_dir)
    loss_root = os.path.join(out_dir, "training_losses")

    for j, n in enumerate(ns):
        subset_w = conv._subset_bundle(work_bundle, perm, int(n), meta_work, bin_idx_all=bin_idx_all)
        subset_n = conv._subset_bundle(native_bundle, perm, int(n), meta_work, bin_idx_all=bin_idx_all)
        bins_n = np.asarray(subset_w.bin_all, dtype=np.int64).reshape(-1)
        x_native_all = np.asarray(subset_n.bundle.x_all, dtype=np.float64)
        train_theta = _as_2d(subset_w.bundle.theta_train)
        val_theta = _as_2d(subset_w.bundle.theta_validation)
        x_train = np.asarray(subset_w.bundle.x_train, dtype=np.float64)
        x_val = np.asarray(subset_w.bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(subset_w.bundle.x_all, dtype=np.float64)
        bins_train = np.asarray(bins_n[: int(subset_w.bundle.theta_train.shape[0])], dtype=np.int64)

        val_h = str(getattr(args, "hellinger_eval_split", "all")).strip().lower() == "validation"
        if val_h:
            n_val = int(subset_w.bundle.x_validation.shape[0])
            if n_val < 1:
                raise ValueError(f"n={int(n)}: empty validation split; cannot use --hellinger-eval-split validation.")
            theta_h = val_theta
            x_h = x_val
            x_native_h = np.asarray(subset_n.bundle.x_validation, dtype=np.float64)
            bins_h = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)
        else:
            theta_h = _as_2d(subset_w.bundle.theta_all)
            x_h = x_all
            x_native_h = x_native_all
            bins_h = bins_n

        true_c = compute_true_conditional_loglik_matrix(x_native_h, theta_h, native_meta)
        true_delta_l = HMatrixEstimator.compute_delta_l(true_c)

        clf_n = conv._pairwise_clf_from_bundle(
            args=args,
            meta=meta_work,
            subset=subset_w,
            output_dir=str(args.output_dir / "decode_shared" / f"n_{int(n):06d}"),
            n_bins=k_cat,
            clf_min_class_count=int(args.clf_min_class_count),
            clf_random_state=clf_rs,
        )
        decode_sweep_list.append(np.asarray(clf_n, dtype=np.float64))
        clf_h_n = binary_classifier_hellinger_sqrt(
            args,
            x_train=x_train,
            bins_train=bins_train,
            x_all=x_all,
            bins_all=bins_n,
            k_cat=k_cat,
        )
        decode_hellinger_sweep_list.append(np.asarray(clf_h_n, dtype=np.float64))

        for i, method_name in enumerate(methods):
            t0 = time.time()
            row_loss_dir = os.path.join(loss_root, _sanitize_row_label(method_name))
            os.makedirs(row_loss_dir, exist_ok=True)
            loss_npz_path = os.path.join(row_loss_dir, f"n_{int(n):06d}.npz")

            if method_name == "bin_gaussian":
                vf = float(
                    getattr(
                        args,
                        "flow_theta_reg_variance_floor",
                        getattr(args, "flow_x_reg_variance_floor", 1e-6),
                    )
                )
                subset_bg = _validation_only_work_sweep_subset(subset_w) if val_h else subset_w
                bg_h2 = conv._binned_gaussian_hellinger_sq(subset_bg, k_cat, variance_floor=vf)
                h_sqrt_sweep[i, j] = np.asarray(conv._sqrt_h_like(np.asarray(bg_h2, dtype=np.float64)), dtype=np.float64)
                np.fill_diagonal(h_sqrt_sweep[i, j], 0.0)
                wall_s[i, j] = time.time() - t0
                result = {"train_out": None}
                nan_delta = np.full(np.asarray(true_delta_l, dtype=np.float64).shape, np.nan, dtype=np.float64)
                m_llr = dbg._llr_comparison_metrics(nan_delta, true_delta_l)
                llr_pearson_offdiag[i, j] = float(m_llr["llr_pearson_r_offdiag"])
                m_h = dbg._hellinger_comparison_metrics_cat(bg_h2, hellinger_gt_sq)
                hellinger_pearson_offdiag_cat[i, j] = float(m_h["hellinger_pearson_r_offdiag_cat"])
                _save_method_training_loss_npz(loss_npz_path, method_name=method_name, result=result)
            else:
                result = _train_one_method(
                    args,
                    dev=dev,
                    method_name=method_name,
                    theta_train=train_theta,
                    x_train=x_train,
                    theta_val=val_theta,
                    x_val=x_val,
                    theta_all=theta_h,
                    x_all=x_h,
                    bins_train=bins_train,
                    bins_all=bins_h,
                    k_cat=k_cat,
                )
                _save_method_training_loss_npz(loss_npz_path, method_name=method_name, result=result)
                delta = np.asarray(result["delta_l"], dtype=np.float64)
                h_dir = h_sq_directed_from_delta_l(delta)
                h_cat_sq = h_sq_category_from_sample_directed(h_dir, bins_h, k_cat=k_cat)
                h_sqrt_sweep[i, j] = np.sqrt(np.clip(h_cat_sq, 0.0, 1.0))
                np.fill_diagonal(h_sqrt_sweep[i, j], 0.0)
                wall_s[i, j] = time.time() - t0

                m_llr = dbg._llr_comparison_metrics(delta, true_delta_l)
                llr_pearson_offdiag[i, j] = float(m_llr["llr_pearson_r_offdiag"])
                m_h = dbg._hellinger_comparison_metrics_cat(h_cat_sq, hellinger_gt_sq)
                hellinger_pearson_offdiag_cat[i, j] = float(m_h["hellinger_pearson_r_offdiag_cat"])

            corr_h[i, j] = vhb.matrix_corr_offdiag_pearson(
                vhb.impute_offdiag_nan_mean(h_sqrt_sweep[i, j]),
                vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64)),
            )
            nmse_h[i, j] = _matrix_nmse_offdiag(
                vhb.impute_offdiag_nan_mean(h_sqrt_sweep[i, j]),
                vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64)),
            )

            if int(n) == int(n_scatter):
                scatter_true_delta = np.asarray(true_delta_l, dtype=np.float64).copy()
                if method_name == "bin_gaussian":
                    scatter_cat_h2[method_name] = np.asarray(bg_h2, dtype=np.float64).copy()
                else:
                    scatter_deltas[method_name] = np.asarray(delta, dtype=np.float64).copy()
                    scatter_cat_h2[method_name] = np.asarray(h_cat_sq, dtype=np.float64).copy()

    decode_sweep = np.stack(decode_sweep_list, axis=0)
    decode_hellinger_sweep = np.stack(decode_hellinger_sweep_list, axis=0)
    corr_decode = np.full((n_cols,), np.nan, dtype=np.float64)
    nmse_decode = np.full((n_cols,), np.nan, dtype=np.float64)
    corr_decode_hellinger = np.full((n_cols,), np.nan, dtype=np.float64)
    nmse_decode_hellinger = np.full((n_cols,), np.nan, dtype=np.float64)
    decode_ref_imp = vhb.impute_offdiag_nan_mean(np.asarray(decode_ref, dtype=np.float64))
    h_gt_imp = vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64))
    for j in range(n_cols):
        corr_decode[j] = vhb.matrix_corr_offdiag_pearson(
            vhb.impute_offdiag_nan_mean(np.asarray(decode_sweep[j], dtype=np.float64)),
            decode_ref_imp,
        )
        corr_decode_hellinger[j] = vhb.matrix_corr_offdiag_pearson(
            vhb.impute_offdiag_nan_mean(np.asarray(decode_hellinger_sweep[j], dtype=np.float64)),
            h_gt_imp,
        )
        nmse_decode_hellinger[j] = _matrix_nmse_offdiag(
            vhb.impute_offdiag_nan_mean(np.asarray(decode_hellinger_sweep[j], dtype=np.float64)),
            h_gt_imp,
        )
    nmse_decode = _nmse_decode_vs_ref(decode_sweep, np.asarray(decode_ref, dtype=np.float64))

    sweep_svg = _render_method_sweep_panel(
        row_labels=methods,
        h_sweep=h_sqrt_sweep,
        clf_sweep_shared=decode_sweep,
        clf_hellinger_sweep_shared=decode_hellinger_sweep,
        n_list=ns,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_sweep.svg"),
        n_bins=k_cat,
        theta_centers=theta_centers,
        clf_ref_decode_limits=np.asarray(decode_ref, dtype=np.float64),
    )
    gt_svg = _render_cat_gt_panel(
        h_gt_sqrt=h_gt_sqrt,
        decode_ref=decode_ref,
        decode_hellinger_ref_sqrt=decode_hellinger_ref_sqrt,
        n_ref=int(args.n_ref),
        k_cat=k_cat,
        theta_centers=theta_centers,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_gt.svg"),
        decode_sweep_for_decode_limits=decode_sweep,
    )
    zdummy = np.zeros_like(corr_h)
    corr_svg = _render_corr_vs_n_panel(
        row_labels=methods,
        n_list=ns,
        corr_h=corr_h,
        corr_decode_shared=corr_decode,
        corr_decode_hellinger_shared=corr_decode_hellinger,
        corr_hellinger_lb=zdummy,
        corr_hellinger_ub=zdummy,
        show_hellinger_bounds=False,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_corr_vs_n.svg"),
    )
    nmse_svg = _render_nmse_vs_n_panel(
        row_labels=methods,
        n_list=ns,
        nmse_h=nmse_h,
        nmse_decode_shared=nmse_decode,
        nmse_decode_hellinger_shared=nmse_decode_hellinger,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_nmse_vs_n.svg"),
    )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=methods,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_training_losses_panel.svg"),
    )

    if not bool(args.no_scatter_diagnostics) and scatter_true_delta is not None and scatter_cat_h2:
        metrics_by_method: dict[str, dict[str, float]] = {}
        hell_cat: dict[str, np.ndarray] = {}
        for name in methods:
            if name not in scatter_cat_h2:
                continue
            hell_cat[name] = scatter_cat_h2[name]
            if name == "bin_gaussian":
                nan_d = np.full_like(scatter_true_delta, np.nan, dtype=np.float64)
                metrics_by_method[name] = dbg._llr_comparison_metrics(nan_d, scatter_true_delta)
            else:
                dlt = scatter_deltas[name]
                metrics_by_method[name] = dbg._llr_comparison_metrics(dlt, scatter_true_delta)
            metrics_by_method[name].update(
                dbg._hellinger_comparison_metrics_cat(scatter_cat_h2[name], hellinger_gt_sq)
            )
        dbg._save_llr_est_vs_true_figure(
            scatter_deltas,
            scatter_true_delta,
            out_base=Path(out_dir) / "llr_est_vs_true_all",
            metrics_by_method=metrics_by_method,
        )
        dbg._save_hellinger_est_vs_gt_figure(
            hell_cat,
            hellinger_gt_sq,
            out_base=Path(out_dir) / "hellinger_est_vs_gt_all",
            metrics_by_method=metrics_by_method,
        )

    source_indices = np.asarray(perm[: int(args.n_ref)], dtype=np.int64)
    out_npz = os.path.join(out_dir, "h_decoding_categorical_twofig_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.asarray(ns, dtype=np.int64),
        n_ref=np.int64(int(args.n_ref)),
        num_categories=np.int64(k_cat),
        method_names=np.asarray(methods, dtype=object),
        theta_bin_centers=np.asarray(theta_centers, dtype=np.float64),
        perm_seed=np.int64(int(args.run_seed)),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        hellinger_gt_sq_category=np.asarray(hellinger_gt_sq, dtype=np.float64),
        decode_ref=np.asarray(decode_ref, dtype=np.float64),
        decode_sweep=np.asarray(decode_sweep, dtype=np.float64),
        decode_hellinger_ref_sqrt=np.asarray(decode_hellinger_ref_sqrt, dtype=np.float64),
        decode_hellinger_sweep_sqrt=np.asarray(decode_hellinger_sweep, dtype=np.float64),
        h_sqrt_sweep=np.asarray(h_sqrt_sweep, dtype=np.float64),
        corr_h_vs_gt=np.asarray(corr_h, dtype=np.float64),
        nmse_h_vs_gt=np.asarray(nmse_h, dtype=np.float64),
        corr_decode_vs_ref=np.asarray(corr_decode, dtype=np.float64),
        nmse_decode_vs_ref=np.asarray(nmse_decode, dtype=np.float64),
        corr_decode_hellinger_vs_gt=np.asarray(corr_decode_hellinger, dtype=np.float64),
        nmse_decode_hellinger_vs_gt=np.asarray(nmse_decode_hellinger, dtype=np.float64),
        llr_pearson_offdiag=np.asarray(llr_pearson_offdiag, dtype=np.float64),
        hellinger_pearson_offdiag_cat=np.asarray(hellinger_pearson_offdiag_cat, dtype=np.float64),
        wall_seconds=np.asarray(wall_s, dtype=np.float64),
        source_indices=source_indices,
        native_dataset_npz=np.asarray([str(native_npz)], dtype=object),
        work_dataset_npz=np.asarray(
            [str(pr_out_resolved) if pr_out_resolved is not None else str(native_npz)], dtype=object
        ),
        pr_projected=np.bool_(pr_project),
        pr_dim=np.int64(int(args.pr_dim) if pr_project else native_x_dim),
        training_losses_root=np.asarray([os.path.abspath(loss_root)], dtype=object),
        hellinger_eval_split=np.asarray([str(getattr(args, "hellinger_eval_split", "all"))], dtype=object),
    )

    summary_path = os.path.join(out_dir, "h_decoding_categorical_twofig_summary.txt")
    _write_summary(
        summary_path,
        args=args,
        out_npz=os.path.abspath(out_npz),
        sweep_svg=os.path.abspath(sweep_svg),
        gt_svg=os.path.abspath(gt_svg),
        corr_svg=os.path.abspath(corr_svg),
        nmse_svg=os.path.abspath(nmse_svg),
        visualization_only=False,
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
    )

    print("[cat-twofig] Saved:", flush=True)
    print(f"  - {os.path.abspath(out_npz)}", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(gt_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_svg)}", flush=True)
    print(f"  - {os.path.abspath(nmse_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
