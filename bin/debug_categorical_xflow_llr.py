#!/usr/bin/env python3
"""Minimal categorical benchmark diagnostic for x-flow log-likelihood ratios.

Loads the native 2D random_mog_categorical NPZ (no PR autoencoder). Default is
``K=2`` mixture components; data live under
``data/random_mog_categorical_xdim2_k2/`` unless ``--dataset-npz`` is set. Saves
a 2D scatter of observations colored by category. Compares x-flow estimated
pairwise LLRs to exact MoG ``log p(x|theta)`` from NPZ metadata (same
``ToyCategoricalRandomMoGDataset`` recipe). Stops before Hellinger / decoding /
benchmark correlation figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import global_setting  # noqa: F401  # matplotlib rc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher.h_decoding_convergence_methods import prepare_categorical_binning_for_convergence
from fisher.h_matrix import HMatrixEstimator
from fisher.lxf_bin_likelihood_hellinger import lxf_bin_likelihood_hellinger
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_conditional_x_velocity_model,
    build_dataset_from_meta,
    require_device,
)
from fisher.trainers import train_conditional_x_flow_model


def _default_dataset_npz(num_categories: int) -> Path:
    return (
        _repo_root
        / "data"
        / f"random_mog_categorical_xdim2_k{int(num_categories)}"
        / "random_mog_categorical.npz"
    )


def _abs_without_resolving_symlinks(path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root / p


def _run(cmd: list[str]) -> None:
    print("[debug-xflow] running: " + " ".join(cmd), flush=True)
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
                str(args.n_total),
                "--output-npz",
                str(ds),
            ]
        )
    else:
        print(f"[debug-xflow] using existing 2D dataset NPZ: {ds}", flush=True)


def _default_debug_dir(args: argparse.Namespace) -> Path:
    if args.output_npz is not None:
        return Path(args.output_npz).resolve().parent
    return _abs_without_resolving_symlinks(Path(args.dataset_npz)).parent / "xflow_llr_debug"


def _save_2d_dataset_scatter(
    x: np.ndarray,
    bin_idx: np.ndarray,
    *,
    k_cat: int,
    out_path: Path,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    bin_idx = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    if x.shape[1] != 2:
        raise ValueError(f"Expected x_dim=2 for scatter plot, got {x.shape[1]}.")
    n = int(x.shape[0])
    if n > int(max_points):
        pick = np.sort(rng.choice(n, size=int(max_points), replace=False))
        x = x[pick]
        bin_idx = bin_idx[pick]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.0, 5.5), layout="constrained")
    for k in range(int(k_cat)):
        m = bin_idx == k
        if not np.any(m):
            continue
        ax.scatter(
            x[m, 0],
            x[m, 1],
            s=6,
            alpha=0.55,
            c=[cmap(k % 10)],
            label=f"category {k}",
            linewidths=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(f"random_mog_categorical, $K={k_cat}$ (2D)")
    ax.legend(loc="best", fontsize=8, markerscale=2.0, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _compute_true_conditional_loglik_matrix(
    x_all: np.ndarray,
    theta_all: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Exact ``c_matrix[i,j] = log p(x_i | theta_j)`` for categorical MoG from NPZ meta."""
    gen_ds = build_dataset_from_meta(dict(meta))
    n = int(np.asarray(x_all).shape[0])
    x_all = np.asarray(x_all, dtype=np.float64)
    theta_all = np.asarray(theta_all, dtype=np.float64)
    true_c = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        th_col = np.tile(theta_all[j : j + 1], (n, 1))
        true_c[:, j] = gen_ds.log_p_x_given_theta(x_all, th_col)
    return true_c


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _llr_comparison_metrics(est_delta: np.ndarray, true_delta: np.ndarray) -> dict[str, float]:
    """RMSE/MAE/bias/std and Pearson for all entries and off-diagonal only."""
    est = np.asarray(est_delta, dtype=np.float64)
    true_m = np.asarray(true_delta, dtype=np.float64)
    diff = est - true_m
    n = int(est.shape[0])
    out: dict[str, float] = {
        "llr_mae_all": float(np.mean(np.abs(diff))),
        "llr_rmse_all": float(np.sqrt(np.mean(diff**2))),
        "llr_bias_all": float(np.mean(diff)),
        "llr_std_err_all": float(np.std(diff, ddof=0)),
        "llr_pearson_r_all": _pearson_r(est, true_m),
    }
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        d_off = diff[mask]
        e_off = est[mask]
        t_off = true_m[mask]
        out["llr_mae_offdiag"] = float(np.mean(np.abs(d_off)))
        out["llr_rmse_offdiag"] = float(np.sqrt(np.mean(d_off**2)))
        out["llr_bias_offdiag"] = float(np.mean(d_off))
        out["llr_std_err_offdiag"] = float(np.std(d_off, ddof=0))
        out["llr_pearson_r_offdiag"] = _pearson_r(e_off, t_off)
    else:
        out["llr_mae_offdiag"] = float("nan")
        out["llr_rmse_offdiag"] = float("nan")
        out["llr_bias_offdiag"] = float("nan")
        out["llr_std_err_offdiag"] = float("nan")
        out["llr_pearson_r_offdiag"] = float("nan")
    return out


def _save_llr_est_vs_true_figure(
    est_delta: np.ndarray,
    true_delta: np.ndarray,
    *,
    out_base: Path,
    metrics: dict[str, float],
) -> None:
    n = int(est_delta.shape[0])
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        x = np.asarray(true_delta, dtype=np.float64)[mask].ravel()
        y = np.asarray(est_delta, dtype=np.float64)[mask].ravel()
    else:
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), layout="constrained")
    if x.size == 0:
        ax.text(
            0.5,
            0.5,
            "No off-diagonal LLR pairs\n(n_ref ≤ 1)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlabel(r"true $\Delta L$")
        ax.set_ylabel(r"x-flow $\Delta L$")
        ax.set_title(r"LLR: estimated vs ground truth")
    else:
        ax.scatter(x, y, s=8, alpha=0.35, c="C0", linewidths=0)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7, label="identity")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"true $\Delta L$")
        ax.set_ylabel(r"x-flow $\Delta L$")
        ax.set_title(r"LLR: estimated vs ground truth (off-diagonal)")
        rmse = metrics.get("llr_rmse_offdiag", float("nan"))
        r = metrics.get("llr_pearson_r_offdiag", float("nan"))
        ax.text(
            0.04,
            0.96,
            f"RMSE(offdiag)={rmse:.4f}\nPearson(offdiag)={r:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _subset_bundle(
    bundle: SharedDatasetBundle,
    *,
    perm: np.ndarray,
    n_ref: int,
    bin_idx_all: np.ndarray,
) -> tuple[SharedDatasetBundle, np.ndarray, np.ndarray]:
    idx = np.asarray(perm[: int(n_ref)], dtype=np.int64)
    theta_all = np.asarray(bundle.theta_all[idx], dtype=np.float64)
    x_all = np.asarray(bundle.x_all[idx], dtype=np.float64)
    bins = np.asarray(bin_idx_all[idx], dtype=np.int64).reshape(-1)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)
    if x_all.ndim != 2:
        raise ValueError("x_all must be 2D.")

    train_frac = float(bundle.meta.get("train_frac", 0.7))
    if train_frac >= 1.0:
        n_train = int(n_ref)
    else:
        n_train = int(train_frac * int(n_ref))
        n_train = min(max(n_train, 1), int(n_ref) - 1)

    sub = SharedDatasetBundle(
        meta=bundle.meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=np.arange(n_train, dtype=np.int64),
        validation_idx=np.arange(n_train, int(n_ref), dtype=np.int64),
        theta_train=theta_all[:n_train],
        x_train=x_all[:n_train],
        theta_validation=theta_all[n_train:],
        x_validation=x_all[n_train:],
    )
    return sub, bins, idx


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D random_mog_categorical: visualize data and compute x_flow log p(x|theta) / DeltaL only."
    )
    p.add_argument(
        "--num-categories",
        type=int,
        default=2,
        help="Mixture cardinality K for random_mog_categorical (passed to make_dataset when generating).",
    )
    p.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Native 2D NPZ (x_dim=2). Default: data/random_mog_categorical_xdim2_kK/random_mog_categorical.npz.",
    )
    p.add_argument("--output-npz", type=Path, default=None)
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--n-total", type=int, default=50000)
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the 2D dataset scatter figure (SVG+PNG).",
    )
    p.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Base path for figures (extensions .svg/.png added). Default: under xflow_llr_debug/.",
    )
    p.add_argument("--plot-max-points", type=int, default=12000)
    p.add_argument("--n-ref", type=int, default=600)
    p.add_argument("--run-seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")

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
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.num_categories) < 2:
        raise ValueError("--num-categories must be >= 2.")
    if args.dataset_npz is None:
        args.dataset_npz = _default_dataset_npz(int(args.num_categories))
    if str(args.device).strip().lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this project requires --device cuda for this run.")

    t0 = time.time()
    _ensure_dataset(args)

    bundle = load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
    if str(meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical NPZ, got {meta.get('dataset_family')!r}.")
    if str(meta.get("theta_type", "")) != "categorical":
        raise ValueError(f"Expected categorical theta_type, got {meta.get('theta_type')!r}.")
    if int(bundle.x_all.shape[1]) != 2:
        raise ValueError(
            f"This script expects native 2D observations (x_dim=2); got x_dim={bundle.x_all.shape[1]}."
        )

    n_pool = int(bundle.theta_all.shape[0])
    if int(args.n_ref) > n_pool:
        raise ValueError(f"--n-ref={args.n_ref} exceeds dataset rows n_total={n_pool}.")

    k_cat = int(meta.get("num_categories", 5))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(bundle.theta_all, k_cat)
    rng = np.random.default_rng(int(args.run_seed))
    if not args.no_plot:
        dbg_dir = _default_debug_dir(args)
        plot_base = args.plot_path
        if plot_base is None:
            plot_base = dbg_dir / "dataset_2d_scatter"
        else:
            plot_base = _abs_without_resolving_symlinks(Path(plot_base))
        plot_rng = np.random.default_rng(int(args.run_seed) + 913_409)
        _save_2d_dataset_scatter(
            bundle.x_all,
            bin_idx_all,
            k_cat=k_cat,
            out_path=Path(plot_base),
            rng=plot_rng,
            max_points=int(args.plot_max_points),
        )
        print(f"[debug-xflow] saved dataset scatter {plot_base}.svg / .png", flush=True)
    perm = rng.permutation(n_pool)
    sub, bins, source_indices = _subset_bundle(bundle, perm=perm, n_ref=int(args.n_ref), bin_idx_all=bin_idx_all)

    dev = require_device(str(args.device))
    torch.manual_seed(int(args.run_seed))
    np.random.seed(int(args.run_seed))

    train_theta = np.asarray(sub.theta_train, dtype=np.float64)
    val_theta = np.asarray(sub.theta_validation, dtype=np.float64)
    theta_all = np.asarray(sub.theta_all, dtype=np.float64)
    x_train = np.asarray(sub.x_train, dtype=np.float64)
    x_val = np.asarray(sub.x_validation, dtype=np.float64)
    x_all = np.asarray(sub.x_all, dtype=np.float64)
    if train_theta.ndim == 1:
        train_theta = train_theta.reshape(-1, 1)
    if val_theta.ndim == 1:
        val_theta = val_theta.reshape(-1, 1)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)

    model_args = SimpleNamespace(
        x_dim=int(x_all.shape[1]),
        flow_hidden_dim=int(args.flow_hidden_dim),
        flow_depth=int(args.flow_depth),
        flow_use_layer_norm=False,
        flow_gated_film=False,
        flow_zero_out_init=False,
        flow_cond_embed_dim=16,
        flow_cond_embed_depth=1,
        flow_cond_embed_act="silu",
        flow_x_theta_fourier_k=4,
        flow_x_theta_fourier_omega=1.0,
        flow_x_theta_fourier_no_linear=False,
        flow_x_theta_fourier_no_bias=False,
    )
    print(
        f"[debug-xflow] training x_flow: n_ref={args.n_ref} train={x_train.shape[0]} "
        f"val={x_val.shape[0]} x_dim={x_all.shape[1]} theta_dim={theta_all.shape[1]}",
        flush=True,
    )
    model = build_conditional_x_velocity_model(
        flow_arch=str(args.flow_arch),
        args=model_args,
        device=dev,
        theta_dim=int(theta_all.shape[1]),
    )
    train_out = train_conditional_x_flow_model(
        model=model,
        theta_train=train_theta,
        x_train=x_train,
        theta_val=val_theta,
        x_val=x_val,
        epochs=int(args.flow_epochs),
        batch_size=int(args.flow_batch_size),
        lr=float(args.flow_lr),
        device=dev,
        log_every=max(1, int(args.log_every)),
        early_stopping_patience=int(args.flow_early_patience),
        early_stopping_min_delta=float(args.flow_early_min_delta),
        early_stopping_ema_alpha=float(args.flow_early_ema_alpha),
        restore_best=bool(args.flow_restore_best),
        scheduler_name=str(args.flow_scheduler),
        fm_t_eps=float(args.flow_fm_t_eps),
    )

    print("[debug-xflow] computing c_matrix = log p(x_i | theta_j)", flush=True)
    estimator = HMatrixEstimator(
        model_post=model,
        model_prior=None,
        sigma_eval=1.0,
        device=dev,
        pair_batch_size=int(args.h_batch_size),
        field_method="flow_x_likelihood",
        flow_scheduler=str(args.flow_scheduler),
        flow_ode_steps=int(args.flow_ode_steps),
        flow_likelihood_exact_divergence=bool(args.flow_likelihood_exact_divergence),
    )
    c_matrix = estimator.compute_x_conditional_loglik_matrix(theta_all, x_all)
    delta_l = HMatrixEstimator.compute_delta_l(c_matrix)
    bin_ll = lxf_bin_likelihood_hellinger(c_matrix, bins, k_cat)

    print("[debug-xflow] computing ground-truth c_matrix from dataset meta", flush=True)
    true_c_matrix = _compute_true_conditional_loglik_matrix(x_all, theta_all, meta)
    true_delta_l = HMatrixEstimator.compute_delta_l(true_c_matrix)
    llr_metrics = _llr_comparison_metrics(delta_l, true_delta_l)

    out_path = args.output_npz
    if out_path is None:
        out_path = _default_debug_dir(args) / f"xflow_llr_debug_n{int(args.n_ref)}.npz"
    out_path = Path(out_path)
    out_path_display = _abs_without_resolving_symlinks(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llr_fig_base = out_path.parent / "llr_est_vs_true"
    _save_llr_est_vs_true_figure(delta_l, true_delta_l, out_base=llr_fig_base, metrics=llr_metrics)
    print(f"[debug-xflow] saved LLR comparison {llr_fig_base}.svg / .png", flush=True)

    np.savez_compressed(
        out_path,
        theta_used=np.asarray(theta_all, dtype=np.float64),
        category_labels=np.asarray(bins, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        c_matrix=np.asarray(c_matrix, dtype=np.float64),
        delta_l_matrix=np.asarray(delta_l, dtype=np.float64),
        true_c_matrix=np.asarray(true_c_matrix, dtype=np.float64),
        true_delta_l_matrix=np.asarray(true_delta_l, dtype=np.float64),
        llr_mae_all=np.float64(llr_metrics["llr_mae_all"]),
        llr_rmse_all=np.float64(llr_metrics["llr_rmse_all"]),
        llr_bias_all=np.float64(llr_metrics["llr_bias_all"]),
        llr_std_err_all=np.float64(llr_metrics["llr_std_err_all"]),
        llr_pearson_r_all=np.float64(llr_metrics["llr_pearson_r_all"]),
        llr_mae_offdiag=np.float64(llr_metrics["llr_mae_offdiag"]),
        llr_rmse_offdiag=np.float64(llr_metrics["llr_rmse_offdiag"]),
        llr_bias_offdiag=np.float64(llr_metrics["llr_bias_offdiag"]),
        llr_std_err_offdiag=np.float64(llr_metrics["llr_std_err_offdiag"]),
        llr_pearson_r_offdiag=np.float64(llr_metrics["llr_pearson_r_offdiag"]),
        bin_log_likelihood=np.asarray(bin_ll["bin_log_likelihood"], dtype=np.float64),
        bin_delta_l_matrix=np.asarray(bin_ll["bin_delta_l"], dtype=np.float64),
        bin_counts=np.asarray(bin_ll["bin_counts"], dtype=np.int64),
        score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
        score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
        score_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
        score_best_epoch=np.int64(train_out["best_epoch"]),
        score_stopped_epoch=np.int64(train_out["stopped_epoch"]),
        score_stopped_early=np.bool_(train_out["stopped_early"]),
        score_best_val_smooth=np.float64(train_out["best_val_loss"]),
        dataset_npz=np.asarray([str(_abs_without_resolving_symlinks(Path(args.dataset_npz)))], dtype=object),
        method=np.asarray(["x_flow"], dtype=object),
        flow_arch=np.asarray([str(args.flow_arch)], dtype=object),
        flow_scheduler=np.asarray([str(args.flow_scheduler)], dtype=object),
        flow_ode_steps=np.int64(int(args.flow_ode_steps)),
        flow_likelihood_exact_divergence=np.bool_(args.flow_likelihood_exact_divergence),
        n_ref=np.int64(int(args.n_ref)),
        run_seed=np.int64(int(args.run_seed)),
        wall_seconds=np.float64(time.time() - t0),
    )
    print(f"[debug-xflow] saved {out_path_display}", flush=True)


if __name__ == "__main__":
    main()
