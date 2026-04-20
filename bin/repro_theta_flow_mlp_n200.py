#!/usr/bin/env python3
"""Minimal reproducer for theta-flow MLP convergence at fixed n=200.

This script intentionally exposes only a tiny CLI surface and fixes the rest:
- dataset_family = randamp_gaussian
- x_dim = --x-dim (default 2)
- obs_noise_scale = 0.5 (TEMPORARY: half the family baseline noise; restore to 1.0 for default)
- n_total = 3000, train_frac = 0.7, seed = 7
- theta_field_method = theta_flow
- flow_arch = mlp
- n_ref = 1000
- n_list = 200
- num_theta_bins = 10

It creates a shared dataset NPZ, then runs ``bin/study_h_decoding_convergence.py``
with those fixed settings and prints the resulting metrics.
Per-n run directories are kept so fitted model checkpoints remain available under
``<output-dir>/sweep_runs/n_000200/`` for diagnostics, and the script auto-runs
the fixed-x posterior+tuning diagnostic then re-renders the combined figure so
the diagnostic panel is embedded in ``h_decoding_convergence_combined``.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_bin_dir = _repo_root / "bin"
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import study_h_decoding_convergence as shdc
import visualize_h_matrix_binned as vhb
from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.hellinger_gt import bin_centers_from_edges, estimate_hellinger_sq_one_sided_mc
from fisher.shared_dataset_io import load_shared_dataset_npz, meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, build_dataset_from_meta

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import zuko

# TEMPORARY: <1.0 reduces Gaussian observation noise (see --obs-noise-scale in make_dataset.py).
# Restore to 1.0 when the low-noise experiment is done.
_TEMP_OBS_NOISE_SCALE = 0.5


def _default_output_dir(x_dim: int) -> str:
    return str(
        Path("data")
        / f"repro_theta_flow_mlp_n200_randamp_gaussian_xdim{x_dim}_obsnoise0p5"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Minimal fixed reproducer for theta_flow + mlp on randamp_gaussian "
            "with convergence n_list=200."
        )
    )
    p.add_argument(
        "--x-dim",
        type=int,
        default=2,
        help="Observation dimension x ∈ R^{d} (randamp_gaussian; default 2).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for generated dataset + study artifacts. "
            "If omitted, uses data/repro_theta_flow_mlp_n200_randamp_gaussian_xdim{d}_obsnoise0p5 "
            "for the chosen --x-dim."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help="Execution device. Per repo policy this reproducer requires CUDA.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["theta-flow", "nf", "both"],
        help=(
            "Which estimator branch to run: theta-flow only, nf only, or both. "
            "Default: both."
        ),
    )
    p.add_argument(
        "--posterior-x-index",
        type=int,
        default=0,
        help=(
            "Global index into shared_dataset.x_all used as the fixed x for "
            "both theta-flow and NF posterior diagnostics."
        ),
    )
    p.add_argument("--nf-epochs", type=int, default=2000, help="NF posterior training epochs.")
    p.add_argument("--nf-batch-size", type=int, default=256, help="NF posterior batch size.")
    p.add_argument("--nf-lr", type=float, default=1e-3, help="NF posterior learning rate.")
    p.add_argument("--nf-hidden-dim", type=int, default=128, help="NF encoder hidden size.")
    p.add_argument("--nf-context-dim", type=int, default=32, help="NF flow conditioning width.")
    p.add_argument("--nf-transforms", type=int, default=5, help="NF spline transform count.")
    p.add_argument(
        "--nf-pair-batch-size",
        type=int,
        default=65536,
        help="Approximate pair budget per NF C-matrix block (rows*cols).",
    )
    p.add_argument("--nf-early-patience", type=int, default=300, help="NF early stopping patience.")
    p.add_argument("--nf-early-min-delta", type=float, default=1e-4, help="NF early min delta.")
    p.add_argument(
        "--nf-early-ema-alpha",
        type=float,
        default=0.05,
        help="NF validation EMA alpha in (0,1].",
    )
    return p


def _normalize_output_dir(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    # Keep repo-relative paths stable for human-facing output (e.g., ./data/...).
    return _repo_root / p


def _write_dataset(dataset_npz: Path, *, x_dim: int) -> None:
    # Build namespace via the shared dataset parser to stay aligned with family recipes.
    ds_parser = argparse.ArgumentParser(add_help=False)
    add_dataset_arguments(ds_parser)
    ds_args = ds_parser.parse_args([])
    ds_args.dataset_family = "randamp_gaussian"
    ds_args.x_dim = int(x_dim)
    ds_args.obs_noise_scale = float(_TEMP_OBS_NOISE_SCALE)
    ds_args.n_total = 3000
    ds_args.train_frac = 0.7
    ds_args.seed = 7

    np.random.seed(int(ds_args.seed))
    rng = np.random.default_rng(int(ds_args.seed))
    dataset = build_dataset_from_args(ds_args)
    n_total = int(ds_args.n_total)
    theta_all, x_all = dataset.sample_joint(n_total)

    perm = rng.permutation(n_total)
    n_train = int(float(ds_args.train_frac) * n_total)
    n_train = min(max(n_train, 1), n_total - 1)
    tr_idx = perm[:n_train].astype(np.int64, copy=False)
    va_idx = perm[n_train:].astype(np.int64, copy=False)

    meta = meta_dict_from_args(ds_args)
    meta["randamp_mu_amp_per_dim"] = dataset._randamp_amp.tolist()
    save_shared_dataset_npz(
        str(dataset_npz),
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr_idx,
        validation_idx=va_idx,
        theta_train=theta_all[tr_idx],
        x_train=x_all[tr_idx],
        theta_validation=theta_all[va_idx],
        x_validation=x_all[va_idx],
    )


def _parse_metrics(results_csv: Path) -> tuple[int, float, float]:
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {results_csv}, found {len(rows)}.")
    row = rows[0]
    n = int(row["n"])
    corr_h = float(row["corr_h_binned_vs_gt_mc"])
    corr_clf = float(row["corr_clf_vs_ref"])
    return n, corr_h, corr_clf


class ConditionalThetaNF(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        context_dim: int,
        hidden_dim: int,
        transforms: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.flow = zuko.flows.NSF(
            features=1,
            context=context_dim,
            transforms=transforms,
            hidden_features=[hidden_dim, hidden_dim],
        )

    def distribution(self, x: torch.Tensor) -> torch.distributions.Distribution:
        return self.flow(self.encoder(x))

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.distribution(x).log_prob(theta.reshape(-1, 1))


def _train_conditional_nf(
    *,
    model: ConditionalThetaNF,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_delta: float,
    ema_alpha: float,
) -> dict[str, np.ndarray | float | int]:
    xtr = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device)
    ttr = torch.from_numpy(np.asarray(theta_train, dtype=np.float32).reshape(-1, 1)).to(device)
    xva = torch.from_numpy(np.asarray(x_val, dtype=np.float32)).to(device)
    tva = torch.from_numpy(np.asarray(theta_val, dtype=np.float32).reshape(-1, 1)).to(device)
    ntr = int(xtr.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ema_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ema = float("inf")
    best_epoch = 0
    bad = 0
    ema = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        idx = torch.randint(0, ntr, (int(batch_size),), device=device)
        loss = -model.log_prob(ttr[idx], xtr[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)

        model.eval()
        with torch.no_grad():
            va = float((-model.log_prob(tva, xva).mean()).detach().cpu().item())
        val_losses.append(va)
        ema = va if ema is None else (float(ema_alpha) * va + (1.0 - float(ema_alpha)) * float(ema))
        val_ema_losses.append(float(ema))

        if float(ema) < (best_ema - float(min_delta)):
            best_ema = float(ema)
            best_epoch = ep
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_ema_losses": np.asarray(val_ema_losses, dtype=np.float64),
        "best_epoch": int(best_epoch if best_epoch > 0 else len(train_losses)),
        "best_val_ema": float(best_ema if np.isfinite(best_ema) else np.nan),
        "stopped_epoch": int(len(train_losses)),
    }


def _compute_c_matrix_nf(
    *,
    model: ConditionalThetaNF,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    pair_batch_size: int,
) -> np.ndarray:
    theta_col = np.asarray(theta_all, dtype=np.float32).reshape(-1, 1)
    x2 = np.asarray(x_all, dtype=np.float32)
    n = int(theta_col.shape[0])
    if n < 1:
        raise ValueError("Need at least one sample for NF C matrix.")
    row_block = max(1, int(pair_batch_size // max(1, n)))
    c = np.zeros((n, n), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_block):
            i1 = min(n, i0 + row_block)
            b = int(i1 - i0)
            xb = np.asarray(x2[i0:i1], dtype=np.float32)
            x_rep = np.repeat(xb, repeats=n, axis=0)
            theta_tile = np.tile(theta_col, (b, 1))
            x_t = torch.from_numpy(x_rep).to(device)
            th_t = torch.from_numpy(theta_tile).to(device)
            logp = model.log_prob(th_t, x_t)
            c[i0:i1, :] = logp.reshape(b, n).detach().cpu().numpy().astype(np.float64)
    return c


def _compute_delta_l(c: np.ndarray) -> np.ndarray:
    diag = np.diag(np.asarray(c, dtype=np.float64)).reshape(-1, 1)
    return np.asarray(c, dtype=np.float64) - diag


def _compute_h_directed(delta_l: np.ndarray) -> np.ndarray:
    z = np.clip(0.5 * np.asarray(delta_l, dtype=np.float64), -60.0, 60.0)
    h = 1.0 - (1.0 / np.cosh(z))
    np.fill_diagonal(h, 0.0)
    return h


def _symmetrize(h_directed: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(h_directed, dtype=np.float64) + np.asarray(h_directed, dtype=np.float64).T)


def _save_heatmap(path: Path, mat: np.ndarray, title: str, cbar_label: str) -> None:
    plt.figure(figsize=(6.2, 5.6))
    im = plt.imshow(np.asarray(mat, dtype=np.float64), aspect="auto", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=cbar_label)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _run_nf_fixed_x_diagnostic(
    *,
    model: ConditionalThetaNF,
    meta: dict,
    x_fixed_global: np.ndarray,
    x_index_global: int,
    n_samples: int,
    run_dir: Path,
    dataset_npz: Path,
    device: torch.device,
) -> dict[str, str]:
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    x_fixed = np.asarray(x_fixed_global, dtype=np.float64).reshape(1, -1)
    if x_fixed.shape[1] < 1:
        raise ValueError("x_fixed_global must have at least one feature.")
    x_fixed_t = torch.from_numpy(x_fixed.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        dist = model.distribution(x_fixed_t)
        theta_samples_t = dist.sample((int(n_samples),))
    theta_samples = np.asarray(theta_samples_t.detach().cpu().numpy(), dtype=np.float64).reshape(int(n_samples), -1)
    dim0 = theta_samples[:, 0]

    out_npz = diag_dir / "nf_single_x_posterior_samples.npz"
    np.savez_compressed(
        out_npz,
        theta_samples=theta_samples,
        theta_samples_dim0=dim0,
        x_fixed=x_fixed,
        x_index=np.int64(int(x_index_global)),
        x_index_global=np.int64(int(x_index_global)),
        n_samples=np.int64(int(n_samples)),
        dataset_npz=np.asarray([str(dataset_npz)], dtype=object),
        model_type=np.asarray(["zuko_nsf_conditional"], dtype=object),
    )

    ds = build_dataset_from_meta(meta)
    n_curve = 500
    t_grid = np.linspace(float(ds.theta_low), float(ds.theta_high), n_curve, dtype=np.float64).reshape(-1, 1)
    mu_grid = np.asarray(ds.tuning_curve(t_grid), dtype=np.float64)
    x_fixed_row = np.asarray(x_fixed, dtype=np.float64).reshape(-1)
    n_dim_plot = int(min(mu_grid.shape[1], x_fixed_row.shape[0]))

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.2), gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    ax.hist(dim0, bins=100, density=True, alpha=0.8, color="#1f77b4", edgecolor="none")
    ax.set_xlabel("theta sample value (dim 0)")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"NF posterior theta samples (x_index={int(x_index_global)}, n={int(n_samples)})")

    ax_tc = axes[1]
    for j in range(n_dim_plot):
        c = f"C{j % 10}"
        ax_tc.plot(
            t_grid[:, 0],
            mu_grid[:, j],
            color=c,
            linewidth=1.6,
            alpha=0.95,
            label=(f"mu_{j+1}(theta)" if j < 10 else None),
        )
        ax_tc.axhline(
            float(x_fixed_row[j]),
            color=c,
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
        )
    ax_tc.set_xlabel("theta")
    ax_tc.set_ylabel("x / tuning value")
    ax_tc.set_title("Dataset tuning curves with fixed x_j overlays (dashed)")
    ax_tc.grid(True, alpha=0.25)
    if n_dim_plot <= 10:
        ax_tc.legend(loc="upper right", fontsize=8, ncol=2)

    x_text = np.array2string(x_fixed_row, precision=3, suppress_small=False, max_line_width=1000)
    title_raw = f"Fixed x value (index={int(x_index_global)}): {x_text}"
    title_wrapped = "\n".join(textwrap.wrap(title_raw, width=95))
    fig.suptitle(title_wrapped, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out_png = diag_dir / "nf_single_x_posterior_hist.png"
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)

    out_txt = diag_dir / "nf_single_x_posterior_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"dataset_npz: {dataset_npz}\n")
        f.write(f"x_index: {int(x_index_global)}\n")
        f.write(f"x_index_global: {int(x_index_global)}\n")
        f.write(f"n_samples: {int(n_samples)}\n")
        f.write("model_type: zuko_nsf_conditional\n")
        f.write(f"theta_dim0_mean: {float(np.mean(dim0)):.8f}\n")
        f.write(f"theta_dim0_std: {float(np.std(dim0)):.8f}\n")
        f.write(f"theta_dim0_q05: {float(np.quantile(dim0, 0.05)):.8f}\n")
        f.write(f"theta_dim0_q50: {float(np.quantile(dim0, 0.50)):.8f}\n")
        f.write(f"theta_dim0_q95: {float(np.quantile(dim0, 0.95)):.8f}\n")

    return {
        "samples_npz": str(out_npz),
        "figure_png": str(out_png),
        "figure_svg": str(out_svg),
        "summary_txt": str(out_txt),
    }


def _subset_train_eval_for_indices(
    x_all: np.ndarray,
    theta_all: np.ndarray,
    bin_all: np.ndarray,
    idx: np.ndarray,
    train_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_sub = np.asarray(x_all[idx], dtype=np.float64)
    theta_sub = np.asarray(theta_all[idx], dtype=np.float64).reshape(-1, 1)
    bin_sub = np.asarray(bin_all[idx], dtype=np.int64).reshape(-1)
    n = int(x_sub.shape[0])
    if n < 2:
        raise RuntimeError("Subset must contain at least 2 rows for train/validation split.")
    tf = float(train_frac)
    if tf >= 1.0:
        n_train = n
    else:
        n_train = int(tf * n)
        n_train = min(max(n_train, 1), n - 1)
    x_tr = x_sub[:n_train]
    theta_tr = theta_sub[:n_train]
    x_va = x_sub[n_train:]
    theta_va = theta_sub[n_train:]
    return x_sub, theta_sub, bin_sub, x_tr, theta_tr, x_va, theta_va


def _load_or_create_reference_for_nf(
    *,
    out_dir: Path,
    dataset_npz: Path,
    n_ref: int,
    n_bins: int,
    base_seed: int,
) -> dict[str, np.ndarray | int | dict]:
    bundle = load_shared_dataset_npz(str(dataset_npz))
    meta = dict(bundle.meta)
    theta_all = np.asarray(bundle.theta_all, dtype=np.float64).reshape(-1)
    x_all = np.asarray(bundle.x_all, dtype=np.float64)
    n_pool = int(theta_all.shape[0])
    if int(n_ref) > n_pool:
        raise ValueError(f"n_ref={n_ref} exceeds dataset size {n_pool}.")

    perm = np.random.default_rng(int(base_seed)).permutation(n_pool)
    theta_ref = theta_all[perm[: int(n_ref)]]

    ref_npz = out_dir / "h_decoding_convergence_reference.npz"
    if ref_npz.is_file():
        with np.load(ref_npz, allow_pickle=True) as ref:
            edges = np.asarray(ref["theta_bin_edges"], dtype=np.float64).reshape(-1)
            h_gt_sqrt = np.asarray(ref["hellinger_gt_sq_mc"], dtype=np.float64)
            clf_ref = np.asarray(ref["clf_acc_ref"], dtype=np.float64)
    else:
        edges, edge_lo, edge_hi = vhb.theta_bin_edges(theta_ref, int(n_bins))
        centers = bin_centers_from_edges(edges)
        gt_n_mc = int(n_ref) // int(n_bins)
        dataset_for_gt = build_dataset_from_meta(meta)
        if hasattr(dataset_for_gt, "rng"):
            dataset_for_gt.rng = np.random.default_rng(int(base_seed))
        h_gt_mc = estimate_hellinger_sq_one_sided_mc(
            dataset_for_gt,
            centers,
            n_mc=int(gt_n_mc),
            symmetrize=False,
        )
        h_gt_sqrt = shdc._sqrt_h_like(h_gt_mc)
        bin_all = vhb.theta_to_bin_index(theta_all, edges, int(n_bins))
        idx_ref = perm[: int(n_ref)]
        x_ref_all = np.asarray(x_all[idx_ref], dtype=np.float64)
        bin_ref_all = np.asarray(bin_all[idx_ref], dtype=np.int64)
        tf = float(meta["train_frac"])
        if tf >= 1.0:
            n_train_ref = int(n_ref)
        else:
            n_train_ref = int(tf * int(n_ref))
            n_train_ref = min(max(n_train_ref, 1), int(n_ref) - 1)
        x_ref_train = x_ref_all[:n_train_ref]
        bin_ref_train = bin_ref_all[:n_train_ref]
        clf_ref, _, _, _ = vhb.pairwise_bin_logistic_accuracy_train_val(
            x_ref_train,
            bin_ref_train,
            x_ref_all,
            bin_ref_all,
            int(n_bins),
            min_class_count=5,
            random_state=int(base_seed),
        )
        np.savez_compressed(
            ref_npz,
            h_binned_ref=np.asarray(h_gt_sqrt, dtype=np.float64),
            clf_acc_ref=np.asarray(clf_ref, dtype=np.float64),
            hellinger_gt_sq_mc=np.asarray(h_gt_sqrt, dtype=np.float64),
            h_binned_ref_is_gt_mc=np.int32(1),
            theta_bin_edges=np.asarray(edges, dtype=np.float64),
            edge_lo=np.float64(edge_lo),
            edge_hi=np.float64(edge_hi),
            theta_bin_centers=np.asarray(centers, dtype=np.float64),
            n_ref=np.int64(int(n_ref)),
            perm_seed=np.int64(int(base_seed)),
            dataset_meta_seed=np.int64(int(meta.get("seed", 7))),
        )

    n_bins_eff = int(np.asarray(edges).size - 1)
    if n_bins_eff < 1:
        raise ValueError("Invalid theta_bin_edges in reference NPZ.")
    centers_eff = bin_centers_from_edges(edges)
    bin_idx_all = vhb.theta_to_bin_index(theta_all, edges, n_bins_eff)
    return {
        "meta": meta,
        "theta_all": theta_all,
        "x_all": x_all,
        "perm": perm,
        "bin_idx_all": bin_idx_all,
        "edges": np.asarray(edges, dtype=np.float64),
        "centers": np.asarray(centers_eff, dtype=np.float64),
        "h_gt_sqrt": np.asarray(h_gt_sqrt, dtype=np.float64),
        "clf_ref": np.asarray(clf_ref, dtype=np.float64),
        "n_bins": int(n_bins_eff),
    }


def _run_theta_flow_branch(
    *,
    out_dir: Path,
    dataset_npz: Path,
    n: int,
    n_ref: int,
    posterior_x_index: int,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    study_py = _repo_root / "bin" / "study_h_decoding_convergence.py"
    if not study_py.is_file():
        raise FileNotFoundError(f"Missing study script: {study_py}")
    cmd = [
        sys.executable,
        str(study_py),
        "--dataset-npz",
        str(dataset_npz),
        "--dataset-family",
        "randamp_gaussian",
        "--output-dir",
        str(out_dir),
        "--theta-field-method",
        "theta_flow",
        "--flow-arch",
        "mlp",
        "--n-ref",
        str(n_ref),
        "--n-list",
        str(n),
        "--num-theta-bins",
        "10",
        "--keep-intermediate",
        "--run-seed",
        "7",
        "--device",
        args.device,
    ]
    print("[repro][theta-flow] running study_h_decoding_convergence...", flush=True)
    result = subprocess.run(cmd, cwd=str(_repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"study_h_decoding_convergence failed with code {result.returncode}.")

    diag_py = _repo_root / "bin" / "diagnose_theta_flow_single_x_samples.py"
    if not diag_py.is_file():
        raise FileNotFoundError(f"Missing diagnostic script: {diag_py}")
    run_dir = out_dir / "sweep_runs" / f"n_{n:06d}"
    diag_cmd = [
        sys.executable,
        str(diag_py),
        "--run-dir",
        str(run_dir),
        "--dataset-npz",
        str(dataset_npz),
        "--x-index",
        str(int(posterior_x_index)),
        "--n-samples",
        "20000",
        "--device",
        args.device,
    ]
    print("[repro][theta-flow] running fixed-x posterior+tuning diagnostic...", flush=True)
    diag_res = subprocess.run(diag_cmd, cwd=str(_repo_root), check=False)
    if diag_res.returncode != 0:
        raise RuntimeError(f"diagnose_theta_flow_single_x_samples failed with code {diag_res.returncode}.")

    viz_cmd = cmd + ["--visualization-only"]
    print("[repro][theta-flow] regenerating combined figure with embedded diagnostic...", flush=True)
    viz_res = subprocess.run(viz_cmd, cwd=str(_repo_root), check=False)
    if viz_res.returncode != 0:
        raise RuntimeError(f"study_h_decoding_convergence --visualization-only failed with code {viz_res.returncode}.")

    results_csv = out_dir / "h_decoding_convergence_results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing expected results CSV: {results_csv}")
    n_row, corr_h, corr_clf = _parse_metrics(results_csv)
    return {
        "n": int(n_row),
        "corr_h_binned_vs_gt_mc": float(corr_h),
        "corr_clf_vs_ref": float(corr_clf),
        "combined_png": str(out_dir / "h_decoding_convergence_combined.png"),
        "results_csv": str(results_csv),
    }


def _run_nf_branch(
    *,
    out_dir: Path,
    dataset_npz: Path,
    x_dim: int,
    n: int,
    n_ref: int,
    posterior_x_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int | str]:
    ref = _load_or_create_reference_for_nf(
        out_dir=out_dir,
        dataset_npz=dataset_npz,
        n_ref=int(n_ref),
        n_bins=10,
        base_seed=7,
    )
    theta_all = np.asarray(ref["theta_all"], dtype=np.float64).reshape(-1)
    x_all = np.asarray(ref["x_all"], dtype=np.float64)
    bin_idx_all = np.asarray(ref["bin_idx_all"], dtype=np.int64)
    perm = np.asarray(ref["perm"], dtype=np.int64)
    meta = dict(ref["meta"])
    edges = np.asarray(ref["edges"], dtype=np.float64)
    centers = np.asarray(ref["centers"], dtype=np.float64)
    h_gt_sqrt = np.asarray(ref["h_gt_sqrt"], dtype=np.float64)
    clf_ref = np.asarray(ref["clf_ref"], dtype=np.float64)
    n_bins = int(ref["n_bins"])

    idx_n = perm[: int(n)]
    x_n_all, theta_n_all, bin_n_all, x_n_tr, theta_n_tr, x_n_va, theta_n_va = _subset_train_eval_for_indices(
        x_all,
        theta_all.reshape(-1, 1),
        bin_idx_all,
        idx_n,
        float(meta["train_frac"]),
    )
    if int(posterior_x_index) < 0 or int(posterior_x_index) >= int(x_all.shape[0]):
        raise ValueError(
            f"posterior_x_index={posterior_x_index} out of range for shared dataset size {x_all.shape[0]}."
        )
    x_fixed_global = np.asarray(x_all[int(posterior_x_index) : int(posterior_x_index) + 1], dtype=np.float64)
    tf = float(meta["train_frac"])
    if tf >= 1.0:
        n_train_n = int(n)
    else:
        n_train_n = int(tf * int(n))
        n_train_n = min(max(n_train_n, 1), int(n) - 1)
    clf_n, _, _, _ = vhb.pairwise_bin_logistic_accuracy_train_val(
        x_n_tr,
        np.asarray(bin_n_all[:n_train_n], dtype=np.int64),
        x_n_all,
        bin_n_all,
        int(n_bins),
        min_class_count=5,
        random_state=7,
    )

    np.random.seed(7)
    torch.manual_seed(7)
    model = ConditionalThetaNF(
        x_dim=int(x_dim),
        context_dim=int(args.nf_context_dim),
        hidden_dim=int(args.nf_hidden_dim),
        transforms=int(args.nf_transforms),
    ).to(device)
    train_out = _train_conditional_nf(
        model=model,
        theta_train=theta_n_tr,
        x_train=x_n_tr,
        theta_val=theta_n_va,
        x_val=x_n_va,
        device=device,
        epochs=int(args.nf_epochs),
        batch_size=int(args.nf_batch_size),
        lr=float(args.nf_lr),
        patience=int(args.nf_early_patience),
        min_delta=float(args.nf_early_min_delta),
        ema_alpha=float(args.nf_early_ema_alpha),
    )
    run_dir = out_dir / "sweep_runs" / f"n_{int(n):06d}"
    nf_diag = _run_nf_fixed_x_diagnostic(
        model=model,
        meta=meta,
        x_fixed_global=x_fixed_global,
        x_index_global=int(posterior_x_index),
        n_samples=20000,
        run_dir=run_dir,
        dataset_npz=dataset_npz,
        device=device,
    )

    c_matrix = _compute_c_matrix_nf(
        model=model,
        theta_all=theta_n_all,
        x_all=x_n_all,
        device=device,
        pair_batch_size=int(args.nf_pair_batch_size),
    )
    delta_l = _compute_delta_l(c_matrix)
    h_directed = _compute_h_directed(delta_l)
    h_sym = _symmetrize(h_directed)

    loss_fig = out_dir / "nf_score_loss_vs_epoch.png"
    ep = np.arange(1, int(np.asarray(train_out["train_losses"]).size) + 1)
    plt.figure(figsize=(8.8, 5.0))
    plt.plot(ep, np.asarray(train_out["train_losses"], dtype=np.float64), linewidth=2.0, label="NF train NLL")
    plt.plot(ep, np.asarray(train_out["val_losses"], dtype=np.float64), linewidth=2.0, label="NF val NLL")
    plt.plot(
        ep,
        np.asarray(train_out["val_ema_losses"], dtype=np.float64),
        linewidth=2.0,
        linestyle="--",
        label=f"NF val EMA (alpha={float(args.nf_early_ema_alpha):g})",
    )
    be = int(train_out["best_epoch"])
    if 1 <= be <= int(ep.size):
        plt.axvline(be, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Best epoch {be}")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log likelihood")
    plt.title("Conditional theta NF training")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=180)
    plt.close()

    h_fig = out_dir / "nf_h_matrix_sym_heatmap.png"
    d_fig = out_dir / "nf_delta_l_heatmap.png"
    _save_heatmap(h_fig, h_sym, "NF symmetric H-matrix heatmap", r"$H^{sym}_{ij}$")
    _save_heatmap(d_fig, delta_l, "NF directed DeltaL heatmap", r"$\Delta L_{ij}$")

    h_npz = out_dir / "nf_h_matrix_results_theta_cov.npz"
    np.savez_compressed(
        h_npz,
        theta_used=np.asarray(theta_n_all, dtype=np.float64).reshape(-1),
        c_matrix=c_matrix,
        delta_l_matrix=delta_l,
        h_directed=h_directed,
        h_sym=h_sym,
        n_samples=np.int32(theta_n_all.shape[0]),
        field_method=np.asarray(["nf_conditional_loglik"], dtype=object),
        prior_assumption=np.asarray(["uniform_constant_canceled_in_deltaL"], dtype=object),
        nf_epochs=np.int32(int(args.nf_epochs)),
        nf_batch_size=np.int32(int(args.nf_batch_size)),
        nf_lr=np.float64(float(args.nf_lr)),
        nf_hidden_dim=np.int32(int(args.nf_hidden_dim)),
        nf_context_dim=np.int32(int(args.nf_context_dim)),
        nf_transforms=np.int32(int(args.nf_transforms)),
        nf_pair_batch_size=np.int32(int(args.nf_pair_batch_size)),
        nf_best_epoch=np.int32(int(train_out["best_epoch"])),
        nf_stopped_epoch=np.int32(int(train_out["stopped_epoch"])),
        nf_best_val_ema=np.float64(float(train_out["best_val_ema"])),
    )

    h_binned, _ = vhb.average_matrix_by_bins(h_sym, bin_n_all, int(n_bins))
    h_binned_sqrt = np.sqrt(np.clip(np.asarray(h_binned, dtype=np.float64), 0.0, 1.0))
    corr_h = vhb.matrix_corr_offdiag_pearson(h_binned_sqrt, h_gt_sqrt)
    corr_clf = vhb.matrix_corr_offdiag_pearson(clf_n, clf_ref)

    binned_npz = out_dir / "nf_h_binned_results.npz"
    np.savez_compressed(
        binned_npz,
        n=np.asarray([int(n)], dtype=np.int64),
        n_ref=np.int64(int(n_ref)),
        theta_bin_edges=edges,
        theta_bin_centers=centers,
        h_binned=np.asarray(h_binned, dtype=np.float64),
        h_binned_sqrt=h_binned_sqrt,
        clf_acc_n=np.asarray(clf_n, dtype=np.float64),
        clf_acc_ref=np.asarray(clf_ref, dtype=np.float64),
        hellinger_gt_sq_mc=np.asarray(h_gt_sqrt, dtype=np.float64),
        corr_h_binned_vs_gt_mc=np.asarray([corr_h], dtype=np.float64),
        corr_clf_vs_ref=np.asarray([corr_clf], dtype=np.float64),
    )

    loss_dir_nf = out_dir / "training_losses_nf"
    loss_dir_nf.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        loss_dir_nf / f"n_{int(n):06d}.npz",
        theta_field_method=np.asarray(["nf"], dtype=object),
        prior_enable=np.bool_(False),
        score_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
        score_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
        score_val_monitor_losses=np.asarray(train_out["val_ema_losses"], dtype=np.float64),
        prior_train_losses=np.asarray([], dtype=np.float64),
        prior_val_losses=np.asarray([], dtype=np.float64),
        prior_val_monitor_losses=np.asarray([], dtype=np.float64),
    )
    with (loss_dir_nf / "manifest.txt").open("w", encoding="utf-8") as mf:
        mf.write("# n\tstatus\trun_dir\tsrc_loss_npz\tdst_loss_npz\tnote\n")
        path = loss_dir_nf / f"n_{int(n):06d}.npz"
        mf.write(f"{n}\tnf\t\t{path}\t{path}\tnf-only training losses\n")

    combined_nf_png = out_dir / "h_decoding_convergence_combined_nf.png"
    combined_nf_svg = shdc._save_combined_convergence_figure(
        h_mats=[h_binned_sqrt, h_gt_sqrt],
        clf_mats=[np.asarray(clf_n, dtype=np.float64), np.asarray(clf_ref, dtype=np.float64)],
        col_labels=[f"n={n}", f"Approx GT, n_ref={n_ref}"],
        n_bins=int(n_bins),
        theta_centers=np.asarray(centers, dtype=np.float64),
        ns=[int(n)],
        corr_h=np.asarray([corr_h], dtype=np.float64),
        corr_clf=np.asarray([corr_clf], dtype=np.float64),
        loss_dir=str(loss_dir_nf),
        diagnostic_png_path=str(nf_diag["figure_png"]),
        out_png_path=str(combined_nf_png),
        dpi=160,
    )

    return {
        "n_bins": int(n_bins),
        "corr_h_binned_vs_gt_mc": float(corr_h),
        "corr_clf_vs_ref": float(corr_clf),
        "loss_fig": str(loss_fig),
        "h_npz": str(h_npz),
        "h_fig": str(h_fig),
        "delta_fig": str(d_fig),
        "binned_npz": str(binned_npz),
        "combined_png": str(combined_nf_png),
        "combined_svg": str(combined_nf_svg),
        "diag_png": str(nf_diag["figure_png"]),
        "diag_svg": str(nf_diag["figure_svg"]),
        "diag_npz": str(nf_diag["samples_npz"]),
        "diag_txt": str(nf_diag["summary_txt"]),
    }


def main() -> None:
    args = _build_parser().parse_args()
    x_dim = int(args.x_dim)
    if x_dim < 2:
        raise ValueError(f"--x-dim must be >= 2, got {x_dim}.")
    out_raw = args.output_dir if args.output_dir is not None else _default_output_dir(x_dim)
    out_dir = _normalize_output_dir(out_raw)
    os.makedirs(out_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")

    n = 200
    n_ref = 1000
    dataset_npz = out_dir / "shared_dataset.npz"
    _write_dataset(dataset_npz, x_dim=x_dim)
    print(f"[repro] x_dim={x_dim} dataset_npz={dataset_npz}", flush=True)
    shared = load_shared_dataset_npz(dataset_npz)
    x_all_shared = np.asarray(shared.x_all, dtype=np.float64)
    posterior_x_index = int(args.posterior_x_index)
    if posterior_x_index < 0 or posterior_x_index >= int(x_all_shared.shape[0]):
        raise ValueError(
            f"--posterior-x-index out of range: {posterior_x_index} not in [0, {x_all_shared.shape[0] - 1}]"
        )
    x_fixed_preview = np.asarray(x_all_shared[posterior_x_index], dtype=np.float64).reshape(-1)
    x_preview = np.array2string(x_fixed_preview, precision=4, suppress_small=False, max_line_width=1000)
    print(f"[repro] shared posterior x-index={posterior_x_index} x={x_preview}", flush=True)
    method = str(args.method).strip().lower()
    run_theta = method in ("theta-flow", "both")
    run_nf = method in ("nf", "both")
    dev = torch.device(args.device)
    theta_out: dict[str, float | int | str] | None = None
    nf_out: dict[str, float | int | str] | None = None

    if run_theta:
        theta_out = _run_theta_flow_branch(
            out_dir=out_dir,
            dataset_npz=dataset_npz,
            n=n,
            n_ref=n_ref,
            posterior_x_index=posterior_x_index,
            args=args,
        )
    if run_nf:
        print("[repro][nf] running NF H-matrix branch...", flush=True)
        nf_out = _run_nf_branch(
            out_dir=out_dir,
            dataset_npz=dataset_npz,
            x_dim=x_dim,
            n=n,
            n_ref=n_ref,
            posterior_x_index=posterior_x_index,
            args=args,
            device=dev,
        )

    print("[repro] completed.", flush=True)
    print(f"[repro] output_dir={out_dir}", flush=True)
    print(f"[repro] method={method}", flush=True)
    if theta_out is not None:
        gap = float(theta_out["corr_h_binned_vs_gt_mc"]) - float(theta_out["corr_clf_vs_ref"])
        print(f"[repro][theta-flow] results_csv={theta_out['results_csv']}", flush=True)
        print(
            "[repro][theta-flow] n={} corr_h_binned_vs_gt_mc={:.6f} corr_clf_vs_ref={:.6f} gap(h-clf)={:.6f}".format(
                int(theta_out["n"]),
                float(theta_out["corr_h_binned_vs_gt_mc"]),
                float(theta_out["corr_clf_vs_ref"]),
                float(gap),
            ),
            flush=True,
        )
        print(f"[repro][theta-flow] combined_png={theta_out['combined_png']}", flush=True)
    if nf_out is not None:
        print(
            "[repro][nf] n_bins={} corr_h_binned_vs_gt_mc={:.6f} corr_clf_vs_ref={:.6f}".format(
                int(nf_out["n_bins"]),
                float(nf_out["corr_h_binned_vs_gt_mc"]),
                float(nf_out["corr_clf_vs_ref"]),
            ),
            flush=True,
        )
        print(f"[repro][nf] loss_fig={nf_out['loss_fig']}", flush=True)
        print(f"[repro][nf] h_npz={nf_out['h_npz']}", flush=True)
        print(f"[repro][nf] h_fig={nf_out['h_fig']}", flush=True)
        print(f"[repro][nf] delta_fig={nf_out['delta_fig']}", flush=True)
        print(f"[repro][nf] binned_npz={nf_out['binned_npz']}", flush=True)
        print(f"[repro][nf] combined_png={nf_out['combined_png']}", flush=True)
        print(f"[repro][nf] combined_svg={nf_out['combined_svg']}", flush=True)
        print(f"[repro][nf] diag_png={nf_out['diag_png']}", flush=True)
        print(f"[repro][nf] diag_svg={nf_out['diag_svg']}", flush=True)
        print(f"[repro][nf] diag_npz={nf_out['diag_npz']}", flush=True)
        print(f"[repro][nf] diag_txt={nf_out['diag_txt']}", flush=True)


if __name__ == "__main__":
    main()
