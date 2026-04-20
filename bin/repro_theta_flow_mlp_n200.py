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
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args

import numpy as np
import torch

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
    print("[repro] running study_h_decoding_convergence with fixed config...", flush=True)
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
        "0",
        "--n-samples",
        "20000",
        "--device",
        args.device,
    ]
    print("[repro] running fixed-x posterior+tuning diagnostic...", flush=True)
    diag_res = subprocess.run(diag_cmd, cwd=str(_repo_root), check=False)
    if diag_res.returncode != 0:
        raise RuntimeError(f"diagnose_theta_flow_single_x_samples failed with code {diag_res.returncode}.")

    # Combined figure is generated during the first study run (before diagnostic exists).
    # Regenerate figures from cached NPZ so the diagnostic panel is embedded.
    viz_cmd = cmd + ["--visualization-only"]
    print("[repro] regenerating combined figures with embedded diagnostic panel...", flush=True)
    viz_res = subprocess.run(viz_cmd, cwd=str(_repo_root), check=False)
    if viz_res.returncode != 0:
        raise RuntimeError(f"study_h_decoding_convergence --visualization-only failed with code {viz_res.returncode}.")

    results_csv = out_dir / "h_decoding_convergence_results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing expected results CSV: {results_csv}")
    n_row, corr_h, corr_clf = _parse_metrics(results_csv)
    gap = corr_h - corr_clf

    print("[repro] completed.", flush=True)
    print(f"[repro] output_dir={out_dir}", flush=True)
    print(f"[repro] results_csv={results_csv}", flush=True)
    print(
        "[repro] n={} corr_h_binned_vs_gt_mc={:.6f} corr_clf_vs_ref={:.6f} gap(h-clf)={:.6f}".format(
            n_row,
            corr_h,
            corr_clf,
            gap,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
