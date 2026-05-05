#!/usr/bin/env python3
"""Run one bounded 2D-theta benchmark loop for experiment-loop mode.

The current two-figure H/decoding study bins scalar theta.  This runner creates
a source archive with two theta columns, then benchmarks the first coordinate
with the second coordinate stored as an inert nuisance coordinate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args


METHODS_ALL = (
    "bin_gaussian",
    "theta_path_integral",
    "theta_flow",
    "x_flow",
    "linear_x_flow_t",
    "linear_x_flow_t_aug",
    "linear_x_flow_t_noise",
    "linear_x_flow_diagonal_t",
    "linear_x_flow_low_rank_t",
)


def _dataset_args(
    *,
    family: str,
    seed: int,
    n_total: int,
    x_dim: int,
    obs_noise_scale: float,
    cov_theta_amp_scale: float,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_dataset_arguments(parser)
    ns = parser.parse_args([])
    ns.dataset_family = family
    ns.seed = int(seed)
    ns.n_total = int(n_total)
    ns.x_dim = int(x_dim)
    ns.train_frac = 0.7
    ns.obs_noise_scale = float(obs_noise_scale)
    ns.cov_theta_amp_scale = float(cov_theta_amp_scale)
    return ns


def _save_dataset_pair(args: argparse.Namespace, loop_dir: Path) -> dict[str, str | int | float]:
    ds_args = _dataset_args(
        family=args.dataset_family,
        seed=args.seed,
        n_total=args.n_total,
        x_dim=args.x_dim,
        obs_noise_scale=args.obs_noise_scale,
        cov_theta_amp_scale=args.cov_theta_amp_scale,
    )
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    dataset = build_dataset_from_args(ds_args)
    theta1, x_all = dataset.sample_joint(args.n_total)
    theta2 = rng.uniform(float(ds_args.theta_low), float(ds_args.theta_high), size=(args.n_total, 1))
    theta2d = np.concatenate([np.asarray(theta1, dtype=np.float64).reshape(-1, 1), theta2], axis=1)
    perm = rng.permutation(args.n_total)
    n_train = min(max(int(float(ds_args.train_frac) * args.n_total), 1), args.n_total - 1)
    train_idx = perm[:n_train].astype(np.int64)
    validation_idx = perm[n_train:].astype(np.int64)

    meta = meta_dict_from_args(ds_args)
    if hasattr(dataset, "_randamp_amp"):
        meta["randamp_mu_amp_per_dim"] = dataset._randamp_amp.tolist()
    if hasattr(dataset, "_cosine_tune_amp"):
        meta["cosine_tune_amp_per_dim"] = dataset._cosine_tune_amp.tolist()

    source_meta = dict(meta)
    source_meta["theta_dim"] = 2
    source_meta["theta2_role"] = "inert_nuisance_coordinate"
    source_meta["benchmark_theta_coordinate"] = 0
    source_meta["benchmark_note"] = (
        "Source dataset stores theta=(theta1, theta2). x is generated from the canonical "
        "one-dimensional family using theta1; theta2 is sampled independently and archived "
        "to exercise two-column theta data while preserving the scalar H target used by "
        "study_h_decoding_twofig.py."
    )
    scalar_meta = dict(meta)
    scalar_meta["source_theta2d_npz"] = str(loop_dir / "dataset_theta2d_source.npz")
    scalar_meta["source_theta_dim"] = 2
    scalar_meta["benchmark_theta_coordinate"] = 0

    source_npz = loop_dir / "dataset_theta2d_source.npz"
    scalar_npz = loop_dir / "dataset_theta1_view_for_twofig.npz"
    save_shared_dataset_npz(
        source_npz,
        meta=source_meta,
        theta_all=theta2d,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta2d[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta2d[validation_idx],
        x_validation=x_all[validation_idx],
    )
    save_shared_dataset_npz(
        scalar_npz,
        meta=scalar_meta,
        theta_all=theta1,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta1[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta1[validation_idx],
        x_validation=x_all[validation_idx],
    )
    return {
        "source_npz": str(source_npz),
        "scalar_view_npz": str(scalar_npz),
        "n_total": int(args.n_total),
        "n_train": int(n_train),
        "n_validation": int(args.n_total - n_train),
        "theta2_std": float(np.std(theta2)),
    }


def _run_study(args: argparse.Namespace, loop_dir: Path, scalar_npz: str) -> dict[str, object]:
    study_dir = loop_dir / "study_twofig"
    study_dir.mkdir(parents=True, exist_ok=True)
    rows = args.theta_field_rows
    cmd = [
        sys.executable,
        "bin/study_h_decoding_twofig.py",
        "--dataset-npz",
        scalar_npz,
        "--dataset-family",
        args.dataset_family,
        "--output-dir",
        str(study_dir),
        "--n-ref",
        str(args.n_ref),
        "--n-list",
        args.n_list,
        "--num-theta-bins",
        str(args.num_theta_bins),
        "--theta-field-rows",
        rows,
        "--run-seed",
        str(args.seed),
        "--gt-hellinger-seed",
        str(args.seed + 101),
        "--score-epochs",
        str(args.epochs),
        "--flow-epochs",
        str(args.epochs),
        "--prior-epochs",
        str(args.epochs),
        "--lxf-epochs",
        str(args.epochs),
        "--lxf-nlpca-epochs",
        str(args.nlpca_epochs),
        "--score-hidden-dim",
        str(args.hidden_dim),
        "--flow-hidden-dim",
        str(args.hidden_dim),
        "--lxf-hidden-dim",
        str(args.hidden_dim),
        "--lxf-nlpca-dim",
        str(min(3, args.x_dim)),
        "--score-depth",
        "1",
        "--flow-depth",
        "1",
        "--lxf-depth",
        "1",
        "--score-batch-size",
        "128",
        "--flow-batch-size",
        "128",
        "--lxf-batch-size",
        "128",
        "--clf-min-class-count",
        "2",
        "--device",
        args.device,
    ]
    t0 = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(_repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    study_log = loop_dir / "study_command.log"
    study_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"study_h_decoding_twofig.py failed; see {study_log}")

    result_npz = study_dir / "h_decoding_twofig_results.npz"
    with np.load(result_npz, allow_pickle=True) as z:
        row_labels = [str(x) for x in np.asarray(z["theta_field_rows"]).reshape(-1)]
        n_values = [int(x) for x in np.asarray(z["n"]).reshape(-1)]
        corr_h = np.asarray(z["corr_h_binned_vs_gt_mc"], dtype=np.float64)
        nmse_h = np.asarray(z["nmse_h_binned_vs_gt_mc"], dtype=np.float64)
        corr_decode = np.asarray(z["corr_decode_vs_ref_shared"], dtype=np.float64).reshape(-1)
        nmse_decode = np.asarray(z["nmse_decode_vs_ref_shared"], dtype=np.float64).reshape(-1)
        wall = np.asarray(z["wall_seconds"], dtype=np.float64)

    by_method = {}
    for i, label in enumerate(row_labels):
        by_method[label] = {
            "corr_h_binned_vs_gt_mc": [float(x) for x in corr_h[i]],
            "nmse_h_binned_vs_gt_mc": [float(x) for x in nmse_h[i]],
            "wall_seconds": [float(x) for x in wall[i]],
        }
    return {
        "study_dir": str(study_dir),
        "study_result_npz": str(result_npz),
        "study_log": str(study_log),
        "study_wall_seconds": float(time.time() - t0),
        "command": cmd,
        "n_values": n_values,
        "methods": row_labels,
        "by_method": by_method,
        "decode": {
            "corr_decode_vs_ref_shared": [float(x) for x in corr_decode],
            "nmse_decode_vs_ref_shared": [float(x) for x in nmse_decode],
        },
        "figures": [
            str(study_dir / "h_decoding_twofig_sweep.svg"),
            str(study_dir / "h_decoding_twofig_gt.svg"),
            str(study_dir / "h_decoding_twofig_corr.svg"),
            str(study_dir / "h_decoding_twofig_nmse.svg"),
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loop-dir", required=True)
    p.add_argument("--dataset-name", required=True, choices=["linearbench", "cosinebench"])
    p.add_argument("--dataset-family", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-total", type=int, default=640)
    p.add_argument("--x-dim", type=int, default=5)
    p.add_argument("--obs-noise-scale", type=float, default=1.0)
    p.add_argument("--cov-theta-amp-scale", type=float, default=1.0)
    p.add_argument("--n-ref", type=int, default=240)
    p.add_argument("--n-list", default="120,240")
    p.add_argument("--num-theta-bins", type=int, default=6)
    p.add_argument("--theta-field-rows", default=",".join(METHODS_ALL))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--nlpca-epochs", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    loop_dir = Path(args.loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = loop_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    dataset_metrics = _save_dataset_pair(args, loop_dir)
    study_metrics = _run_study(args, loop_dir, str(dataset_metrics["scalar_view_npz"]))

    metrics = {
        "loop_dir": str(loop_dir),
        "dataset_name": args.dataset_name,
        "dataset_family": args.dataset_family,
        "seed": int(args.seed),
        "theta_dim_source": 2,
        "benchmark_theta_coordinate": 0,
        "dataset": dataset_metrics,
        "study": study_metrics,
    }
    (loop_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    best = []
    for method, vals in study_metrics["by_method"].items():
        corr = vals["corr_h_binned_vs_gt_mc"][-1]
        nmse = vals["nmse_h_binned_vs_gt_mc"][-1]
        best.append((method, corr, nmse))
    lines = [
        f"# Loop notes: {args.dataset_name}",
        "",
        "Hypothesis: a two-column theta source archive can be created while the existing scalar two-figure study benchmarks theta1.",
        "",
        f"Source dataset: `{dataset_metrics['source_npz']}`",
        f"Scalar study view: `{dataset_metrics['scalar_view_npz']}`",
        "",
        "Final-column H metrics:",
    ]
    for method, corr, nmse in best:
        lines.append(f"- `{method}`: corr={corr:.4g}, nmse={nmse:.4g}")
    lines.extend(
        [
            "",
            "Interpretation: these metrics assess recovery of the first theta coordinate. The second coordinate is archived as an independent nuisance coordinate and is not part of the scalar H target.",
            "",
            "Next-loop decision: compare the other benchmark alias, or if both aliases have run, summarize limitations and suggested full 2D H extensions.",
        ]
    )
    (loop_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
