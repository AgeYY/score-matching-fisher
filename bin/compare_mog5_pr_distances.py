#!/usr/bin/env python3
"""Compare classical, flow-matching, and ground-truth distances on MoG5 PR data."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import (
    METRIC_NAMES,
    FlowComparisonConfig,
    assemble_comparison_result,
    classical_metric_matrices,
    condition_labels,
    flow_metric_matrices,
    labels_from_theta,
    pr_autoencoder_ground_truth_matrices,
    write_pairs_csv,
    write_results_npz,
    write_summary_json,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


def _load_make_mog5_module() -> Any:
    path = _REPO_ROOT / "bin" / "make_mog5_pr_dataset.py"
    spec = importlib.util.spec_from_file_location("make_mog5_pr_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_dataset_dir(*, n_total: int = 1_000, pr_dim: int = 5) -> Path:
    mod = _load_make_mog5_module()
    return Path(mod.default_output_dir(n_total=int(n_total), pr_dim=int(pr_dim)))


def default_output_dir(*, n_total: int = 1_000, pr_dim: int = 5) -> Path:
    return default_dataset_dir(n_total=int(n_total), pr_dim=int(pr_dim)) / "distance_comparison_flow_skl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-total", type=int, default=1_000, help="MoG5 PR dataset row count.")
    p.add_argument("--pr-dim", type=int, default=5, help="MoG5 PR embedded dimension.")
    p.add_argument("--seed", type=int, default=7, help="Dataset, flow, and estimation seed.")
    p.add_argument("--device", type=str, default="cuda", help="Execution device for PR GT encoding and flows.")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="MoG5 PR dataset directory. Defaults to <repo-root>/data/mog_5pr{pr_dim}_n{n_total}.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Comparison output directory. Defaults to <dataset-dir>/distance_comparison_flow_skl.",
    )
    p.add_argument("--force-dataset", action="store_true", help="Regenerate native and projected MoG5 PR NPZs.")
    p.add_argument(
        "--dataset-use-cache",
        action="store_true",
        help="Allow the PR projection wrapper to reuse a matching PR autoencoder checkpoint if generation is needed.",
    )
    p.add_argument("--skip-dataset-viz", action="store_true", help="Skip dataset visualization if generation is needed.")
    p.add_argument(
        "--pr-cache-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "pr_autoencoder_cache",
        help="PR autoencoder cache directory used for ground-truth projected sampling.",
    )

    p.add_argument("--gt-samples-per-class", type=int, default=100_000)
    p.add_argument("--gt-batch-size", type=int, default=8192)
    p.add_argument("--mahalanobis-ridge", type=float, default=1e-6)
    p.add_argument("--skl-folds", type=int, default=5)
    p.add_argument("--skl-logistic-c", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=20_000)
    p.add_argument("--early-patience", type=int, default=1_000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--low-rank-dim", type=int, default=4)
    p.add_argument("--radius", type=float, default=1.0, help="Fixed norm radius for cosine/correlation flow rows.")
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--t-eps", type=float, default=0.05)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--mc-jeffreys-sample", dest="mc_jeffreys_sample", type=int, default=4096)
    p.add_argument("--mc-samples", dest="mc_jeffreys_sample", type=int, help=argparse.SUPPRESS)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    return p


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir).expanduser()
    return default_dataset_dir(n_total=int(args.n_total), pr_dim=int(args.pr_dim))


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    return resolve_dataset_dir(args) / "distance_comparison_flow_skl"


def _dataset_wrapper_args(args: argparse.Namespace, dataset_dir: Path) -> argparse.Namespace:
    mod = _load_make_mog5_module()
    argv = [
        "--n-total",
        str(int(args.n_total)),
        "--pr-dim",
        str(int(args.pr_dim)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--output-dir",
        str(dataset_dir),
        "--pr-cache-dir",
        str(Path(args.pr_cache_dir)),
    ]
    if bool(args.force_dataset):
        argv.append("--force")
    if bool(args.dataset_use_cache):
        argv.append("--use-cache")
    if bool(args.skip_dataset_viz):
        argv.append("--skip-viz")
    return mod.parse_args(argv)


def ensure_dataset(args: argparse.Namespace, dataset_dir: Path) -> tuple[Path, Path]:
    mod = _load_make_mog5_module()
    return mod.run(_dataset_wrapper_args(args, dataset_dir))


def _flow_config_from_args(args: argparse.Namespace) -> FlowComparisonConfig:
    return FlowComparisonConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        early_min_delta=float(args.early_min_delta),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        low_rank_dim=int(args.low_rank_dim),
        radius=float(args.radius),
        path_schedule=str(args.path_schedule),
        t_eps=float(args.t_eps),
        quadrature_steps=int(args.quadrature_steps),
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        solve_jitter=float(args.solve_jitter),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
    )


def _validate_bundle(bundle, *, n_total: int, pr_dim: int) -> None:
    meta = dict(bundle.meta)
    if str(meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical, got {meta.get('dataset_family')!r}.")
    if int(meta.get("num_categories", -1)) != 5:
        raise ValueError(f"Expected num_categories=5, got {meta.get('num_categories')!r}.")
    if int(meta.get("x_dim", -1)) != int(pr_dim):
        raise ValueError(f"Expected projected x_dim={pr_dim}, got {meta.get('x_dim')!r}.")
    if int(np.asarray(bundle.x_all).shape[0]) != int(n_total):
        raise ValueError(f"Expected n_total={n_total}, got {np.asarray(bundle.x_all).shape[0]}.")
    if not bool(meta.get("pr_autoencoder_embedded", False)):
        raise ValueError("Projected dataset must have pr_autoencoder_embedded=True.")


def run(args: argparse.Namespace) -> dict[str, Path]:
    dev = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    dataset_dir = resolve_dataset_dir(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_npz, projected_npz = ensure_dataset(args, dataset_dir)

    native_bundle = load_shared_dataset_npz(native_npz)
    projected_bundle = load_shared_dataset_npz(projected_npz)
    _validate_bundle(projected_bundle, n_total=int(args.n_total), pr_dim=int(args.pr_dim))

    k = int(projected_bundle.meta.get("num_categories", 5))
    labels = labels_from_theta(projected_bundle.theta_all, num_categories=k)
    names = condition_labels(k)

    print("[distance-comparison] computing classical finite-sample metrics", flush=True)
    classical = classical_metric_matrices(
        projected_bundle.x_all,
        labels,
        num_categories=k,
        metrics=METRIC_NAMES,
        mahalanobis_ridge=float(args.mahalanobis_ridge),
        skl_folds=int(args.skl_folds),
        skl_seed=int(args.seed),
        skl_logistic_c=float(args.skl_logistic_c),
    )

    print("[distance-comparison] computing projected-coordinate Monte Carlo ground truth", flush=True)
    ground_truth = pr_autoencoder_ground_truth_matrices(
        native_meta=dict(native_bundle.meta),
        projected_meta=dict(projected_bundle.meta),
        device=dev,
        cache_dir=Path(args.pr_cache_dir),
        samples_per_class=int(args.gt_samples_per_class),
        seed=int(args.seed) + 12345,
        batch_size=int(args.gt_batch_size),
        mahalanobis_ridge=float(args.mahalanobis_ridge),
    )

    print("[distance-comparison] training flow-matching metrics", flush=True)
    flow, flow_paths = flow_metric_matrices(
        bundle=projected_bundle,
        device=dev,
        output_dir=output_dir / "flow",
        config=_flow_config_from_args(args),
        seed=int(args.seed),
        metrics=METRIC_NAMES,
    )

    result = assemble_comparison_result(
        metrics=METRIC_NAMES,
        condition_names=names,
        classical=classical,
        flow_matching=flow,
        ground_truth=ground_truth,
        flow_npz_paths=flow_paths,
    )

    results_npz = write_results_npz(output_dir / "mog5_pr_distance_comparison_results.npz", result)
    pairs_csv = write_pairs_csv(output_dir / "mog5_pr_distance_comparison_pairs.csv", result.rows)
    summary_json = write_summary_json(
        output_dir / "mog5_pr_distance_comparison_summary.json",
        result=result,
        extra={
            "script": "bin/compare_mog5_pr_distances.py",
            "device": str(dev),
            "dataset_dir": str(dataset_dir),
            "native_npz": str(native_npz),
            "projected_npz": str(projected_npz),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "pairs_csv": str(pairs_csv),
            "gt_samples_per_class": int(args.gt_samples_per_class),
            "flow_defaults": vars(_flow_config_from_args(args)),
        },
    )
    print(f"results_npz: {results_npz}", flush=True)
    print(f"pairs_csv: {pairs_csv}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    return {
        "output_dir": output_dir,
        "results_npz": results_npz,
        "pairs_csv": pairs_csv,
        "summary_json": summary_json,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
