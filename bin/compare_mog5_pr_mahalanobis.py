#!/usr/bin/env python3
"""Compare Mahalanobis distance on the MoG5 PR dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BIN_DIR = Path(__file__).resolve().parent
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

from fisher.distance_comparison import METRIC_MAHALANOBIS_SQ

import compare_mog5_pr_distances as _base

FlowComparisonConfig = _base.FlowComparisonConfig
assemble_comparison_result = _base.assemble_comparison_result
classical_metric_matrices = _base.classical_metric_matrices
condition_labels = _base.condition_labels
default_dataset_dir = _base.default_dataset_dir
ensure_dataset = _base.ensure_dataset
flow_metric_matrices = _base.flow_metric_matrices
labels_from_theta = _base.labels_from_theta
load_shared_dataset_npz = _base.load_shared_dataset_npz
pr_autoencoder_ground_truth_matrices = _base.pr_autoencoder_ground_truth_matrices
require_device = _base.require_device
write_pairs_csv = _base.write_pairs_csv
write_results_npz = _base.write_results_npz
write_summary_json = _base.write_summary_json
_flow_config_from_args = _base._flow_config_from_args
_validate_bundle = _base._validate_bundle


def build_parser():
    p = _base.build_parser()
    p.set_defaults(metric=METRIC_MAHALANOBIS_SQ, output_dir=None)
    return p


def resolve_dataset_dir(args) -> Path:
    return _base.resolve_dataset_dir(args)


def resolve_output_dir(args) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    return resolve_dataset_dir(args) / "mahalanobis_comparison_flow_skl"


def resolve_metric_names(args) -> tuple[str, ...]:
    del args
    return (METRIC_MAHALANOBIS_SQ,)


def run(args) -> dict[str, Path]:
    dev = require_device(str(args.device))
    metrics = resolve_metric_names(args)
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

    classical = classical_metric_matrices(
        projected_bundle.x_all,
        labels,
        num_categories=k,
        metrics=metrics,
        mahalanobis_ridge=float(args.mahalanobis_ridge),
        skl_folds=int(args.skl_folds),
        skl_seed=int(args.seed),
        skl_logistic_c=float(args.skl_logistic_c),
    )
    ground_truth = pr_autoencoder_ground_truth_matrices(
        native_meta=dict(native_bundle.meta),
        projected_meta=dict(projected_bundle.meta),
        device=dev,
        cache_dir=Path(args.pr_cache_dir),
        samples_per_class=int(args.gt_samples_per_class),
        seed=int(args.seed) + 12345,
        batch_size=int(args.gt_batch_size),
        mahalanobis_ridge=float(args.mahalanobis_ridge),
        metrics=metrics,
    )
    ground_truth = {metric: ground_truth[metric] for metric in metrics}
    flow, flow_paths = flow_metric_matrices(
        bundle=projected_bundle,
        device=dev,
        output_dir=output_dir / "flow",
        config=_flow_config_from_args(args),
        seed=int(args.seed),
        metrics=metrics,
    )

    result = assemble_comparison_result(
        metrics=metrics,
        condition_names=names,
        classical=classical,
        flow_matching=flow,
        ground_truth=ground_truth,
        flow_npz_paths=flow_paths,
    )
    results_npz = write_results_npz(output_dir / "mog5_pr_mahalanobis_comparison_results.npz", result)
    pairs_csv = write_pairs_csv(output_dir / "mog5_pr_mahalanobis_comparison_pairs.csv", result.rows)
    summary_json = write_summary_json(
        output_dir / "mog5_pr_mahalanobis_comparison_summary.json",
        result=result,
        extra={
            "script": "bin/compare_mog5_pr_mahalanobis.py",
            "device": str(dev),
            "n_total": int(args.n_total),
            "pr_dim": int(args.pr_dim),
            "seed": int(args.seed),
            "dataset_dir": str(dataset_dir),
            "native_npz": str(native_npz),
            "projected_npz": str(projected_npz),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "pairs_csv": str(pairs_csv),
            "metric": METRIC_MAHALANOBIS_SQ,
            "metrics": list(metrics),
            "gt_samples_per_class": int(args.gt_samples_per_class),
            "gt_batch_size": int(args.gt_batch_size),
            "mahalanobis_ridge": float(args.mahalanobis_ridge),
            "skl_folds": int(args.skl_folds),
            "skl_logistic_c": float(args.skl_logistic_c),
            "pr_cache_dir": str(Path(args.pr_cache_dir)),
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
