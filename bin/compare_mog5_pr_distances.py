#!/usr/bin/env python3
"""Compare classical, flow-matching, and ground-truth distances on MoG5 PR data."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import (
    DEFAULT_DEVICE,
    TRAINING_EARLY_STOPPING_PATIENCE,
    TRAINING_MAX_EPOCHS,
)
from fisher.ctsm_distance import (
    CTSMVBinaryJeffreysConfig,
    CTSMVJeffreysConfig,
    save_pairwise_binary_ctsm_v_jeffreys_result,
    save_ctsm_v_jeffreys_result,
    train_and_estimate_pairwise_binary_ctsm_v_jeffreys,
    train_and_estimate_ctsm_v_jeffreys,
)
from fisher.distance_comparison import (
    METRIC_NAMES,
    FlowComparisonConfig,
    assemble_comparison_result,
    classical_metric_matrices,
    condition_labels,
    flow_metric_matrices,
    flow_metric_variants,
    labels_from_theta,
    native_mog_ground_truth_matrices,
    pr_autoencoder_ground_truth_matrices,
    velocity_family_for_metric,
    write_pairs_csv,
    write_results_npz,
    write_summary_json,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from fisher.tre_distance import (
    TRE_ARCHITECTURES,
    TRE_WAYMARK_SCHEDULES,
    TREDensityRatioConfig,
    save_pairwise_tre_jeffreys_result,
    train_and_estimate_pairwise_tre_jeffreys,
)


def _load_make_mog5_module() -> Any:
    path = _REPO_ROOT / "bin" / "make_mog5_pr_dataset.py"
    spec = importlib.util.spec_from_file_location("make_mog5_pr_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_pr_dim(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pr-dim must be an integer, 'none', or 'null'.") from exc


def default_dataset_dir(*, n_total: int = 1_000, pr_dim: int | None = None, native_x_dim: int = 3) -> Path:
    mod = _load_make_mog5_module()
    return Path(mod.default_output_dir(n_total=int(n_total), pr_dim=pr_dim, native_x_dim=int(native_x_dim)))


def default_output_dir(*, n_total: int = 1_000, pr_dim: int | None = None, native_x_dim: int = 3) -> Path:
    return default_dataset_dir(n_total=int(n_total), pr_dim=pr_dim, native_x_dim=int(native_x_dim)) / "distance_comparison_flow_skl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-total", type=int, default=1_000, help="MoG5 PR dataset row count.")
    p.add_argument(
        "--native-x-dim",
        type=int,
        default=3,
        help="Native MoG5 x dimension before optional PR projection.",
    )
    p.add_argument(
        "--pr-dim",
        type=parse_pr_dim,
        default=None,
        help="MoG5 PR embedded dimension. Use 'none' or 'null' for native mode.",
    )
    p.add_argument("--seed", type=int, default=19, help="Dataset, flow, and estimation seed.")
    p.add_argument(
        "--dataset-train-frac",
        type=float,
        default=0.8,
        help="Fraction of generated rows assigned to flow training; the remainder is validation.",
    )
    p.add_argument(
        "--dataset-obs-noise-scale",
        type=float,
        default=1.0,
        help="Scale factor for the random-MoG baseline observation noise.",
    )
    p.add_argument(
        "--dataset-cov-theta-amp-scale",
        type=float,
        default=1.0,
        help="Scale factor for the random-MoG mean-dependent variance term.",
    )
    p.add_argument(
        "--dataset-mog-mean-min-dist",
        type=float,
        default=None,
        help="Minimum pairwise component-mean distance; omitted uses 0.5*sqrt(native_x_dim).",
    )
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Execution device for PR GT encoding and flows.")
    p.add_argument(
        "--native-template-npz",
        type=Path,
        default=None,
        help="Optional native MoG5 NPZ whose fixed component metadata is reused when generating this dataset.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="MoG5 PR dataset directory. Defaults to a dimension-aware <repo-root>/data/mog_5... path.",
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
    p.add_argument(
        "--metric",
        default="all",
        help="Distance metric, comma-separated metric subset, or 'all'.",
    )

    p.add_argument("--gt-samples-per-class", type=int, default=100_000)
    p.add_argument("--gt-batch-size", type=int, default=8192)
    p.add_argument("--mahalanobis-ridge", type=float, default=1e-6)
    p.add_argument("--skl-folds", type=int, default=5)
    p.add_argument("--skl-logistic-c", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    p.add_argument(
        "--early-patience", type=int, default=TRAINING_EARLY_STOPPING_PATIENCE
    )
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument(
        "--lr-schedule-epochs",
        type=int,
        default=None,
        help="Cosine decay horizon; defaults to --epochs and stays at --min-lr afterward.",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--network-architecture", choices=("mlp", "residual_mlp"), default="mlp")
    p.add_argument("--fm-checkpoint-selection", choices=("best", "last"), default="best")
    validation_group = p.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--fixed-validation",
        dest="fixed_validation",
        action="store_true",
        help="Reuse one fixed set of validation base samples and interpolation times across FM epochs.",
    )
    validation_group.add_argument(
        "--no-fixed-validation",
        dest="fixed_validation",
        action="store_false",
        help="Resample validation flow paths at every epoch.",
    )
    p.set_defaults(fixed_validation=True)
    p.add_argument(
        "--fixed-validation-paths",
        type=int,
        default=10,
        help="Number of fixed stratified flow paths per validation observation.",
    )
    p.add_argument("--low-rank-dim", type=int, default=4)
    p.add_argument("--radius", type=float, default=1.0, help="Fixed norm radius for cosine/correlation flow rows.")
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--mc-jeffreys-sample", dest="mc_jeffreys_sample", type=int, default=4096)
    p.add_argument("--mc-samples", dest="mc_jeffreys_sample", type=int, help=argparse.SUPPRESS)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--shared-affine-a-diag-jitter", type=float, default=1e-3)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--flow-normalize-x",
        action="store_true",
        help="Fit one train-only affine normalizer on projected x and use normalized x for flow matching.",
    )
    p.add_argument(
        "--flow-normalize-x-eps",
        type=float,
        default=1e-8,
        help="Minimum std threshold for --flow-normalize-x; smaller stds are treated as constant dimensions.",
    )
    p.add_argument(
        "--flow-likelihood-finetune-epochs",
        type=int,
        default=TRAINING_MAX_EPOCHS,
        help="CNF endpoint-NLL fine-tuning epochs; 0 disables the fine-tuned estimator.",
    )
    p.add_argument(
        "--flow-likelihood-finetune-batch-size",
        type=int,
        default=3000,
        help="NLL fine-tuning batch size; 0 reuses --batch-size.",
    )
    p.add_argument("--flow-likelihood-finetune-lr", type=float, default=3e-5)
    p.add_argument("--flow-likelihood-finetune-weight-decay", type=float, default=0.0)
    p.add_argument("--flow-likelihood-finetune-ode-steps", type=int, default=32)
    p.add_argument("--flow-likelihood-finetune-ode-method", type=str, default="midpoint")
    p.add_argument(
        "--flow-likelihood-finetune-patience",
        type=int,
        default=TRAINING_EARLY_STOPPING_PATIENCE,
    )
    p.add_argument("--flow-likelihood-finetune-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-likelihood-finetune-ema-alpha", type=float, default=0.05)
    p.add_argument(
        "--flow-likelihood-finetune-checkpoint-selection",
        choices=("best", "last"),
        default="best",
    )
    p.add_argument(
        "--flow-likelihood-finetune-divergence-estimator",
        choices=("exact", "hutchinson"),
        default="exact",
    )
    p.add_argument("--flow-likelihood-finetune-hutchinson-probes", type=int, default=1)
    p.add_argument(
        "--flow-likelihood-augment-existing",
        action="store_true",
        help=(
            "Reuse classical, ground-truth, and optional TRE/CTSM-v estimators in the output NPZ; "
            "retrain only flow matching and add its CNF NLL-fine-tuned variant."
        ),
    )
    p.add_argument(
        "--include-tre",
        action="store_true",
        help="Fit one Torch TRE model per condition pair and add its Jeffreys-divergence estimate.",
    )
    p.add_argument(
        "--tre-augment-existing",
        action="store_true",
        help="Reuse estimators in the output NPZ and train only the requested TRE estimator.",
    )
    p.add_argument("--tre-num-bridges", type=int, default=8)
    p.add_argument("--tre-waymark-schedule", choices=TRE_WAYMARK_SCHEDULES, default="angle")
    p.add_argument("--tre-architecture", choices=TRE_ARCHITECTURES, default="mlp")
    p.add_argument("--tre-hidden-dim", type=int, default=128)
    p.add_argument("--tre-depth", type=int, default=3)
    p.add_argument("--tre-epochs", type=int, default=TRAINING_MAX_EPOCHS)
    p.add_argument("--tre-batch-size", type=int, default=512)
    p.add_argument("--tre-lr", type=float, default=1e-3)
    p.add_argument("--tre-weight-decay", type=float, default=0.0)
    p.add_argument(
        "--tre-early-patience", type=int, default=TRAINING_EARLY_STOPPING_PATIENCE
    )
    p.add_argument("--tre-early-min-delta", type=float, default=1e-5)
    p.add_argument("--tre-max-grad-norm", type=float, default=10.0)
    p.add_argument("--tre-validation-pairs", type=int, default=2_048)
    p.add_argument("--tre-eval-batch-size", type=int, default=4_096)
    tre_normalization = p.add_mutually_exclusive_group()
    tre_normalization.add_argument("--tre-standardize", dest="tre_standardize", action="store_true")
    tre_normalization.add_argument("--no-tre-standardize", dest="tre_standardize", action="store_false")
    p.set_defaults(tre_standardize=True)
    p.add_argument(
        "--include-ctsm-v",
        action="store_true",
        help="Train pair-conditioned CTSM-v and add its Jeffreys-divergence estimate.",
    )
    p.add_argument(
        "--ctsm-v-augment-existing",
        action="store_true",
        help="Reuse estimators in the output NPZ and train only requested CTSM-v variants.",
    )
    p.add_argument(
        "--include-ctsm-v-binary",
        action="store_true",
        help="Fit one unconditioned CTSM-v model per condition pair and estimate Jeffreys divergence.",
    )
    p.add_argument("--ctsm-v-binary-epochs", type=int, default=TRAINING_MAX_EPOCHS)
    p.add_argument("--ctsm-v-epochs", type=int, default=TRAINING_MAX_EPOCHS)
    p.add_argument("--ctsm-v-batch-size", type=int, default=512)
    p.add_argument("--ctsm-v-lr", type=float, default=2e-3)
    p.add_argument("--ctsm-v-weight-decay", type=float, default=0.0)
    p.add_argument("--ctsm-v-hidden-dim", type=int, default=256)
    p.add_argument("--ctsm-v-architecture", choices=("mlp", "film"), default="film")
    p.add_argument("--ctsm-v-film-depth", type=int, default=3)
    p.add_argument("--ctsm-v-gated-film", action="store_true")
    p.add_argument("--ctsm-v-raw-time", action="store_true")
    p.add_argument("--ctsm-v-m-scale", type=float, default=1.0)
    p.add_argument("--ctsm-v-delta-scale", type=float, default=0.5)
    p.add_argument("--ctsm-v-two-sb-var", type=float, default=2.0)
    p.add_argument("--ctsm-v-path-schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument("--ctsm-v-path-eps", type=float, default=1e-12)
    p.add_argument("--ctsm-v-factor", type=float, default=1.0)
    p.add_argument("--ctsm-v-t-eps", type=float, default=1e-4)
    p.add_argument("--ctsm-v-integration-steps", type=int, default=300)
    p.add_argument("--ctsm-v-eval-batch-size", type=int, default=4096)
    p.add_argument(
        "--ctsm-v-early-patience",
        type=int,
        default=TRAINING_EARLY_STOPPING_PATIENCE,
    )
    p.add_argument("--ctsm-v-early-min-delta", type=float, default=1e-4)
    p.add_argument("--ctsm-v-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--ctsm-v-validation-batches", type=int, default=8)
    p.add_argument("--ctsm-v-normalize-x", action="store_true")
    return p


def resolve_metric_names(args: argparse.Namespace) -> tuple[str, ...]:
    selection = str(getattr(args, "metric", "all")).strip()
    if selection == "all":
        return tuple(METRIC_NAMES)
    metrics = tuple(dict.fromkeys(part.strip() for part in selection.split(",") if part.strip()))
    invalid = tuple(metric for metric in metrics if metric not in METRIC_NAMES)
    if not metrics or invalid:
        valid = ", ".join(METRIC_NAMES)
        invalid_text = selection if not metrics else ", ".join(invalid)
        raise ValueError(f"Unknown --metric value(s): {invalid_text}. Expected 'all' or a subset of: {valid}.")
    return metrics


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir).expanduser()
    return default_dataset_dir(n_total=int(args.n_total), pr_dim=args.pr_dim, native_x_dim=int(args.native_x_dim))


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    return resolve_dataset_dir(args) / "distance_comparison_flow_skl"


def validate_args(args: argparse.Namespace) -> None:
    metrics = resolve_metric_names(args)
    native_x_dim = int(args.native_x_dim)
    if native_x_dim < 2:
        raise ValueError(f"--native-x-dim must be >= 2; got {args.native_x_dim}.")
    if args.pr_dim is not None and int(args.pr_dim) < native_x_dim:
        raise ValueError(f"--pr-dim must be >= native x_dim={native_x_dim}; got {args.pr_dim}.")
    if not (0.0 < float(args.dataset_train_frac) < 1.0):
        raise ValueError("--dataset-train-frac must be in (0, 1) for train/validation selection.")
    for name in ("dataset_obs_noise_scale", "dataset_cov_theta_amp_scale"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.dataset_mog_mean_min_dist is not None:
        value = float(args.dataset_mog_mean_min_dist)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("--dataset-mog-mean-min-dist must be finite and non-negative.")
    if int(args.fixed_validation_paths) < 1:
        raise ValueError("--fixed-validation-paths must be >= 1.")
    if args.lr_schedule_epochs is not None and int(args.lr_schedule_epochs) < 1:
        raise ValueError("--lr-schedule-epochs must be >= 1.")
    if int(args.flow_likelihood_finetune_epochs) < 0:
        raise ValueError("--flow-likelihood-finetune-epochs must be >= 0.")
    if int(args.flow_likelihood_finetune_batch_size) < 0:
        raise ValueError("--flow-likelihood-finetune-batch-size must be >= 0.")
    if float(args.flow_likelihood_finetune_lr) <= 0.0:
        raise ValueError("--flow-likelihood-finetune-lr must be > 0.")
    if int(args.flow_likelihood_finetune_ode_steps) < 1:
        raise ValueError("--flow-likelihood-finetune-ode-steps must be >= 1.")
    if int(args.flow_likelihood_finetune_patience) < 0:
        raise ValueError("--flow-likelihood-finetune-patience must be >= 0.")
    if float(args.flow_likelihood_finetune_min_delta) < 0.0:
        raise ValueError("--flow-likelihood-finetune-min-delta must be >= 0.")
    if not (0.0 < float(args.flow_likelihood_finetune_ema_alpha) <= 1.0):
        raise ValueError("--flow-likelihood-finetune-ema-alpha must be in (0, 1].")
    if int(args.flow_likelihood_finetune_hutchinson_probes) < 1:
        raise ValueError("--flow-likelihood-finetune-hutchinson-probes must be >= 1.")
    if bool(args.flow_likelihood_augment_existing) and int(args.flow_likelihood_finetune_epochs) <= 0:
        raise ValueError(
            "--flow-likelihood-augment-existing requires --flow-likelihood-finetune-epochs > 0."
        )
    if bool(args.flow_likelihood_augment_existing) and (
        bool(args.tre_augment_existing) or bool(args.ctsm_v_augment_existing)
    ):
        raise ValueError(
            "--flow-likelihood-augment-existing cannot be combined with another augmentation mode."
        )
    includes_ctsm = bool(args.include_ctsm_v) or bool(args.include_ctsm_v_binary)
    if includes_ctsm and "symmetric_kl" not in metrics:
        raise ValueError("CTSM-v estimators require --metric to include symmetric_kl.")
    if bool(args.ctsm_v_augment_existing) and not includes_ctsm:
        raise ValueError("--ctsm-v-augment-existing requires a CTSM-v estimator.")
    if bool(args.tre_augment_existing) and not bool(args.include_tre):
        raise ValueError("--tre-augment-existing requires --include-tre.")
    if bool(args.include_tre) and "symmetric_kl" not in metrics:
        raise ValueError("TRE requires --metric to include symmetric_kl.")
    TREDensityRatioConfig(
        num_bridges=int(args.tre_num_bridges),
        waymark_schedule=str(args.tre_waymark_schedule),
        architecture=str(args.tre_architecture),
        hidden_dim=int(args.tre_hidden_dim),
        depth=int(args.tre_depth),
        epochs=int(args.tre_epochs),
        batch_size=int(args.tre_batch_size),
        lr=float(args.tre_lr),
        weight_decay=float(args.tre_weight_decay),
        early_patience=int(args.tre_early_patience),
        early_min_delta=float(args.tre_early_min_delta),
        max_grad_norm=float(args.tre_max_grad_norm),
        validation_pairs=int(args.tre_validation_pairs),
        standardize=bool(args.tre_standardize),
        log_every=int(args.log_every),
    ).validate()
    if int(args.tre_eval_batch_size) < 1:
        raise ValueError("--tre-eval-batch-size must be >= 1.")
    if int(args.ctsm_v_epochs) < 1:
        raise ValueError("--ctsm-v-epochs must be >= 1.")
    if int(args.ctsm_v_binary_epochs) < 1:
        raise ValueError("--ctsm-v-binary-epochs must be >= 1.")
    if int(args.ctsm_v_batch_size) < 2:
        raise ValueError("--ctsm-v-batch-size must be >= 2.")
    if float(args.ctsm_v_lr) <= 0.0:
        raise ValueError("--ctsm-v-lr must be > 0.")
    if int(args.ctsm_v_hidden_dim) < 1 or int(args.ctsm_v_film_depth) < 1:
        raise ValueError("CTSM-v hidden dimension and FiLM depth must be >= 1.")
    if float(args.ctsm_v_two_sb_var) <= 0.0:
        raise ValueError("--ctsm-v-two-sb-var must be > 0.")
    if not (0.0 <= float(args.ctsm_v_t_eps) < 0.5):
        raise ValueError("--ctsm-v-t-eps must be in [0, 0.5).")
    if int(args.ctsm_v_integration_steps) < 2:
        raise ValueError("--ctsm-v-integration-steps must be >= 2.")
    if int(args.ctsm_v_eval_batch_size) < 1:
        raise ValueError("--ctsm-v-eval-batch-size must be >= 1.")
    if int(args.ctsm_v_early_patience) < 0 or int(args.ctsm_v_validation_batches) < 1:
        raise ValueError("CTSM-v patience must be non-negative and validation batches must be >= 1.")
    if not (0.0 < float(args.ctsm_v_early_ema_alpha) <= 1.0):
        raise ValueError("--ctsm-v-early-ema-alpha must be in (0, 1].")


def _dataset_wrapper_args(args: argparse.Namespace, dataset_dir: Path) -> argparse.Namespace:
    mod = _load_make_mog5_module()
    argv = [
        "--n-total",
        str(int(args.n_total)),
        "--native-x-dim",
        str(int(args.native_x_dim)),
        "--pr-dim",
        "none" if args.pr_dim is None else str(int(args.pr_dim)),
        "--seed",
        str(int(args.seed)),
        "--train-frac",
        str(float(args.dataset_train_frac)),
        "--obs-noise-scale",
        str(float(args.dataset_obs_noise_scale)),
        "--cov-theta-amp-scale",
        str(float(args.dataset_cov_theta_amp_scale)),
        "--device",
        str(args.device),
        "--output-dir",
        str(dataset_dir),
        "--pr-cache-dir",
        str(Path(args.pr_cache_dir)),
    ]
    if args.dataset_mog_mean_min_dist is not None:
        argv.extend(["--mog-mean-min-dist", str(float(args.dataset_mog_mean_min_dist))])
    if args.native_template_npz is not None:
        argv.extend(["--native-template-npz", str(Path(args.native_template_npz))])
    if bool(args.force_dataset):
        argv.append("--force")
    if bool(args.dataset_use_cache):
        argv.append("--use-cache")
    if bool(args.skip_dataset_viz):
        argv.append("--skip-viz")
    return mod.parse_args(argv)


def ensure_dataset(args: argparse.Namespace, dataset_dir: Path) -> tuple[Path, Path | None]:
    mod = _load_make_mog5_module()
    return mod.run(_dataset_wrapper_args(args, dataset_dir))


def _flow_config_from_args(args: argparse.Namespace) -> FlowComparisonConfig:
    return FlowComparisonConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        early_min_delta=float(args.early_min_delta),
        early_ema_alpha=float(args.early_ema_alpha),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_schedule=str(args.lr_schedule),
        min_lr=float(args.min_lr),
        lr_schedule_epochs=None if args.lr_schedule_epochs is None else int(args.lr_schedule_epochs),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        network_architecture=str(args.network_architecture),
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
        shared_affine_a_diag_jitter=float(args.shared_affine_a_diag_jitter),
        solve_jitter=float(args.solve_jitter),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        checkpoint_selection=str(args.fm_checkpoint_selection),
        fixed_validation=bool(args.fixed_validation),
        fixed_validation_paths=int(args.fixed_validation_paths),
        normalize_x=bool(args.flow_normalize_x),
        normalize_x_eps=float(args.flow_normalize_x_eps),
        likelihood_finetune_epochs=int(args.flow_likelihood_finetune_epochs),
        likelihood_finetune_batch_size=int(args.flow_likelihood_finetune_batch_size),
        likelihood_finetune_lr=float(args.flow_likelihood_finetune_lr),
        likelihood_finetune_weight_decay=float(args.flow_likelihood_finetune_weight_decay),
        likelihood_finetune_ode_steps=int(args.flow_likelihood_finetune_ode_steps),
        likelihood_finetune_ode_method=str(args.flow_likelihood_finetune_ode_method),
        likelihood_finetune_patience=int(args.flow_likelihood_finetune_patience),
        likelihood_finetune_min_delta=float(args.flow_likelihood_finetune_min_delta),
        likelihood_finetune_ema_alpha=float(args.flow_likelihood_finetune_ema_alpha),
        likelihood_finetune_checkpoint_selection=str(args.flow_likelihood_finetune_checkpoint_selection),
        likelihood_finetune_divergence_estimator=str(
            args.flow_likelihood_finetune_divergence_estimator
        ),
        likelihood_finetune_hutchinson_probes=int(
            args.flow_likelihood_finetune_hutchinson_probes
        ),
    )


def _ctsm_v_config_from_args(args: argparse.Namespace) -> CTSMVJeffreysConfig:
    return CTSMVJeffreysConfig(
        epochs=int(args.ctsm_v_epochs),
        batch_size=int(args.ctsm_v_batch_size),
        lr=float(args.ctsm_v_lr),
        weight_decay=float(args.ctsm_v_weight_decay),
        hidden_dim=int(args.ctsm_v_hidden_dim),
        architecture=str(args.ctsm_v_architecture),
        film_depth=int(args.ctsm_v_film_depth),
        gated_film=bool(args.ctsm_v_gated_film),
        raw_time=bool(args.ctsm_v_raw_time),
        m_scale=float(args.ctsm_v_m_scale),
        delta_scale=float(args.ctsm_v_delta_scale),
        two_sb_var=float(args.ctsm_v_two_sb_var),
        path_schedule=str(args.ctsm_v_path_schedule),
        path_eps=float(args.ctsm_v_path_eps),
        factor=float(args.ctsm_v_factor),
        t_eps=float(args.ctsm_v_t_eps),
        integration_steps=int(args.ctsm_v_integration_steps),
        eval_batch_size=int(args.ctsm_v_eval_batch_size),
        early_patience=int(args.ctsm_v_early_patience),
        early_min_delta=float(args.ctsm_v_early_min_delta),
        early_ema_alpha=float(args.ctsm_v_early_ema_alpha),
        validation_batches_per_epoch=int(args.ctsm_v_validation_batches),
        normalize_x=bool(args.ctsm_v_normalize_x),
        log_every=int(args.log_every),
    )


def _tre_config_from_args(args: argparse.Namespace) -> TREDensityRatioConfig:
    return TREDensityRatioConfig(
        num_bridges=int(args.tre_num_bridges),
        waymark_schedule=str(args.tre_waymark_schedule),
        architecture=str(args.tre_architecture),
        hidden_dim=int(args.tre_hidden_dim),
        depth=int(args.tre_depth),
        epochs=int(args.tre_epochs),
        batch_size=int(args.tre_batch_size),
        lr=float(args.tre_lr),
        weight_decay=float(args.tre_weight_decay),
        early_patience=int(args.tre_early_patience),
        early_min_delta=float(args.tre_early_min_delta),
        max_grad_norm=float(args.tre_max_grad_norm),
        validation_pairs=int(args.tre_validation_pairs),
        standardize=bool(args.tre_standardize),
        log_every=int(args.log_every),
    )


def _ctsm_v_binary_config_from_args(args: argparse.Namespace) -> CTSMVBinaryJeffreysConfig:
    return CTSMVBinaryJeffreysConfig(
        epochs=int(args.ctsm_v_binary_epochs),
        batch_size=int(args.ctsm_v_batch_size),
        lr=float(args.ctsm_v_lr),
        weight_decay=float(args.ctsm_v_weight_decay),
        hidden_dim=int(args.ctsm_v_hidden_dim),
        two_sb_var=float(args.ctsm_v_two_sb_var),
        path_schedule=str(args.ctsm_v_path_schedule),
        path_eps=float(args.ctsm_v_path_eps),
        factor=float(args.ctsm_v_factor),
        t_eps=float(args.ctsm_v_t_eps),
        integration_steps=int(args.ctsm_v_integration_steps),
        eval_batch_size=int(args.ctsm_v_eval_batch_size),
        early_patience=int(args.ctsm_v_early_patience),
        early_min_delta=float(args.ctsm_v_early_min_delta),
        early_ema_alpha=float(args.ctsm_v_early_ema_alpha),
        validation_batches_per_epoch=int(args.ctsm_v_validation_batches),
        normalize_x=bool(args.ctsm_v_normalize_x),
        log_every=int(args.log_every),
    )


def _load_existing_estimators(
    path: Path,
    *,
    metrics: tuple[str, ...],
    require_nll_finetuned: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Estimator augmentation requires an existing result NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "metric_names",
            "classical_matrices",
            "flow_matching_matrices",
            "ground_truth_matrices",
        )
        missing_fields = [field for field in required if field not in data.files]
        if missing_fields:
            raise KeyError(f"Existing comparison NPZ {path} is missing: {', '.join(missing_fields)}")
        available = tuple(str(value) for value in data["metric_names"].tolist())
        missing_metrics = [metric for metric in metrics if metric not in available]
        if missing_metrics:
            raise ValueError(f"Existing comparison NPZ {path} is missing metrics: {', '.join(missing_metrics)}")
        indices = [available.index(metric) for metric in metrics]

        def unpack(field: str) -> dict[str, np.ndarray]:
            stack = np.asarray(data[field], dtype=np.float64)[indices]
            return {metric: stack[index] for index, metric in enumerate(metrics)}

        classical = unpack("classical_matrices")
        flow = unpack("flow_matching_matrices")
        ground_truth = unpack("ground_truth_matrices")
        fine = None
        if "flow_matching_nll_finetuned_matrices" in data.files:
            fine = unpack("flow_matching_nll_finetuned_matrices")
        elif bool(require_nll_finetuned):
            raise ValueError(f"Existing comparison NPZ {path} has no FM+NLL matrices.")
        tre = unpack("tre_matrices") if "tre_matrices" in data.files else None
        ctsm_v = unpack("ctsm_v_matrices") if "ctsm_v_matrices" in data.files else None
        ctsm_v_binary = (
            unpack("ctsm_v_binary_matrices") if "ctsm_v_binary_matrices" in data.files else None
        )
    return classical, flow, fine, ground_truth, tre, ctsm_v, ctsm_v_binary


def _validate_bundle(
    bundle,
    *,
    n_total: int,
    native_x_dim: int,
    pr_dim: int | None,
    pr_projected: bool,
) -> None:
    meta = dict(bundle.meta)
    if str(meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical, got {meta.get('dataset_family')!r}.")
    if int(meta.get("num_categories", -1)) != 5:
        raise ValueError(f"Expected num_categories=5, got {meta.get('num_categories')!r}.")
    expected_x_dim = int(native_x_dim) if pr_dim is None else int(pr_dim)
    if int(meta.get("x_dim", -1)) != int(expected_x_dim):
        raise ValueError(f"Expected work x_dim={expected_x_dim}, got {meta.get('x_dim')!r}.")
    if int(np.asarray(bundle.x_all).shape[0]) != int(n_total):
        raise ValueError(f"Expected n_total={n_total}, got {np.asarray(bundle.x_all).shape[0]}.")
    embedded = bool(meta.get("pr_autoencoder_embedded", False))
    if bool(pr_projected) and not embedded:
        raise ValueError("Projected dataset must have pr_autoencoder_embedded=True.")
    if not bool(pr_projected) and embedded:
        raise ValueError("Native dataset must not have pr_autoencoder_embedded=True.")
    if bool(pr_projected) and int(meta.get("pr_autoencoder_z_dim", -1)) != int(native_x_dim):
        raise ValueError(
            f"Expected pr_autoencoder_z_dim={int(native_x_dim)}, got {meta.get('pr_autoencoder_z_dim')!r}."
        )


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
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
    pr_projected = args.pr_dim is not None
    work_npz = projected_npz if pr_projected else native_npz
    if work_npz is None:
        raise RuntimeError("Native mode did not produce a work NPZ.")

    native_bundle = load_shared_dataset_npz(native_npz)
    work_bundle = load_shared_dataset_npz(work_npz)
    _validate_bundle(
        work_bundle,
        n_total=int(args.n_total),
        native_x_dim=int(args.native_x_dim),
        pr_dim=args.pr_dim,
        pr_projected=pr_projected,
    )

    k = int(work_bundle.meta.get("num_categories", 5))
    labels = labels_from_theta(work_bundle.theta_all, num_categories=k)
    names = condition_labels(k)
    flow_config = _flow_config_from_args(args)
    existing_results_path = output_dir / "mog5_pr_distance_comparison_results.npz"
    existing_tre = None
    existing_ctsm_v = None
    existing_ctsm_v_binary = None
    flow_likelihood_augment_existing = bool(args.flow_likelihood_augment_existing)
    augment_existing = (
        bool(args.ctsm_v_augment_existing)
        or bool(args.tre_augment_existing)
        or flow_likelihood_augment_existing
    )
    if augment_existing:
        print(f"[distance-comparison] reusing existing estimators from {existing_results_path}", flush=True)
        (
            classical,
            flow,
            flow_finetuned,
            ground_truth,
            existing_tre,
            existing_ctsm_v,
            existing_ctsm_v_binary,
        ) = _load_existing_estimators(
            existing_results_path,
            metrics=metrics,
            require_nll_finetuned=(
                int(flow_config.likelihood_finetune_epochs) > 0
                and not flow_likelihood_augment_existing
            ),
        )
        flow_paths = None
        flow_finetuned_paths = None
        if flow_likelihood_augment_existing:
            print(
                "[distance-comparison] retraining flow matching and adding CNF NLL fine-tuning "
                f"for {int(flow_config.likelihood_finetune_epochs)} epochs",
                flush=True,
            )
            flow_variants = flow_metric_variants(
                bundle=work_bundle,
                device=dev,
                output_dir=output_dir / "flow",
                config=flow_config,
                seed=int(args.seed),
                metrics=metrics,
            )
            flow = flow_variants.flow_matching
            flow_finetuned = flow_variants.flow_matching_nll_finetuned
            flow_paths = flow_variants.flow_npz_paths
            flow_finetuned_paths = flow_variants.flow_nll_finetuned_npz_paths
    else:
        print("[distance-comparison] computing classical finite-sample metrics", flush=True)
        classical = classical_metric_matrices(
            work_bundle.x_all,
            labels,
            num_categories=k,
            metrics=metrics,
            mahalanobis_ridge=float(args.mahalanobis_ridge),
            skl_folds=int(args.skl_folds),
            skl_seed=int(args.seed),
            skl_logistic_c=float(args.skl_logistic_c),
        )

        if pr_projected:
            print("[distance-comparison] computing projected-coordinate Monte Carlo ground truth", flush=True)
            ground_truth = pr_autoencoder_ground_truth_matrices(
                native_meta=dict(native_bundle.meta),
                projected_meta=dict(work_bundle.meta),
                device=dev,
                cache_dir=Path(args.pr_cache_dir),
                samples_per_class=int(args.gt_samples_per_class),
                seed=int(args.seed) + 12345,
                batch_size=int(args.gt_batch_size),
                mahalanobis_ridge=float(args.mahalanobis_ridge),
                metrics=metrics,
            )
        else:
            print("[distance-comparison] computing native-coordinate Monte Carlo ground truth", flush=True)
            ground_truth = native_mog_ground_truth_matrices(
                native_meta=dict(native_bundle.meta),
                samples_per_class=int(args.gt_samples_per_class),
                seed=int(args.seed) + 12345,
                mahalanobis_ridge=float(args.mahalanobis_ridge),
                metrics=metrics,
            )

        print("[distance-comparison] training flow-matching metrics", flush=True)
        if int(flow_config.likelihood_finetune_epochs) > 0:
            print(
                "[distance-comparison] enabling CNF NLL fine-tuned flow estimator "
                f"for {int(flow_config.likelihood_finetune_epochs)} epochs",
                flush=True,
            )
            flow_variants = flow_metric_variants(
                bundle=work_bundle,
                device=dev,
                output_dir=output_dir / "flow",
                config=flow_config,
                seed=int(args.seed),
                metrics=metrics,
            )
            flow = flow_variants.flow_matching
            flow_finetuned = flow_variants.flow_matching_nll_finetuned
            flow_paths = flow_variants.flow_npz_paths
            flow_finetuned_paths = flow_variants.flow_nll_finetuned_npz_paths
        else:
            flow, flow_paths = flow_metric_matrices(
                bundle=work_bundle,
                device=dev,
                output_dir=output_dir / "flow",
                config=flow_config,
                seed=int(args.seed),
                metrics=metrics,
            )
            flow_finetuned = None
            flow_finetuned_paths = None

    tre = existing_tre
    tre_paths = None
    tre_checkpoint_paths = None
    tre_config = _tre_config_from_args(args)
    if bool(args.include_tre) and not flow_likelihood_augment_existing:
        print("[distance-comparison] training pairwise Torch TRE models", flush=True)
        labels_train = labels_from_theta(work_bundle.theta_train, num_categories=k)
        labels_validation = labels_from_theta(work_bundle.theta_validation, num_categories=k)
        tre_states, tre_result = train_and_estimate_pairwise_tre_jeffreys(
            x_train=work_bundle.x_train,
            labels_train=labels_train,
            x_validation=work_bundle.x_validation,
            labels_validation=labels_validation,
            x_eval=work_bundle.x_all,
            labels_eval=labels,
            num_categories=k,
            device=dev,
            seed=int(args.seed),
            config=tre_config,
            eval_batch_size=int(args.tre_eval_batch_size),
        )
        tre_npz, tre_checkpoint = save_pairwise_tre_jeffreys_result(
            output_dir / "tre" / "symmetric_kl_tre_results.npz",
            output_dir / "tre" / "symmetric_kl_tre_models.pt",
            pair_state_dicts=tre_states,
            result=tre_result,
        )
        tre = {"symmetric_kl": tre_result.symmetric_kl_matrix}
        tre_paths = {"symmetric_kl": tre_npz}
        tre_checkpoint_paths = {"symmetric_kl": tre_checkpoint}

    ctsm_v = existing_ctsm_v
    ctsm_v_paths = None
    ctsm_v_checkpoint_paths = None
    ctsm_v_config = _ctsm_v_config_from_args(args)
    if bool(args.include_ctsm_v):
        print("[distance-comparison] training pair-conditioned CTSM-v", flush=True)
        ctsm_model, ctsm_result = train_and_estimate_ctsm_v_jeffreys(
            theta_train=work_bundle.theta_train,
            x_train=work_bundle.x_train,
            theta_val=work_bundle.theta_validation,
            x_val=work_bundle.x_validation,
            theta_eval=work_bundle.theta_all,
            x_eval=work_bundle.x_all,
            labels_eval=labels,
            num_categories=k,
            device=dev,
            seed=int(args.seed),
            config=ctsm_v_config,
        )
        ctsm_npz, ctsm_checkpoint = save_ctsm_v_jeffreys_result(
            output_dir / "ctsm_v" / "symmetric_kl_ctsm_v_results.npz",
            output_dir / "ctsm_v" / "symmetric_kl_ctsm_v_model.pt",
            model=ctsm_model,
            result=ctsm_result,
        )
        ctsm_v = {"symmetric_kl": ctsm_result.symmetric_kl_matrix}
        ctsm_v_paths = {"symmetric_kl": ctsm_npz}
        ctsm_v_checkpoint_paths = {"symmetric_kl": ctsm_checkpoint}

    ctsm_v_binary = existing_ctsm_v_binary
    ctsm_v_binary_paths = None
    ctsm_v_binary_checkpoint_paths = None
    ctsm_v_binary_config = _ctsm_v_binary_config_from_args(args)
    if bool(args.include_ctsm_v_binary):
        print("[distance-comparison] training pairwise CTSM-v-binary models", flush=True)
        labels_train = labels_from_theta(work_bundle.theta_train, num_categories=k)
        labels_val = labels_from_theta(work_bundle.theta_validation, num_categories=k)
        pair_states, ctsm_binary_result = train_and_estimate_pairwise_binary_ctsm_v_jeffreys(
            x_train=work_bundle.x_train,
            labels_train=labels_train,
            x_val=work_bundle.x_validation,
            labels_val=labels_val,
            x_eval=work_bundle.x_all,
            labels_eval=labels,
            num_categories=k,
            device=dev,
            seed=int(args.seed),
            config=ctsm_v_binary_config,
        )
        ctsm_binary_npz, ctsm_binary_checkpoint = save_pairwise_binary_ctsm_v_jeffreys_result(
            output_dir / "ctsm_v_binary" / "symmetric_kl_ctsm_v_binary_results.npz",
            output_dir / "ctsm_v_binary" / "symmetric_kl_ctsm_v_binary_models.pt",
            pair_state_dicts=pair_states,
            result=ctsm_binary_result,
        )
        ctsm_v_binary = {"symmetric_kl": ctsm_binary_result.symmetric_kl_matrix}
        ctsm_v_binary_paths = {"symmetric_kl": ctsm_binary_npz}
        ctsm_v_binary_checkpoint_paths = {"symmetric_kl": ctsm_binary_checkpoint}

    result = assemble_comparison_result(
        metrics=metrics,
        condition_names=names,
        classical=classical,
        flow_matching=flow,
        flow_matching_nll_finetuned=flow_finetuned,
        tre=tre,
        ctsm_v=ctsm_v,
        ctsm_v_binary=ctsm_v_binary,
        ground_truth=ground_truth,
        flow_npz_paths=flow_paths,
        flow_nll_finetuned_npz_paths=flow_finetuned_paths,
        tre_npz_paths=tre_paths,
        tre_checkpoint_paths=tre_checkpoint_paths,
        ctsm_v_npz_paths=ctsm_v_paths,
        ctsm_v_checkpoint_paths=ctsm_v_checkpoint_paths,
        ctsm_v_binary_npz_paths=ctsm_v_binary_paths,
        ctsm_v_binary_checkpoint_paths=ctsm_v_binary_checkpoint_paths,
        flow_velocity_families={metric: velocity_family_for_metric(metric, flow_config) for metric in metrics},
    )

    results_npz = write_results_npz(output_dir / "mog5_pr_distance_comparison_results.npz", result)
    pairs_csv = write_pairs_csv(output_dir / "mog5_pr_distance_comparison_pairs.csv", result.rows)
    summary_json = write_summary_json(
        output_dir / "mog5_pr_distance_comparison_summary.json",
        result=result,
        extra={
            "script": "bin/compare_mog5_pr_distances.py",
            "device": str(dev),
            "n_total": int(args.n_total),
            "native_x_dim": int(args.native_x_dim),
            "pr_projected": bool(pr_projected),
            "pr_dim": None if args.pr_dim is None else int(args.pr_dim),
            "seed": int(args.seed),
            "dataset_dir": str(dataset_dir),
            "native_npz": str(native_npz),
            "work_npz": str(work_npz),
            "projected_npz": None if projected_npz is None else str(projected_npz),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "pairs_csv": str(pairs_csv),
            "metric": str(args.metric),
            "metrics": list(metrics),
            "gt_samples_per_class": int(args.gt_samples_per_class),
            "gt_batch_size": int(args.gt_batch_size),
            "mahalanobis_ridge": float(args.mahalanobis_ridge),
            "skl_folds": int(args.skl_folds),
            "skl_logistic_c": float(args.skl_logistic_c),
            "pr_cache_dir": str(Path(args.pr_cache_dir)),
            "flow_defaults": vars(flow_config),
            "flow_likelihood_augment_existing": flow_likelihood_augment_existing,
            "include_tre": bool(args.include_tre),
            "tre_defaults": asdict(tre_config),
            "tre_eval_batch_size": int(args.tre_eval_batch_size),
            "tre_augment_existing": bool(args.tre_augment_existing),
            "include_ctsm_v": bool(args.include_ctsm_v),
            "ctsm_v_defaults": asdict(ctsm_v_config),
            "include_ctsm_v_binary": bool(args.include_ctsm_v_binary),
            "ctsm_v_binary_defaults": asdict(ctsm_v_binary_config),
            "ctsm_v_augment_existing": bool(args.ctsm_v_augment_existing),
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
