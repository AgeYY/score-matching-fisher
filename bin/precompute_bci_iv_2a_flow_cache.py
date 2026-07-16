#!/usr/bin/env python3
"""Precompute one recording's query flow-RDM caches for the identification run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS  # noqa: E402
from fisher.bci_iv_2a_dataset import load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    FlowRDMConfig,
    QUERY_RUNS,
    per_class_counts,
    select_half,
    subsample_balanced_trials,
)

from compare_bci_iv_2a_session_identification import (  # noqa: E402
    METHOD_SEED_OFFSETS,
    N_LABELS,
    fit_or_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    dataset = load_features_npz(args.feature_path)
    expected_index = int(dataset.session_key[1:3]) - 1
    if args.subject_index != expected_index:
        raise ValueError(
            f"subject-index {args.subject_index} does not match {dataset.session_key} "
            f"in the canonical A01T...A09T ordering ({expected_index})."
        )
    config = FlowRDMConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    cache_dir = args.output_dir / "rdm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_query, y_query, _ = select_half(dataset, QUERY_RUNS)
    max_balanced = int(np.min(per_class_counts(y_query)))
    requested = [4, 8, 12, 18, 24, max_balanced]
    method = "time_varying_shared_affine_flow"
    for n_index, n_per_class in enumerate(requested):
        for repeat in range(args.repeats):
            subset_seed = (
                args.seed + args.subject_index * 100_000 + n_index * 1_000 + repeat
            )
            selected = subsample_balanced_trials(y_query, n_per_class, subset_seed)
            cache_path = cache_dir / (
                f"query_{dataset.session_key}_n{N_LABELS[n_index]}_rep{repeat:02d}_{method}.npz"
            )
            fit_or_load(
                cache_path,
                method=method,
                x=x_query[selected],
                labels=y_query[selected],
                time_centers=dataset.time_centers,
                device=device,
                seed=subset_seed + METHOD_SEED_OFFSETS[method],
                config=config,
                context={
                    "role": "query",
                    "recording": dataset.session_key,
                    "n_label": N_LABELS[n_index],
                    "n_per_class": int(n_per_class),
                    "repeat": int(repeat),
                    "selected_trial_indices": selected.tolist(),
                },
            )
        print(
            f"[precompute] {dataset.session_key} n={N_LABELS[n_index]} "
            f"effective={n_per_class} repeats={args.repeats} complete",
            flush=True,
        )
    print(f"=== {dataset.session_key} flow-cache precompute complete ===", flush=True)


if __name__ == "__main__":
    main()
