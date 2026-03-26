#!/usr/bin/env python3
"""Unified CLI for Fisher estimation experiments."""

from __future__ import annotations

import argparse

from fisher.config import DatasetConfig, DecoderRunConfig, ScoreRunConfig
from fisher.pipelines import run_decoder_pipeline, run_score_pipeline


def add_shared_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--theta-low", type=float, default=-3.0)
    p.add_argument("--theta-high", type=float, default=3.0)
    p.add_argument("--sigma-x1", type=float, default=0.30)
    p.add_argument("--sigma-x2", type=float, default=0.22)
    p.add_argument("--rho", type=float, default=0.15)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run toy Fisher estimation methods.")
    sub = parser.add_subparsers(dest="method", required=True)

    score = sub.add_parser("score", help="Score matching Fisher with multi-sigma extrapolation.")
    add_shared_dataset_args(score)
    score.add_argument("--epochs", type=int, default=120)
    score.add_argument("--batch-size", type=int, default=256)
    score.add_argument("--lr", type=float, default=1e-3)
    score.add_argument("--hidden-dim", type=int, default=128)
    score.add_argument("--depth", type=int, default=3)
    score.add_argument("--sigma-alpha-list", type=float, nargs="+", default=[0.08, 0.06, 0.045, 0.03, 0.02])
    score.add_argument("--n-train", type=int, default=28000)
    score.add_argument("--n-eval", type=int, default=18000)
    score.add_argument("--fd-delta", type=float, default=0.03)
    score.add_argument("--n-bins", type=int, default=35)
    score.add_argument("--min-bin-count", type=int, default=80)
    score.add_argument("--eval-margin", type=float, default=0.30)
    score.add_argument("--log-every", type=int, default=25)
    score.add_argument("--output-dir", type=str, default="outputs_step3_multi_sigma")
    score.add_argument("--device", type=str, default="cpu")

    dec = sub.add_parser("decoder", help="Decoder-based Fisher via local classification.")
    add_shared_dataset_args(dec)
    dec.add_argument("--epsilon", type=float, default=0.12)
    dec.add_argument("--fd-delta", type=float, default=0.03)
    dec.add_argument("--n-bins", type=int, default=35)
    dec.add_argument("--eval-margin", type=float, default=0.30)
    dec.add_argument("--n-train-local", type=int, default=1200)
    dec.add_argument("--n-eval-local", type=int, default=1200)
    dec.add_argument("--epochs", type=int, default=80)
    dec.add_argument("--batch-size", type=int, default=256)
    dec.add_argument("--lr", type=float, default=1e-3)
    dec.add_argument("--hidden-dim", type=int, default=64)
    dec.add_argument("--depth", type=int, default=2)
    dec.add_argument("--log-every", type=int, default=5)
    dec.add_argument("--output-dir", type=str, default="outputs_step4_decoder")
    dec.add_argument("--device", type=str, default="cpu")

    return parser


def _dataset_cfg(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        sigma_x1=args.sigma_x1,
        sigma_x2=args.sigma_x2,
        rho=args.rho,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_cfg = _dataset_cfg(args)

    if args.method == "score":
        run_cfg = ScoreRunConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            sigma_alpha_list=args.sigma_alpha_list,
            n_train=args.n_train,
            n_eval=args.n_eval,
            fd_delta=args.fd_delta,
            n_bins=args.n_bins,
            min_bin_count=args.min_bin_count,
            eval_margin=args.eval_margin,
            log_every=args.log_every,
            output_dir=args.output_dir,
            device=args.device,
        )
        paths = run_score_pipeline(dataset_cfg, run_cfg)
    else:
        run_cfg = DecoderRunConfig(
            epsilon=args.epsilon,
            fd_delta=args.fd_delta,
            n_bins=args.n_bins,
            eval_margin=args.eval_margin,
            n_train_local=args.n_train_local,
            n_eval_local=args.n_eval_local,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            log_every=args.log_every,
            output_dir=args.output_dir,
            device=args.device,
        )
        paths = run_decoder_pipeline(dataset_cfg, run_cfg)

    print("Saved artifacts:")
    for p in paths.values():
        print(f"  - {p}")


if __name__ == "__main__":
    main()
