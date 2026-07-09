#!/usr/bin/env python3
"""Train the two-square geometric-base model panel and assemble one figure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR


@dataclass(frozen=True)
class ModelCase:
    label: str
    output_name: str
    base_geometry: str
    velocity_family: str

    @property
    def output_dir(self) -> Path:
        return Path(DATA_DIR) / "geometric_base_fit_check" / self.output_name

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "geometric_base_fit_check_summary.json"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "geometric_base_fit_check_model_best.pt"


MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase(
        label="Square + Lie sim.",
        output_name="two-square__base-square__lie-similarity-2d__nf_best",
        base_geometry="square",
        velocity_family="lie-similarity-2d",
    ),
    ModelCase(
        label="Square + uncon.",
        output_name="two-square__base-square__unconstrained__nf_best",
        base_geometry="square",
        velocity_family="unconstrained",
    ),
    ModelCase(
        label="Normal + affine",
        output_name="two-square__base-standard-normal__centered-affine__nf_best",
        base_geometry="standard-normal",
        velocity_family="centered-affine",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-per-condition", type=int, default=3000)
    p.add_argument("--side-length", type=float, default=2.0)
    p.add_argument("--target-sigma", type=float, default=0.2)
    p.add_argument("--base-noise-sigma", type=float, default=0.1)
    p.add_argument("--path-schedule", choices=("linear", "straight", "cosine"), default="cosine")
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--nf-epochs", type=int, default=500)
    p.add_argument("--nf-batch-size", type=int, default=0, help="0 reuses --batch-size.")
    p.add_argument("--nf-lr", type=float, default=1e-4)
    p.add_argument("--nf-density-points", type=int, default=512)
    p.add_argument("--ode-steps", type=int, default=32)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--plot-n-per-condition", type=int, default=600)
    p.add_argument("--generated-samples-per-condition", type=int, default=600)
    p.add_argument("--contour-target-samples-per-condition", type=int, default=6000)
    p.add_argument("--contour-generated-samples-per-condition", type=int, default=6000)
    p.add_argument("--base-samples-per-condition", type=int, default=250)
    p.add_argument("--contour-grid-size", type=int, default=140)
    p.add_argument("--contour-levels", type=int, default=6)
    p.add_argument("--contour-low-quantile", type=float, default=0.78)
    p.add_argument("--contour-high-quantile", type=float, default=0.985)
    p.add_argument(
        "--figure-output-dir",
        type=Path,
        default=Path(DATA_DIR) / "geometric_base_dataset_visualizations" / "two_square_target_vs_all_models",
    )
    p.add_argument("--figure-prefix", type=str, default="two_square_target_vs_all_models")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip a model case when its summary and best checkpoint already exist.",
    )
    return p


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def _train_case(args: argparse.Namespace, case: ModelCase) -> None:
    if bool(args.reuse_existing) and case.summary_path.exists() and case.best_checkpoint_path.exists():
        print(f"[reuse] {case.label}: {case.output_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        "bin/run_geometric_base_fit_check.py",
        "--dataset",
        "two-square",
        "--base-geometry",
        case.base_geometry,
        "--velocity-family",
        case.velocity_family,
        "--output-dir",
        str(case.output_dir),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--n-per-condition",
        str(args.n_per_condition),
        "--side-length",
        str(args.side_length),
        "--target-sigma",
        str(args.target_sigma),
        "--base-noise-sigma",
        str(args.base_noise_sigma),
        "--path-schedule",
        str(args.path_schedule),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--depth",
        str(args.depth),
        "--early-patience",
        str(args.early_patience),
        "--nf-likelihood-finetune",
        "--nf-checkpoint-selection",
        "best",
        "--nf-epochs",
        str(args.nf_epochs),
        "--nf-batch-size",
        str(args.nf_batch_size),
        "--nf-lr",
        str(args.nf_lr),
        "--nf-density-points",
        str(args.nf_density_points),
        "--ode-steps",
        str(args.ode_steps),
        "--ode-method",
        str(args.ode_method),
        "--generated-samples-per-condition",
        str(args.generated_samples_per_condition),
    ]
    _run(cmd)


def _plot_figure(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "bin/visualize_two_square_target_dataset.py",
        "--layout",
        "overlap",
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--n-per-condition",
        str(args.plot_n_per_condition),
        "--side-length",
        str(args.side_length),
        "--target-sigma",
        str(args.target_sigma),
        "--ode-steps",
        str(args.ode_steps),
        "--ode-method",
        str(args.ode_method),
        "--generated-samples-per-condition",
        str(args.generated_samples_per_condition),
        "--contour-target-samples-per-condition",
        str(args.contour_target_samples_per_condition),
        "--contour-generated-samples-per-condition",
        str(args.contour_generated_samples_per_condition),
        "--base-samples-per-condition",
        str(args.base_samples_per_condition),
        "--contour-grid-size",
        str(args.contour_grid_size),
        "--contour-levels",
        str(args.contour_levels),
        "--contour-low-quantile",
        str(args.contour_low_quantile),
        "--contour-high-quantile",
        str(args.contour_high_quantile),
        "--output-dir",
        str(Path(args.figure_output_dir)),
        "--prefix",
        str(args.figure_prefix),
    ]
    for case in MODEL_CASES:
        if not case.summary_path.exists():
            raise FileNotFoundError(f"Missing summary for {case.label}: {case.summary_path}")
        if not case.best_checkpoint_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint for {case.label}: {case.best_checkpoint_path}")
        cmd.extend(["--model-summary", str(case.summary_path)])
        cmd.extend(["--model-checkpoint", str(case.best_checkpoint_path)])
        cmd.extend(["--model-label", case.label])
    _run(cmd)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for case in MODEL_CASES:
        _train_case(args, case)
    _plot_figure(args)
    png = Path(args.figure_output_dir).expanduser().resolve() / f"{args.figure_prefix}.png"
    svg = Path(args.figure_output_dir).expanduser().resolve() / f"{args.figure_prefix}.svg"
    summary = Path(args.figure_output_dir).expanduser().resolve() / f"{args.figure_prefix}_summary.json"
    print(f"png: {png}", flush=True)
    print(f"svg: {svg}", flush=True)
    print(f"summary_json: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
