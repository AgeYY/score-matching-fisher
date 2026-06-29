#!/usr/bin/env python3
"""Identify Stringer sessions by matching Fisher curves between split halves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import ContinuousFlowConfig
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import list_stringer_sessions
from fisher.stringer_session_identification import (
    ALL_DISTANCE_SUMMARY_PNG_NAME,
    ALL_DISTANCE_SUMMARY_SVG_NAME,
    DIRECTION_A_TO_B,
    CURVES_CSV_NAME,
    DISTANCES,
    HEATMAPS_PNG_NAME,
    HEATMAPS_SVG_NAME,
    METHODS,
    PAIRS_CSV_NAME,
    RANKS_PNG_NAME,
    RANKS_SVG_NAME,
    RESULTS_NPZ_NAME,
    SUMMARY_JSON_NAME,
    IdentificationResult,
    parse_optional_int,
    plot_all_distance_summary,
    plot_primary_heatmaps,
    plot_ranks,
    run_session_identification,
    theta_grid_periodic,
    write_curves_csv,
    write_pairs_csv,
    write_results_npz,
    write_summary_json,
)
from global_setting import DATA_DIR, DEFAULT_DEVICE


def _default_data_root() -> Path:
    root = Path(DATA_DIR).expanduser()
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root.resolve()


def default_output_dir(session_stimuli_type: str = "gratings_static") -> Path:
    return _default_data_root() / "stringer_fisher_session_identification" / str(session_stimuli_type)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--session-stimuli-type", default="gratings_static")
    p.add_argument("--max-sessions", type=parse_optional_int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--orientation-period", type=float, default=float(np.pi))
    p.add_argument("--theta-grid-size", type=int, default=17)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--pca-random-state", type=int, default=0)
    p.add_argument("--no-pca-whiten", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--classical-window-radius", type=float, default=None)
    p.add_argument("--classical-min-endpoint-samples", type=int, default=8)
    p.add_argument("--classical-linear-ridge", type=float, default=1e-6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-flow-npz", action="store_true")
    p.add_argument("--visualization-only", action="store_true")

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--shared-affine-a-diag-jitter", type=float, default=1e-3)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--affine-ridge", type=float, default=1e-6)
    return p


def flow_config_from_args(args: argparse.Namespace) -> ContinuousFlowConfig:
    return ContinuousFlowConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        early_min_delta=float(args.early_min_delta),
        early_ema_alpha=float(args.early_ema_alpha),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule=str(args.path_schedule),
        t_eps=float(args.t_eps),
        quadrature_steps=int(args.quadrature_steps),
        mc_jeffreys_sample=0,
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        shared_affine_a_diag_jitter=float(args.shared_affine_a_diag_jitter),
        solve_jitter=float(args.solve_jitter),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        affine_ridge=float(args.affine_ridge),
    )


def _load_visualization_result(output_dir: Path) -> IdentificationResult:
    import json

    output_dir = Path(output_dir)
    results_path = output_dir / RESULTS_NPZ_NAME
    summary_path = output_dir / SUMMARY_JSON_NAME
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results NPZ for visualization-only mode: {results_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary JSON for visualization-only mode: {summary_path}")
    data = np.load(results_path, allow_pickle=False)
    summary = json.loads(summary_path.read_text())
    session_keys = [str(v) for v in np.asarray(data["session_keys"]).reshape(-1)]
    distances = {method: {} for method in METHODS}
    for method in METHODS:
        for distance_name in DISTANCES:
            distances[method][distance_name] = {}
            distances[method][distance_name][DIRECTION_A_TO_B] = np.asarray(
                data[f"{method}_{distance_name}_{DIRECTION_A_TO_B}"],
                dtype=np.float64,
            )
    return IdentificationResult(
        session_keys=session_keys,
        theta_grid=np.asarray(data["theta_grid"], dtype=np.float64),
        theta_midpoints=np.asarray(data["theta_midpoints"], dtype=np.float64),
        half_results=[],
        distances=distances,
        pair_rows=[],
        curve_rows=[],
        summary=summary,
    )


def validate_args(args: argparse.Namespace) -> None:
    if int(args.theta_grid_size) < 2:
        raise ValueError("--theta-grid-size must be >= 2.")
    if int(args.pca_dim) < 1:
        raise ValueError("--pca-dim must be >= 1.")
    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")
    if float(args.orientation_period) <= 0.0:
        raise ValueError("--orientation-period must be positive.")


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else default_output_dir(str(args.session_stimuli_type))
    )
    if not output_dir.is_absolute():
        output_dir = (_REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.visualization_only):
        result = _load_visualization_result(output_dir)
        heatmaps_svg, heatmaps_png = plot_primary_heatmaps(output_dir / HEATMAPS_SVG_NAME, output_dir / HEATMAPS_PNG_NAME, result)
        ranks_svg, ranks_png = plot_ranks(output_dir / RANKS_SVG_NAME, output_dir / RANKS_PNG_NAME, result)
        all_summary_svg, all_summary_png = plot_all_distance_summary(
            output_dir / ALL_DISTANCE_SUMMARY_SVG_NAME,
            output_dir / ALL_DISTANCE_SUMMARY_PNG_NAME,
            result,
        )
        for label, path in (
            ("heatmaps_svg", heatmaps_svg),
            ("ranks_svg", ranks_svg),
            ("all_distance_summary_svg", all_summary_svg),
        ):
            print(f"{label}: {path}", flush=True)
        return {
            "output_dir": output_dir,
            "heatmaps_svg": heatmaps_svg,
            "heatmaps_png": heatmaps_png,
            "ranks_svg": ranks_svg,
            "ranks_png": ranks_png,
            "all_distance_summary_svg": all_summary_svg,
            "all_distance_summary_png": all_summary_png,
        }

    device = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    sessions = list_stringer_sessions(str(args.session_stimuli_type), data_dir=args.data_dir)
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]
    if len(sessions) < 2:
        raise ValueError("Session identification requires at least two sessions.")

    print(
        f"[stringer-identification] sessions={len(sessions)} "
        f"stimuli_type={args.session_stimuli_type} output_dir={output_dir}",
        flush=True,
    )
    grid = theta_grid_periodic(float(args.orientation_period), int(args.theta_grid_size))
    result = run_session_identification(
        sessions=sessions,
        theta_grid=grid,
        period=float(args.orientation_period),
        pca_dim=int(args.pca_dim),
        pca_random_state=int(args.pca_random_state),
        pca_whiten=not bool(args.no_pca_whiten),
        train_frac=float(args.train_frac),
        seed=int(args.seed),
        device=device,
        flow_config=flow_config_from_args(args),
        output_dir=output_dir,
        force=bool(args.force),
        save_flow_npz=not bool(args.skip_flow_npz),
        classical_ridge=float(args.classical_linear_ridge),
        classical_window_radius=args.classical_window_radius,
        classical_min_endpoint_samples=int(args.classical_min_endpoint_samples),
    )

    results_npz = write_results_npz(output_dir / RESULTS_NPZ_NAME, result)
    curves_csv = write_curves_csv(output_dir / CURVES_CSV_NAME, result.curve_rows)
    pairs_csv = write_pairs_csv(output_dir / PAIRS_CSV_NAME, result.pair_rows)
    heatmaps_svg, heatmaps_png = plot_primary_heatmaps(output_dir / HEATMAPS_SVG_NAME, output_dir / HEATMAPS_PNG_NAME, result)
    ranks_svg, ranks_png = plot_ranks(output_dir / RANKS_SVG_NAME, output_dir / RANKS_PNG_NAME, result)
    all_summary_svg, all_summary_png = plot_all_distance_summary(
        output_dir / ALL_DISTANCE_SUMMARY_SVG_NAME,
        output_dir / ALL_DISTANCE_SUMMARY_PNG_NAME,
        result,
    )
    summary_json = write_summary_json(
        output_dir / SUMMARY_JSON_NAME,
        result,
        extra={
            "script": "bin/compare_stringer_fisher_session_identification.py",
            "device": str(device),
            "data_dir": None if args.data_dir is None else str(args.data_dir),
            "session_stimuli_type": str(args.session_stimuli_type),
            "max_sessions": None if args.max_sessions is None else int(args.max_sessions),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "curves_csv": str(curves_csv),
            "pairs_csv": str(pairs_csv),
            "heatmaps_svg": str(heatmaps_svg),
            "heatmaps_png": str(heatmaps_png),
            "ranks_svg": str(ranks_svg),
            "ranks_png": str(ranks_png),
            "all_distance_summary_svg": str(all_summary_svg),
            "all_distance_summary_png": str(all_summary_png),
            "flow_config": vars(flow_config_from_args(args)),
            "classical_config": {
                "linear_ridge": float(args.classical_linear_ridge),
                "window_radius": args.classical_window_radius,
                "min_endpoint_samples": int(args.classical_min_endpoint_samples),
            },
            "seed": int(args.seed),
            "train_frac": float(args.train_frac),
            "pca_dim": int(args.pca_dim),
            "pca_whiten": not bool(args.no_pca_whiten),
            "pca_random_state": int(args.pca_random_state),
        },
    )
    for label, path in (
        ("results_npz", results_npz),
        ("curves_csv", curves_csv),
        ("pairs_csv", pairs_csv),
        ("summary_json", summary_json),
        ("heatmaps_svg", heatmaps_svg),
        ("ranks_svg", ranks_svg),
        ("all_distance_summary_svg", all_summary_svg),
    ):
        print(f"{label}: {path}", flush=True)
    return {
        "output_dir": output_dir,
        "results_npz": results_npz,
        "curves_csv": curves_csv,
        "pairs_csv": pairs_csv,
        "summary_json": summary_json,
        "heatmaps_svg": heatmaps_svg,
        "heatmaps_png": heatmaps_png,
        "ranks_svg": ranks_svg,
        "ranks_png": ranks_png,
        "all_distance_summary_svg": all_summary_svg,
        "all_distance_summary_png": all_summary_png,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
