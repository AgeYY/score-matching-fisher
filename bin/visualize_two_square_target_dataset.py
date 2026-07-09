#!/usr/bin/env python3
"""Visualize the target two-square geometric-base dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR

from fisher.geometric_base_flow_skl import (
    NoisyGeometricBase,
    SquarePerimeterBase,
    StandardNormalBase,
    build_geometric_base_velocity_model,
    estimate_pushed_base_symmetric_kl,
    push_base_curve,
    push_initial_points,
)
from fisher.noisy_square_dataset import NoisySquareBoundaryDataset
from fisher.shared_fisher_est import require_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theta-values", type=str, default="0.0,0.7853981633974483")
    p.add_argument("--n-per-condition", "--n-per-theta", dest="n_per_condition", type=int, default=600)
    p.add_argument("--side-length", type=float, default=2.0)
    p.add_argument("--target-sigma", type=float, default=0.2)
    p.add_argument("--center-x", type=float, default=0.0)
    p.add_argument("--center-y", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--points-per-edge", type=int, default=160)
    p.add_argument("--point-size", type=float, default=30.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--legend-font-size", type=float, default=14.0)
    p.add_argument("--panel-label-font-size", type=float, default=18.0)
    p.add_argument("--layout", choices=("overlap", "condition-rows"), default="overlap")
    p.add_argument("--contour-grid-size", type=int, default=140)
    p.add_argument("--contour-levels", type=int, default=6)
    p.add_argument("--contour-low-quantile", type=float, default=0.78)
    p.add_argument("--contour-high-quantile", type=float, default=0.985)
    p.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "geometric_base_dataset_visualizations" / "two_square_target")
    p.add_argument("--prefix", type=str, default="two_square_target_dataset")
    p.add_argument("--model-summary", type=Path, action="append", default=None)
    p.add_argument("--model-checkpoint", type=Path, action="append", default=None)
    p.add_argument("--model-label", type=str, action="append", default=None)
    p.add_argument("--generated-samples-per-condition", type=int, default=600)
    p.add_argument("--contour-target-samples-per-condition", type=int, default=6000)
    p.add_argument("--contour-generated-samples-per-condition", type=int, default=6000)
    p.add_argument("--target-skl-samples", type=int, default=4096)
    p.add_argument("--target-skl-density-samples", type=int, default=4096)
    p.add_argument("--base-samples-per-condition", type=int, default=250)
    p.add_argument("--ode-steps", type=int, default=None)
    p.add_argument("--ode-method", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    return p


def _parse_theta_values(text: str) -> np.ndarray:
    vals = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(vals) != 2:
        raise ValueError("--theta-values must contain exactly two comma-separated values.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("--theta-values must be finite.")
    return np.asarray(vals, dtype=np.float64)


def _axis_limits(arrays: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in arrays if np.asarray(arr).size > 0], axis=0)
    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-6)
    pad = 0.035 * radius
    return (
        (float(center[0] - radius - pad), float(center[0] + radius + pad)),
        (float(center[1] - radius - pad), float(center[1] + radius + pad)),
    )


def _set_equal_axes(
    ax: Any,
    arrays: list[np.ndarray],
    *,
    axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    xlim, ylim = _axis_limits(arrays) if axis_limits is None else axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def _style_panel(
    ax: Any,
    arrays: list[np.ndarray],
    *,
    legend_font_size: float,
    show_legend: bool = True,
    axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    if show_legend:
        ax.legend(frameon=False, loc="lower right", fontsize=float(legend_font_size), markerscale=2.0)
    ax.grid(False)
    ax.set_axis_off()
    _set_equal_axes(ax, arrays, axis_limits=axis_limits)


def _panel_label(ax: Any, text: str, *, font_size: float) -> None:
    ax.text(
        0.04,
        0.96,
        str(text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=float(font_size),
        color="#222222",
    )


def _format_skl(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "SKL = n/a"
    if not math.isfinite(val):
        return "SKL = n/a"
    if abs(val) < 0.01 and val != 0.0:
        return f"SKL = {val:.2e}"
    return f"SKL = {val:.3f}"


def _square_boundary_torch(
    u: torch.Tensor,
    *,
    theta: float,
    side_length: float,
    center: tuple[float, float],
) -> torch.Tensor:
    if u.ndim == 1:
        u = u.unsqueeze(-1)
    s = torch.remainder(u, 4.0)
    side = float(side_length)
    h = 0.5 * side
    out = torch.empty(int(u.shape[0]), 2, device=u.device, dtype=u.dtype)
    m0 = s[:, 0] < 1.0
    m1 = (s[:, 0] >= 1.0) & (s[:, 0] < 2.0)
    m2 = (s[:, 0] >= 2.0) & (s[:, 0] < 3.0)
    m3 = s[:, 0] >= 3.0
    out[m0, 0] = -h + side * s[m0, 0]
    out[m0, 1] = -h
    out[m1, 0] = h
    out[m1, 1] = -h + side * (s[m1, 0] - 1.0)
    out[m2, 0] = h - side * (s[m2, 0] - 2.0)
    out[m2, 1] = h
    out[m3, 0] = -h
    out[m3, 1] = h - side * (s[m3, 0] - 3.0)

    c = math.cos(float(theta))
    st = math.sin(float(theta))
    rot = torch.tensor([[c, -st], [st, c]], device=u.device, dtype=u.dtype)
    center_t = torch.tensor(center, device=u.device, dtype=u.dtype).reshape(1, 2)
    return out @ rot.T + center_t


def _log_noisy_square_density(
    x: torch.Tensor,
    *,
    theta: float,
    side_length: float,
    sigma: float,
    center: tuple[float, float],
    u_density: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    sig = float(sigma)
    if sig <= 0.0:
        return torch.full((int(x.shape[0]),), float("nan"), device=x.device, dtype=x.dtype)
    means = _square_boundary_torch(u_density, theta=theta, side_length=side_length, center=center)
    log_norm = -0.5 * int(x.shape[1]) * math.log(2.0 * math.pi * sig * sig)
    out_parts: list[torch.Tensor] = []
    for start in range(0, int(x.shape[0]), int(chunk_size)):
        x_chunk = x[start : start + int(chunk_size)]
        d2 = ((x_chunk[:, None, :] - means[None, :, :]) ** 2).sum(dim=-1)
        out_parts.append(float(log_norm) + torch.logsumexp(-0.5 * d2 / (sig * sig), dim=1) - math.log(int(means.shape[0])))
    return torch.cat(out_parts, dim=0)


def _estimate_target_skl(
    *,
    theta_values: np.ndarray,
    side_length: float,
    sigma: float,
    center: tuple[float, float],
    n_samples: int,
    density_samples: int,
    seed: int,
    device_text: str,
) -> float:
    if int(n_samples) < 1 or int(density_samples) < 1:
        return float("nan")
    if float(sigma) <= 0.0:
        return float("nan")
    dev = require_device(str(device_text))
    dtype = torch.float32
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    x_parts: list[torch.Tensor] = []
    u_density_parts: list[torch.Tensor] = []
    for idx, theta in enumerate(theta_values):
        u_x = 4.0 * torch.rand(int(n_samples), 1, device=dev, dtype=dtype, generator=gen)
        clean = _square_boundary_torch(u_x, theta=float(theta), side_length=float(side_length), center=center)
        x_parts.append(clean + float(sigma) * torch.randn(int(n_samples), 2, device=dev, dtype=dtype, generator=gen))
        u_density_parts.append(4.0 * torch.rand(int(density_samples), 1, device=dev, dtype=dtype, generator=gen))

    log_11 = _log_noisy_square_density(
        x_parts[0],
        theta=float(theta_values[0]),
        side_length=float(side_length),
        sigma=float(sigma),
        center=center,
        u_density=u_density_parts[0],
    )
    log_12 = _log_noisy_square_density(
        x_parts[0],
        theta=float(theta_values[1]),
        side_length=float(side_length),
        sigma=float(sigma),
        center=center,
        u_density=u_density_parts[1],
    )
    log_22 = _log_noisy_square_density(
        x_parts[1],
        theta=float(theta_values[1]),
        side_length=float(side_length),
        sigma=float(sigma),
        center=center,
        u_density=u_density_parts[1],
    )
    log_21 = _log_noisy_square_density(
        x_parts[1],
        theta=float(theta_values[0]),
        side_length=float(side_length),
        sigma=float(sigma),
        center=center,
        u_density=u_density_parts[0],
    )
    skl = torch.mean(log_11 - log_12) + torch.mean(log_22 - log_21)
    return float(torch.clamp(skl, min=0.0).detach().cpu().item())


def _draw_density_contours(
    ax: Any,
    points: np.ndarray,
    *,
    arrays: list[np.ndarray],
    color: str,
    grid_size: int,
    levels: int,
    low_quantile: float,
    high_quantile: float,
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if int(pts.shape[0]) < 5:
        return {"levels": [], "grid_size": int(grid_size), "sample_count": int(pts.shape[0])}
    grid_n = max(30, int(grid_size))
    n_levels = max(2, int(levels))
    xlim, ylim = _axis_limits(arrays)
    xs = np.linspace(xlim[0], xlim[1], grid_n, dtype=np.float64)
    ys = np.linspace(ylim[0], ylim[1], grid_n, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    coords = np.vstack([xx.ravel(), yy.ravel()])
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(pts.T)(coords).reshape(grid_n, grid_n)
    except Exception:
        return {"levels": [], "grid_size": int(grid_n), "sample_count": int(pts.shape[0])}
    finite = np.asarray(density[np.isfinite(density) & (density > 0.0)], dtype=np.float64)
    if int(finite.size) < n_levels:
        return {"levels": [], "grid_size": int(grid_n), "sample_count": int(pts.shape[0])}
    q_low = float(low_quantile)
    q_high = float(high_quantile)
    if not math.isfinite(q_low) or not math.isfinite(q_high) or q_low <= 0.0 or q_high >= 1.0 or q_low >= q_high:
        raise ValueError("contour quantiles must satisfy 0 < low < high < 1.")
    contour_levels = np.quantile(finite, np.linspace(q_low, q_high, n_levels))
    contour_levels = np.unique(contour_levels)
    if int(contour_levels.size) < 2:
        return {"levels": [], "grid_size": int(grid_n), "sample_count": int(pts.shape[0])}
    ax.contour(xx, yy, density, levels=contour_levels, colors=[color], linewidths=1.35, alpha=0.95)
    return {
        "levels": [float(v) for v in contour_levels],
        "grid_size": int(grid_n),
        "sample_count": int(pts.shape[0]),
        "low_quantile": q_low,
        "high_quantile": q_high,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_base_from_summary(summary: dict[str, Any]) -> tuple[NoisyGeometricBase | StandardNormalBase, str]:
    params = dict(summary.get("training_parameters", {}))
    base_geometry = str(params.get("base_geometry", summary.get("base_geometry", "none"))).strip().lower().replace("_", "-")
    if base_geometry == "standard-normal":
        return StandardNormalBase(ambient_dim=2), base_geometry
    inferred_geometry = "square" if base_geometry == "none" else base_geometry
    return (
        NoisyGeometricBase(
            SquarePerimeterBase(center=(0.0, 0.0), side_length=float(params.get("base_side_length", 1.0))),
            sigma=float(params.get("base_noise_sigma", 0.1)),
        ),
        inferred_geometry,
    )


def _generated_from_model(
    *,
    summary_path: Path,
    checkpoint_path: Path,
    n_per_condition: int,
    contour_n_per_condition: int,
    base_samples_per_condition: int,
    device_text: str,
    ode_steps_override: int | None,
    ode_method_override: str | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, Any]]:
    summary = _load_json(summary_path)
    params = dict(summary.get("training_parameters", {}))
    nf_meta = summary.get("nf_likelihood_finetune_metadata") or {}
    condition_eval = np.asarray(summary["condition_eval"], dtype=np.float64)
    if condition_eval.shape[0] != 2:
        raise ValueError("Expected exactly two model conditions.")

    dev = require_device(str(device_text))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_geometric_base_velocity_model(
        velocity_family=str(params.get("velocity_family", summary.get("velocity_family", "unconstrained"))),
        theta_dim=int(condition_eval.shape[1]),
        x_dim=2,
        hidden_dim=int(params.get("hidden_dim", 64)),
        depth=int(params.get("depth", 2)),
        path_schedule=str(params.get("path_schedule", "cosine")),
    ).to(dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    base, resolved_base_geometry = _make_base_from_summary(summary)
    ode_steps = int(ode_steps_override if ode_steps_override is not None else params.get("ode_steps", 32))
    ode_method = str(ode_method_override if ode_method_override is not None else params.get("ode_method", "midpoint"))
    curve_points_per_edge = int(params.get("curve_points_per_edge", 100))
    curve_u = None
    if hasattr(base, "u_low") and hasattr(base, "u_high"):
        curve_u = torch.linspace(base.u_low, base.u_high, 4 * curve_points_per_edge + 1, dtype=torch.float32).reshape(-1, 1)
    checkpoint_sigmas = checkpoint.get("base_noise_sigmas")
    if checkpoint_sigmas is not None:
        selected_sigmas = np.asarray(checkpoint_sigmas, dtype=np.float64).reshape(-1)
    else:
        selected_sigmas = np.asarray(nf_meta.get("selected_base_noise_sigmas", []), dtype=np.float64).reshape(-1)
    skl_sigmas = selected_sigmas if isinstance(base, NoisyGeometricBase) and int(selected_sigmas.size) == int(condition_eval.shape[0]) else None
    corrected_skl = estimate_pushed_base_symmetric_kl(
        model=model,
        base=base,
        theta_all=condition_eval,
        device=dev,
        base_noise_sigmas=skl_sigmas,
        mc_skl_samples=int(params.get("mc_skl_samples", 1024)),
        density_mc_samples=int(params.get("density_mc_samples", 512)),
        ode_steps=ode_steps,
        ode_method=ode_method,
        batch_size=int(params.get("batch_size", 256)),
    )

    generated: list[np.ndarray] = []
    contour_generated: list[np.ndarray] = []
    fitted_boundaries: list[np.ndarray] = []
    base_samples: list[np.ndarray] = []
    base_mean = np.empty((0, 2), dtype=np.float64)
    if curve_u is not None and isinstance(base, NoisyGeometricBase):
        base_mean = base.points_from_u(curve_u.to(device=dev)).detach().cpu().numpy().astype(np.float64)
    for idx, theta_row in enumerate(condition_eval):
        if curve_u is not None:
            curve, _ = push_base_curve(
                model=model,
                base=base,  # type: ignore[arg-type]
                theta=theta_row.reshape(1, -1),
                device=dev,
                u=curve_u,
                ode_steps=ode_steps,
                ode_method=ode_method,
            )
            fitted_boundaries.append(curve.detach().cpu().numpy().astype(np.float64))
        else:
            fitted_boundaries.append(np.empty((0, 2), dtype=np.float64))

        if isinstance(base, NoisyGeometricBase):
            u_gen = base.sample_u(int(n_per_condition), device=dev, dtype=torch.float32)
            x0_gen = base.points_from_u(u_gen)
            sigma = float(selected_sigmas[idx]) if int(selected_sigmas.size) == int(condition_eval.shape[0]) else float(base.sigma)
            if sigma > 0.0:
                x0_gen = x0_gen + sigma * torch.randn_like(x0_gen)
            u_contour = base.sample_u(int(contour_n_per_condition), device=dev, dtype=torch.float32)
            x0_contour = base.points_from_u(u_contour)
            if sigma > 0.0:
                x0_contour = x0_contour + sigma * torch.randn_like(x0_contour)
            u_base = base.sample_u(int(base_samples_per_condition), device=dev, dtype=torch.float32)
            x0_base = base.points_from_u(u_base)
            if sigma > 0.0:
                x0_base = x0_base + sigma * torch.randn_like(x0_base)
        else:
            x0_gen = base.sample(int(n_per_condition), device=dev, dtype=torch.float32)
            x0_contour = base.sample(int(contour_n_per_condition), device=dev, dtype=torch.float32)
            x0_base = base.sample(int(base_samples_per_condition), device=dev, dtype=torch.float32)
        base_samples.append(x0_base.detach().cpu().numpy().astype(np.float64))
        pushed = push_initial_points(
            model=model,
            x0=x0_gen,
            theta=theta_row.reshape(1, -1),
            device=dev,
            ode_steps=ode_steps,
            ode_method=ode_method,
        )
        generated.append(pushed.detach().cpu().numpy().astype(np.float64))
        pushed_contour = push_initial_points(
            model=model,
            x0=x0_contour,
            theta=theta_row.reshape(1, -1),
            device=dev,
            ode_steps=ode_steps,
            ode_method=ode_method,
        )
        contour_generated.append(pushed_contour.detach().cpu().numpy().astype(np.float64))
    summary = dict(summary)
    summary["resolved_base_geometry"] = resolved_base_geometry
    summary["summary_skl_value"] = summary.get("skl_value")
    summary["skl_value"] = float(corrected_skl.symmetric_kl_matrix[0, 1]) if int(condition_eval.shape[0]) > 1 else 0.0
    summary["canonical_metric_name"] = corrected_skl.canonical_metric_name
    summary["symmetric_kl_matrix"] = corrected_skl.symmetric_kl_matrix
    summary["skl_metric_source"] = "recomputed_general_cnf"
    return generated, contour_generated, fitted_boundaries, base_samples, base_mean, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    theta_values = _parse_theta_values(str(args.theta_values))
    n_per_condition = int(args.n_per_condition)
    if n_per_condition < 1:
        raise ValueError("--n-per-condition must be >= 1.")
    if int(args.generated_samples_per_condition) < 1:
        raise ValueError("--generated-samples-per-condition must be >= 1.")
    if int(args.contour_target_samples_per_condition) < 5:
        raise ValueError("--contour-target-samples-per-condition must be >= 5.")
    if int(args.contour_generated_samples_per_condition) < 5:
        raise ValueError("--contour-generated-samples-per-condition must be >= 5.")
    if int(args.target_skl_samples) < 1:
        raise ValueError("--target-skl-samples must be >= 1.")
    if int(args.target_skl_density_samples) < 1:
        raise ValueError("--target-skl-density-samples must be >= 1.")
    if int(args.base_samples_per_condition) < 1:
        raise ValueError("--base-samples-per-condition must be >= 1.")
    if int(args.contour_grid_size) < 30:
        raise ValueError("--contour-grid-size must be >= 30.")
    if int(args.contour_levels) < 2:
        raise ValueError("--contour-levels must be >= 2.")
    if not (
        0.0 < float(args.contour_low_quantile) < float(args.contour_high_quantile) < 1.0
    ):
        raise ValueError("--contour-low-quantile and --contour-high-quantile must satisfy 0 < low < high < 1.")
    if int(args.points_per_edge) < 2:
        raise ValueError("--points-per-edge must be >= 2.")
    if float(args.side_length) <= 0.0:
        raise ValueError("--side-length must be > 0.")
    if float(args.target_sigma) < 0.0:
        raise ValueError("--target-sigma must be >= 0.")

    datasets: list[NoisySquareBoundaryDataset] = []
    samples: list[np.ndarray] = []
    contour_samples: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    center = (float(args.center_x), float(args.center_y))
    for idx, theta in enumerate(theta_values):
        ds = NoisySquareBoundaryDataset(
            theta=float(theta),
            side_length=float(args.side_length),
            sigma=float(args.target_sigma),
            center=center,
            seed=int(args.seed) + idx,
        )
        batch = ds.sample(n_per_condition)
        contour_batch = ds.sample(int(args.contour_target_samples_per_condition))
        datasets.append(ds)
        samples.append(batch.x1.astype(np.float64, copy=False))
        contour_samples.append(contour_batch.x1.astype(np.float64, copy=False))
        boundaries.append(ds.boundary(points_per_edge=int(args.points_per_edge)).astype(np.float64, copy=False))
    target_skl_value = _estimate_target_skl(
        theta_values=theta_values,
        side_length=float(args.side_length),
        sigma=float(args.target_sigma),
        center=center,
        n_samples=int(args.target_skl_samples),
        density_samples=int(args.target_skl_density_samples),
        seed=int(args.seed) + 991,
        device_text=str(args.device),
    )
    target_skl_text = _format_skl(target_skl_value)
    torch.manual_seed(int(args.seed) + 12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + 12345)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518"]
    model_summaries = [] if args.model_summary is None else list(args.model_summary)
    model_checkpoints = [] if args.model_checkpoint is None else list(args.model_checkpoint)
    model_labels = [] if args.model_label is None else list(args.model_label)
    has_model = bool(model_summaries or model_checkpoints)
    if len(model_summaries) != len(model_checkpoints):
        raise ValueError("--model-summary and --model-checkpoint must be provided the same number of times.")
    if model_labels and len(model_labels) != len(model_summaries):
        raise ValueError("--model-label must be omitted or provided once per model.")
    panel_count = 1 + len(model_summaries)
    layout = str(args.layout).strip().lower()
    if layout == "overlap":
        fig, axes_arr = plt.subplots(1, panel_count, figsize=(5.2 * panel_count, 5.4), squeeze=False)
        axes = list(axes_arr[0])
        target_arrays: list[np.ndarray] = [*samples, *boundaries]
        shared_axis_limits = _axis_limits(target_arrays)
        model_summaries_out: list[dict[str, Any]] = []
        generated_shapes: list[list[list[int]]] | None = None
        base_sample_shapes: list[list[list[int]]] | None = None
        if has_model:
            generated_shapes = []
            base_sample_shapes = []
            generated_panels: list[tuple[str, str, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]] = []
            for model_idx, (summary_arg, checkpoint_arg) in enumerate(zip(model_summaries, model_checkpoints, strict=True)):
                generated, _contour_generated, fitted_boundaries, base_samples, base_mean, model_summary = _generated_from_model(
                    summary_path=Path(summary_arg).expanduser().resolve(),
                    checkpoint_path=Path(checkpoint_arg).expanduser().resolve(),
                    n_per_condition=int(args.generated_samples_per_condition),
                    contour_n_per_condition=max(5, int(args.generated_samples_per_condition)),
                    base_samples_per_condition=int(args.base_samples_per_condition),
                    device_text=str(args.device),
                    ode_steps_override=args.ode_steps,
                    ode_method_override=args.ode_method,
                )
                panel_name = str(model_labels[model_idx]) if model_labels else (
                    f"{model_summary.get('base_geometry', 'base')} + {model_summary.get('velocity_family', 'velocity')}"
                )
                generated_shapes.append([list(arr.shape) for arr in generated])
                base_sample_shapes.append([list(arr.shape) for arr in base_samples])
                model_summaries_out.append(
                    {
                        "label": panel_name,
                        "summary": str(Path(summary_arg).expanduser().resolve()),
                        "checkpoint": str(Path(checkpoint_arg).expanduser().resolve()),
                        "base_geometry": model_summary.get("base_geometry"),
                        "velocity_family": model_summary.get("velocity_family"),
                        "skl_value": model_summary.get("skl_value"),
                        "summary_skl_value": model_summary.get("summary_skl_value"),
                        "canonical_metric_name": model_summary.get("canonical_metric_name"),
                        "skl_metric_source": model_summary.get("skl_metric_source"),
                        "resolved_base_geometry": model_summary.get("resolved_base_geometry"),
                    }
                )
                generated_panels.append((panel_name, _format_skl(model_summary.get("skl_value")), generated, fitted_boundaries, base_samples, base_mean))

            ax_target = axes[0]
            for idx in range(2):
                color = colors[idx % len(colors)]
                ax_target.scatter(
                    samples[idx][:, 0],
                    samples[idx][:, 1],
                    s=float(args.point_size),
                    marker="x",
                    alpha=1.0,
                    color=color,
                    linewidths=1.15,
                    label=f"Dataset {idx + 1}",
                )
                ax_target.plot(
                    boundaries[idx][:, 0],
                    boundaries[idx][:, 1],
                    color=color,
                    linewidth=3.2,
                    alpha=0.95,
                    label=f"Mean {idx + 1}",
                )
            _panel_label(ax_target, f"Target\n{target_skl_text}", font_size=float(args.panel_label_font_size))
            _style_panel(
                ax_target,
                target_arrays,
                legend_font_size=float(args.legend_font_size),
                show_legend=False,
                axis_limits=shared_axis_limits,
            )
            legend_handles = [
                plt.Line2D([0], [0], marker="x", linestyle="None", color=colors[0], alpha=1.0, label="Dataset 1"),
                plt.Line2D([0], [0], marker="x", linestyle="None", color=colors[1], alpha=1.0, label="Dataset 2"),
                plt.Line2D([0], [0], marker="o", linestyle="None", color=colors[0], alpha=float(args.alpha), label="Generated 1"),
                plt.Line2D([0], [0], marker="o", linestyle="None", color=colors[1], alpha=float(args.alpha), label="Generated 2"),
                plt.Line2D([0], [0], color=colors[0], linewidth=3.2, label="Mean 1"),
                plt.Line2D([0], [0], color=colors[1], linewidth=3.2, label="Mean 2"),
                plt.Line2D([0], [0], color="#111111", linewidth=3.0, linestyle="--", label="Base mean"),
            ]
            ax_target.legend(
                handles=legend_handles,
                frameon=False,
                loc="center",
                bbox_to_anchor=(0.56, 0.43),
                fontsize=max(9.0, 0.78 * float(args.legend_font_size)),
                markerscale=1.55,
                ncol=2,
                columnspacing=1.1,
                handlelength=1.8,
                handletextpad=0.5,
            )

            for model_idx, (panel_name, skl_text, generated, fitted_boundaries, base_samples, base_mean) in enumerate(generated_panels):
                ax_gen = axes[1 + model_idx]
                panel_arrays = [*generated]
                if int(base_mean.size) > 0:
                    panel_arrays.append(base_mean)
                    ax_gen.plot(
                        base_mean[:, 0],
                        base_mean[:, 1],
                        color="#111111",
                        linewidth=3.0,
                        linestyle="--",
                        label="Base mean",
                    )
                for idx in range(2):
                    color = colors[idx % len(colors)]
                    ax_gen.scatter(
                        generated[idx][:, 0],
                        generated[idx][:, 1],
                        s=float(args.point_size),
                        alpha=float(args.alpha),
                        color=color,
                        linewidths=0,
                        label=f"Generated {idx + 1}",
                    )
                    if int(fitted_boundaries[idx].size) > 0:
                        ax_gen.plot(
                            fitted_boundaries[idx][:, 0],
                            fitted_boundaries[idx][:, 1],
                            color=color,
                            linewidth=3.2,
                            alpha=0.95,
                            label="_nolegend_",
                        )
                        panel_arrays.append(fitted_boundaries[idx])
                _panel_label(ax_gen, f"{panel_name}\n{skl_text}", font_size=float(args.panel_label_font_size))
                _style_panel(
                    ax_gen,
                    panel_arrays,
                    legend_font_size=float(args.legend_font_size),
                    show_legend=False,
                    axis_limits=shared_axis_limits,
                )
        else:
            ax_target = axes[0]
            for idx in range(2):
                color = colors[idx % len(colors)]
                ax_target.scatter(
                    samples[idx][:, 0],
                    samples[idx][:, 1],
                    s=float(args.point_size),
                    marker="x",
                    alpha=1.0,
                    color=color,
                    linewidths=1.15,
                    label=f"Dataset {idx + 1}",
                )
                ax_target.plot(
                    boundaries[idx][:, 0],
                    boundaries[idx][:, 1],
                    color=color,
                    linewidth=3.2,
                    alpha=0.95,
                    label=f"Mean {idx + 1}",
                )
            _panel_label(ax_target, f"Target\n{target_skl_text}", font_size=float(args.panel_label_font_size))
            _style_panel(ax_target, target_arrays, legend_font_size=float(args.legend_font_size), axis_limits=shared_axis_limits)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.03)

        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(args.prefix).strip() or "two_square_target_dataset"
        png_path = out_dir / f"{prefix}.png"
        svg_path = out_dir / f"{prefix}.svg"
        summary_path = out_dir / f"{prefix}_summary.json"
        fig.savefig(png_path, dpi=180)
        fig.savefig(svg_path)
        plt.close(fig)

        summary = {
            "script": "bin/visualize_two_square_target_dataset.py",
            "theta_values": theta_values.tolist(),
            "n_per_condition": n_per_condition,
            "side_length": float(args.side_length),
            "target_sigma": float(args.target_sigma),
            "target_skl_value": target_skl_value,
            "target_skl_samples": int(args.target_skl_samples),
            "target_skl_density_samples": int(args.target_skl_density_samples),
            "center": [float(args.center_x), float(args.center_y)],
            "seed": int(args.seed),
            "points_per_edge": int(args.points_per_edge),
            "layout": "overlap",
            "shared_axis_limits": [[float(v) for v in shared_axis_limits[0]], [float(v) for v in shared_axis_limits[1]]],
            "sample_shapes": [list(arr.shape) for arr in samples],
            "model_panels": model_summaries_out,
            "generated_sample_shapes": generated_shapes,
            "base_sample_shapes": base_sample_shapes,
            "png": str(png_path),
            "svg": str(svg_path),
            "summary": str(summary_path),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

        print(f"png: {png_path}", flush=True)
        print(f"svg: {svg_path}", flush=True)
        print(f"summary_json: {summary_path}", flush=True)
        return 0

    fig, axes_arr = plt.subplots(2, panel_count, figsize=(4.8 * panel_count, 8.2), squeeze=False)
    all_arrays: list[np.ndarray] = [*samples, *boundaries]
    model_summaries_out: list[dict[str, Any]] = []
    generated_shapes: list[list[list[int]]] | None = None
    base_sample_shapes: list[list[list[int]]] | None = None
    contour_metadata: list[dict[str, Any]] = []
    if has_model:
        shared_arrays = list(all_arrays)
        generated_shapes = []
        base_sample_shapes = []
        generated_panels: list[tuple[str, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]] = []
        for model_idx, (summary_arg, checkpoint_arg) in enumerate(zip(model_summaries, model_checkpoints, strict=True)):
            generated, contour_generated, fitted_boundaries, base_samples, base_mean, model_summary = _generated_from_model(
                summary_path=Path(summary_arg).expanduser().resolve(),
                checkpoint_path=Path(checkpoint_arg).expanduser().resolve(),
                n_per_condition=int(args.generated_samples_per_condition),
                contour_n_per_condition=int(args.contour_generated_samples_per_condition),
                base_samples_per_condition=int(args.base_samples_per_condition),
                device_text=str(args.device),
                ode_steps_override=args.ode_steps,
                ode_method_override=args.ode_method,
            )
            if model_labels:
                panel_name = str(model_labels[model_idx])
            else:
                panel_name = f"{model_summary.get('base_geometry', 'base')} + {model_summary.get('velocity_family', 'velocity')}"
            panel_arrays = [*generated, *fitted_boundaries, *base_samples]
            if int(base_mean.size) > 0:
                panel_arrays.append(base_mean)
            shared_arrays.extend(panel_arrays)
            generated_shapes.append([list(arr.shape) for arr in generated])
            base_sample_shapes.append([list(arr.shape) for arr in base_samples])
            model_summaries_out.append(
                {
                    "label": panel_name,
                    "summary": str(Path(summary_arg).expanduser().resolve()),
                    "checkpoint": str(Path(checkpoint_arg).expanduser().resolve()),
                    "base_geometry": model_summary.get("base_geometry"),
                    "velocity_family": model_summary.get("velocity_family"),
                    "skl_value": model_summary.get("skl_value"),
                    "summary_skl_value": model_summary.get("summary_skl_value"),
                    "canonical_metric_name": model_summary.get("canonical_metric_name"),
                    "skl_metric_source": model_summary.get("skl_metric_source"),
                    "resolved_base_geometry": model_summary.get("resolved_base_geometry"),
                }
            )
            generated_panels.append((panel_name, generated, contour_generated, fitted_boundaries, base_samples, base_mean))

        for row_idx in range(2):
            color = colors[row_idx % len(colors)]
            ax_target = axes_arr[row_idx, 0]
            ax_target.scatter(
                samples[row_idx][:, 0],
                samples[row_idx][:, 1],
                s=float(args.point_size),
                alpha=float(args.alpha),
                color=color,
                linewidths=0,
                label=f"Dataset {row_idx + 1}",
            )
            target_contours = _draw_density_contours(
                ax_target,
                contour_samples[row_idx],
                arrays=shared_arrays,
                color=color,
                grid_size=int(args.contour_grid_size),
                levels=int(args.contour_levels),
                low_quantile=float(args.contour_low_quantile),
                high_quantile=float(args.contour_high_quantile),
            )
            ax_target.plot([], [], color=color, linewidth=1.35, label="density contours")
            contour_metadata.append({"panel": "Target", "condition": int(row_idx + 1), **target_contours})
            if row_idx == 0:
                _panel_label(ax_target, f"Target\n{target_skl_text}", font_size=float(args.legend_font_size))
            _style_panel(ax_target, shared_arrays, legend_font_size=float(args.legend_font_size))

            for model_idx, (panel_name, generated, contour_generated, fitted_boundaries, base_samples, base_mean) in enumerate(generated_panels):
                ax_gen = axes_arr[row_idx, 1 + model_idx]
                ax_gen.scatter(
                    base_samples[row_idx][:, 0],
                    base_samples[row_idx][:, 1],
                    s=max(3.0, 0.55 * float(args.point_size)),
                    alpha=0.22,
                    color="#111111",
                    linewidths=0,
                    label="Base",
                )
                if int(base_mean.size) > 0:
                    ax_gen.plot(
                        base_mean[:, 0],
                        base_mean[:, 1],
                        color="#111111",
                        linewidth=1.6,
                        linestyle="--",
                        label="Base mean",
                    )
                ax_gen.scatter(
                    generated[row_idx][:, 0],
                    generated[row_idx][:, 1],
                    s=float(args.point_size),
                    alpha=float(args.alpha),
                    color=color,
                    linewidths=0,
                    label=f"Generated {row_idx + 1}",
                )
                generated_contours = _draw_density_contours(
                    ax_gen,
                    contour_generated[row_idx],
                    arrays=shared_arrays,
                    color=color,
                    grid_size=int(args.contour_grid_size),
                    levels=int(args.contour_levels),
                    low_quantile=float(args.contour_low_quantile),
                    high_quantile=float(args.contour_high_quantile),
                )
                ax_gen.plot([], [], color=color, linewidth=1.35, label="density contours")
                contour_metadata.append(
                    {"panel": panel_name, "condition": int(row_idx + 1), **generated_contours}
                )
                if row_idx == 0:
                    _panel_label(ax_gen, panel_name, font_size=float(args.legend_font_size))
                _style_panel(ax_gen, shared_arrays, legend_font_size=float(args.legend_font_size))
    else:
        shared_arrays = list(all_arrays)
        for row_idx in range(2):
            color = colors[row_idx % len(colors)]
            ax_target = axes_arr[row_idx, 0]
            ax_target.scatter(
                samples[row_idx][:, 0],
                samples[row_idx][:, 1],
                s=float(args.point_size),
                alpha=float(args.alpha),
                color=color,
                linewidths=0,
                label=f"Dataset {row_idx + 1}",
            )
            target_contours = _draw_density_contours(
                ax_target,
                contour_samples[row_idx],
                arrays=shared_arrays,
                color=color,
                grid_size=int(args.contour_grid_size),
                levels=int(args.contour_levels),
                low_quantile=float(args.contour_low_quantile),
                high_quantile=float(args.contour_high_quantile),
            )
            ax_target.plot([], [], color=color, linewidth=1.35, label="density contours")
            contour_metadata.append({"panel": "Target", "condition": int(row_idx + 1), **target_contours})
            if row_idx == 0:
                _panel_label(ax_target, f"Target\n{target_skl_text}", font_size=float(args.legend_font_size))
            _style_panel(ax_target, shared_arrays, legend_font_size=float(args.legend_font_size))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.03, hspace=0.03)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip() or "two_square_target_dataset"
    png_path = out_dir / f"{prefix}.png"
    svg_path = out_dir / f"{prefix}.svg"
    summary_path = out_dir / f"{prefix}_summary.json"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)

    summary = {
        "script": "bin/visualize_two_square_target_dataset.py",
        "theta_values": theta_values.tolist(),
        "n_per_condition": n_per_condition,
        "side_length": float(args.side_length),
        "target_sigma": float(args.target_sigma),
        "target_skl_value": target_skl_value,
        "target_skl_samples": int(args.target_skl_samples),
        "target_skl_density_samples": int(args.target_skl_density_samples),
        "center": [float(args.center_x), float(args.center_y)],
        "seed": int(args.seed),
        "points_per_edge": int(args.points_per_edge),
        "layout": "condition_rows",
        "contour_grid_size": int(args.contour_grid_size),
        "contour_levels": int(args.contour_levels),
        "contour_low_quantile": float(args.contour_low_quantile),
        "contour_high_quantile": float(args.contour_high_quantile),
        "contour_target_samples_per_condition": int(args.contour_target_samples_per_condition),
        "contour_generated_samples_per_condition": int(args.contour_generated_samples_per_condition),
        "contour_metadata": contour_metadata,
        "sample_shapes": [list(arr.shape) for arr in samples],
        "model_panels": model_summaries_out,
        "generated_sample_shapes": generated_shapes,
        "base_sample_shapes": base_sample_shapes,
        "png": str(png_path),
        "svg": str(svg_path),
        "summary": str(summary_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"png: {png_path}", flush=True)
    print(f"svg: {svg_path}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
