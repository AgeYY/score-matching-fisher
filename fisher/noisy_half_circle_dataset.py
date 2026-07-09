"""Noisy half-circle boundary dataset for rigid-base flow experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NoisyHalfCircleBoundaryBatch:
    """Matched base half-circle and noisy shifted target samples."""

    x0: np.ndarray
    x1: np.ndarray
    u: np.ndarray
    eta: np.ndarray


HALF_CIRCLE_ARCS = ("upper", "lower")


def _normalize_arc(value: str) -> str:
    arc = str(value).strip().lower()
    if arc not in HALF_CIRCLE_ARCS:
        raise ValueError(f"arc must be one of {HALF_CIRCLE_ARCS}; got {value!r}.")
    return arc


def _half_circle_from_u(u: np.ndarray, *, radius: float = 1.0, arc: str = "upper") -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be finite and positive.")
    arc_norm = _normalize_arc(arc)
    theta = math.pi * np.clip(u_arr, 0.0, 1.0)
    y_sign = 1.0 if arc_norm == "upper" else -1.0
    return np.column_stack((r * np.cos(theta), y_sign * r * np.sin(theta))).astype(np.float64, copy=False)


@dataclass
class NoisyHalfCircleBoundaryDataset:
    """Noisy shifted half-circle target distribution in two dimensions."""

    radius: float = 1.0
    sigma: float = 0.2
    center: tuple[float, float] = (0.0, 0.0)
    arc: str = "upper"
    seed: int = 7

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be > 0.")
        if float(self.sigma) < 0.0:
            raise ValueError("sigma must be >= 0.")
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("center must contain exactly two values.")
        if not np.all(np.isfinite(center)):
            raise ValueError("center must be finite.")
        self.center = (float(center[0]), float(center[1]))
        self.arc = _normalize_arc(str(self.arc))
        self.rng = np.random.default_rng(int(self.seed))

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    def boundary(self, u: np.ndarray | None = None, *, n_points: int = 200) -> np.ndarray:
        """Return ordered shifted half-circle boundary points."""

        if u is None:
            count = int(n_points)
            if count < 2:
                raise ValueError("n_points must be >= 2.")
            u_arr = np.linspace(0.0, 1.0, count, dtype=np.float64)
        else:
            u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        base = _half_circle_from_u(u_arr, radius=float(self.radius), arc=str(self.arc))
        return base + self.center_array.reshape(1, 2)

    def sample(self, num: int, *, rng: np.random.Generator | None = None) -> NoisyHalfCircleBoundaryBatch:
        """Draw matched base half-circle and noisy shifted target samples."""

        count = int(num)
        if count < 1:
            raise ValueError("num must be >= 1.")
        gen = self.rng if rng is None else rng
        u = gen.uniform(0.0, 1.0, size=(count, 1)).astype(np.float64, copy=False)
        eta = gen.standard_normal(size=(count, 2)).astype(np.float64, copy=False)
        x0 = _half_circle_from_u(u, radius=float(self.radius), arc="upper")
        target_clean = _half_circle_from_u(u, radius=float(self.radius), arc=str(self.arc))
        x1 = target_clean + self.center_array.reshape(1, 2) + float(self.sigma) * eta
        return NoisyHalfCircleBoundaryBatch(
            x0=x0.astype(np.float64, copy=False),
            x1=x1.astype(np.float64, copy=False),
            u=u.reshape(-1).astype(np.float64, copy=False),
            eta=eta.astype(np.float64, copy=False),
        )


def generate_noisy_half_circle_boundary_batch(
    *,
    num: int,
    radius: float = 1.0,
    sigma: float = 0.2,
    center: tuple[float, float] = (0.0, 0.0),
    arc: str = "upper",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NoisyHalfCircleBoundaryBatch:
    """Functional wrapper for drawing a noisy half-circle boundary batch."""

    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    dataset = NoisyHalfCircleBoundaryDataset(
        radius=radius,
        sigma=sigma,
        center=center,
        arc=arc,
        seed=0 if seed is None else int(seed),
    )
    return dataset.sample(int(num), rng=rng)


def plot_noisy_half_circle_boundary_datasets(
    batches: list[NoisyHalfCircleBoundaryBatch] | tuple[NoisyHalfCircleBoundaryBatch, ...],
    datasets: list[NoisyHalfCircleBoundaryDataset] | tuple[NoisyHalfCircleBoundaryDataset, ...],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a scatter plot of base and noisy shifted half-circle datasets."""

    if len(batches) != len(datasets):
        raise ValueError("batches and datasets must have the same length.")
    if len(batches) < 1:
        raise ValueError("At least one dataset is required.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(
        batches[0].x0[:, 0],
        batches[0].x0[:, 1],
        s=10,
        alpha=0.35,
        label="base half-circle",
        color="#2f2f2f",
    )
    for idx, (batch, dataset) in enumerate(zip(batches, datasets, strict=True)):
        color = colors[idx % len(colors)]
        ax.scatter(batch.x1[:, 0], batch.x1[:, 1], s=10, alpha=0.45, label=f"target {idx + 1}", color=color)
        boundary = dataset.boundary(n_points=240)
        ax.plot(boundary[:, 0], boundary[:, 1], color=color, linewidth=2.0, label=f"target boundary {idx + 1}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Noisy half-circle boundary datasets")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
