"""Noisy square-boundary geometric dataset for rigid-base flow experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NoisySquareBoundaryBatch:
    """Matched base-boundary and noisy rotated square-boundary samples."""

    x0: np.ndarray
    x1: np.ndarray
    u: np.ndarray
    eta: np.ndarray


def _square_boundary_from_u(u: np.ndarray, *, side_length: float = 1.0) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    side = float(side_length)
    if not math.isfinite(side) or side <= 0.0:
        raise ValueError("side_length must be finite and positive.")
    s = np.mod(u_arr, 4.0)
    h = 0.5 * side
    out = np.empty((int(s.size), 2), dtype=np.float64)
    m0 = s < 1.0
    m1 = (s >= 1.0) & (s < 2.0)
    m2 = (s >= 2.0) & (s < 3.0)
    m3 = s >= 3.0
    out[m0, 0] = -h + side * s[m0]
    out[m0, 1] = -h
    out[m1, 0] = h
    out[m1, 1] = -h + side * (s[m1] - 1.0)
    out[m2, 0] = h - side * (s[m2] - 2.0)
    out[m2, 1] = h
    out[m3, 0] = -h
    out[m3, 1] = h - side * (s[m3] - 3.0)
    return out


def square_rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2D rotation matrix for angle ``theta`` in radians."""

    th = float(theta)
    c = math.cos(th)
    s = math.sin(th)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


@dataclass
class NoisySquareBoundaryDataset:
    """Noisy rotated square-boundary target distribution in two dimensions."""

    theta: float = math.pi / 4.0
    side_length: float = 1.0
    sigma: float = 0.03
    center: tuple[float, float] = (0.0, 0.0)
    seed: int = 7

    def __post_init__(self) -> None:
        side = float(self.side_length)
        if not math.isfinite(side) or side <= 0.0:
            raise ValueError("side_length must be > 0.")
        if float(self.sigma) < 0.0:
            raise ValueError("sigma must be >= 0.")
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("center must contain exactly two values.")
        if not np.all(np.isfinite(center)):
            raise ValueError("center must be finite.")
        self.center = (float(center[0]), float(center[1]))
        self.rng = np.random.default_rng(int(self.seed))

    @property
    def rotation(self) -> np.ndarray:
        return square_rotation_matrix(float(self.theta))

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    def boundary(self, u: np.ndarray | None = None, *, points_per_edge: int = 100) -> np.ndarray:
        """Return ordered rotated square-boundary points."""

        if u is None:
            count = int(points_per_edge)
            if count < 2:
                raise ValueError("points_per_edge must be >= 2.")
            u_arr = np.linspace(0.0, 4.0, 4 * count + 1, dtype=np.float64)
        else:
            u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        base = _square_boundary_from_u(u_arr, side_length=float(self.side_length))
        return base @ self.rotation.T + self.center_array.reshape(1, 2)

    def sample(self, num: int, *, rng: np.random.Generator | None = None) -> NoisySquareBoundaryBatch:
        """Draw matched base-boundary and noisy rotated target samples."""

        count = int(num)
        if count < 1:
            raise ValueError("num must be >= 1.")
        gen = self.rng if rng is None else rng
        u = gen.uniform(0.0, 4.0, size=(count, 1)).astype(np.float64, copy=False)
        eta = gen.standard_normal(size=(count, 2)).astype(np.float64, copy=False)
        x0 = _square_boundary_from_u(u, side_length=float(self.side_length))
        x_clean = x0 @ self.rotation.T + self.center_array.reshape(1, 2)
        x1 = x_clean + float(self.sigma) * eta
        return NoisySquareBoundaryBatch(
            x0=x0.astype(np.float64, copy=False),
            x1=x1.astype(np.float64, copy=False),
            u=u.reshape(-1).astype(np.float64, copy=False),
            eta=eta.astype(np.float64, copy=False),
        )


def generate_noisy_square_boundary_batch(
    *,
    num: int,
    theta: float = math.pi / 4.0,
    side_length: float = 1.0,
    sigma: float = 0.03,
    center: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NoisySquareBoundaryBatch:
    """Functional wrapper for drawing a noisy square-boundary batch."""

    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    dataset = NoisySquareBoundaryDataset(
        theta=theta,
        side_length=side_length,
        sigma=sigma,
        center=center,
        seed=0 if seed is None else int(seed),
    )
    return dataset.sample(int(num), rng=rng)


def plot_noisy_square_boundary_dataset(
    batch: NoisySquareBoundaryBatch,
    dataset: NoisySquareBoundaryDataset,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a scatter plot of base square, noisy target square, and target boundary."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(batch.x0[:, 0], batch.x0[:, 1], s=10, alpha=0.35, label="source x0", color="#4c78a8")
    ax.scatter(batch.x1[:, 0], batch.x1[:, 1], s=10, alpha=0.35, label="target x1", color="#f58518")
    boundary = dataset.boundary(points_per_edge=120)
    ax.plot(boundary[:, 0], boundary[:, 1], color="#222222", linewidth=2.0, label="target boundary")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Noisy square-boundary dataset")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
