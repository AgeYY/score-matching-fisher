"""Noisy-line geometric dataset used for rigid-base flow experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NoisyLineBatch:
    """Matched source/target samples for the noisy-line construction."""

    x0: np.ndarray
    x1: np.ndarray
    u: np.ndarray
    eta: np.ndarray


@dataclass
class NoisyLineDataset:
    """Noiseless line source and noisy rotated target line in two dimensions.

    Source samples are ``x0 = (u, 0)`` with ``u ~ Uniform[-1/2, 1/2]``. Target
    samples use the same latent ``u``:

    ``x1 = shift + ell * u * q + sigma * eta * n``,

    where ``q=(cos(theta), sin(theta))`` and ``n=(-sin(theta), cos(theta))``.
    """

    theta: float = math.pi / 6.0
    ell: float = 1.5
    sigma: float = 0.12
    shift: tuple[float, float] = (0.0, 0.0)
    seed: int = 7

    def __post_init__(self) -> None:
        if float(self.ell) <= 0.0:
            raise ValueError("ell must be > 0.")
        if float(self.sigma) < 0.0:
            raise ValueError("sigma must be >= 0.")
        shift = np.asarray(self.shift, dtype=np.float64)
        if shift.shape != (2,):
            raise ValueError("shift must contain exactly two values.")
        self.shift = (float(shift[0]), float(shift[1]))
        self.rng = np.random.default_rng(int(self.seed))

    @property
    def q(self) -> np.ndarray:
        th = float(self.theta)
        return np.asarray([math.cos(th), math.sin(th)], dtype=np.float64)

    @property
    def n(self) -> np.ndarray:
        th = float(self.theta)
        return np.asarray([-math.sin(th), math.cos(th)], dtype=np.float64)

    @property
    def shift_array(self) -> np.ndarray:
        return np.asarray(self.shift, dtype=np.float64)

    def centerline(self, u: np.ndarray | None = None, *, num: int = 200) -> np.ndarray:
        """Return target centerline points for supplied or evenly spaced latent values."""

        if u is None:
            if int(num) < 2:
                raise ValueError("num must be >= 2 when u is not supplied.")
            u_arr = np.linspace(-0.5, 0.5, int(num), dtype=np.float64)
        else:
            u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        return self.shift_array.reshape(1, 2) + float(self.ell) * u_arr.reshape(-1, 1) * self.q.reshape(1, 2)

    def sample(self, num: int, *, rng: np.random.Generator | None = None) -> NoisyLineBatch:
        """Draw matched source and target samples."""

        count = int(num)
        if count < 1:
            raise ValueError("num must be >= 1.")
        gen = self.rng if rng is None else rng
        u = gen.uniform(-0.5, 0.5, size=(count, 1)).astype(np.float64, copy=False)
        eta = gen.standard_normal(size=(count, 1)).astype(np.float64, copy=False)
        x0 = np.concatenate([u, np.zeros_like(u)], axis=1)
        x1 = (
            self.shift_array.reshape(1, 2)
            + float(self.ell) * u * self.q.reshape(1, 2)
            + float(self.sigma) * eta * self.n.reshape(1, 2)
        )
        return NoisyLineBatch(
            x0=x0.astype(np.float64, copy=False),
            x1=x1.astype(np.float64, copy=False),
            u=u.reshape(-1).astype(np.float64, copy=False),
            eta=eta.reshape(-1).astype(np.float64, copy=False),
        )

    def target_coordinates(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project target-space points onto tangent and normal line coordinates."""

        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or int(arr.shape[1]) != 2:
            raise ValueError(f"x must have shape (N, 2); got {arr.shape}.")
        centered = arr - self.shift_array.reshape(1, 2)
        tangent = centered @ self.q
        normal = centered @ self.n
        return tangent, normal


def noisy_line_basis(theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the target tangent and normal basis vectors."""

    ds = NoisyLineDataset(theta=float(theta))
    return ds.q, ds.n


def generate_noisy_line_batch(
    *,
    num: int,
    theta: float = math.pi / 6.0,
    ell: float = 1.5,
    sigma: float = 0.12,
    shift: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NoisyLineBatch:
    """Functional wrapper for drawing a matched noisy-line batch."""

    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    dataset = NoisyLineDataset(theta=theta, ell=ell, sigma=sigma, shift=shift, seed=0 if seed is None else int(seed))
    return dataset.sample(int(num), rng=rng)


def plot_noisy_line_dataset(
    batch: NoisyLineBatch,
    dataset: NoisyLineDataset,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a scatter plot of source line, noisy target line, and target centerline."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(batch.x0[:, 0], batch.x0[:, 1], s=10, alpha=0.35, label="source x0", color="#4c78a8")
    ax.scatter(batch.x1[:, 0], batch.x1[:, 1], s=10, alpha=0.35, label="target x1", color="#f58518")
    line = dataset.centerline(num=200)
    ax.plot(line[:, 0], line[:, 1], color="#222222", linewidth=2.0, label="target centerline")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Noisy-line dataset")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
