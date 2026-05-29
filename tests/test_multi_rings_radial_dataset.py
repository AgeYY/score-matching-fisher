from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def make_multi_rings(
    n_samples: int = 900,
    n_classes: int = 3,
    noise: float = 0.20,
    random_state: int | None = 0,
    *,
    radius_start: float = 1.0,
    radius_step: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Make nonlinear radial classes, one class per concentric ring."""
    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive.")
    if int(n_classes) < 2:
        raise ValueError("n_classes must be at least 2.")
    if float(radius_start) <= 0.0:
        raise ValueError("radius_start must be positive.")
    if float(radius_step) <= 0.0:
        raise ValueError("radius_step must be positive.")
    if float(noise) < 0.0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(random_state)
    n_samples = int(n_samples)
    n_classes = int(n_classes)
    counts = np.full(n_classes, n_samples // n_classes, dtype=np.int64)
    counts[: n_samples % n_classes] += 1

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for k, count in enumerate(counts):
        radius = float(radius_start) + float(k) * float(radius_step)
        theta = rng.uniform(0.0, 2.0 * np.pi, int(count))
        x_k = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        if noise > 0.0:
            x_k = x_k + float(noise) * rng.normal(size=x_k.shape)
        x_parts.append(np.asarray(x_k, dtype=np.float64))
        y_parts.append(np.full(int(count), k, dtype=np.int64))

    return np.vstack(x_parts), np.concatenate(y_parts)


def save_multi_rings_figure(
    x: np.ndarray,
    y: np.ndarray,
    *,
    out_base: Path,
    title: str = "Multi-class concentric rings",
) -> tuple[Path, Path]:
    """Save a scatter visualization for a multi-ring dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must have shape (n_samples, 2); got {x.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"y length {y.shape[0]} does not match x rows {x.shape[0]}.")

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg").resolve()
    png_path = out_base.with_suffix(".png").resolve()

    fig, ax = plt.subplots(figsize=(5.2, 5.2), layout="constrained")
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, s=15, cmap="tab10", alpha=0.82, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="class", loc="upper right", framealpha=0.9)
    ax.add_artist(legend)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def save_multi_rings_panel(*, out_base: Path) -> tuple[Path, Path]:
    """Save a 3/4/5-class comparison panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg").resolve()
    png_path = out_base.with_suffix(".png").resolve()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")
    for ax, n_classes in zip(axes, (3, 4, 5), strict=True):
        x, y = make_multi_rings(n_samples=900, n_classes=n_classes, random_state=n_classes)
        ax.scatter(x[:, 0], x[:, 1], c=y, s=12, cmap="tab10", alpha=0.82, linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(f"{n_classes} radial classes")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def test_make_multi_rings_shapes_and_balanced_labels() -> None:
    x, y = make_multi_rings(n_samples=902, n_classes=4, random_state=0)

    assert x.shape == (902, 2)
    assert y.shape == (902,)
    assert set(np.unique(y).tolist()) == {0, 1, 2, 3}
    assert np.bincount(y).tolist() == [226, 226, 225, 225]


def test_make_multi_rings_is_reproducible() -> None:
    x0, y0 = make_multi_rings(n_samples=120, n_classes=3, random_state=4)
    x1, y1 = make_multi_rings(n_samples=120, n_classes=3, random_state=4)
    x2, y2 = make_multi_rings(n_samples=120, n_classes=3, random_state=5)

    np.testing.assert_allclose(x0, x1)
    np.testing.assert_array_equal(y0, y1)
    np.testing.assert_array_equal(y0, y2)
    assert not np.allclose(x0, x2)


def test_multi_rings_are_radially_ordered() -> None:
    x, y = make_multi_rings(n_samples=900, n_classes=5, noise=0.04, random_state=0)
    radii = np.linalg.norm(x, axis=1)
    mean_radii = np.array([np.mean(radii[y == k]) for k in range(5)])

    assert np.all(np.diff(mean_radii) > 0.6)


def test_linear_vs_rbf_classifier_accuracy_gap() -> None:
    x, y = make_multi_rings(n_samples=900, n_classes=4, random_state=0)

    linear_clf = LogisticRegression(max_iter=1000, random_state=0)
    linear_clf.fit(x, y)
    linear_acc = accuracy_score(y, linear_clf.predict(x))

    rbf_clf = SVC(kernel="rbf", gamma="scale")
    rbf_clf.fit(x, y)
    rbf_acc = accuracy_score(y, rbf_clf.predict(x))

    assert linear_acc < 0.45
    assert rbf_acc > 0.93
    assert rbf_acc - linear_acc > 0.5


def test_save_multi_rings_figure_writes_svg_and_png(tmp_path: Path) -> None:
    x, y = make_multi_rings(n_samples=240, n_classes=4, random_state=0)

    svg_path, png_path = save_multi_rings_figure(x, y, out_base=tmp_path / "multi_rings_4class")

    assert svg_path == (tmp_path / "multi_rings_4class.svg").resolve()
    assert png_path == (tmp_path / "multi_rings_4class.png").resolve()
    assert svg_path.is_file() and svg_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0


def test_save_multi_rings_panel_writes_svg_and_png(tmp_path: Path) -> None:
    svg_path, png_path = save_multi_rings_panel(out_base=tmp_path / "multi_rings_panel")

    assert svg_path == (tmp_path / "multi_rings_panel.svg").resolve()
    assert png_path == (tmp_path / "multi_rings_panel.png").resolve()
    assert svg_path.is_file() and svg_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "tests" / "multi_rings_radial"
    x, y = make_multi_rings(n_samples=900, n_classes=4, random_state=0)
    svg_path, png_path = save_multi_rings_figure(x, y, out_base=out_dir / "multi_rings_4class")
    panel_svg, panel_png = save_multi_rings_panel(out_base=out_dir / "multi_rings_3_4_5_panel")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {panel_svg}")
    print(f"Saved: {panel_png}")


if __name__ == "__main__":
    main()
