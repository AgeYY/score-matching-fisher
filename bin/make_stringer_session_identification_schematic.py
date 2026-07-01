#!/usr/bin/env python3
"""Draw a paper-style schematic for Stringer session identification."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def default_output_dir() -> Path:
    return (
        _REPO_ROOT
        / "report"
        / "notes"
        / "figures"
        / "2026-06-29-stringer-session-identification"
    )


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fc: str,
    ec: str,
    fontsize: float = 9.5,
    weight: str = "normal",
    lw: float = 1.4,
    radius: float = 0.025,
    text_color: str = "#111827",
    shadow: bool = True,
) -> None:
    if shadow:
        ax.add_patch(
            patches.FancyBboxPatch(
                (xy[0] + 0.006, xy[1] - 0.006),
                width,
                height,
                boxstyle=f"round,pad=0.014,rounding_size={radius}",
                facecolor="#CBD5E1",
                edgecolor="none",
                alpha=0.28,
                zorder=1,
            )
        )
    ax.add_patch(
        patches.FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0.014,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        zorder=3,
    )


def add_step(ax, x: float, y: float, number: int, label: str, *, color: str = "#111827") -> None:
    ax.add_patch(patches.Circle((x, y), 0.018, facecolor=color, edgecolor="none", zorder=5))
    ax.text(x, y, str(number), ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=6)
    ax.text(x + 0.026, y, label, ha="left", va="center", fontsize=10.2, fontweight="bold", color=color, zorder=6)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#374151", lw: float = 1.7) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=4, shrinkB=4),
        zorder=4,
    )


def draw_small_curve(ax, x0: float, y0: float, w: float, h: float, *, color: str, phase: float) -> None:
    t = np.linspace(0, 1, 80)
    y = 0.46 + 0.24 * np.sin(2 * math.pi * (t + phase)) + 0.10 * np.sin(4 * math.pi * (t + 0.12))
    y = np.clip(y, 0.10, 0.90)
    ax.plot(x0 + w * t, y0 + h * y, color=color, lw=1.8, zorder=5)
    ax.plot([x0, x0], [y0, y0 + h], color="#9CA3AF", lw=0.7, zorder=4)
    ax.plot([x0, x0 + w], [y0, y0], color="#9CA3AF", lw=0.7, zorder=4)


def draw_session_split(ax) -> None:
    x, y, w, h = 0.045, 0.315, 0.185, 0.48
    add_box(ax, (x, y), w, h, "", fc="#F8FAFC", ec="#475569", lw=1.25)
    ax.text(x + w / 2, y + h - 0.055, "recording sessions", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(x + 0.060, y + h - 0.105, "A query", ha="center", va="center", fontsize=8.1, color="#065F46")
    ax.text(x + 0.127, y + h - 0.105, "B ref.", ha="center", va="center", fontsize=8.1, color="#1D4ED8")
    colors = ["#67A9CF", "#A6BDDB", "#7FC97F", "#BEAED4", "#FDB462", "#FDC086"]
    for i, color in enumerate(colors):
        yy = y + h - 0.145 - i * 0.048
        ax.text(x + 0.020, yy, f"S{i + 1}", ha="left", va="center", fontsize=8.8)
        ax.add_patch(patches.Rectangle((x + 0.050, yy - 0.015), 0.060, 0.030, facecolor=color, edgecolor="white", lw=0.8, zorder=4))
        ax.add_patch(patches.Rectangle((x + 0.112, yy - 0.015), 0.060, 0.030, facecolor=color, alpha=0.45, edgecolor="white", lw=0.8, zorder=4))
        ax.text(x + 0.080, yy, "A", ha="center", va="center", fontsize=8)
        ax.text(x + 0.142, yy, "B", ha="center", va="center", fontsize=8)


def draw_subset_and_reference(ax) -> None:
    add_box(ax, (0.300, 0.640), 0.155, 0.145, "", fc="#ECFDF5", ec="#047857", fontsize=10.2, weight="bold")
    ax.text(0.377, 0.747, "subsample", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#064E3B")
    ax.text(0.377, 0.720, "query A", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#064E3B")
    rng = np.random.default_rng(4)
    for _ in range(35):
        ax.plot(0.322 + 0.110 * rng.random(), 0.674 + 0.033 * rng.random(), "o", ms=2.0, color="#059669", alpha=0.70, zorder=5)
    ax.text(0.377, 0.615, r"$A_i^{(n)}$,  $n=200,\ldots,2000$", ha="center", va="center", fontsize=8.2, color="#065F46")
    ax.text(0.377, 0.590, "stratified, no replacement", ha="center", va="center", fontsize=7.8, color="#065F46")

    add_box(ax, (0.300, 0.360), 0.155, 0.145, "", fc="#EFF6FF", ec="#2563EB", fontsize=10.2, weight="bold")
    ax.text(0.377, 0.468, "fixed B", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E3A8A")
    ax.text(0.377, 0.440, "reference bank", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E3A8A")
    for k in range(6):
        yy = 0.385 + k * 0.012
        ax.plot([0.328, 0.425], [yy, yy], color="#2563EB", alpha=0.22 + 0.08 * (k % 2), lw=2.2, zorder=5)
    ax.text(0.377, 0.335, r"$\{B_j\}_{j=1}^6$ never changes", ha="center", va="center", fontsize=8.2, color="#1E3A8A")


def draw_estimation(ax) -> None:
    add_box(
        ax,
        (0.520, 0.515),
        0.180,
        0.220,
        "",
        fc="#FFF7ED",
        ec="#C2410C",
        fontsize=11.0,
        weight="bold",
    )
    ax.text(0.610, 0.675, "fit Fisher estimator", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#111827")
    ax.text(0.610, 0.642, "independently", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#111827")
    ax.text(0.610, 0.602, "classical linear", ha="center", va="center", fontsize=8.8, color="#7C2D12")
    ax.text(0.610, 0.575, "or", ha="center", va="center", fontsize=8.0, color="#7C2D12")
    ax.text(0.610, 0.548, r"periodic flow: $(\cos2\theta,\sin2\theta)$", ha="center", va="center", fontsize=8.5, color="#7C2D12")
    ax.text(0.610, 0.492, r"output: scalar curve $I(\theta)$", ha="center", va="center", fontsize=8.5, color="#374151")


def draw_curves(ax) -> None:
    add_box(ax, (0.735, 0.605), 0.120, 0.150, "", fc="#FFFFFF", ec="#94A3B8", fontsize=9.5, weight="bold")
    ax.text(0.795, 0.720, "query curve", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#111827")
    ax.text(0.795, 0.696, r"$I_{A_i^{(n)}}(\theta)$", ha="center", va="center", fontsize=8.0)
    draw_small_curve(ax, 0.758, 0.632, 0.075, 0.048, color="#111827", phase=0.02)

    add_box(ax, (0.735, 0.330), 0.120, 0.185, "", fc="#FFFFFF", ec="#94A3B8", fontsize=9.5, weight="bold")
    ax.text(0.795, 0.482, "reference curves", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#111827")
    ax.text(0.795, 0.458, r"$\{I_{B_j}(\theta)\}_{j=1}^6$", ha="center", va="center", fontsize=8.0)
    for k, phase in enumerate([0.02, 0.11, 0.23, 0.35]):
        color = "#2563EB" if k == 1 else "#94A3B8"
        draw_small_curve(ax, 0.758, 0.356 + 0.027 * k, 0.075, 0.019, color=color, phase=phase)


def draw_matching(ax) -> None:
    add_box(ax, (0.890, 0.430), 0.080, 0.315, "", fc="#FDF2F8", ec="#BE185D", lw=1.25)
    ax.text(0.930, 0.708, "match", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#831843")
    ax.text(0.930, 0.680, r"$d_{ij}=d(I_A,I_B)$", ha="center", va="center", fontsize=7.6, color="#831843")
    x0, y0, cell = 0.903, 0.505, 0.010
    vals = np.array(
        [
            [0.18, 0.82, 0.70, 0.62, 0.76, 0.81],
            [0.74, 0.20, 0.66, 0.58, 0.63, 0.71],
            [0.69, 0.73, 0.19, 0.75, 0.77, 0.60],
            [0.80, 0.67, 0.64, 0.22, 0.72, 0.68],
            [0.72, 0.62, 0.71, 0.65, 0.20, 0.69],
            [0.76, 0.70, 0.66, 0.74, 0.61, 0.19],
        ]
    )
    cmap = plt.get_cmap("YlGnBu")
    for r in range(6):
        for c in range(6):
            ax.add_patch(
                patches.Rectangle(
                    (x0 + c * cell, y0 + (5 - r) * cell),
                    cell * 0.9,
                    cell * 0.9,
                    facecolor=cmap(1.0 - vals[r, c]),
                    edgecolor="white",
                    lw=0.4,
                    zorder=5,
                )
            )
    ax.add_patch(patches.Rectangle((x0 + cell, y0 + 4 * cell), cell * 0.9, cell * 0.9, fill=False, edgecolor="#DC2626", lw=1.2, zorder=6))
    ax.text(0.930, 0.480, "distance matrix", ha="center", va="center", fontsize=7.4, color="#374151")
    ax.text(0.930, 0.462, r"rank paired $B_i$", ha="center", va="center", fontsize=7.4, color="#374151")

    add_box(ax, (0.880, 0.235), 0.100, 0.120, "score\nTop-1 / Top-3", fc="#FFFFFF", ec="#BE185D", fontsize=9.0, weight="bold", text_color="#831843")
    ax.text(0.930, 0.205, "repeat over A subsets", ha="center", va="center", fontsize=7.6, color="#831843")


def draw_no_oracle(ax) -> None:
    add_box(
        ax,
        (0.335, 0.115),
        0.355,
        0.080,
        "No ground-truth Fisher curve is used",
        fc="#F8FAFC",
        ec="#94A3B8",
        fontsize=9.8,
        weight="bold",
    )
    ax.text(0.512, 0.092, "session identity is used only after matching, to score accuracy", ha="center", va="center", fontsize=8.2, color="#475569")


def draw_schematic(output_dir: Path, stem: str) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(15.8, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.945, "Session identification without ground-truth Fisher information", ha="center", va="center", fontsize=18, fontweight="bold", color="#111827")
    ax.text(
        0.5,
        0.905,
        "A Fisher curve estimated from a small A subset should retrieve the paired B half from the same recording.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )

    ax.add_patch(patches.Rectangle((0.285, 0.300), 0.185, 0.505, facecolor="#ECFDF5", edgecolor="none", alpha=0.18, zorder=0))
    ax.add_patch(patches.Rectangle((0.505, 0.300), 0.215, 0.505, facecolor="#FFF7ED", edgecolor="none", alpha=0.23, zorder=0))
    ax.add_patch(patches.Rectangle((0.865, 0.195), 0.130, 0.610, facecolor="#FDF2F8", edgecolor="none", alpha=0.17, zorder=0))
    ax.text(0.292, 0.810, "QUERY", ha="left", va="center", fontsize=7.5, fontweight="bold", color="#047857")
    ax.text(0.292, 0.535, "REFERENCE", ha="left", va="center", fontsize=7.5, fontweight="bold", color="#2563EB")

    add_step(ax, 0.045, 0.835, 1, "split recordings")
    add_step(ax, 0.300, 0.835, 2, "query and reference")
    add_step(ax, 0.520, 0.835, 3, "fit Fisher curves")
    add_step(ax, 0.865, 0.835, 4, "match and score")

    draw_session_split(ax)
    draw_subset_and_reference(ax)
    draw_estimation(ax)
    draw_curves(ax)
    draw_matching(ax)
    draw_no_oracle(ax)

    arrow(ax, (0.230, 0.665), (0.300, 0.710), color="#047857")
    arrow(ax, (0.230, 0.460), (0.300, 0.430), color="#2563EB")
    arrow(ax, (0.455, 0.710), (0.520, 0.650), color="#047857")
    arrow(ax, (0.455, 0.430), (0.520, 0.585), color="#2563EB")
    arrow(ax, (0.700, 0.650), (0.735, 0.685), color="#111827")
    arrow(ax, (0.700, 0.545), (0.735, 0.430), color="#2563EB")
    arrow(ax, (0.855, 0.682), (0.890, 0.645), color="#111827")
    arrow(ax, (0.855, 0.425), (0.890, 0.560), color="#2563EB")
    arrow(ax, (0.930, 0.430), (0.930, 0.355), color="#BE185D")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--stem", default="session_identification_schematic")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in draw_schematic(Path(args.output_dir), str(args.stem)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
