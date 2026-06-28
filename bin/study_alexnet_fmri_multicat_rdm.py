#!/usr/bin/env python3
"""Compute AlexNet-EcoSet simulated fMRI RDMs for multiple image categories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE, ECOSET_VALIDATION_DIR
from fisher.alexnet_ecoset_catdog_decoding import EcosetValidationDirAction, ensure_sampled_images
from fisher.alexnet_fmri_simulation import AlexNetFMRISimulator, FMRISimulationConfig, LayerKey


DEFAULT_CLASSES: tuple[str, ...] = (
    "dog",
    "cat",
    "boat",
    "car",
    "airplane",
    "chair",
    "table",
    "man",
    "woman",
    "apple",
    "banana",
    "bird",
)
DEFAULT_LAYERS = "2,5,8,10,12,classifier.4"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RDM_METRICS: tuple[str, ...] = ("correlation", "euclidean")


def _abs_without_resolving_symlinks(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _default_ecoset_root() -> Path:
    return _abs_without_resolving_symlinks(DATA_DIR) / "ecoset"


def parse_classes(raw: str) -> tuple[str, ...]:
    classes = tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())
    if len(classes) < 2:
        raise ValueError("--classes must contain at least two comma-separated EcoSet class names.")
    duplicates = sorted({class_name for class_name in classes if classes.count(class_name) > 1})
    if duplicates:
        raise ValueError(f"--classes contains duplicate class names: {duplicates}.")
    return classes


def parse_layers(raw: str) -> tuple[LayerKey, ...]:
    layers: list[LayerKey] = []
    for tok in str(raw).split(","):
        layer = tok.strip()
        if not layer:
            continue
        try:
            layers.append(int(layer))
        except ValueError:
            layers.append(layer)
    if not layers:
        raise ValueError("--layers must contain at least one AlexNet-EcoSet fMRI layer id.")
    return tuple(layers)


def _count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def fmri_patterns_from_beta_hat(beta_hat: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta_hat, dtype=np.float64)
    if beta.ndim != 4:
        raise ValueError(f"beta_hat must have shape (subjects, runs, images, voxels); got {beta.shape}.")
    patterns = np.mean(beta, axis=(0, 1))
    if patterns.ndim != 2:
        raise ValueError(f"Averaged fMRI patterns must be 2D; got {patterns.shape}.")
    return patterns


def fmri_patterns_from_b_true(b_true: np.ndarray) -> np.ndarray:
    b = np.asarray(b_true, dtype=np.float64)
    if b.ndim != 3:
        raise ValueError(f"b_true must have shape (subjects, images, voxels); got {b.shape}.")
    patterns = np.mean(b, axis=0)
    if patterns.ndim != 2:
        raise ValueError(f"Averaged noise-free fMRI patterns must be 2D; got {patterns.shape}.")
    return patterns


def _validate_pattern_matrix(patterns: np.ndarray) -> np.ndarray:
    x = np.asarray(patterns, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"patterns must be 2D with shape (items, features); got {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError("At least two items are required to compute an RDM.")
    if not np.all(np.isfinite(x)):
        raise ValueError("patterns contains non-finite values.")
    return x


def compute_correlation_rdm(patterns: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = _validate_pattern_matrix(patterns)
    centered = x - np.mean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    zero_variance = np.flatnonzero(norms <= float(eps))
    if zero_variance.size:
        raise ValueError(f"Cannot compute correlation RDM for zero-variance item rows: {zero_variance.tolist()}.")

    z = centered / norms[:, None]
    corr = np.clip(z @ z.T, -1.0, 1.0)
    rdm = 1.0 - corr
    rdm = 0.5 * (rdm + rdm.T)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def compute_euclidean_rdm(patterns: np.ndarray) -> np.ndarray:
    x = _validate_pattern_matrix(patterns)
    sq_norms = np.sum(x * x, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    rdm = np.sqrt(np.maximum(sq_dist, 0.0))
    rdm = 0.5 * (rdm + rdm.T)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def normalize_rdm_metric(metric: str) -> str:
    value = str(metric).strip().lower()
    if value not in RDM_METRICS:
        raise ValueError(f"rdm_metric must be one of {RDM_METRICS}; got {metric!r}.")
    return value


def compute_rdm(patterns: np.ndarray, metric: str) -> np.ndarray:
    metric_name = normalize_rdm_metric(metric)
    if metric_name == "correlation":
        return compute_correlation_rdm(patterns)
    if metric_name == "euclidean":
        return compute_euclidean_rdm(patterns)
    raise AssertionError(f"Unhandled RDM metric: {metric_name}")


def category_mean_patterns(patterns: np.ndarray, labels: np.ndarray, n_categories: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(patterns, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"patterns must be 2D with shape (items, features); got {x.shape}.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("labels length must match number of pattern rows.")
    if int(n_categories) < 2:
        raise ValueError("n_categories must be at least 2.")

    means = np.empty((int(n_categories), x.shape[1]), dtype=np.float64)
    counts = np.empty(int(n_categories), dtype=np.int64)
    for label in range(int(n_categories)):
        mask = y == label
        count = int(np.count_nonzero(mask))
        if count < 1:
            raise ValueError(f"Category label {label} has no examples.")
        means[label] = np.mean(x[mask], axis=0)
        counts[label] = count
    return means, counts


def _class_centers_and_boundaries(labels: np.ndarray, n_categories: int) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    centers: list[float] = []
    boundaries: list[int] = []
    start = 0
    for label in range(int(n_categories)):
        count = int(np.count_nonzero(labels_arr == label))
        if count < 1:
            raise ValueError(f"Category label {label} has no examples.")
        centers.append(start + (count - 1) / 2.0)
        start += count
        boundaries.append(start)
    return np.asarray(centers, dtype=np.float64), np.asarray(boundaries[:-1], dtype=np.int64)


def _rdm_metric_title_name(rdm_metric: str) -> str:
    metric = normalize_rdm_metric(rdm_metric)
    return "correlation" if metric == "correlation" else "Euclidean"


def _rdm_colorbar_label(rdm_metric: str) -> str:
    metric = normalize_rdm_metric(rdm_metric)
    return "correlation distance (1 - r)" if metric == "correlation" else "Euclidean distance"


def _rdm_plot_vmax(rdms: Sequence[np.ndarray], rdm_metric: str) -> float:
    metric = normalize_rdm_metric(rdm_metric)
    if metric == "correlation":
        return 2.0
    vmax = 0.0
    for rdm in rdms:
        arr = np.asarray(rdm, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("RDM contains non-finite values.")
        if arr.size:
            vmax = max(vmax, float(np.max(arr)))
    return vmax if vmax > 0.0 else 1.0


def _save_rdm_panel(
    rdms: Sequence[np.ndarray],
    layer_names: Sequence[str],
    output_dir: Path,
    *,
    stem: str,
    class_names: Sequence[str],
    labels: np.ndarray | None,
    title: str,
    rdm_metric: str,
) -> tuple[Path, Path]:
    if len(rdms) != len(layer_names):
        raise ValueError("rdms and layer_names must have the same length.")
    if not rdms:
        raise ValueError("At least one RDM is required.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(rdms)
    ncols = min(3, n_layers)
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 4.05 * nrows),
        layout="constrained",
        squeeze=False,
    )
    axes_flat = list(axes.ravel())
    images = []

    if labels is None:
        tick_positions = np.arange(len(class_names), dtype=np.float64)
        tick_labels = list(class_names)
        boundaries = np.asarray([], dtype=np.int64)
        tick_fontsize = 7
    else:
        tick_positions, boundaries = _class_centers_and_boundaries(labels, len(class_names))
        tick_labels = list(class_names)
        tick_fontsize = 5

    vmax = _rdm_plot_vmax(rdms, rdm_metric)
    for ax, rdm, layer_name in zip(axes_flat[:n_layers], rdms, layer_names, strict=True):
        arr = np.asarray(rdm, dtype=np.float64)
        image = ax.imshow(arr, vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
        images.append(image)
        ax.set_title(str(layer_name), fontsize=10)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
        ax.tick_params(length=0)
        for boundary in boundaries:
            ax.axhline(float(boundary) - 0.5, color="white", linewidth=0.45, alpha=0.75)
            ax.axvline(float(boundary) - 0.5, color="white", linewidth=0.45, alpha=0.75)

    for ax in axes_flat[n_layers:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.colorbar(images[0], ax=axes_flat[:n_layers], shrink=0.82, label=_rdm_colorbar_label(rdm_metric))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _save_rdm_comparison_panel(
    estimated_rdms: Sequence[np.ndarray],
    noise_free_rdms: Sequence[np.ndarray],
    layer_names: Sequence[str],
    output_dir: Path,
    *,
    stem: str,
    class_names: Sequence[str],
    labels: np.ndarray | None,
    title: str,
    rdm_metric: str,
) -> tuple[Path, Path]:
    if len(estimated_rdms) != len(noise_free_rdms) or len(estimated_rdms) != len(layer_names):
        raise ValueError("estimated_rdms, noise_free_rdms, and layer_names must have the same length.")
    if not estimated_rdms:
        raise ValueError("At least one RDM is required.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(estimated_rdms)
    layer_cols = min(3, n_layers)
    nrows = int(np.ceil(n_layers / layer_cols))
    ncols = 2 * layer_cols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.75 * nrows),
        layout="constrained",
        squeeze=False,
    )
    images = []
    axes_with_images = []

    if labels is None:
        tick_positions = np.arange(len(class_names), dtype=np.float64)
        tick_labels = list(class_names)
        boundaries = np.asarray([], dtype=np.int64)
        tick_fontsize = 7
    else:
        tick_positions, boundaries = _class_centers_and_boundaries(labels, len(class_names))
        tick_labels = list(class_names)
        tick_fontsize = 5

    vmax = _rdm_plot_vmax([*estimated_rdms, *noise_free_rdms], rdm_metric)
    for layer_idx, (estimated, noise_free, layer_name) in enumerate(
        zip(estimated_rdms, noise_free_rdms, layer_names, strict=True)
    ):
        row = layer_idx // layer_cols
        col = 2 * (layer_idx % layer_cols)
        for ax, rdm, suffix in (
            (axes[row, col], estimated, "estimated"),
            (axes[row, col + 1], noise_free, "noise-free"),
        ):
            arr = np.asarray(rdm, dtype=np.float64)
            image = ax.imshow(arr, vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
            images.append(image)
            axes_with_images.append(ax)
            ax.set_title(f"{layer_name}\n{suffix}", fontsize=9)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=tick_fontsize)
            ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
            ax.tick_params(length=0)
            for boundary in boundaries:
                ax.axhline(float(boundary) - 0.5, color="white", linewidth=0.45, alpha=0.75)
                ax.axvline(float(boundary) - 0.5, color="white", linewidth=0.45, alpha=0.75)

    for layer_idx in range(n_layers, nrows * layer_cols):
        row = layer_idx // layer_cols
        col = 2 * (layer_idx % layer_cols)
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.colorbar(images[0], ax=axes_with_images, shrink=0.82, label=_rdm_colorbar_label(rdm_metric))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def save_rdm_figures(
    *,
    image_rdms: Sequence[np.ndarray],
    category_rdms: Sequence[np.ndarray],
    noise_free_image_rdms: Sequence[np.ndarray] | None = None,
    noise_free_category_rdms: Sequence[np.ndarray] | None = None,
    layer_names: Sequence[str],
    labels: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    rdm_metric: str = "correlation",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    metric_name = normalize_rdm_metric(rdm_metric)
    metric_title = _rdm_metric_title_name(metric_name)
    image_png, image_svg = _save_rdm_panel(
        image_rdms,
        layer_names,
        out_dir,
        stem="alexnet_fmri_image_rdms",
        class_names=class_names,
        labels=labels,
        title=f"Image-level fMRI {metric_title} RDMs",
        rdm_metric=metric_name,
    )
    category_png, category_svg = _save_rdm_panel(
        category_rdms,
        layer_names,
        out_dir,
        stem="alexnet_fmri_category_rdms",
        class_names=class_names,
        labels=None,
        title=f"Category-mean fMRI {metric_title} RDMs",
        rdm_metric=metric_name,
    )
    paths = {
        "image_png": image_png,
        "image_svg": image_svg,
        "category_png": category_png,
        "category_svg": category_svg,
    }
    if noise_free_image_rdms is not None:
        image_comp_png, image_comp_svg = _save_rdm_comparison_panel(
            image_rdms,
            noise_free_image_rdms,
            layer_names,
            out_dir,
            stem="alexnet_fmri_image_rdms_estimated_vs_noise_free",
            class_names=class_names,
            labels=labels,
            title=f"Image-level fMRI {metric_title} RDMs: estimated vs noise-free",
            rdm_metric=metric_name,
        )
        paths["image_comparison_png"] = image_comp_png
        paths["image_comparison_svg"] = image_comp_svg
    if noise_free_category_rdms is not None:
        category_comp_png, category_comp_svg = _save_rdm_comparison_panel(
            category_rdms,
            noise_free_category_rdms,
            layer_names,
            out_dir,
            stem="alexnet_fmri_category_rdms_estimated_vs_noise_free",
            class_names=class_names,
            labels=None,
            title=f"Category-mean fMRI {metric_title} RDMs: estimated vs noise-free",
            rdm_metric=metric_name,
        )
        paths["category_comparison_png"] = category_comp_png
        paths["category_comparison_svg"] = category_comp_svg
    return paths


def _save_npz(
    *,
    output_dir: Path,
    layer_ids: Sequence[LayerKey],
    layer_names: Sequence[str],
    class_names: Sequence[str],
    labels: np.ndarray,
    image_paths: Sequence[Path],
    b_true: Sequence[np.ndarray],
    fmri_patterns: Sequence[np.ndarray],
    category_patterns: Sequence[np.ndarray],
    image_rdms: Sequence[np.ndarray],
    category_rdms: Sequence[np.ndarray],
    noise_free_fmri_patterns: Sequence[np.ndarray],
    noise_free_category_patterns: Sequence[np.ndarray],
    noise_free_image_rdms: Sequence[np.ndarray],
    noise_free_category_rdms: Sequence[np.ndarray],
    category_counts: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "alexnet_fmri_rdm_results.npz"
    np.savez_compressed(
        npz_path,
        layer_ids=np.asarray([str(layer_id) for layer_id in layer_ids], dtype=object),
        layer_names=np.asarray(layer_names, dtype=object),
        classes=np.asarray(class_names, dtype=object),
        labels=np.asarray(labels, dtype=np.int64),
        image_paths=np.asarray([str(path) for path in image_paths], dtype=object),
        b_true=np.stack([np.asarray(x, dtype=np.float64) for x in b_true], axis=0),
        fmri_patterns=np.stack([np.asarray(x, dtype=np.float64) for x in fmri_patterns], axis=0),
        category_patterns=np.stack([np.asarray(x, dtype=np.float64) for x in category_patterns], axis=0),
        image_rdms=np.stack([np.asarray(x, dtype=np.float64) for x in image_rdms], axis=0),
        category_rdms=np.stack([np.asarray(x, dtype=np.float64) for x in category_rdms], axis=0),
        noise_free_fmri_patterns=np.stack(
            [np.asarray(x, dtype=np.float64) for x in noise_free_fmri_patterns], axis=0
        ),
        noise_free_category_patterns=np.stack(
            [np.asarray(x, dtype=np.float64) for x in noise_free_category_patterns], axis=0
        ),
        noise_free_image_rdms=np.stack([np.asarray(x, dtype=np.float64) for x in noise_free_image_rdms], axis=0),
        noise_free_category_rdms=np.stack([np.asarray(x, dtype=np.float64) for x in noise_free_category_rdms], axis=0),
        rdm_metric=np.asarray([normalize_rdm_metric(getattr(args, "rdm_metric", "correlation"))], dtype=np.str_),
        category_counts=np.asarray(category_counts, dtype=np.int64),
        n_per_class=np.asarray([int(args.n_per_class)], dtype=np.int64),
        n_subjects=np.asarray([int(args.n_subjects)], dtype=np.int64),
        n_voxels=np.asarray([int(args.n_voxels)], dtype=np.int64),
        n_runs=np.asarray([int(args.n_runs)], dtype=np.int64),
        noise_lambda=np.asarray([float(args.noise_lambda)], dtype=np.float64),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
    )
    return npz_path


def _save_summary(
    *,
    output_dir: Path,
    figure_paths: dict[str, Path],
    npz_path: Path,
    class_names: Sequence[str],
    layer_names: Sequence[str],
    labels: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.txt"
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("AlexNet-EcoSet simulated fMRI multi-category RDM run\n")
        f.write(f"classes: {','.join(class_names)}\n")
        f.write(f"class_counts: {dict(zip(class_names, counts.tolist(), strict=True))}\n")
        f.write(f"n_images: {int(labels.shape[0])}\n")
        f.write(f"layers: {','.join(layer_names)}\n")
        f.write(f"n_subjects: {int(args.n_subjects)}\n")
        f.write(f"n_voxels: {int(args.n_voxels)}\n")
        f.write(f"n_runs: {int(args.n_runs)}\n")
        f.write(f"noise_lambda: {float(args.noise_lambda)}\n")
        f.write(f"rdm_metric: {normalize_rdm_metric(getattr(args, 'rdm_metric', 'correlation'))}\n")
        f.write(f"seed: {int(args.seed)}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"image_root: {_abs_without_resolving_symlinks(args.image_root)}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"results_npz: {npz_path}\n")
        for name, path in figure_paths.items():
            f.write(f"{name}: {path}\n")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    ecoset_root = _default_ecoset_root()
    validation_dir = _abs_without_resolving_symlinks(ECOSET_VALIDATION_DIR)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.set_defaults(hf_cache_dir=str(validation_dir))
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated EcoSet validation class names.")
    p.add_argument("--n-per-class", type=int, default=50)
    p.add_argument(
        "--image-root",
        default=str(ecoset_root / "validation_12cat_50"),
        help="Folder where sampled EcoSet validation images are exported or reused.",
    )
    p.add_argument(
        "--ecoset-validation-dir",
        "--hf-cache-dir",
        dest="ecoset_validation_dir",
        action=EcosetValidationDirAction,
        default=str(validation_dir),
        help="Local EcoSet validation Arrow file, build directory, or parent cache directory; no downloads are attempted.",
    )
    p.add_argument("--output-dir", default=str(ecoset_root / "alexnet_fmri_multicat_rdm"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--layers", default=DEFAULT_LAYERS, help="Comma-separated AlexNet-EcoSet layer ids or module names.")
    p.add_argument("--n-subjects", type=int, default=1)
    p.add_argument("--n-voxels", type=int, default=100)
    p.add_argument("--n-runs", type=int, default=4)
    p.add_argument("--noise-lambda", type=float, default=0.3)
    p.add_argument("--rdm-metric", choices=RDM_METRICS, default="correlation")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--force-export", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, Path]:
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise ValueError("This project analysis requires a CUDA --device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AGENTS.md requires a CUDA device for project runs.")

    class_names = parse_classes(args.classes)
    layer_ids = parse_layers(args.layers)
    rdm_metric = normalize_rdm_metric(args.rdm_metric)
    image_root = _abs_without_resolving_symlinks(args.image_root)
    output_dir = _abs_without_resolving_symlinks(args.output_dir)
    validation_dir = getattr(args, "ecoset_validation_dir", getattr(args, "hf_cache_dir", None))

    samples = ensure_sampled_images(
        image_root=image_root,
        cache_dir=None if validation_dir is None else _abs_without_resolving_symlinks(validation_dir),
        classes=class_names,
        n_per_class=int(args.n_per_class),
        seed=int(args.seed),
        force_export=bool(args.force_export),
    )
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    image_paths = [sample.path for sample in samples]
    expected_counts = np.full(len(class_names), int(args.n_per_class), dtype=np.int64)
    counts = np.bincount(labels, minlength=len(class_names))
    if not np.array_equal(counts, expected_counts):
        raise RuntimeError(
            f"Expected {int(args.n_per_class)} images per class under {image_root}; got {dict(zip(class_names, counts.tolist(), strict=True))}."
        )
    if not all(_count_images(image_root / class_name) >= int(args.n_per_class) for class_name in class_names):
        raise RuntimeError(f"EcoSet export did not produce {int(args.n_per_class)} images per class under {image_root}.")

    cfg = FMRISimulationConfig(
        candidate_layer_ids=layer_ids,
        n_subjects=int(args.n_subjects),
        n_voxels=int(args.n_voxels),
        n_runs=int(args.n_runs),
        noise_lambda=float(args.noise_lambda),
        seed=int(args.seed),
        device=str(args.device),
    )
    sim = AlexNetFMRISimulator(cfg)
    activations = sim.extract_activations(image_paths, batch_size=int(args.batch_size))

    layer_names: list[str] = []
    b_true_by_layer: list[np.ndarray] = []
    fmri_patterns: list[np.ndarray] = []
    category_patterns_by_layer: list[np.ndarray] = []
    image_rdms: list[np.ndarray] = []
    category_rdms: list[np.ndarray] = []
    noise_free_fmri_patterns: list[np.ndarray] = []
    noise_free_category_patterns_by_layer: list[np.ndarray] = []
    noise_free_image_rdms: list[np.ndarray] = []
    noise_free_category_rdms: list[np.ndarray] = []
    category_counts: np.ndarray | None = None

    for layer_id in layer_ids:
        layer_result = sim.simulate_layer(activations[layer_id], labels, layer_id=layer_id)
        patterns = fmri_patterns_from_beta_hat(layer_result.beta_hat)
        cat_patterns, cat_counts = category_mean_patterns(patterns, labels, len(class_names))
        true_patterns = fmri_patterns_from_b_true(layer_result.b_true)
        true_cat_patterns, true_cat_counts = category_mean_patterns(true_patterns, labels, len(class_names))
        if not np.array_equal(cat_counts, true_cat_counts):
            raise RuntimeError("Noise-free category counts differ from estimated category counts.")
        image_rdms.append(compute_rdm(patterns, rdm_metric))
        category_rdms.append(compute_rdm(cat_patterns, rdm_metric))
        noise_free_image_rdms.append(compute_rdm(true_patterns, rdm_metric))
        noise_free_category_rdms.append(compute_rdm(true_cat_patterns, rdm_metric))
        b_true_by_layer.append(layer_result.b_true)
        fmri_patterns.append(patterns)
        category_patterns_by_layer.append(cat_patterns)
        noise_free_fmri_patterns.append(true_patterns)
        noise_free_category_patterns_by_layer.append(true_cat_patterns)
        layer_names.append(layer_result.layer_name)
        if category_counts is None:
            category_counts = cat_counts
        elif not np.array_equal(category_counts, cat_counts):
            raise RuntimeError("Category counts changed across layers.")

    assert category_counts is not None
    figure_paths = save_rdm_figures(
        image_rdms=image_rdms,
        category_rdms=category_rdms,
        noise_free_image_rdms=noise_free_image_rdms,
        noise_free_category_rdms=noise_free_category_rdms,
        layer_names=layer_names,
        labels=labels,
        class_names=class_names,
        output_dir=output_dir,
        rdm_metric=rdm_metric,
    )
    npz_path = _save_npz(
        output_dir=output_dir,
        layer_ids=layer_ids,
        layer_names=layer_names,
        class_names=class_names,
        labels=labels,
        image_paths=image_paths,
        b_true=b_true_by_layer,
        fmri_patterns=fmri_patterns,
        category_patterns=category_patterns_by_layer,
        image_rdms=image_rdms,
        category_rdms=category_rdms,
        noise_free_fmri_patterns=noise_free_fmri_patterns,
        noise_free_category_patterns=noise_free_category_patterns_by_layer,
        noise_free_image_rdms=noise_free_image_rdms,
        noise_free_category_rdms=noise_free_category_rdms,
        category_counts=category_counts,
        args=args,
    )
    summary_path = _save_summary(
        output_dir=output_dir,
        figure_paths=figure_paths,
        npz_path=npz_path,
        class_names=class_names,
        layer_names=layer_names,
        labels=labels,
        args=args,
    )
    return {"results_npz": npz_path, "summary": summary_path, **figure_paths}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = run(args)
    print("[alexnet-fmri-rdm] Saved:", flush=True)
    for path in paths.values():
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
