#!/usr/bin/env python3
"""Compute per-subject noise-free AlexNet-EcoSet simulated fMRI RDMs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from global_setting import ECOSET_VALIDATION_DIR
from fisher.alexnet_ecoset_catdog_decoding import EcosetValidationDirAction, ensure_sampled_images
from fisher.alexnet_fmri_simulation import AlexNetFMRISimulator, FMRISimulationConfig, LayerKey
from study_alexnet_fmri_multicat_rdm import (
    DEFAULT_CLASSES,
    DEFAULT_LAYERS,
    RDM_METRICS,
    _abs_without_resolving_symlinks,
    _class_centers_and_boundaries,
    _count_images,
    _default_ecoset_root,
    _rdm_colorbar_label,
    _rdm_metric_title_name,
    _rdm_plot_vmax,
    category_mean_patterns,
    compute_rdm,
    normalize_rdm_metric,
    parse_classes,
    parse_layers,
)


class SubjectRDMResult(NamedTuple):
    image_patterns: np.ndarray
    category_patterns: np.ndarray
    image_rdms: np.ndarray
    category_rdms: np.ndarray
    category_counts: np.ndarray


def _subject_ids(n_subjects: int) -> tuple[str, ...]:
    if int(n_subjects) < 1:
        raise ValueError("n_subjects must be positive.")
    return tuple(f"S{subj + 1:02d}" for subj in range(int(n_subjects)))


def compute_subject_noise_free_rdms(
    b_true: np.ndarray,
    labels: np.ndarray,
    n_categories: int,
    rdm_metric: str = "correlation",
) -> SubjectRDMResult:
    """Compute subject-specific image and category RDMs from layer_result.b_true."""
    b = np.asarray(b_true, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    metric_name = normalize_rdm_metric(rdm_metric)
    if b.ndim != 3:
        raise ValueError(f"b_true must have shape (subjects, images, voxels); got {b.shape}.")
    if y.shape[0] != b.shape[1]:
        raise ValueError("labels length must match the number of image patterns in b_true.")
    if int(n_categories) < 2:
        raise ValueError("n_categories must be at least 2.")
    if not np.all(np.isfinite(b)):
        raise ValueError("b_true contains non-finite values.")

    n_subjects, n_images, n_voxels = b.shape
    category_patterns = np.empty((n_subjects, int(n_categories), n_voxels), dtype=np.float64)
    category_counts = np.empty((n_subjects, int(n_categories)), dtype=np.int64)
    image_rdms = np.empty((n_subjects, n_images, n_images), dtype=np.float64)
    category_rdms = np.empty((n_subjects, int(n_categories), int(n_categories)), dtype=np.float64)

    for subj in range(n_subjects):
        subj_patterns = b[subj]
        subj_category_patterns, subj_counts = category_mean_patterns(subj_patterns, y, int(n_categories))
        category_patterns[subj] = subj_category_patterns
        category_counts[subj] = subj_counts
        image_rdms[subj] = compute_rdm(subj_patterns, metric_name)
        category_rdms[subj] = compute_rdm(subj_category_patterns, metric_name)

    return SubjectRDMResult(
        image_patterns=b,
        category_patterns=category_patterns,
        image_rdms=image_rdms,
        category_rdms=category_rdms,
        category_counts=category_counts,
    )


def _validate_subject_layer_rdms(
    rdms: np.ndarray,
    *,
    name: str,
    n_subjects: int | None = None,
    n_layers: int | None = None,
) -> np.ndarray:
    arr = np.asarray(rdms, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"{name} must have shape (subjects, layers, items, items); got {arr.shape}.")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one subject and one layer.")
    if arr.shape[2] != arr.shape[3]:
        raise ValueError(f"{name} must contain square RDMs; got trailing shape {arr.shape[2:]}.")
    if n_subjects is not None and arr.shape[0] != int(n_subjects):
        raise ValueError(f"{name} has {arr.shape[0]} subjects but expected {int(n_subjects)}.")
    if n_layers is not None and arr.shape[1] != int(n_layers):
        raise ValueError(f"{name} has {arr.shape[1]} layers but expected {int(n_layers)}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _save_subject_rdm_grid(
    rdms: np.ndarray,
    layer_names: Sequence[str],
    subject_ids: Sequence[str],
    output_dir: Path,
    *,
    stem: str,
    class_names: Sequence[str],
    labels: np.ndarray | None,
    title: str,
    rdm_metric: str,
) -> tuple[Path, Path]:
    metric_name = normalize_rdm_metric(rdm_metric)
    arr = _validate_subject_layer_rdms(
        rdms,
        name=stem,
        n_subjects=len(subject_ids),
        n_layers=len(layer_names),
    )
    if labels is None:
        if arr.shape[2] != len(class_names):
            raise ValueError(
                f"{stem} category RDM size {arr.shape[2]} does not match {len(class_names)} class names."
            )
        tick_positions = np.arange(len(class_names), dtype=np.float64)
        tick_labels = list(class_names)
        boundaries = np.asarray([], dtype=np.int64)
        tick_fontsize = 5
        row_label_x = -0.62
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape[0] != arr.shape[2]:
            raise ValueError(f"{stem} image RDM size {arr.shape[2]} does not match labels length {labels_arr.shape[0]}.")
        tick_positions, boundaries = _class_centers_and_boundaries(labels_arr, len(class_names))
        tick_labels = list(class_names)
        tick_fontsize = 4
        row_label_x = -0.72

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_subjects, n_layers = arr.shape[:2]
    fig_width = max(6.8, 1.55 * n_layers + 1.6)
    fig_height = max(4.2, 1.42 * n_subjects + 1.4)
    fig, axes = plt.subplots(
        n_subjects,
        n_layers,
        figsize=(fig_width, fig_height),
        layout="constrained",
        squeeze=False,
    )
    vmax = _rdm_plot_vmax([arr], metric_name)
    images = []

    for subj in range(n_subjects):
        for layer_idx, layer_name in enumerate(layer_names):
            ax = axes[subj, layer_idx]
            image = ax.imshow(arr[subj, layer_idx], vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
            images.append(image)
            if subj == 0:
                ax.set_title(str(layer_name), fontsize=9, pad=4)
            if layer_idx == 0:
                ax.text(
                    row_label_x,
                    0.5,
                    str(subject_ids[subj]),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            if subj == n_subjects - 1:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=tick_fontsize)
            else:
                ax.set_xticks([])
            if layer_idx == 0:
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
            else:
                ax.set_yticks([])
            ax.tick_params(length=0)
            for boundary in boundaries:
                ax.axhline(float(boundary) - 0.5, color="white", linewidth=0.35, alpha=0.75)
                ax.axvline(float(boundary) - 0.5, color="white", linewidth=0.35, alpha=0.75)

    fig.suptitle(title, fontsize=12)
    fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.78, label=_rdm_colorbar_label(metric_name))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def save_subject_rdm_figures(
    *,
    noise_free_image_rdms: np.ndarray,
    noise_free_category_rdms: np.ndarray,
    layer_names: Sequence[str],
    class_names: Sequence[str],
    labels: np.ndarray,
    output_dir: str | Path,
    rdm_metric: str = "correlation",
    subject_ids: Sequence[str] | None = None,
) -> dict[str, Path]:
    image_arr = _validate_subject_layer_rdms(noise_free_image_rdms, name="noise_free_image_rdms")
    category_arr = _validate_subject_layer_rdms(
        noise_free_category_rdms,
        name="noise_free_category_rdms",
        n_subjects=image_arr.shape[0],
        n_layers=image_arr.shape[1],
    )
    if len(layer_names) != image_arr.shape[1]:
        raise ValueError("layer_names length must match the layer dimension of the RDM arrays.")
    if subject_ids is None:
        subject_ids = _subject_ids(image_arr.shape[0])
    if len(subject_ids) != image_arr.shape[0]:
        raise ValueError("subject_ids length must match the subject dimension of the RDM arrays.")

    out_dir = Path(output_dir)
    metric_name = normalize_rdm_metric(rdm_metric)
    metric_title = _rdm_metric_title_name(metric_name)
    category_png, category_svg = _save_subject_rdm_grid(
        category_arr,
        layer_names,
        subject_ids,
        out_dir,
        stem="alexnet_fmri_subject_category_rdms",
        class_names=class_names,
        labels=None,
        title=f"Subject category-mean noise-free fMRI {metric_title} RDMs",
        rdm_metric=metric_name,
    )
    image_png, image_svg = _save_subject_rdm_grid(
        image_arr,
        layer_names,
        subject_ids,
        out_dir,
        stem="alexnet_fmri_subject_image_rdms",
        class_names=class_names,
        labels=labels,
        title=f"Subject image-level noise-free fMRI {metric_title} RDMs",
        rdm_metric=metric_name,
    )
    return {
        "category_png": category_png,
        "category_svg": category_svg,
        "image_png": image_png,
        "image_svg": image_svg,
    }


def _save_npz(
    *,
    output_dir: Path,
    layer_ids: Sequence[LayerKey],
    layer_names: Sequence[str],
    class_names: Sequence[str],
    labels: np.ndarray,
    image_paths: Sequence[Path],
    subject_ids: Sequence[str],
    noise_free_image_patterns: np.ndarray,
    noise_free_category_patterns: np.ndarray,
    noise_free_image_rdms: np.ndarray,
    noise_free_category_rdms: np.ndarray,
    category_counts: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    image_patterns = np.asarray(noise_free_image_patterns, dtype=np.float64)
    category_patterns = np.asarray(noise_free_category_patterns, dtype=np.float64)
    image_rdms = _validate_subject_layer_rdms(noise_free_image_rdms, name="noise_free_image_rdms")
    category_rdms = _validate_subject_layer_rdms(
        noise_free_category_rdms,
        name="noise_free_category_rdms",
        n_subjects=image_rdms.shape[0],
        n_layers=image_rdms.shape[1],
    )
    if image_patterns.ndim != 4:
        raise ValueError(
            f"noise_free_image_patterns must have shape (subjects, layers, images, voxels); got {image_patterns.shape}."
        )
    if category_patterns.ndim != 4:
        raise ValueError(
            "noise_free_category_patterns must have shape "
            f"(subjects, layers, classes, voxels); got {category_patterns.shape}."
        )
    if image_patterns.shape[:3] != image_rdms.shape[:3]:
        raise ValueError("noise_free_image_patterns shape is inconsistent with noise_free_image_rdms.")
    if category_patterns.shape[:3] != category_rdms.shape[:3]:
        raise ValueError("noise_free_category_patterns shape is inconsistent with noise_free_category_rdms.")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "alexnet_fmri_subject_rdm_results.npz"
    np.savez_compressed(
        npz_path,
        noise_free_image_rdms=image_rdms,
        noise_free_category_rdms=category_rdms,
        noise_free_image_patterns=image_patterns,
        noise_free_category_patterns=category_patterns,
        layer_ids=np.asarray([str(layer_id) for layer_id in layer_ids], dtype=np.str_),
        layer_names=np.asarray(layer_names, dtype=np.str_),
        classes=np.asarray(class_names, dtype=np.str_),
        labels=np.asarray(labels, dtype=np.int64),
        image_paths=np.asarray([str(path) for path in image_paths], dtype=np.str_),
        rdm_metric=np.asarray([normalize_rdm_metric(getattr(args, "rdm_metric", "correlation"))], dtype=np.str_),
        subject_ids=np.asarray(subject_ids, dtype=np.str_),
        category_counts=np.asarray(category_counts, dtype=np.int64),
        n_subjects=np.asarray([int(args.n_subjects)], dtype=np.int64),
        n_layers=np.asarray([len(layer_names)], dtype=np.int64),
        n_images=np.asarray([len(labels)], dtype=np.int64),
        n_classes=np.asarray([len(class_names)], dtype=np.int64),
        n_per_class=np.asarray([int(args.n_per_class)], dtype=np.int64),
        n_voxels=np.asarray([int(args.n_voxels)], dtype=np.int64),
        n_runs=np.asarray([int(args.n_runs)], dtype=np.int64),
        noise_lambda=np.asarray([float(args.noise_lambda)], dtype=np.float64),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
        device=np.asarray([str(args.device)], dtype=np.str_),
        noise_free_source=np.asarray(["layer_result.b_true[subj]"], dtype=np.str_),
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
    subject_ids: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.txt"
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("AlexNet-EcoSet simulated fMRI per-subject noise-free RDM run\n")
        f.write(f"classes: {','.join(class_names)}\n")
        f.write(f"class_counts: {dict(zip(class_names, counts.tolist(), strict=True))}\n")
        f.write(f"n_images: {int(labels.shape[0])}\n")
        f.write(f"layers: {','.join(layer_names)}\n")
        f.write(f"n_subjects: {len(subject_ids)}\n")
        f.write(f"subject_ids: {','.join(subject_ids)}\n")
        f.write(f"n_voxels: {int(args.n_voxels)}\n")
        f.write(f"n_runs: {int(args.n_runs)}\n")
        f.write(f"noise_lambda: {float(args.noise_lambda)}\n")
        f.write(f"rdm_metric: {normalize_rdm_metric(getattr(args, 'rdm_metric', 'correlation'))}\n")
        f.write(f"seed: {int(args.seed)}\n")
        f.write(f"device: {args.device}\n")
        f.write("rdm_source: layer_result.b_true[subj]\n")
        f.write(
            "noise_free_note: saved RDMs use b_true, so n_runs and noise_lambda are simulator "
            "compatibility settings and do not affect the requested noise-free RDM values.\n"
        )
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
    p.add_argument("--output-dir", default=str(ecoset_root / "alexnet_fmri_multicat_subject_rdms"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--layers", default=DEFAULT_LAYERS, help="Comma-separated AlexNet-EcoSet layer ids or module names.")
    p.add_argument("--n-subjects", type=int, default=10)
    p.add_argument("--n-voxels", type=int, default=100)
    p.add_argument("--n-runs", type=int, default=4)
    p.add_argument("--noise-lambda", type=float, default=0.3)
    p.add_argument("--rdm-metric", choices=RDM_METRICS, default="correlation")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force-export", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, Path]:
    if str(args.device) != "cuda":
        raise ValueError("This project analysis requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AGENTS.md requires --device cuda for project runs.")

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
            f"Expected {int(args.n_per_class)} images per class under {image_root}; "
            f"got {dict(zip(class_names, counts.tolist(), strict=True))}."
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

    subject_ids = _subject_ids(int(args.n_subjects))
    layer_names: list[str] = []
    image_patterns_by_layer: list[np.ndarray] = []
    category_patterns_by_layer: list[np.ndarray] = []
    image_rdms_by_layer: list[np.ndarray] = []
    category_rdms_by_layer: list[np.ndarray] = []
    category_counts: np.ndarray | None = None

    for layer_id in layer_ids:
        layer_result = sim.simulate_layer(activations[layer_id], labels, layer_id=layer_id)
        subject_result = compute_subject_noise_free_rdms(
            layer_result.b_true,
            labels,
            len(class_names),
            rdm_metric,
        )
        if category_counts is None:
            category_counts = subject_result.category_counts
        elif not np.array_equal(category_counts, subject_result.category_counts):
            raise RuntimeError("Category counts changed across layers.")

        image_patterns_by_layer.append(subject_result.image_patterns)
        category_patterns_by_layer.append(subject_result.category_patterns)
        image_rdms_by_layer.append(subject_result.image_rdms)
        category_rdms_by_layer.append(subject_result.category_rdms)
        layer_names.append(layer_result.layer_name)

    assert category_counts is not None
    noise_free_image_patterns = np.stack(image_patterns_by_layer, axis=1)
    noise_free_category_patterns = np.stack(category_patterns_by_layer, axis=1)
    noise_free_image_rdms = np.stack(image_rdms_by_layer, axis=1)
    noise_free_category_rdms = np.stack(category_rdms_by_layer, axis=1)

    figure_paths = save_subject_rdm_figures(
        noise_free_image_rdms=noise_free_image_rdms,
        noise_free_category_rdms=noise_free_category_rdms,
        layer_names=layer_names,
        class_names=class_names,
        labels=labels,
        output_dir=output_dir,
        rdm_metric=rdm_metric,
        subject_ids=subject_ids,
    )
    npz_path = _save_npz(
        output_dir=output_dir,
        layer_ids=layer_ids,
        layer_names=layer_names,
        class_names=class_names,
        labels=labels,
        image_paths=image_paths,
        subject_ids=subject_ids,
        noise_free_image_patterns=noise_free_image_patterns,
        noise_free_category_patterns=noise_free_category_patterns,
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
        subject_ids=subject_ids,
        args=args,
    )
    return {"results_npz": npz_path, "summary": summary_path, **figure_paths}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = run(args)
    print("[alexnet-fmri-subject-rdms] Saved:", flush=True)
    for path in paths.values():
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
