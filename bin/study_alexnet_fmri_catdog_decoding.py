#!/usr/bin/env python3
"""Run AlexNet-EcoSet fMRI cat-vs-dog decoding and save accuracy by layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR
from fisher.alexnet_ecoset_catdog_decoding import ensure_sampled_images
from fisher.alexnet_fmri_simulation import (
    AlexNetFMRISimulator,
    FMRISimulationConfig,
    save_layer_decoding_accuracy_figure,
)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _abs_without_resolving_symlinks(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _default_ecoset_root() -> Path:
    return _abs_without_resolving_symlinks(DATA_DIR) / "ecoset"


def _count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _ensure_class_dirs(
    *,
    image_root: Path,
    hf_cache_dir: Path,
    n_per_class: int,
    seed: int,
    force_export: bool,
) -> tuple[Path, Path]:
    class_dirs = (image_root / "cat", image_root / "dog")
    if not force_export and all(_count_images(class_dir) >= int(n_per_class) for class_dir in class_dirs):
        return class_dirs

    ensure_sampled_images(
        image_root=image_root,
        cache_dir=hf_cache_dir,
        classes=("cat", "dog"),
        n_per_class=int(n_per_class),
        seed=int(seed),
        force_export=bool(force_export),
    )
    if not all(_count_images(class_dir) >= int(n_per_class) for class_dir in class_dirs):
        raise RuntimeError(f"EcoSet export did not produce {int(n_per_class)} images per class under {image_root}.")
    return class_dirs


def _parse_layers(raw: str) -> tuple[int | str, ...]:
    layers: list[int | str] = []
    for tok in str(raw).split(","):
        layer = tok.strip()
        if not layer:
            continue
        try:
            layers.append(int(layer))
        except ValueError:
            layers.append(layer)
    if not layers:
        raise ValueError("--layers must contain at least one layer id.")
    return tuple(layers)


def build_parser() -> argparse.ArgumentParser:
    ecoset_root = _default_ecoset_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--image-root",
        default=str(ecoset_root / "validation_catdog"),
        help="Folder where sampled EcoSet cat/dog validation images are exported or reused.",
    )
    p.add_argument("--hf-cache-dir", default=str(ecoset_root / "hf_cache"))
    p.add_argument("--output-dir", default=str(ecoset_root / "alexnet_fmri_catdog_decoding"))
    p.add_argument("--n-per-class", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--layers",
        default="2,5,8,10,12,classifier.4",
        help="Comma-separated AlexNet-EcoSet layer ids or module names.",
    )
    p.add_argument("--n-subjects", type=int, default=1)
    p.add_argument("--n-voxels", type=int, default=100)
    p.add_argument("--n-runs", type=int, default=4)
    p.add_argument("--noise-lambda", type=float, default=0.3)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--clf-max-iter", type=int, default=5000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force-export", action="store_true")
    return p


def run(args: argparse.Namespace) -> Path:
    if str(args.device) != "cuda":
        raise ValueError("This project analysis requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AGENTS.md requires --device cuda for project runs.")

    n_per_class = int(args.n_per_class)
    class_dirs = _ensure_class_dirs(
        image_root=_abs_without_resolving_symlinks(args.image_root),
        hf_cache_dir=_abs_without_resolving_symlinks(args.hf_cache_dir),
        n_per_class=n_per_class,
        seed=int(args.seed),
        force_export=bool(args.force_export),
    )
    cfg = FMRISimulationConfig(
        candidate_layer_ids=_parse_layers(str(args.layers)),
        n_subjects=int(args.n_subjects),
        n_voxels=int(args.n_voxels),
        n_runs=int(args.n_runs),
        noise_lambda=float(args.noise_lambda),
        seed=int(args.seed),
        device=str(args.device),
        cv_folds=int(args.cv_folds),
        max_iter=int(args.clf_max_iter),
    )
    sim = AlexNetFMRISimulator(cfg)
    out = sim.run_image_decoding(class_dirs, max_images_per_class=n_per_class, batch_size=int(args.batch_size))
    return save_layer_decoding_accuracy_figure(out, _abs_without_resolving_symlinks(args.output_dir))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fig_path = run(args)
    print(f"[alexnet-ecoset-fmri] Saved figure: {fig_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
