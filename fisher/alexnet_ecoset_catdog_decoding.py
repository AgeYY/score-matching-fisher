"""Cat-vs-dog linear decoding across AlexNet-Ecoset layers."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, ECOSET_VALIDATION_DIR
from fisher.alexnet_ecoset_model import load_alexnet_ecoset


ECOSET_VALIDATION_ARROW = "ecoset-validation.arrow"
ALEXNET_ECOSET_LAYERS: tuple[str, ...] = (
    "features.0",
    "features.3",
    "features.6",
    "features.8",
    "features.10",
    "classifier.1",
    "classifier.4",
    "classifier.6",
)
PLOT_LAYERS: tuple[str, ...] = ("raw_pixel",) + ALEXNET_ECOSET_LAYERS
DEFAULT_CLASSES: tuple[str, str] = ("cat", "dog")


@dataclass(frozen=True)
class SampledImage:
    path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class DecodingResult:
    layer_name: str
    accuracy: float
    n_train: int
    n_test: int


class EcosetValidationDirAction(argparse.Action):
    """Store the local validation path under canonical and legacy argparse names."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, "hf_cache_dir", values)


def _repo_root() -> Path:
    return _REPO_ROOT


def _abs_without_resolving_symlinks(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _repo_root() / p


def _default_output_root() -> Path:
    return _abs_without_resolving_symlinks(DATA_DIR) / "ecoset"


def class_label_name(example_label: object, label_names: Sequence[str] | None) -> str:
    if isinstance(example_label, str):
        return example_label
    if label_names is None:
        return str(example_label)
    return str(label_names[int(example_label)])


def stratified_sample_indices(
    labels: Sequence[str],
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    n_per_class: int = 100,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(int(seed))
    out: dict[str, np.ndarray] = {}
    for class_name in classes:
        idx = np.flatnonzero(labels_arr == str(class_name))
        if idx.shape[0] < int(n_per_class):
            raise ValueError(f"Class {class_name!r} has only {idx.shape[0]} examples; need {int(n_per_class)}.")
        chosen = rng.choice(idx, size=int(n_per_class), replace=False)
        out[str(class_name)] = np.sort(chosen)
    return out


def make_train_test_indices(
    labels: Sequence[int],
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    idx = np.arange(labels_arr.shape[0], dtype=np.int64)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=labels_arr,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def fit_linear_decoders(
    features_by_layer: Mapping[str, np.ndarray],
    labels: Sequence[int],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    seed: int = 0,
    max_iter: int = 5000,
) -> list[DecodingResult]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    train_idx_arr = np.asarray(train_idx, dtype=np.int64)
    test_idx_arr = np.asarray(test_idx, dtype=np.int64)
    results: list[DecodingResult] = []
    for layer_name in PLOT_LAYERS:
        if layer_name not in features_by_layer:
            continue
        x = np.asarray(features_by_layer[layer_name], dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Features for {layer_name!r} must be 2D after flattening; got {x.shape}.")
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                C=1.0,
                max_iter=int(max_iter),
                random_state=int(seed),
            ),
        )
        clf.fit(x[train_idx_arr], labels_arr[train_idx_arr])
        acc = float(clf.score(x[test_idx_arr], labels_arr[test_idx_arr]))
        results.append(
            DecodingResult(
                layer_name=layer_name,
                accuracy=acc,
                n_train=int(train_idx_arr.shape[0]),
                n_test=int(test_idx_arr.shape[0]),
            )
        )
    return results


def save_accuracy_figure(results: Sequence[DecodingResult], output_dir: str | os.PathLike[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "catdog_layer_accuracy.png"
    names = [r.layer_name for r in results]
    accs = [r.accuracy for r in results]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    x = np.arange(len(names))
    ax.plot(x, accs, marker="o", linewidth=2.0, color="#2f6f9f")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("held-out accuracy")
    ax.set_xlabel("AlexNet-Ecoset representation")
    ax.set_title("Cat vs dog linear decoding")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _missing_ecoset_validation_error(root: Path) -> FileNotFoundError:
    direct = root if root.name == ECOSET_VALIDATION_ARROW else root / ECOSET_VALIDATION_ARROW
    return FileNotFoundError(
        "EcoSet validation Arrow data is missing locally. "
        f"Expected {ECOSET_VALIDATION_ARROW!r} at {direct} or somewhere below {root}. "
        "Set SCORE_MATCHING_FISHER_ECOSET_VALIDATION_DIR to a local Arrow file, build directory, "
        "or parent cache directory. Downloads are not attempted from this sampling path."
    )


def _find_ecoset_validation_arrow(root: Path) -> Path | None:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in {"downloads", "__pycache__"})
        if ECOSET_VALIDATION_ARROW in filenames:
            return Path(dirpath) / ECOSET_VALIDATION_ARROW
    return None


def resolve_ecoset_validation_arrow(validation_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a local EcoSet validation Arrow file from a file, build dir, or cache parent."""
    root = _abs_without_resolving_symlinks(ECOSET_VALIDATION_DIR if validation_dir is None else validation_dir)
    if root.is_file():
        if root.name == ECOSET_VALIDATION_ARROW:
            return root
        raise _missing_ecoset_validation_error(root)
    if root.is_dir():
        direct = root / ECOSET_VALIDATION_ARROW
        if direct.is_file():
            return direct
        builder_root = root / "kietzmannlab___ecoset" / "Full"
        if builder_root.is_dir():
            arrow_path = _find_ecoset_validation_arrow(builder_root)
            if arrow_path is not None:
                return arrow_path
        arrow_path = _find_ecoset_validation_arrow(root)
        if arrow_path is not None:
            return arrow_path
    raise _missing_ecoset_validation_error(root)


def ecoset_validation_cache_ready(cache_dir: str | os.PathLike[str]) -> bool:
    """True when a local EcoSet validation Arrow table can be found under cache_dir."""
    try:
        arrow_path = resolve_ecoset_validation_arrow(cache_dir)
        return arrow_path.is_file() and arrow_path.stat().st_size > 0
    except FileNotFoundError:
        return False


def load_ecoset_validation(cache_dir: str | os.PathLike[str] | None = None):
    """Load EcoSet validation from a locally cached Arrow file without downloading."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to load local EcoSet validation Arrow data.") from exc

    arrow_path = resolve_ecoset_validation_arrow(cache_dir)
    return Dataset.from_file(str(arrow_path))


def _dataset_label_names(ds) -> Sequence[str] | None:
    try:
        return ds.features["label"].names
    except Exception:
        return None


def export_sampled_images(
    ds,
    *,
    image_root: str | os.PathLike[str],
    classes: Sequence[str] = DEFAULT_CLASSES,
    n_per_class: int = 100,
    seed: int = 0,
) -> list[SampledImage]:
    label_names = _dataset_label_names(ds)
    label_column = ds["label"]
    all_labels = [class_label_name(label, label_names) for label in label_column]
    sampled = stratified_sample_indices(all_labels, classes=classes, n_per_class=n_per_class, seed=seed)

    root = Path(image_root)
    root.mkdir(parents=True, exist_ok=True)
    out: list[SampledImage] = []
    for label, class_name in enumerate(classes):
        class_dir = root / str(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        for rank, ds_idx in enumerate(sampled[str(class_name)]):
            ex = ds[int(ds_idx)]
            img = ex["image"].convert("RGB")
            path = class_dir / f"{str(class_name)}_{rank:04d}_ds{int(ds_idx):07d}.jpg"
            if not path.exists():
                img.save(path, quality=95)
            out.append(SampledImage(path=path, label=int(label), class_name=str(class_name)))
    return out


def load_exported_samples(image_root: str | os.PathLike[str], *, classes: Sequence[str] = DEFAULT_CLASSES) -> list[SampledImage]:
    samples: list[SampledImage] = []
    for label, class_name in enumerate(classes):
        class_dir = Path(image_root) / str(class_name)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected exported class directory: {class_dir}")
        paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not paths:
            raise ValueError(f"No exported images found in {class_dir}.")
        samples.extend(SampledImage(path=p, label=int(label), class_name=str(class_name)) for p in paths)
    return samples


def ensure_sampled_images(
    *,
    image_root: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
    n_per_class: int = 100,
    seed: int = 0,
    force_export: bool = False,
) -> list[SampledImage]:
    root = Path(image_root)
    if not force_export:
        try:
            samples = load_exported_samples(root, classes=classes)
            counts = {c: sum(s.class_name == c for s in samples) for c in classes}
            if all(counts[str(c)] >= int(n_per_class) for c in classes):
                trimmed: list[SampledImage] = []
                for class_name in classes:
                    class_samples = [s for s in samples if s.class_name == str(class_name)]
                    trimmed.extend(class_samples[: int(n_per_class)])
                return trimmed
        except (FileNotFoundError, ValueError):
            pass

    ds = load_ecoset_validation(cache_dir)
    return export_sampled_images(ds, image_root=root, classes=classes, n_per_class=n_per_class, seed=seed)


def extract_representations(
    image_paths: Sequence[str | os.PathLike[str]],
    *,
    device: str = "cuda",
    batch_size: int = 64,
    layers: Sequence[str] = ALEXNET_ECOSET_LAYERS,
) -> dict[str, np.ndarray]:
    if str(device) != "cuda":
        raise ValueError("This project analysis requires device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AGENTS.md requires --device cuda for project runs.")
    model, transform = load_alexnet_ecoset(device)
    return extract_representations_from_model(
        model,
        transform,
        image_paths,
        device=device,
        batch_size=batch_size,
        layers=layers,
    )


def extract_representations_from_model(
    model: torch.nn.Module,
    transform,
    image_paths: Sequence[str | os.PathLike[str]],
    *,
    device: str = "cuda",
    batch_size: int = 64,
    layers: Sequence[str] = ALEXNET_ECOSET_LAYERS,
) -> dict[str, np.ndarray]:
    module_by_name = dict(model.named_modules())
    missing = [layer for layer in layers if layer not in module_by_name]
    if missing:
        raise ValueError(f"Requested layers are absent from AlexNet-EcoSet model: {missing}")

    captured: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    hooks = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output) -> None:
            captured[layer_name].append(output.detach().cpu().flatten(start_dim=1))

        return hook

    for layer in layers:
        hooks.append(module_by_name[layer].register_forward_hook(make_hook(layer)))

    raw_pixels: list[torch.Tensor] = []
    paths = [Path(p) for p in image_paths]
    dev = torch.device(device)
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(paths), int(batch_size)):
                batch_paths = paths[start : start + int(batch_size)]
                batch = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch_paths])
                raw_pixels.append(batch.detach().cpu().flatten(start_dim=1))
                model(batch.to(dev))
    finally:
        for handle in hooks:
            handle.remove()

    out: dict[str, np.ndarray] = {"raw_pixel": torch.cat(raw_pixels, dim=0).numpy()}
    for layer, parts in captured.items():
        out[layer] = torch.cat(parts, dim=0).numpy()
    return out


def parse_classes(raw: str) -> tuple[str, str]:
    classes = tuple(tok.strip() for tok in str(raw).split(",") if tok.strip())
    if len(classes) != 2:
        raise ValueError("--classes must contain exactly two comma-separated class names.")
    return classes  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    root = _default_output_root()
    validation_dir = _abs_without_resolving_symlinks(ECOSET_VALIDATION_DIR)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.set_defaults(hf_cache_dir=str(validation_dir))
    p.add_argument("--classes", default="cat,dog", help="Two comma-separated Ecoset validation class names.")
    p.add_argument("--n-per-class", type=int, default=100, help="Number of validation images sampled per class.")
    p.add_argument("--test-size", type=float, default=0.2, help="Held-out stratified test fraction.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling, split, and classifier.")
    p.add_argument("--batch-size", type=int, default=64, help="Image batch size for AlexNet-EcoSet extraction.")
    p.add_argument("--device", default="cuda", help="Execution device; this project run requires cuda.")
    p.add_argument(
        "--ecoset-validation-dir",
        "--hf-cache-dir",
        dest="ecoset_validation_dir",
        action=EcosetValidationDirAction,
        default=str(validation_dir),
        help="Local EcoSet validation Arrow file, build directory, or parent cache directory; no downloads are attempted.",
    )
    p.add_argument(
        "--image-root",
        default=str(root / "validation_catdog_100"),
        help="Folder where sampled validation images are exported or reused.",
    )
    p.add_argument(
        "--output-dir",
        default=str(root / "alexnet_ecoset_catdog_decoding"),
        help="Directory for the accuracy figure.",
    )
    p.add_argument("--force-export", action="store_true", help="Re-sample/re-export images even if image-root exists.")
    p.add_argument("--clf-max-iter", type=int, default=5000, help="LogisticRegression max_iter.")
    return p


def run(args: argparse.Namespace) -> Path:
    if str(args.device) != "cuda":
        raise ValueError("This project analysis requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; AGENTS.md requires --device cuda for project runs.")
    classes = parse_classes(args.classes)
    validation_dir = getattr(args, "ecoset_validation_dir", getattr(args, "hf_cache_dir", None))
    samples = ensure_sampled_images(
        image_root=args.image_root,
        cache_dir=validation_dir,
        classes=classes,
        n_per_class=int(args.n_per_class),
        seed=int(args.seed),
        force_export=bool(args.force_export),
    )
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    train_idx, test_idx = make_train_test_indices(labels, test_size=float(args.test_size), seed=int(args.seed))
    features = extract_representations(
        [s.path for s in samples],
        device=str(args.device),
        batch_size=int(args.batch_size),
    )
    results = fit_linear_decoders(
        features,
        labels,
        train_idx,
        test_idx,
        seed=int(args.seed),
        max_iter=int(args.clf_max_iter),
    )
    return save_accuracy_figure(results, args.output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fig_path = run(args)
    print(f"[alexnet-ecoset] Saved figure: {fig_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
