"""AlexNet-based fMRI beta-pattern simulation and layer decoding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from fisher.alexnet_ecoset_model import load_alexnet_ecoset


LayerKey = int | str


CONV_LAYER_IDS: dict[int, str] = {
    2: "conv1",
    5: "conv2",
    8: "conv3",
    10: "conv4",
    12: "conv5",
}


@dataclass(frozen=True)
class AlexNetFMRILayerSpec:
    module_name: str
    layer_name: str
    sort_order: int
    seed_offset: int


ALEXNET_FMRI_LAYER_SPECS: dict[LayerKey, AlexNetFMRILayerSpec] = {
    2: AlexNetFMRILayerSpec("features.0", "conv1", 0, 2),
    5: AlexNetFMRILayerSpec("features.3", "conv2", 1, 5),
    8: AlexNetFMRILayerSpec("features.6", "conv3", 2, 8),
    10: AlexNetFMRILayerSpec("features.8", "conv4", 3, 10),
    12: AlexNetFMRILayerSpec("features.10", "conv5", 4, 12),
    "classifier.4": AlexNetFMRILayerSpec("classifier.4", "classifier.4", 5, 14),
}


def alexnet_fmri_layer_spec(layer_id: LayerKey) -> AlexNetFMRILayerSpec:
    try:
        return ALEXNET_FMRI_LAYER_SPECS[layer_id]
    except KeyError as exc:
        allowed = ", ".join(str(key) for key in ALEXNET_FMRI_LAYER_SPECS)
        raise ValueError(f"Unknown AlexNet-EcoSet fMRI layer {layer_id!r}; available layers: {allowed}.") from exc


def _layer_seed_offset(layer_id: LayerKey) -> int:
    spec = ALEXNET_FMRI_LAYER_SPECS.get(layer_id)
    if spec is not None:
        return int(spec.seed_offset)
    if isinstance(layer_id, int):
        return int(layer_id)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(str(layer_id)))


def _layer_sort_key(layer_id: LayerKey) -> tuple[int, str]:
    spec = ALEXNET_FMRI_LAYER_SPECS.get(layer_id)
    if spec is not None:
        return int(spec.sort_order), str(layer_id)
    if isinstance(layer_id, int):
        return 1000 + int(layer_id), str(layer_id)
    return 2000, str(layer_id)


def _layer_display_name(layer_id: LayerKey) -> str:
    spec = ALEXNET_FMRI_LAYER_SPECS.get(layer_id)
    if spec is not None:
        return spec.layer_name
    return f"layer_{layer_id}"


@dataclass(frozen=True)
class FMRISimulationConfig:
    candidate_layer_ids: tuple[LayerKey, ...] = (2, 5, 8, 10, 12)
    n_subjects: int = 1
    n_voxels: int = 100
    n_runs: int = 4
    tr: float = 2.0
    stim_interval: float = 4.0
    blank_interval: float = 2.0
    sigma_voxel: float = 0.05
    rho: float = 0.5
    noise_lambda: float = 0.3
    seed: int = 0
    device: str = "cuda"
    cv_folds: int = 3
    max_iter: int = 1000
    hrf_duration: float = 32.0
    ridge_jitter: float = 1e-6

    def validate(self) -> None:
        if self.n_subjects < 1:
            raise ValueError("n_subjects must be positive.")
        if self.n_voxels < 1:
            raise ValueError("n_voxels must be positive.")
        if self.n_runs < 1:
            raise ValueError("n_runs must be positive.")
        if self.tr <= 0.0:
            raise ValueError("tr must be positive.")
        if self.stim_interval <= 0.0 or self.blank_interval < 0.0:
            raise ValueError("stim_interval must be positive and blank_interval must be non-negative.")
        if self.sigma_voxel <= 0.0:
            raise ValueError("sigma_voxel must be positive.")
        if not (0.0 <= self.rho < 1.0):
            raise ValueError("rho must satisfy 0 <= rho < 1.")
        if self.noise_lambda < 0.0:
            raise ValueError("noise_lambda must be non-negative.")
        unknown = [layer_id for layer_id in self.candidate_layer_ids if layer_id not in ALEXNET_FMRI_LAYER_SPECS]
        if unknown:
            allowed = ", ".join(str(key) for key in ALEXNET_FMRI_LAYER_SPECS)
            raise ValueError(f"Unknown AlexNet-EcoSet fMRI layers: {unknown}. Available layers: {allowed}.")


@dataclass(frozen=True)
class LayerSimulationResult:
    layer_id: LayerKey
    layer_name: str
    beta_hat: np.ndarray
    b_true: np.ndarray
    labels: np.ndarray
    sigma_v: np.ndarray
    design_matrices: tuple[np.ndarray, ...]
    decoding_accuracy: float | None = None
    image_paths: tuple[Path, ...] = ()
    class_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerDecodingResult:
    layer_id: LayerKey
    layer_name: str
    accuracy: float
    scores: np.ndarray


def load_two_class_image_paths(
    class_dirs: Sequence[str | Path],
    *,
    max_images_per_class: int | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Load image paths from exactly two class directories."""
    if len(class_dirs) != 2:
        raise ValueError("class_dirs must contain exactly two directories.")
    paths: list[Path] = []
    labels: list[int] = []
    class_names: list[str] = []
    suffixes = {ext.lower() for ext in extensions}
    for label, class_dir in enumerate(class_dirs):
        root = Path(class_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {root}")
        class_names.append(root.name)
        found = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
        if max_images_per_class is not None:
            found = found[: int(max_images_per_class)]
        if not found:
            raise ValueError(f"No image files found in {root}.")
        paths.extend(found)
        labels.extend([label] * len(found))
    return paths, np.asarray(labels, dtype=np.int64), class_names


def save_layer_decoding_accuracy_figure(
    decoding_by_layer: Mapping[LayerKey, tuple[LayerSimulationResult, LayerDecodingResult]],
    output_dir: str | Path,
    *,
    filename: str = "catdog_fmri_layer_accuracy.png",
) -> Path:
    """Save cat-vs-dog fMRI linear decoding accuracy as a function of AlexNet layer."""
    if not decoding_by_layer:
        raise ValueError("decoding_by_layer must contain at least one layer.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_ids = sorted(decoding_by_layer, key=_layer_sort_key)
    decoding_results = [decoding_by_layer[layer_id][1] for layer_id in layer_ids]
    layer_names = [result.layer_name for result in decoding_results]
    accuracies = np.asarray([result.accuracy for result in decoding_results], dtype=np.float64)
    if not np.all(np.isfinite(accuracies)):
        raise ValueError("Layer decoding accuracies must be finite.")

    class_names: tuple[str, ...] = ()
    for layer_id in layer_ids:
        class_names = decoding_by_layer[layer_id][0].class_names
        if class_names:
            break
    if len(class_names) == 2:
        title = f"{class_names[0].capitalize()} vs {class_names[1]} fMRI linear decoding"
    else:
        title = "Two-class fMRI linear decoding"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / filename

    x = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    ax.plot(x, accuracies, marker="o", linewidth=2.0, color="#2f6f9f")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.1, alpha=0.8)
    for xpos, acc in zip(x, accuracies, strict=True):
        ax.text(float(xpos), min(float(acc) + 0.035, 0.985), f"{float(acc):.2f}", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("cross-validated accuracy")
    ax.set_xlabel("AlexNet-EcoSet layer")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def spm_hrf(tr: float, *, duration: float = 32.0) -> np.ndarray:
    """Return a normalized double-gamma HRF sampled at TR resolution."""
    from scipy.stats import gamma

    times = np.arange(0.0, float(duration), float(tr), dtype=np.float64)
    hrf = gamma.pdf(times, 6.0) - (1.0 / 6.0) * gamma.pdf(times, 16.0)
    total = np.sum(hrf)
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Could not normalize HRF.")
    return hrf / total


class FMRIBetaSimulator:
    """Simulate GLM-estimated beta patterns from layer activations."""

    def __init__(self, config: FMRISimulationConfig | None = None) -> None:
        self.config = config or FMRISimulationConfig()
        self.config.validate()

    def simulate_layer(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        *,
        layer_id: LayerKey,
    ) -> LayerSimulationResult:
        activations = np.asarray(activations, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if activations.ndim == 2:
            activations = activations[:, :, None, None]
        elif activations.ndim != 4:
            raise ValueError(f"activations must have shape (K, C, H, W) or (K, D); got {activations.shape}.")
        if labels.shape[0] != activations.shape[0]:
            raise ValueError("labels length must match number of activations.")
        cfg = self.config
        k_total, n_channels, height, width = activations.shape
        rng = np.random.default_rng(cfg.seed + 1009 * _layer_seed_offset(layer_id))

        b_true = np.empty((cfg.n_subjects, k_total, cfg.n_voxels), dtype=np.float64)
        sigmas = np.empty((cfg.n_subjects, cfg.n_voxels, cfg.n_voxels), dtype=np.float64)
        for subj in range(cfg.n_subjects):
            filters = self._sample_measurement_filters(
                rng,
                n_voxels=cfg.n_voxels,
                n_channels=n_channels,
                height=height,
                width=width,
            )
            flat_filters = filters.reshape(cfg.n_voxels, -1)
            b_true[subj] = activations.reshape(k_total, -1) @ flat_filters.T
            sigmas[subj] = measurement_filter_correlation(flat_filters, jitter=cfg.ridge_jitter)

        design_matrices = tuple(
            make_event_design(
                k_total,
                tr=cfg.tr,
                stim_interval=cfg.stim_interval,
                blank_interval=cfg.blank_interval,
                hrf=spm_hrf(cfg.tr, duration=cfg.hrf_duration),
                rng=rng,
            )
            for _ in range(cfg.n_runs)
        )
        beta_hat = np.empty((cfg.n_subjects, cfg.n_runs, k_total, cfg.n_voxels), dtype=np.float64)
        for subj in range(cfg.n_subjects):
            chol = np.linalg.cholesky(sigmas[subj])
            for run, x_design in enumerate(design_matrices):
                clean = x_design @ b_true[subj]
                noise = sample_ar1_noise(
                    rng,
                    n_time=x_design.shape[0],
                    chol=chol,
                    rho=cfg.rho,
                )
                y_obs = clean + cfg.noise_lambda * noise
                beta_hat[subj, run] = np.linalg.pinv(x_design) @ y_obs

        return LayerSimulationResult(
            layer_id=layer_id,
            layer_name=_layer_display_name(layer_id),
            beta_hat=beta_hat,
            b_true=b_true,
            labels=labels.copy(),
            sigma_v=sigmas,
            design_matrices=design_matrices,
        )

    def decode_layer(self, result: LayerSimulationResult) -> LayerDecodingResult:
        labels = np.asarray(result.labels, dtype=np.int64)
        features = np.mean(result.beta_hat, axis=(0, 1))
        _, counts = np.unique(labels, return_counts=True)
        n_splits = min(int(self.config.cv_folds), int(np.min(counts)))
        if n_splits < 2:
            raise ValueError("At least two examples per class are required for cross-validated decoding.")
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=int(self.config.max_iter), random_state=int(self.config.seed)),
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(self.config.seed))
        scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
        return LayerDecodingResult(
            layer_id=result.layer_id,
            layer_name=result.layer_name,
            accuracy=float(np.mean(scores)),
            scores=np.asarray(scores, dtype=np.float64),
        )

    def simulate_and_decode(
        self,
        layer_activations: Mapping[LayerKey, np.ndarray],
        labels: np.ndarray,
    ) -> dict[LayerKey, tuple[LayerSimulationResult, LayerDecodingResult]]:
        out: dict[LayerKey, tuple[LayerSimulationResult, LayerDecodingResult]] = {}
        for layer_id in self.config.candidate_layer_ids:
            sim = self.simulate_layer(layer_activations[layer_id], labels, layer_id=layer_id)
            dec = self.decode_layer(sim)
            out[layer_id] = (sim, dec)
        return out

    def _sample_measurement_filters(
        self,
        rng: np.random.Generator,
        *,
        n_voxels: int,
        n_channels: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
        sigma_y = max(float(self.config.sigma_voxel) * float(height), 1e-6)
        sigma_x = max(float(self.config.sigma_voxel) * float(width), 1e-6)
        filters = np.empty((n_voxels, n_channels, height, width), dtype=np.float64)
        for voxel in range(n_voxels):
            cx = rng.uniform(0.0, max(float(width - 1), 0.0))
            cy = rng.uniform(0.0, max(float(height - 1), 0.0))
            spatial = np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
            spatial_sum = np.sum(spatial)
            if spatial_sum > 0.0:
                spatial = spatial / spatial_sum
            channel_weights = rng.uniform(0.0, 1.0, size=(n_channels, 1, 1))
            filt = channel_weights * spatial[None, :, :]
            norm = np.linalg.norm(filt.reshape(-1))
            filters[voxel] = filt / max(norm, 1e-12)
        return filters


class AlexNetFMRISimulator(FMRIBetaSimulator):
    """End-to-end simulator from image paths through AlexNet-EcoSet features."""

    def __init__(self, config: FMRISimulationConfig | None = None) -> None:
        super().__init__(config)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; AGENTS.md requires --device cuda for project runs.")
        self.device = torch.device(self.config.device)
        self.model, self.preprocess = load_alexnet_ecoset(self.device)

    def extract_activations(
        self,
        image_paths: Sequence[str | Path],
        *,
        batch_size: int = 16,
    ) -> dict[LayerKey, np.ndarray]:
        image_paths = [Path(p) for p in image_paths]
        requested = tuple(self.config.candidate_layer_ids)
        captured: dict[LayerKey, list[torch.Tensor]] = {layer_id: [] for layer_id in requested}
        hooks = []
        module_by_name = dict(self.model.named_modules())

        def make_hook(layer_id: LayerKey) -> Callable:
            def hook(_module, _inputs, output) -> None:
                captured[layer_id].append(output.detach().cpu())

            return hook

        for layer_id in requested:
            spec = alexnet_fmri_layer_spec(layer_id)
            if spec.module_name not in module_by_name:
                raise ValueError(f"AlexNet-EcoSet model does not contain layer {spec.module_name!r}.")
            hooks.append(module_by_name[spec.module_name].register_forward_hook(make_hook(layer_id)))
        try:
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(image_paths), int(batch_size)):
                    batch_paths = image_paths[start : start + int(batch_size)]
                    batch = torch.stack([self.preprocess(Image.open(path).convert("RGB")) for path in batch_paths])
                    self.model(batch.to(self.device))
        finally:
            for handle in hooks:
                handle.remove()
        return {layer_id: torch.cat(parts, dim=0).numpy() for layer_id, parts in captured.items()}

    def run_image_decoding(
        self,
        class_dirs: Sequence[str | Path],
        *,
        max_images_per_class: int | None = None,
        batch_size: int = 16,
    ) -> dict[LayerKey, tuple[LayerSimulationResult, LayerDecodingResult]]:
        paths, labels, class_names = load_two_class_image_paths(
            class_dirs,
            max_images_per_class=max_images_per_class,
        )
        activations = self.extract_activations(paths, batch_size=batch_size)
        out = self.simulate_and_decode(activations, labels)
        return {
            layer_id: (
                replace(sim, image_paths=tuple(paths), class_names=tuple(class_names)),
                decoding,
            )
            for layer_id, (sim, decoding) in out.items()
        }


def load_pretrained_alexnet(device: torch.device):
    """Compatibility wrapper returning AlexNet-EcoSet, not ImageNet AlexNet."""
    return load_alexnet_ecoset(device)


def make_event_design(
    n_stimuli: int,
    *,
    tr: float,
    stim_interval: float,
    blank_interval: float,
    hrf: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a run design with one presentation of each stimulus."""
    order = rng.permutation(int(n_stimuli))
    step = max(1, int(round((float(stim_interval) + float(blank_interval)) / float(tr))))
    onset_offset = max(0, int(round(float(blank_interval) / float(tr))))
    n_time = onset_offset + step * int(n_stimuli) + len(hrf)
    impulses = np.zeros((n_time, int(n_stimuli)), dtype=np.float64)
    for pos, stimulus_idx in enumerate(order):
        onset = onset_offset + pos * step
        impulses[onset, int(stimulus_idx)] = 1.0
    design = np.empty_like(impulses)
    for col in range(int(n_stimuli)):
        design[:, col] = fftconvolve(impulses[:, col], hrf, mode="full")[:n_time]
    return design


def measurement_filter_correlation(flat_filters: np.ndarray, *, jitter: float = 1e-6) -> np.ndarray:
    gram = np.asarray(flat_filters, dtype=np.float64) @ np.asarray(flat_filters, dtype=np.float64).T
    diag = np.sqrt(np.maximum(np.diag(gram), 1e-12))
    corr = gram / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)
    corr.flat[:: corr.shape[0] + 1] += float(jitter)
    return corr


def sample_ar1_noise(
    rng: np.random.Generator,
    *,
    n_time: int,
    chol: np.ndarray,
    rho: float,
) -> np.ndarray:
    n_voxels = int(chol.shape[0])
    innovations = rng.normal(size=(int(n_time), n_voxels)) @ chol.T
    noise = np.empty_like(innovations)
    noise[0] = innovations[0]
    scale = float(np.sqrt(1.0 - float(rho) ** 2))
    for t in range(1, int(n_time)):
        noise[t] = float(rho) * noise[t - 1] + scale * innovations[t]
    return noise
