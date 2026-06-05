from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from global_setting import DATA_DIR
from fisher import alexnet_fmri_simulation as fmri_module
from fisher.alexnet_ecoset_catdog_decoding import ensure_sampled_images
from fisher.alexnet_fmri_simulation import (
    AlexNetFMRISimulator,
    FMRIBetaSimulator,
    FMRISimulationConfig,
    make_event_design,
    measurement_filter_correlation,
    save_layer_decoding_accuracy_figure,
)


def _load_fmri_study_module():
    script_path = Path(__file__).resolve().parents[1] / "bin" / "study_alexnet_fmri_catdog_decoding.py"
    spec = importlib.util.spec_from_file_location("_study_alexnet_fmri_catdog_decoding", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alexnet_fmri_simulator_uses_ecoset_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.device] = []

    class TinyAlexNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.ModuleList(torch.nn.Identity() for _ in range(11))

    model = TinyAlexNet()

    def fake_load_alexnet_ecoset(device: str | torch.device):
        calls.append(torch.device(device))
        return model, lambda image: torch.zeros(3, 224, 224)

    monkeypatch.setattr(fmri_module, "load_alexnet_ecoset", fake_load_alexnet_ecoset)

    cfg = FMRISimulationConfig(candidate_layer_ids=(2,), device="cpu")
    sim = AlexNetFMRISimulator(cfg)

    assert calls == [torch.device("cpu")]
    assert sim.model is model


def test_alexnet_fmri_extract_activations_supports_named_classifier_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyAlexNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.Flatten())
            self.classifier = torch.nn.Sequential(
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Linear(3 * 4 * 4, 7),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    def fake_load_alexnet_ecoset(device: str | torch.device):
        model = TinyAlexNet().to(torch.device(device))
        return model, lambda _image: torch.ones(3, 4, 4)

    monkeypatch.setattr(fmri_module, "load_alexnet_ecoset", fake_load_alexnet_ecoset)
    image_paths = []
    for idx in range(2):
        path = tmp_path / f"image_{idx}.png"
        Image.fromarray(np.full((6, 6, 3), idx, dtype=np.uint8)).save(path)
        image_paths.append(path)

    cfg = FMRISimulationConfig(candidate_layer_ids=("classifier.4",), device="cpu")
    sim = AlexNetFMRISimulator(cfg)

    activations = sim.extract_activations(image_paths, batch_size=1)

    assert set(activations) == {"classifier.4"}
    assert activations["classifier.4"].shape == (2, 7)


def test_fmri_beta_simulator_shapes_finite_and_reproducible() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    class_signal = labels[:, None, None, None].astype(np.float64) * 0.5
    activations = rng.normal(size=(6, 4, 5, 5)) + class_signal
    cfg = FMRISimulationConfig(
        candidate_layer_ids=(8,),
        n_subjects=1,
        n_voxels=10,
        n_runs=2,
        noise_lambda=0.1,
        seed=11,
        device="cuda",
        cv_folds=2,
    )

    sim0 = FMRIBetaSimulator(cfg).simulate_layer(activations, labels, layer_id=8)
    sim1 = FMRIBetaSimulator(cfg).simulate_layer(activations, labels, layer_id=8)

    assert sim0.beta_hat.shape == (1, 2, 6, 10)
    assert sim0.b_true.shape == (1, 6, 10)
    assert sim0.sigma_v.shape == (1, 10, 10)
    assert len(sim0.design_matrices) == 2
    assert np.all(np.isfinite(sim0.beta_hat))
    assert np.all(np.isfinite(sim0.b_true))
    np.testing.assert_allclose(np.diag(sim0.sigma_v[0]), np.full(10, 1.0 + cfg.ridge_jitter))
    np.testing.assert_allclose(sim0.beta_hat, sim1.beta_hat)
    np.testing.assert_allclose(sim0.b_true, sim1.b_true)

    decoding = FMRIBetaSimulator(cfg).decode_layer(sim0)
    assert decoding.layer_id == 8
    assert 0.0 <= decoding.accuracy <= 1.0
    assert decoding.scores.shape == (2,)


def test_fmri_beta_simulator_accepts_classifier_vector_activations() -> None:
    rng = np.random.default_rng(4)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    activations = rng.normal(size=(6, 8)) + labels[:, None].astype(np.float64) * 0.5
    cfg = FMRISimulationConfig(
        candidate_layer_ids=("classifier.4",),
        n_subjects=1,
        n_voxels=4,
        n_runs=2,
        noise_lambda=0.1,
        seed=4,
        device="cuda",
        cv_folds=2,
    )

    sim = FMRIBetaSimulator(cfg)
    result = sim.simulate_layer(activations, labels, layer_id="classifier.4")
    decoding = sim.decode_layer(result)

    assert result.layer_id == "classifier.4"
    assert result.layer_name == "classifier.4"
    assert result.beta_hat.shape == (1, 2, 6, 4)
    assert np.all(np.isfinite(result.beta_hat))
    assert 0.0 <= decoding.accuracy <= 1.0


def test_measurement_filter_correlation_is_symmetric_unit_diagonal() -> None:
    flat = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    corr = measurement_filter_correlation(flat, jitter=0.0)

    assert corr.shape == (3, 3)
    np.testing.assert_allclose(corr, corr.T)
    np.testing.assert_allclose(np.diag(corr), np.ones(3))


def test_make_event_design_uses_stimulus_duration_boxcars() -> None:
    design = make_event_design(
        3,
        tr=2.0,
        stim_interval=4.0,
        blank_interval=2.0,
        hrf=np.array([1.0], dtype=np.float64),
        rng=np.random.default_rng(2),
    )

    assert design.shape == (11, 3)
    np.testing.assert_allclose(np.sum(design, axis=0), np.full(3, 2.0))
    assert np.all(np.sum(design > 0.0, axis=1) <= 1)
    for col in range(design.shape[1]):
        active = np.flatnonzero(design[:, col] > 0.0)
        assert active.shape == (2,)
        assert int(active[1] - active[0]) == 1


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _data_dir_path(*parts: str) -> Path:
    root = Path(DATA_DIR)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    return root.joinpath(*parts)


def test_fmri_cli_defaults_to_validation_catdog_folder() -> None:
    study_module = _load_fmri_study_module()

    args = study_module.build_parser().parse_args([])

    assert Path(args.image_root) == _data_dir_path("ecoset", "validation_catdog")
    assert int(args.n_per_class) == 50
    assert args.layers == "2,5,8,10,12,classifier.4"
    assert study_module._parse_layers(args.layers) == (2, 5, 8, 10, 12, "classifier.4")


def _count_images(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _ecoset_catdog_class_dirs(n_per_class: int) -> tuple[Path, Path]:
    ecoset_root = _data_dir_path("ecoset")
    for image_root in (
        ecoset_root / "validation_catdog",
        ecoset_root / "validation_catdog_100",
        ecoset_root / "validation_catdog_smoke",
    ):
        class_dirs = (image_root / "cat", image_root / "dog")
        if all(_count_images(class_dir) >= int(n_per_class) for class_dir in class_dirs):
            return class_dirs

    image_root = ecoset_root / "validation_catdog_smoke"
    try:
        ensure_sampled_images(
            image_root=image_root,
            cache_dir=ecoset_root / "hf_cache",
            classes=("cat", "dog"),
            n_per_class=int(n_per_class),
            seed=3,
        )
    except Exception as exc:
        pytest.skip(f"EcoSet cat/dog smoke images could not be exported: {exc}")

    class_dirs = (image_root / "cat", image_root / "dog")
    if not all(_count_images(class_dir) >= int(n_per_class) for class_dir in class_dirs):
        pytest.skip(f"EcoSet cat/dog smoke export did not produce {int(n_per_class)} images per class.")
    return class_dirs


def test_alexnet_fmri_simulator_two_class_image_decoding_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for project AlexNet fMRI runs.")

    class_dirs = _ecoset_catdog_class_dirs(n_per_class=3)
    cfg = FMRISimulationConfig(
        candidate_layer_ids=(2, 5, 8, 10, 12),
        n_subjects=1,
        n_voxels=4,
        n_runs=2,
        noise_lambda=0.05,
        seed=3,
        device="cuda",
        cv_folds=2,
    )
    try:
        sim = AlexNetFMRISimulator(cfg)
    except Exception as exc:
        pytest.skip(f"AlexNet-EcoSet could not be loaded: {exc}")

    out = sim.run_image_decoding(class_dirs, max_images_per_class=3, batch_size=2)
    fig_path = save_layer_decoding_accuracy_figure(out, tmp_path)

    assert set(out) == {2, 5, 8, 10, 12}
    assert fig_path == tmp_path / "catdog_fmri_layer_accuracy.png"
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0
    for layer_id, (layer_sim, decoding) in out.items():
        assert layer_sim.layer_id == layer_id
        assert layer_sim.beta_hat.shape == (1, 2, 6, 4)
        assert layer_sim.class_names == ("cat", "dog")
        assert np.all(np.isfinite(layer_sim.beta_hat))
        assert 0.0 <= decoding.accuracy <= 1.0
