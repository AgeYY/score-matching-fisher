from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from global_setting import DATA_DIR, ECOSET_VALIDATION_DIR


def _load_subject_rdm_module():
    script_path = Path(__file__).resolve().parents[1] / "bin" / "study_alexnet_fmri_multicat_subject_rdms.py"
    spec = importlib.util.spec_from_file_location("_study_alexnet_fmri_multicat_subject_rdms", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data_dir_path(*parts: str) -> Path:
    root = Path(DATA_DIR)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    return root.joinpath(*parts)


def _abs_repo_path(path: str | Path) -> Path:
    root = Path(path)
    if root.is_absolute():
        return root
    return Path(__file__).resolve().parents[1] / root


def test_subject_rdm_cli_defaults() -> None:
    module = _load_subject_rdm_module()

    args = module.build_parser().parse_args([])

    assert module.parse_classes(args.classes) == module.DEFAULT_CLASSES
    assert module.parse_layers(args.layers) == (2, 5, 8, 10, 12, "classifier.4")
    assert Path(args.image_root) == _data_dir_path("ecoset", "validation_12cat_50")
    assert Path(args.ecoset_validation_dir) == _abs_repo_path(ECOSET_VALIDATION_DIR)
    assert Path(args.hf_cache_dir) == _abs_repo_path(ECOSET_VALIDATION_DIR)
    assert Path(args.output_dir) == _data_dir_path("ecoset", "alexnet_fmri_multicat_subject_rdms")
    assert int(args.n_subjects) == 10
    assert int(args.n_per_class) == 50
    assert int(args.n_voxels) == 100
    assert args.rdm_metric == "correlation"
    assert args.device == "cuda:0"


def test_compute_subject_noise_free_rdms_shapes() -> None:
    module = _load_subject_rdm_module()
    b_true = np.arange(2 * 6 * 4, dtype=np.float64).reshape(2, 6, 4)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    result = module.compute_subject_noise_free_rdms(b_true, labels, n_categories=3, rdm_metric="euclidean")

    assert result.image_patterns.shape == (2, 6, 4)
    assert result.category_patterns.shape == (2, 3, 4)
    assert result.image_rdms.shape == (2, 6, 6)
    assert result.category_rdms.shape == (2, 3, 3)
    np.testing.assert_allclose(result.image_patterns, b_true)
    np.testing.assert_array_equal(result.category_counts, np.array([[2, 2, 2], [2, 2, 2]], dtype=np.int64))
    np.testing.assert_allclose(result.category_patterns[0, 0], np.mean(b_true[0, :2], axis=0))


def test_subject_category_count_and_shape_validation() -> None:
    module = _load_subject_rdm_module()
    b_true = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    with pytest.raises(ValueError, match="labels length"):
        module.compute_subject_noise_free_rdms(b_true, np.array([0, 0, 1]), n_categories=2, rdm_metric="euclidean")

    with pytest.raises(ValueError, match="no examples"):
        module.compute_subject_noise_free_rdms(
            b_true,
            np.array([0, 0, 1, 1], dtype=np.int64),
            n_categories=3,
            rdm_metric="euclidean",
        )

    with pytest.raises(ValueError, match="square RDMs"):
        module.save_subject_rdm_figures(
            noise_free_image_rdms=np.zeros((2, 2, 4, 4), dtype=np.float64),
            noise_free_category_rdms=np.zeros((2, 2, 3, 2), dtype=np.float64),
            layer_names=("conv1", "conv2"),
            class_names=("cat", "dog"),
            labels=np.array([0, 0, 1, 1], dtype=np.int64),
            output_dir=Path("/tmp"),
            rdm_metric="euclidean",
            subject_ids=("S01", "S02"),
        )


def test_save_subject_rdm_figures_creates_png_and_svg(tmp_path: Path) -> None:
    module = _load_subject_rdm_module()
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    b_true0 = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [1.0, 0.8, 0.2], [1.2, 0.9, 0.4]],
            [[0.1, 0.0, 0.0], [0.3, 0.1, 0.2], [1.1, 0.7, 0.3], [1.3, 1.0, 0.5]],
        ],
        dtype=np.float64,
    )
    b_true1 = 1.5 * b_true0 + 0.25
    result0 = module.compute_subject_noise_free_rdms(b_true0, labels, n_categories=2, rdm_metric="euclidean")
    result1 = module.compute_subject_noise_free_rdms(b_true1, labels, n_categories=2, rdm_metric="euclidean")

    paths = module.save_subject_rdm_figures(
        noise_free_image_rdms=np.stack([result0.image_rdms, result1.image_rdms], axis=1),
        noise_free_category_rdms=np.stack([result0.category_rdms, result1.category_rdms], axis=1),
        layer_names=("conv1", "conv2"),
        class_names=("cat", "dog"),
        labels=labels,
        output_dir=tmp_path,
        rdm_metric="euclidean",
        subject_ids=("S01", "S02"),
    )

    assert set(paths) == {"category_png", "category_svg", "image_png", "image_svg"}
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_save_npz_stores_subject_first_arrays(tmp_path: Path) -> None:
    module = _load_subject_rdm_module()
    args = module.build_parser().parse_args(
        [
            "--n-per-class",
            "2",
            "--n-subjects",
            "2",
            "--n-voxels",
            "3",
            "--n-runs",
            "4",
            "--noise-lambda",
            "0.25",
            "--rdm-metric",
            "euclidean",
            "--seed",
            "9",
        ]
    )
    n_subjects, n_layers, n_images, n_classes, n_voxels = 2, 2, 4, 2, 3
    image_patterns = np.arange(n_subjects * n_layers * n_images * n_voxels, dtype=np.float64).reshape(
        n_subjects,
        n_layers,
        n_images,
        n_voxels,
    )
    category_patterns = np.arange(n_subjects * n_layers * n_classes * n_voxels, dtype=np.float64).reshape(
        n_subjects,
        n_layers,
        n_classes,
        n_voxels,
    )
    image_base = np.abs(np.subtract.outer(np.arange(n_images), np.arange(n_images))).astype(np.float64)
    category_base = np.abs(np.subtract.outer(np.arange(n_classes), np.arange(n_classes))).astype(np.float64)
    image_rdms = np.stack(
        [
            np.stack([image_base + 0.1 * subj + 0.01 * layer for layer in range(n_layers)], axis=0)
            for subj in range(n_subjects)
        ],
        axis=0,
    )
    category_rdms = np.stack(
        [
            np.stack([category_base + 0.1 * subj + 0.01 * layer for layer in range(n_layers)], axis=0)
            for subj in range(n_subjects)
        ],
        axis=0,
    )

    npz_path = module._save_npz(
        output_dir=tmp_path,
        layer_ids=(2, 5),
        layer_names=("conv1", "conv2"),
        class_names=("cat", "dog"),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        image_paths=(
            tmp_path / "cat0.jpg",
            tmp_path / "cat1.jpg",
            tmp_path / "dog0.jpg",
            tmp_path / "dog1.jpg",
        ),
        subject_ids=("S01", "S02"),
        noise_free_image_patterns=image_patterns,
        noise_free_category_patterns=category_patterns,
        noise_free_image_rdms=image_rdms,
        noise_free_category_rdms=category_rdms,
        category_counts=np.array([[2, 2], [2, 2]], dtype=np.int64),
        args=args,
    )

    with np.load(npz_path) as data:
        expected_keys = {
            "noise_free_image_rdms",
            "noise_free_category_rdms",
            "noise_free_image_patterns",
            "noise_free_category_patterns",
            "layer_ids",
            "layer_names",
            "classes",
            "labels",
            "image_paths",
            "rdm_metric",
            "subject_ids",
            "seed",
            "n_voxels",
        }
        assert expected_keys.issubset(set(data.files))
        assert data["noise_free_image_rdms"].shape == (2, 2, 4, 4)
        assert data["noise_free_category_rdms"].shape == (2, 2, 2, 2)
        assert data["noise_free_image_patterns"].shape == (2, 2, 4, 3)
        assert data["noise_free_category_patterns"].shape == (2, 2, 2, 3)
        assert data["rdm_metric"][0] == "euclidean"
        assert data["noise_free_source"][0] == "layer_result.b_true[subj]"
        assert data["subject_ids"].tolist() == ["S01", "S02"]
        np.testing.assert_allclose(data["noise_free_image_patterns"], image_patterns)
        np.testing.assert_array_equal(data["category_counts"], np.array([[2, 2], [2, 2]], dtype=np.int64))
