from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from global_setting import DATA_DIR, ECOSET_VALIDATION_DIR


def _load_multicat_rdm_module():
    script_path = Path(__file__).resolve().parents[1] / "bin" / "study_alexnet_fmri_multicat_rdm.py"
    spec = importlib.util.spec_from_file_location("_study_alexnet_fmri_multicat_rdm", script_path)
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


def test_multicat_rdm_cli_defaults() -> None:
    module = _load_multicat_rdm_module()

    args = module.build_parser().parse_args([])

    assert module.DEFAULT_CLASSES == (
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
    assert module.parse_classes(args.classes) == module.DEFAULT_CLASSES
    assert module.parse_layers(args.layers) == (2, 5, 8, 10, 12, "classifier.4")
    assert Path(args.image_root) == _data_dir_path("ecoset", "validation_12cat_50")
    assert Path(args.ecoset_validation_dir) == _abs_repo_path(ECOSET_VALIDATION_DIR)
    assert Path(args.hf_cache_dir) == _abs_repo_path(ECOSET_VALIDATION_DIR)
    assert Path(args.output_dir) == _data_dir_path("ecoset", "alexnet_fmri_multicat_rdm")
    assert int(args.n_per_class) == 50
    assert args.rdm_metric == "correlation"
    assert args.device == "cuda"

    euclidean_args = module.build_parser().parse_args(["--rdm-metric", "euclidean"])
    assert euclidean_args.rdm_metric == "euclidean"


def test_multicat_rdm_cli_hf_cache_alias_maps_to_validation_dir() -> None:
    module = _load_multicat_rdm_module()

    args = module.build_parser().parse_args(["--hf-cache-dir", "/tmp/ecoset-cache"])

    assert args.ecoset_validation_dir == "/tmp/ecoset-cache"
    assert args.hf_cache_dir == "/tmp/ecoset-cache"


def test_parse_classes_rejects_duplicates() -> None:
    module = _load_multicat_rdm_module()

    with pytest.raises(ValueError, match="duplicate"):
        module.parse_classes("cat,dog,cat")


def test_compute_correlation_rdm_known_values() -> None:
    module = _load_multicat_rdm_module()
    patterns = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [-1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    rdm = module.compute_correlation_rdm(patterns)

    assert rdm.shape == (4, 4)
    np.testing.assert_allclose(rdm, rdm.T)
    np.testing.assert_allclose(np.diag(rdm), np.zeros(4))
    assert np.isclose(rdm[0, 1], 0.0)
    assert np.isclose(rdm[0, 2], 2.0)
    assert np.all((0.0 <= rdm) & (rdm <= 2.0))


def test_compute_correlation_rdm_rejects_zero_variance_rows() -> None:
    module = _load_multicat_rdm_module()
    patterns = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="zero-variance"):
        module.compute_correlation_rdm(patterns)


def test_compute_euclidean_rdm_known_values() -> None:
    module = _load_multicat_rdm_module()
    patterns = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float64,
    )

    rdm = module.compute_euclidean_rdm(patterns)

    expected = np.array(
        [
            [0.0, 5.0, 10.0],
            [5.0, 0.0, 5.0],
            [10.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(rdm, expected)


def test_compute_rdm_dispatches_metric() -> None:
    module = _load_multicat_rdm_module()
    patterns = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.5, 0.0, -1.5],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(module.compute_rdm(patterns, "correlation"), module.compute_correlation_rdm(patterns))
    np.testing.assert_allclose(module.compute_rdm(patterns, "euclidean"), module.compute_euclidean_rdm(patterns))
    with pytest.raises(ValueError, match="rdm_metric"):
        module.compute_rdm(patterns, "cosine")


def test_category_mean_patterns_and_rdm() -> None:
    module = _load_multicat_rdm_module()
    patterns = np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    means, counts = module.category_mean_patterns(patterns, labels, n_categories=2)
    rdm = module.compute_correlation_rdm(means)

    np.testing.assert_array_equal(counts, np.array([2, 2], dtype=np.int64))
    np.testing.assert_allclose(means[0], np.array([1.5, 0.0, -1.5]))
    np.testing.assert_allclose(means[1], np.array([-1.5, 0.0, 1.5]))
    np.testing.assert_allclose(rdm, np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_fmri_patterns_from_beta_hat_averages_subjects_and_runs() -> None:
    module = _load_multicat_rdm_module()
    beta_hat = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)

    patterns = module.fmri_patterns_from_beta_hat(beta_hat)

    assert patterns.shape == (4, 5)
    np.testing.assert_allclose(patterns, np.mean(beta_hat, axis=(0, 1)))


def test_fmri_patterns_from_b_true_averages_subjects() -> None:
    module = _load_multicat_rdm_module()
    b_true = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)

    patterns = module.fmri_patterns_from_b_true(b_true)

    assert patterns.shape == (4, 5)
    np.testing.assert_allclose(patterns, np.mean(b_true, axis=0))


def test_save_rdm_figures_creates_png_and_svg(tmp_path: Path) -> None:
    module = _load_multicat_rdm_module()
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    image_rdms = [
        np.array(
            [
                [0.0, 0.1, 1.2, 1.4],
                [0.1, 0.0, 1.1, 1.3],
                [1.2, 1.1, 0.0, 0.2],
                [1.4, 1.3, 0.2, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.0, 0.3, 1.0, 1.1],
                [0.3, 0.0, 1.2, 1.5],
                [1.0, 1.2, 0.0, 0.4],
                [1.1, 1.5, 0.4, 0.0],
            ],
            dtype=np.float64,
        ),
    ]
    category_rdms = [
        np.array([[0.0, 1.5], [1.5, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.2], [1.2, 0.0]], dtype=np.float64),
    ]
    noise_free_image_rdms = [
        np.array(
            [
                [0.0, 0.2, 1.0, 1.2],
                [0.2, 0.0, 1.2, 1.4],
                [1.0, 1.2, 0.0, 0.1],
                [1.2, 1.4, 0.1, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.0, 0.1, 1.3, 1.4],
                [0.1, 0.0, 1.4, 1.6],
                [1.3, 1.4, 0.0, 0.2],
                [1.4, 1.6, 0.2, 0.0],
            ],
            dtype=np.float64,
        ),
    ]
    noise_free_category_rdms = [
        np.array([[0.0, 1.7], [1.7, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.4], [1.4, 0.0]], dtype=np.float64),
    ]

    paths = module.save_rdm_figures(
        image_rdms=image_rdms,
        category_rdms=category_rdms,
        noise_free_image_rdms=noise_free_image_rdms,
        noise_free_category_rdms=noise_free_category_rdms,
        layer_names=("conv1", "conv2"),
        labels=labels,
        class_names=("cat", "dog"),
        output_dir=tmp_path,
        rdm_metric="euclidean",
    )

    assert set(paths) == {
        "image_png",
        "image_svg",
        "category_png",
        "category_svg",
        "image_comparison_png",
        "image_comparison_svg",
        "category_comparison_png",
        "category_comparison_svg",
    }
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_save_npz_includes_noise_free_outputs(tmp_path: Path) -> None:
    module = _load_multicat_rdm_module()
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
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    b_true = [
        np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3),
        100.0 + np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3),
    ]
    fmri_patterns = [
        np.arange(4 * 3, dtype=np.float64).reshape(4, 3),
        50.0 + np.arange(4 * 3, dtype=np.float64).reshape(4, 3),
    ]
    category_patterns = [
        np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[2.0, 0.0, -2.0], [-2.0, 0.0, 2.0]], dtype=np.float64),
    ]
    image_rdms = [
        np.array(
            [
                [0.0, 0.1, 1.2, 1.3],
                [0.1, 0.0, 1.1, 1.4],
                [1.2, 1.1, 0.0, 0.2],
                [1.3, 1.4, 0.2, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.0, 0.2, 1.0, 1.5],
                [0.2, 0.0, 1.3, 1.6],
                [1.0, 1.3, 0.0, 0.3],
                [1.5, 1.6, 0.3, 0.0],
            ],
            dtype=np.float64,
        ),
    ]
    category_rdms = [
        np.array([[0.0, 1.6], [1.6, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.4], [1.4, 0.0]], dtype=np.float64),
    ]
    noise_free_fmri_patterns = [np.mean(arr, axis=0) for arr in b_true]
    noise_free_category_patterns = [
        np.array([[10.0, 0.0, -10.0], [-10.0, 0.0, 10.0]], dtype=np.float64),
        np.array([[20.0, 0.0, -20.0], [-20.0, 0.0, 20.0]], dtype=np.float64),
    ]
    noise_free_image_rdms = [
        rdm + 0.05 * (np.ones_like(rdm) - np.eye(rdm.shape[0])) for rdm in image_rdms
    ]
    noise_free_category_rdms = [
        np.array([[0.0, 1.8], [1.8, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.5], [1.5, 0.0]], dtype=np.float64),
    ]

    npz_path = module._save_npz(
        output_dir=tmp_path,
        layer_ids=(2, 5),
        layer_names=("conv1", "conv2"),
        class_names=("cat", "dog"),
        labels=labels,
        image_paths=(
            tmp_path / "cat.png",
            tmp_path / "dog.png",
            tmp_path / "boat.png",
            tmp_path / "car.png",
        ),
        b_true=b_true,
        fmri_patterns=fmri_patterns,
        category_patterns=category_patterns,
        image_rdms=image_rdms,
        category_rdms=category_rdms,
        noise_free_fmri_patterns=noise_free_fmri_patterns,
        noise_free_category_patterns=noise_free_category_patterns,
        noise_free_image_rdms=noise_free_image_rdms,
        noise_free_category_rdms=noise_free_category_rdms,
        category_counts=np.array([2, 2], dtype=np.int64),
        args=args,
    )

    with np.load(npz_path, allow_pickle=True) as data:
        assert data["b_true"].shape == (2, 2, 4, 3)
        assert data["noise_free_fmri_patterns"].shape == (2, 4, 3)
        assert data["noise_free_category_patterns"].shape == (2, 2, 3)
        assert data["noise_free_image_rdms"].shape == (2, 4, 4)
        assert data["noise_free_category_rdms"].shape == (2, 2, 2)
        assert data["rdm_metric"][0] == "euclidean"
        np.testing.assert_allclose(data["b_true"], np.stack(b_true, axis=0))
        np.testing.assert_allclose(
            data["noise_free_fmri_patterns"], np.stack(noise_free_fmri_patterns, axis=0)
        )
        np.testing.assert_array_equal(data["category_counts"], np.array([2, 2], dtype=np.int64))
