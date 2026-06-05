from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fisher.alexnet_ecoset_catdog_decoding import (
    DecodingResult,
    export_sampled_images,
    fit_linear_decoders,
    make_train_test_indices,
    save_accuracy_figure,
    stratified_sample_indices,
)


class _FakeFeatures:
    def __init__(self, names):
        self.names = names


class _FakeDataset:
    def __init__(self, labels: list[int], names: list[str]):
        self._labels = labels
        self.features = {"label": _FakeFeatures(names)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx):
        if idx == "label":
            return list(self._labels)
        arr = np.full((12, 12, 3), int(self._labels[idx]) * 100, dtype=np.uint8)
        return {"label": self._labels[idx], "image": Image.fromarray(arr)}


def test_stratified_sample_indices_balanced_reproducible() -> None:
    labels = ["cat"] * 120 + ["dog"] * 110 + ["chair"] * 20

    a = stratified_sample_indices(labels, classes=("cat", "dog"), n_per_class=100, seed=7)
    b = stratified_sample_indices(labels, classes=("cat", "dog"), n_per_class=100, seed=7)

    assert set(a) == {"cat", "dog"}
    assert a["cat"].shape == (100,)
    assert a["dog"].shape == (100,)
    np.testing.assert_array_equal(a["cat"], b["cat"])
    np.testing.assert_array_equal(a["dog"], b["dog"])
    assert np.all(np.asarray(labels, dtype=object)[a["cat"]] == "cat")
    assert np.all(np.asarray(labels, dtype=object)[a["dog"]] == "dog")


def test_make_train_test_indices_is_stratified_80_20() -> None:
    labels = np.array([0] * 100 + [1] * 100, dtype=np.int64)

    train_idx, test_idx = make_train_test_indices(labels, test_size=0.2, seed=3)

    assert train_idx.shape == (160,)
    assert test_idx.shape == (40,)
    assert np.bincount(labels[train_idx]).tolist() == [80, 80]
    assert np.bincount(labels[test_idx]).tolist() == [20, 20]


def test_export_sampled_images_writes_class_folders(tmp_path: Path) -> None:
    ds = _FakeDataset(labels=[0] * 6 + [1] * 6 + [2] * 3, names=["cat", "dog", "chair"])

    samples = export_sampled_images(
        ds,
        image_root=tmp_path,
        classes=("cat", "dog"),
        n_per_class=4,
        seed=0,
    )

    assert len(samples) == 8
    assert sum(s.class_name == "cat" for s in samples) == 4
    assert sum(s.class_name == "dog" for s in samples) == 4
    assert len(list((tmp_path / "cat").glob("*.jpg"))) == 4
    assert len(list((tmp_path / "dog").glob("*.jpg"))) == 4


def test_fit_linear_decoders_returns_accuracy_per_layer() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    signal = labels[:, None].astype(np.float64)
    features = {
        "raw_pixel": np.hstack([signal, rng.normal(scale=0.05, size=(40, 2))]),
        "features.0": np.hstack([signal, rng.normal(scale=0.05, size=(40, 3))]),
        "features.3": np.hstack([signal, rng.normal(scale=0.05, size=(40, 4))]),
    }
    train_idx, test_idx = make_train_test_indices(labels, test_size=0.25, seed=1)

    results = fit_linear_decoders(features, labels, train_idx, test_idx, seed=1)

    assert [r.layer_name for r in results] == ["raw_pixel", "features.0", "features.3"]
    assert all(0.0 <= r.accuracy <= 1.0 for r in results)
    assert all(r.n_train == 30 and r.n_test == 10 for r in results)


def test_save_accuracy_figure_creates_png(tmp_path: Path) -> None:
    results = [
        DecodingResult("raw_pixel", 0.7, 160, 40),
        DecodingResult("features.0", 0.85, 160, 40),
    ]

    path = save_accuracy_figure(results, tmp_path)

    assert path == tmp_path / "catdog_layer_accuracy.png"
    assert path.is_file()
    assert path.stat().st_size > 0
