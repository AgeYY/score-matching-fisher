from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD


def _load_script():
    path = Path(__file__).resolve().parents[1] / "bin" / "run_fisher_validation_directions.py"
    spec = importlib.util.spec_from_file_location("run_fisher_validation_directions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_session() -> SimpleNamespace:
    rng = np.random.default_rng(4)
    return SimpleNamespace(
        grating_orientation=np.linspace(0.0, np.pi, 120, endpoint=False),
        neural_responses=rng.normal(size=(120, 14)) + np.linspace(2.0, 8.0, 14),
        session_file=Path("fake_stringer.npy"),
    )


def _args(*, center: bool, whiten: bool) -> SimpleNamespace:
    return SimpleNamespace(
        direction="train-test-allocation",
        stringer_grid_size=17,
        pca_dim=4,
        pca_center=center,
        pca_whiten=whiten,
    )


def test_prepare_stringer_supports_train_fitted_uncentered_projection(tmp_path: Path) -> None:
    module = _load_script()
    session = _fake_session()
    prepared = module._prepare_stringer(
        _args(center=False, whiten=False),
        session=session,
        seed=7,
        train_fraction=0.6,
        validation_fraction=0.1,
        case_dir=tmp_path,
    )

    with np.load(tmp_path / "split_and_pca.npz") as saved:
        train = np.asarray(saved["train_index"], dtype=np.int64)
        assert saved["projection_method"].item() == "truncated_svd_uncentered"
        assert saved["pca_center"].item() is False
        assert saved["pca_whiten"].item() is False
        np.testing.assert_allclose(saved["pca_mean"], 0.0)
        expected = TruncatedSVD(
            n_components=4,
            algorithm="randomized",
            random_state=7,
        ).fit_transform(session.neural_responses[train])

    np.testing.assert_allclose(prepared["x_train"], expected)
    assert prepared["x_train"].shape == (72, 4)
    assert prepared["x_validation"].shape[1] == 4
    assert prepared["x_test"].shape[1] == 4
    assert sum(
        part.shape[0]
        for part in (prepared["x_train"], prepared["x_validation"], prepared["x_test"])
    ) == 120


def test_uncentered_projection_rejects_whitening(tmp_path: Path) -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="does not support whitening"):
        module._prepare_stringer(
            _args(center=False, whiten=True),
            session=_fake_session(),
            seed=7,
            train_fraction=0.6,
            validation_fraction=0.1,
            case_dir=tmp_path,
        )
