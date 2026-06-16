from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shear_rank_dataset import (
    centered_cosine_feature,
    centered_cosine_nu,
    generate_shear_rank_dataset,
    save_shear_rank_dataset_npz,
    shear_symmetric_kl,
)


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "run_shear_rank_skl_experiment.py"
    spec = importlib.util.spec_from_file_location("run_shear_rank_skl_experiment", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_centered_cosine_nu_matches_monte_carlo() -> None:
    omega = 2.5
    rng = np.random.default_rng(123)
    z = rng.standard_normal(500_000)
    empirical = float(np.var(centered_cosine_feature(z, omega=omega)))
    assert centered_cosine_nu(omega) == pytest.approx(empirical, abs=2e-3)


def test_sign_flip_and_null_true_skl_formulas() -> None:
    omega = 2.5
    nu = centered_cosine_nu(omega)
    amplitude = 0.7
    sign_flip = generate_shear_rank_dataset(
        n_per_condition=12,
        x_dim=10,
        r_star=8,
        amplitude=amplitude,
        omega=omega,
        seed=7,
        q_seed=123,
        train_frac=0.75,
        mode="sign_flip",
    )
    null = generate_shear_rank_dataset(
        n_per_condition=12,
        x_dim=10,
        r_star=8,
        amplitude=amplitude,
        omega=omega,
        seed=7,
        q_seed=123,
        train_frac=0.75,
        mode="null",
    )

    assert sign_flip.true_skl_matrix[0, 1] == pytest.approx(16.0 * nu * amplitude * amplitude)
    assert sign_flip.true_skl_matrix[0, 1] == pytest.approx(
        shear_symmetric_kl(sign_flip.condition_shear_a[0], sign_flip.condition_shear_a[1], nu=nu)
    )
    assert null.true_skl_matrix[0, 1] == pytest.approx(0.0)


def test_generated_rotation_and_shared_npz_extra_arrays(tmp_path: Path) -> None:
    dataset = generate_shear_rank_dataset(
        n_per_condition=10,
        x_dim=12,
        r_star=8,
        seed=9,
        q_seed=99,
        train_frac=0.8,
    )
    q = dataset.q_matrix
    np.testing.assert_allclose(q.T @ q, np.eye(12), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dataset.u_star, q[:, :8], rtol=1e-12, atol=1e-12)
    assert dataset.bundle.theta_all.shape == (20, 2)
    assert dataset.bundle.x_all.shape == (20, 12)
    assert dataset.bundle.theta_train.shape[0] == 16
    assert dataset.bundle.theta_validation.shape[0] == 4

    out = save_shear_rank_dataset_npz(tmp_path / "dataset.npz", dataset)
    loaded = load_shared_dataset_npz(out)
    assert loaded.meta["dataset_family"] == "two_condition_hidden_shear_rank"
    assert loaded.x_all.shape == (20, 12)
    with np.load(out, allow_pickle=False) as data:
        assert "orthogonal_Q" in data.files
        assert "u_star" in data.files
        assert "true_skl_matrix" in data.files
        np.testing.assert_allclose(data["u_star"], q[:, :8])


def test_cli_smoke_aggregates_mocked_model_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    calls: list[tuple[str, int, str]] = []

    def fake_train_one_model(*, dataset, spec, args, device, seed, output_dir):
        del args, device
        calls.append((dataset.bundle.meta["mode"], seed, spec.name))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "flow_matching_skl_results.npz"
        truth = float(dataset.true_skl_matrix[0, 1])
        estimate = truth + (0.1 if spec.name == "affine" else 0.05)
        np.savez_compressed(
            path,
            estimate_skl=np.asarray([estimate], dtype=np.float64),
            true_skl=np.asarray([truth], dtype=np.float64),
            relative_error=np.asarray([abs(estimate - truth) / truth if truth > 0.0 else np.nan]),
            best_epoch=np.asarray([1], dtype=np.int64),
            best_val_loss=np.asarray([0.25], dtype=np.float64),
        )
        return path, {
            "estimate": estimate,
            "true_skl": truth,
            "rel_error": abs(estimate - truth) / truth if truth > 0.0 else float("nan"),
            "best_epoch": 1,
            "best_val_loss": 0.25,
        }

    monkeypatch.setattr(mod, "train_one_model", fake_train_one_model)
    args = mod.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--n-list",
            "4",
            "--n-seeds",
            "2",
            "--ranks",
            "0,2",
            "--no-full",
            "--x-dim",
            "8",
            "--r-star",
            "4",
            "--fixed-n",
            "4",
        ]
    )

    paths = mod.run(args)

    assert len(calls) == 8
    assert paths["results_npz"].is_file()
    assert paths["results_csv"].is_file()
    assert paths["summary_json"].is_file()
    with np.load(paths["results_npz"], allow_pickle=False) as data:
        assert data["estimates"].shape == (2, 1, 2, 2)
        assert data["true_skl"].shape == (2, 1, 2)
        assert data["model_names"].tolist() == ["affine", "rank_2"]
    assert (tmp_path / "shear_rank_dataset_geometry.svg").is_file()
    assert (tmp_path / "shear_rank_null_false_positive.svg").is_file()


def test_cli_defaults_and_parallel_device_parsing() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])
    assert args.n_list == [50, 100, 500, 1000, 2000]
    assert args.n_seeds == 1
    assert args.ranks == [0, 2, 4]
    assert [spec.name for spec in mod.model_specs(args.ranks, include_full=not args.no_full)] == [
        "affine",
        "rank_2",
        "rank_4",
        "full",
    ]
    assert mod._resolve_parallel_devices("") == []
    assert mod._resolve_parallel_devices("0,1") == ["cuda:0", "cuda:1"]
    assert mod._resolve_parallel_devices("cuda:0,cuda:1") == ["cuda:0", "cuda:1"]
    with pytest.raises(ValueError, match="duplicate"):
        mod._resolve_parallel_devices("0,cuda:0")
