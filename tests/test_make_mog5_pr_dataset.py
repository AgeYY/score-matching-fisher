from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fisher.shared_dataset_io import save_shared_dataset_npz
from fisher.shared_dataset_io import load_shared_dataset_npz


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "make_mog5_pr_dataset.py"
    spec = importlib.util.spec_from_file_location("make_mog5_pr_dataset", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_parser_defaults_and_output_path_naming() -> None:
    mod = _load_module()
    repo_root = Path(__file__).resolve().parent.parent

    default_args = mod.parse_args([])
    assert default_args.n_total == 1_000
    assert default_args.native_x_dim == 3
    assert default_args.pr_dim is None
    assert mod.resolve_output_dir(default_args) == repo_root / "data" / "mog_5native_xdim3_n1000"

    args = mod.parse_args(["--n-total", "123", "--pr-dim", "7"])
    out_dir = mod.resolve_output_dir(args)

    assert args.seed == 7
    assert args.train_frac == 0.8
    assert args.obs_noise_scale == pytest.approx(1.0)
    assert args.cov_theta_amp_scale == pytest.approx(1.0)
    assert args.mog_mean_min_dist is None
    assert args.device == "cuda:0"
    assert args.force is False
    assert args.use_cache is False
    assert args.skip_viz is False
    assert out_dir == repo_root / "data" / "mog_5native_xdim3_pr7_n123"
    assert mod.native_npz_path(out_dir) == out_dir / "random_mog_categorical.npz"
    assert mod.projected_npz_path(out_dir, pr_dim=7) == out_dir / "random_mog_categorical_pr7.npz"

    native_args = mod.parse_args(["--n-total", "123", "--native-x-dim", "2", "--pr-dim", "none"])
    assert native_args.pr_dim is None
    assert mod.parse_args(["--pr-dim", "NULL"]).pr_dim is None
    assert mod.resolve_output_dir(native_args) == repo_root / "data" / "mog_5native_n123"
    legacy_pr_args = mod.parse_args(["--n-total", "123", "--native-x-dim", "2", "--pr-dim", "7"])
    assert mod.resolve_output_dir(legacy_pr_args) == repo_root / "data" / "mog_5pr7_n123"

    native3_args = mod.parse_args(["--n-total", "123", "--native-x-dim", "3", "--pr-dim", "none"])
    assert mod.resolve_output_dir(native3_args) == repo_root / "data" / "mog_5native_xdim3_n123"

    native3_pr_args = mod.parse_args(["--n-total", "123", "--native-x-dim", "3", "--pr-dim", "5"])
    assert mod.resolve_output_dir(native3_pr_args) == repo_root / "data" / "mog_5native_xdim3_pr5_n123"


def test_existing_files_are_skipped_unless_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "mog_5pr7_n123"
    out_dir.mkdir()
    mod.native_npz_path(out_dir).write_bytes(b"native")
    mod.projected_npz_path(out_dir, pr_dim=7).write_bytes(b"projected")
    validations: list[tuple[str, Path]] = []

    def fake_validate_native(path: Path, *, n_total: int, native_x_dim: int = 2):
        validations.append(("native", path))
        assert n_total == 123
        assert native_x_dim == 3
        return None

    def fake_validate_projected(path: Path, *, n_total: int, pr_dim: int, native_x_dim: int = 2):
        validations.append(("projected", path))
        assert n_total == 123
        assert pr_dim == 7
        assert native_x_dim == 3
        return None

    monkeypatch.setattr(mod, "validate_native_npz", fake_validate_native)
    monkeypatch.setattr(mod, "validate_projected_npz", fake_validate_projected)

    args = mod.parse_args(["--n-total", "123", "--pr-dim", "7", "--output-dir", str(out_dir)])
    commands: list[list[str]] = []
    mod.run(args, runner=lambda cmd: commands.append(list(cmd)))

    assert commands == []
    assert validations == [
        ("native", out_dir / "random_mog_categorical.npz"),
        ("projected", out_dir / "random_mog_categorical_pr7.npz"),
    ]

    validations.clear()
    force_args = mod.parse_args(
        ["--n-total", "123", "--pr-dim", "7", "--output-dir", str(out_dir), "--force"]
    )
    mod.run(force_args, runner=lambda cmd: commands.append(list(cmd)))

    assert len(commands) == 2
    assert validations == [
        ("native", out_dir / "random_mog_categorical.npz"),
        ("projected", out_dir / "random_mog_categorical_pr7.npz"),
    ]


def test_command_construction_for_native_and_pr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "mog_5native_xdim3_pr9_n456"
    commands: list[list[str]] = []
    monkeypatch.setattr(mod, "validate_native_npz", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "validate_projected_npz", lambda *args, **kwargs: None)

    args = mod.parse_args(
        [
            "--n-total",
            "456",
            "--pr-dim",
            "9",
            "--output-dir",
            str(out_dir),
            "--pr-train-epochs",
            "3",
            "--pr-train-samples",
            "40",
            "--pr-train-batch-size",
            "8",
            "--pr-hidden1",
            "16",
            "--pr-hidden2",
            "24",
            "--pr-cache-dir",
            str(tmp_path / "cache"),
            "--use-cache",
            "--skip-viz",
        ]
    )
    mod.run(args, runner=lambda cmd: commands.append(list(cmd)))

    assert len(commands) == 2
    native, projected = commands
    assert _flag_value(native, "--dataset-family") == "random_mog_categorical"
    assert _flag_value(native, "--num-categories") == "5"
    assert _flag_value(native, "--x-dim") == "3"
    assert _flag_value(native, "--n-total") == "456"
    assert _flag_value(native, "--train-frac") == "0.8"
    assert _flag_value(native, "--obs-noise-scale") == "1.0"
    assert _flag_value(native, "--cov-theta-amp-scale") == "1.0"
    assert _flag_value(native, "--seed") == "7"

    assert _flag_value(projected, "--h-dim") == "9"
    assert "--allow-non-randamp-sqrtd" in projected
    assert _flag_value(projected, "--device") == "cuda:0"
    assert _flag_value(projected, "--seed") == "7"
    assert "--use-cache" in projected
    assert "--skip-viz" in projected
    assert _flag_value(projected, "--pr-train-epochs") == "3"
    assert _flag_value(projected, "--pr-train-samples") == "40"
    assert _flag_value(projected, "--pr-train-batch-size") == "8"
    assert _flag_value(projected, "--pr-hidden1") == "16"
    assert _flag_value(projected, "--pr-hidden2") == "24"
    assert _flag_value(projected, "--cache-dir") == str(tmp_path / "cache")


def test_native_x_dim_three_native_mode_builds_native_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "mog_5native_xdim3_n456"
    commands: list[list[str]] = []
    validations: list[tuple[str, int]] = []

    def fake_validate_native(path: Path, *, n_total: int, native_x_dim: int = 2):
        validations.append(("native", native_x_dim))
        assert path == out_dir / "random_mog_categorical.npz"
        assert n_total == 456
        return None

    def fake_build_project_command(*args, **kwargs):
        raise AssertionError("native mode must not build a projection command")

    monkeypatch.setattr(mod, "validate_native_npz", fake_validate_native)
    monkeypatch.setattr(mod, "validate_projected_npz", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "build_project_command", fake_build_project_command)

    args = mod.parse_args(["--n-total", "456", "--native-x-dim", "3", "--pr-dim", "none", "--output-dir", str(out_dir)])
    native_npz, projected_npz = mod.run(args, runner=lambda cmd: commands.append(list(cmd)))

    assert projected_npz is None
    assert native_npz == out_dir / "random_mog_categorical.npz"
    assert len(commands) == 1
    assert _flag_value(commands[0], "--x-dim") == "3"
    assert validations == [("native", 3)]


def test_native_x_dim_three_pr_mode_threads_projection_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "mog_5native_xdim3_pr5_n456"
    commands: list[list[str]] = []
    validations: list[tuple[str, int | None, int]] = []

    def fake_validate_native(path: Path, *, n_total: int, native_x_dim: int = 2):
        validations.append(("native", None, native_x_dim))
        assert n_total == 456
        return None

    def fake_validate_projected(path: Path, *, n_total: int, pr_dim: int, native_x_dim: int = 2):
        validations.append(("projected", pr_dim, native_x_dim))
        assert n_total == 456
        return None

    monkeypatch.setattr(mod, "validate_native_npz", fake_validate_native)
    monkeypatch.setattr(mod, "validate_projected_npz", fake_validate_projected)

    args = mod.parse_args(["--n-total", "456", "--native-x-dim", "3", "--pr-dim", "5", "--output-dir", str(out_dir)])
    mod.run(args, runner=lambda cmd: commands.append(list(cmd)))

    assert len(commands) == 2
    assert _flag_value(commands[0], "--x-dim") == "3"
    assert _flag_value(commands[1], "--h-dim") == "5"
    assert validations == [("native", None, 3), ("projected", 5, 3)]


def test_native_mode_builds_only_native_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "mog_5native_xdim3_n456"
    commands: list[list[str]] = []
    validations: list[tuple[str, Path]] = []

    def fake_validate_native(path: Path, *, n_total: int, native_x_dim: int = 2):
        validations.append(("native", path))
        assert n_total == 456
        assert native_x_dim == 3
        return None

    def fake_validate_projected(*args, **kwargs):
        raise AssertionError("native mode must not validate a projected NPZ")

    def fake_build_project_command(*args, **kwargs):
        raise AssertionError("native mode must not build a projection command")

    monkeypatch.setattr(mod, "validate_native_npz", fake_validate_native)
    monkeypatch.setattr(mod, "validate_projected_npz", fake_validate_projected)
    monkeypatch.setattr(mod, "build_project_command", fake_build_project_command)

    args = mod.parse_args(["--n-total", "456", "--pr-dim", "none", "--output-dir", str(out_dir)])
    native_npz, projected_npz = mod.run(args, runner=lambda cmd: commands.append(list(cmd)))

    assert projected_npz is None
    assert native_npz == out_dir / "random_mog_categorical.npz"
    assert len(commands) == 1
    assert _flag_value(commands[0], "--output-npz") == str(native_npz)
    assert validations == [("native", native_npz)]


def test_pr_dim_must_be_at_least_native_dim() -> None:
    mod = _load_module()
    valid_args = mod.parse_args(["--native-x-dim", "2", "--pr-dim", "2"])
    mod.validate_args(valid_args)

    invalid_args = mod.parse_args(["--native-x-dim", "2", "--pr-dim", "1"])
    with pytest.raises(ValueError, match="--pr-dim must be >= native x_dim=2"):
        mod.validate_args(invalid_args)

    invalid_native3_args = mod.parse_args(["--native-x-dim", "3", "--pr-dim", "2"])
    with pytest.raises(ValueError, match="--pr-dim must be >= native x_dim=3"):
        mod.validate_args(invalid_native3_args)


def test_projected_validation_uses_native_x_dim_for_pr_autoencoder_z_dim(tmp_path: Path) -> None:
    mod = _load_module()
    k = 5
    n_total = 10
    labels = np.arange(n_total, dtype=np.int64) % k
    theta_all = np.eye(k, dtype=np.float64)[labels]
    x_all = np.zeros((n_total, 5), dtype=np.float64)
    train_idx = np.arange(5, dtype=np.int64)
    validation_idx = np.arange(5, n_total, dtype=np.int64)
    path = tmp_path / "random_mog_categorical_pr5.npz"
    save_shared_dataset_npz(
        path,
        meta={
            "dataset_family": "random_mog_categorical",
            "num_categories": k,
            "x_dim": 5,
            "pr_autoencoder_embedded": True,
            "pr_autoencoder_z_dim": 3,
        },
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )

    mod.validate_projected_npz(path, n_total=n_total, pr_dim=5, native_x_dim=3)
    with pytest.raises(ValueError, match="expected 2"):
        mod.validate_projected_npz(path, n_total=n_total, pr_dim=5, native_x_dim=2)


def _write_template_npz(path: Path, *, x_dim: int = 3, n_total: int = 20) -> Path:
    k = 5
    labels = np.arange(n_total, dtype=np.int64) % k
    theta_all = np.eye(k, dtype=np.float64)[labels]
    gains = np.arange(k * x_dim, dtype=np.float64).reshape(k, x_dim) + 1.0
    means = gains * 0.1
    variances = np.full((k, x_dim), 0.25, dtype=np.float64)
    x_all = means[labels]
    train_idx = np.arange(n_total // 2, dtype=np.int64)
    validation_idx = np.arange(n_total // 2, n_total, dtype=np.int64)
    save_shared_dataset_npz(
        path,
        meta={
            "dataset_family": "random_mog_categorical",
            "theta_type": "categorical",
            "theta_encoding": "one_hot",
            "num_categories": k,
            "x_dim": x_dim,
            "n_total": n_total,
            "train_frac": 0.5,
            "seed": 11,
            "mog_component_gains": gains.tolist(),
            "mog_component_means": means.tolist(),
            "mog_component_variances": variances.tolist(),
        },
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )
    return path


def test_native_template_npz_reuses_components_and_varies_samples(tmp_path: Path) -> None:
    mod = _load_module()
    template = _write_template_npz(tmp_path / "template.npz", x_dim=3)
    out_a = tmp_path / "repeat_a"
    out_b = tmp_path / "repeat_b"

    args_a = mod.parse_args(
        [
            "--n-total",
            "30",
            "--native-x-dim",
            "3",
            "--pr-dim",
            "none",
            "--seed",
            "101",
            "--output-dir",
            str(out_a),
            "--native-template-npz",
            str(template),
        ]
    )
    args_b = mod.parse_args(
        [
            "--n-total",
            "30",
            "--native-x-dim",
            "3",
            "--pr-dim",
            "none",
            "--seed",
            "102",
            "--output-dir",
            str(out_b),
            "--native-template-npz",
            str(template),
        ]
    )
    native_a, _ = mod.run(args_a)
    native_b, _ = mod.run(args_b)

    template_bundle = load_shared_dataset_npz(template)
    bundle_a = load_shared_dataset_npz(native_a)
    bundle_b = load_shared_dataset_npz(native_b)
    for key in ("mog_component_gains", "mog_component_means", "mog_component_variances"):
        np.testing.assert_allclose(bundle_a.meta[key], template_bundle.meta[key])
        np.testing.assert_allclose(bundle_b.meta[key], template_bundle.meta[key])
    assert not np.array_equal(bundle_a.x_all, bundle_b.x_all)
    assert not np.array_equal(bundle_a.train_idx, bundle_b.train_idx)


def test_native_template_npz_rejects_dimension_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    template = _write_template_npz(tmp_path / "template_x2.npz", x_dim=2)
    args = mod.parse_args(
        [
            "--n-total",
            "30",
            "--native-x-dim",
            "3",
            "--pr-dim",
            "none",
            "--output-dir",
            str(tmp_path / "out"),
            "--native-template-npz",
            str(template),
        ]
    )

    with pytest.raises(ValueError, match="expected 3"):
        mod.validate_args(args)
