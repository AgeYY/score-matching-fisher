from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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
    assert mod.resolve_output_dir(default_args) == repo_root / "data" / "mog_5pr5_n1000"

    args = mod.parse_args(["--n-total", "123", "--pr-dim", "7"])
    out_dir = mod.resolve_output_dir(args)

    assert args.seed == 7
    assert args.train_frac == 0.7
    assert args.device == "cuda"
    assert args.force is False
    assert args.use_cache is False
    assert args.skip_viz is False
    assert out_dir == repo_root / "data" / "mog_5pr7_n123"
    assert mod.native_npz_path(out_dir) == out_dir / "random_mog_categorical.npz"
    assert mod.projected_npz_path(out_dir, pr_dim=7) == out_dir / "random_mog_categorical_pr7.npz"


def test_existing_files_are_skipped_unless_force(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "mog_5pr7_n123"
    out_dir.mkdir()
    mod.native_npz_path(out_dir).write_bytes(b"native")
    mod.projected_npz_path(out_dir, pr_dim=7).write_bytes(b"projected")
    validations: list[tuple[str, Path]] = []

    def fake_validate_native(path: Path, *, n_total: int):
        validations.append(("native", path))
        assert n_total == 123
        return None

    def fake_validate_projected(path: Path, *, n_total: int, pr_dim: int):
        validations.append(("projected", path))
        assert n_total == 123
        assert pr_dim == 7
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
    out_dir = tmp_path / "mog_5pr9_n456"
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
    assert _flag_value(native, "--x-dim") == "2"
    assert _flag_value(native, "--n-total") == "456"
    assert _flag_value(native, "--train-frac") == "0.7"
    assert _flag_value(native, "--seed") == "7"

    assert _flag_value(projected, "--h-dim") == "9"
    assert "--allow-non-randamp-sqrtd" in projected
    assert _flag_value(projected, "--device") == "cuda"
    assert _flag_value(projected, "--seed") == "7"
    assert "--use-cache" in projected
    assert "--skip-viz" in projected
    assert _flag_value(projected, "--pr-train-epochs") == "3"
    assert _flag_value(projected, "--pr-train-samples") == "40"
    assert _flag_value(projected, "--pr-train-batch-size") == "8"
    assert _flag_value(projected, "--pr-hidden1") == "16"
    assert _flag_value(projected, "--pr-hidden2") == "24"
    assert _flag_value(projected, "--cache-dir") == str(tmp_path / "cache")


def test_pr_dim_must_exceed_native_dim() -> None:
    mod = _load_module()
    args = mod.parse_args(["--pr-dim", "2"])

    with pytest.raises(ValueError, match="--pr-dim must be > 2"):
        mod.validate_args(args)
