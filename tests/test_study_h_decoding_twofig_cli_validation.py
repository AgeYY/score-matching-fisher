"""CLI validation tests for ``bin/study_h_decoding_twofig.py``."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args


def _ns(**overrides: object) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_dataset_arguments(p)
    ns = p.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _make_dataset(path: Path, *, n_total: int = 120, family: str = "cosine_gaussian_sqrtd", seed: int = 13) -> None:
    ns_ds = _ns(
        dataset_family=family,
        x_dim=2,
        n_total=n_total,
        train_frac=0.5,
        seed=seed,
    )
    ds = build_dataset_from_args(ns_ds)
    theta_all, x_all = ds.sample_joint(n_total)
    meta = meta_dict_from_args(ns_ds)
    n_train = int(0.5 * n_total)
    n_train = min(max(n_train, 1), n_total - 1)
    tr = np.arange(0, n_train, dtype=np.int64)
    va = np.arange(n_train, n_total, dtype=np.int64)
    save_shared_dataset_npz(
        path,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr,
        validation_idx=va,
        theta_train=theta_all[tr],
        x_train=x_all[tr],
        theta_validation=theta_all[va],
        x_validation=x_all[va],
    )


def test_rejects_n_ref_theta_bin_budget() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-npz",
        str(repo / "AGENTS.md"),
        "--n-ref",
        "5",
        "--n-list",
        "2",
        "--num-theta-bins",
        "10",
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert r.returncode != 0
    assert "n_mc = n_ref // total_theta_bins" in (r.stderr + r.stdout)


def test_rejects_max_n_list_exceeds_n_ref() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.npz"
        _make_dataset(ds_path, n_total=120, family="cosine_gaussian_sqrtd")
        cmd = [
            sys.executable,
            str(script),
            "--dataset-npz",
            str(ds_path),
            "--dataset-family",
            "cosine_gaussian_sqrtd",
            "--n-ref",
            "60",
            "--n-list",
            "80",
            "--num-theta-bins",
            "5",
            "--device",
            "cpu",
        ]
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        assert r.returncode != 0
        assert "Require max(n-list) <= n-ref" in (r.stderr + r.stdout)


def test_rejects_dataset_family_mismatch() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.npz"
        _make_dataset(ds_path, n_total=120, family="cosine_gaussian_sqrtd")
        cmd = [
            sys.executable,
            str(script),
            "--dataset-npz",
            str(ds_path),
            "--dataset-family",
            "randamp_gaussian_sqrtd",
            "--n-ref",
            "80",
            "--n-list",
            "40",
            "--num-theta-bins",
            "5",
            "--device",
            "cpu",
        ]
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        assert r.returncode != 0
        assert "does not match --dataset-family" in (r.stderr + r.stdout)


def test_rejects_invalid_theta_field_methods_token() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-npz",
        str(repo / "AGENTS.md"),
        "--n-ref",
        "10",
        "--n-list",
        "2",
        "--num-theta-bins",
        "5",
        "--theta-field-methods",
        "theta_flow,not_a_method",
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert r.returncode != 0
    assert "--theta-field-method must be one of" in (r.stderr + r.stdout)


def test_rejects_incompatible_flag_method_combo_multi_method() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-npz",
        str(repo / "AGENTS.md"),
        "--n-ref",
        "20",
        "--n-list",
        "5",
        "--num-theta-bins",
        "5",
        "--theta-flow-onehot-state",
        "--theta-field-methods",
        "theta_flow,x_flow",
        "--flow-arch",
        "mlp",
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert r.returncode != 0
    text = r.stderr + r.stdout
    assert "row=x_flow" in text
    assert "--theta-flow-onehot-state requires --theta-field-method theta_flow" in text


def test_rejects_invalid_theta_field_rows_token_format() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-npz",
        str(repo / "AGENTS.md"),
        "--n-ref",
        "10",
        "--n-list",
        "2",
        "--num-theta-bins",
        "5",
        "--theta-field-rows",
        "theta_flow:mlp:film",
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert r.returncode != 0
    assert "expected method or method:arch" in (r.stderr + r.stdout)


def test_rejects_invalid_theta_field_rows_arch() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-npz",
        str(repo / "AGENTS.md"),
        "--n-ref",
        "10",
        "--n-list",
        "2",
        "--num-theta-bins",
        "5",
        "--theta-field-rows",
        "theta_flow:not_an_arch",
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert r.returncode != 0
    assert "--flow-arch must be one of {'mlp','soft_moe','film','film_fourier'}." in (r.stderr + r.stdout)


def test_rejects_theta_field_rows_arch_on_non_flow_method() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-npz",
        str(repo / "AGENTS.md"),
        "--n-ref",
        "10",
        "--n-list",
        "2",
        "--num-theta-bins",
        "5",
        "--theta-field-rows",
        "ctsm_v:film",
        "--device",
        "cpu",
    ]
    r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert r.returncode != 0
    assert "arch suffix is only allowed for flow methods" in (r.stderr + r.stdout)


def test_theta_field_methods_takes_precedence_over_theta_field_method() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.npz"
        out_dir = Path(tmp) / "out"
        _make_dataset(ds_path, n_total=120, family="cosine_gaussian_sqrtd", seed=11)
        cmd = [
            sys.executable,
            str(script),
            "--dataset-npz",
            str(ds_path),
            "--dataset-family",
            "cosine_gaussian_sqrtd",
            "--n-ref",
            "80",
            "--n-list",
            "40",
            "--num-theta-bins",
            "5",
            "--theta-field-method",
            "ctsm_v",
            "--theta-field-methods",
            "theta_flow",
            "--flow-arch",
            "mlp",
            "--flow-epochs",
            "2",
            "--prior-epochs",
            "2",
            "--flow-batch-size",
            "32",
            "--prior-batch-size",
            "32",
            "--flow-hidden-dim",
            "32",
            "--prior-hidden-dim",
            "32",
            "--flow-early-patience",
            "3",
            "--prior-early-patience",
            "3",
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
        ]
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout, r.stderr)
        z = np.load(out_dir / "h_decoding_twofig_results.npz", allow_pickle=True)
        np.testing.assert_array_equal(z["theta_field_methods"], np.asarray(["theta_flow"], dtype=np.str_))


def test_theta_field_rows_takes_precedence_over_methods_and_method() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "bin" / "study_h_decoding_twofig.py"
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.npz"
        out_dir = Path(tmp) / "out"
        _make_dataset(ds_path, n_total=120, family="cosine_gaussian_sqrtd", seed=11)
        cmd = [
            sys.executable,
            str(script),
            "--dataset-npz",
            str(ds_path),
            "--dataset-family",
            "cosine_gaussian_sqrtd",
            "--n-ref",
            "80",
            "--n-list",
            "40",
            "--num-theta-bins",
            "5",
            "--theta-field-method",
            "ctsm_v",
            "--theta-field-methods",
            "x_flow",
            "--theta-field-rows",
            "theta_flow:mlp,theta_flow:film",
            "--flow-epochs",
            "2",
            "--prior-epochs",
            "2",
            "--flow-batch-size",
            "32",
            "--prior-batch-size",
            "32",
            "--flow-hidden-dim",
            "32",
            "--prior-hidden-dim",
            "32",
            "--flow-early-patience",
            "3",
            "--prior-early-patience",
            "3",
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
        ]
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout, r.stderr)
        z = np.load(out_dir / "h_decoding_twofig_results.npz", allow_pickle=True)
        np.testing.assert_array_equal(
            z["theta_field_rows"], np.asarray(["theta_flow:mlp", "theta_flow:film"], dtype=np.str_)
        )
