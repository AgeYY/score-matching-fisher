from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from fisher.distance_comparison import analytic_diagonal_gaussian_skl_matrix
from fisher.flow_matching_skl import FlowSKLResult
from fisher.llr_divergence import symmetric_kl_gaussian_diag_matrix
from fisher.shared_dataset_io import save_shared_dataset_npz


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "diagnose_mog5_symmetric_kl_endpoint.py"
    spec = importlib.util.spec_from_file_location("diagnose_mog5_symmetric_kl_endpoint", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_native_mog5_npz(path: Path, *, n_total: int = 25, x_dim: int = 2) -> Path:
    labels = np.arange(n_total, dtype=np.int64) % 5
    theta_all = np.eye(5, dtype=np.float64)[labels]
    means = np.asarray([[float(k), 0.5 * float(k)] for k in range(5)], dtype=np.float64)
    if x_dim > 2:
        means = np.pad(means, ((0, 0), (0, x_dim - 2)), mode="constant")
    variances = np.full((5, x_dim), 0.5, dtype=np.float64)
    x_all = means[labels] + 0.01 * np.arange(n_total, dtype=np.float64).reshape(-1, 1)
    train_idx = np.arange(15, dtype=np.int64)
    val_idx = np.arange(15, n_total, dtype=np.int64)
    save_shared_dataset_npz(
        path,
        meta={
            "dataset_family": "random_mog_categorical",
            "theta_type": "categorical",
            "theta_encoding": "one_hot",
            "num_categories": 5,
            "x_dim": x_dim,
            "n_total": n_total,
            "pr_autoencoder_embedded": False,
            "mog_component_means": means.tolist(),
            "mog_component_variances": variances.tolist(),
        },
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=val_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[val_idx],
        x_validation=x_all[val_idx],
    )
    return path


def test_diagonal_gaussian_log_prob_matches_manual_formula() -> None:
    mod = _load_cli_module()
    means = np.array([[1.0, -1.0]], dtype=np.float64)
    variances = np.array([[2.0, 0.5]], dtype=np.float64)
    x = np.array([[1.0, -1.0], [3.0, 0.0]], dtype=np.float64)

    got = mod.diagonal_gaussian_log_prob(x, means, variances, 0)
    expected = -0.5 * (
        2.0 * np.log(2.0 * np.pi)
        + np.log(2.0)
        + np.log(0.5)
        + np.array([0.0, (2.0**2) / 2.0 + (1.0**2) / 0.5], dtype=np.float64)
    )
    np.testing.assert_allclose(got, expected)


def test_mog5_analytic_jeffreys_matches_closed_form_helper() -> None:
    means = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]], dtype=np.float64)
    variances = np.array([[1.0, 2.0], [0.5, 3.0], [2.0, 0.75]], dtype=np.float64)
    np.testing.assert_allclose(
        analytic_diagonal_gaussian_skl_matrix(means, variances),
        2.0 * symmetric_kl_gaussian_diag_matrix(means, variances),
    )


def test_endpoint_diagnostic_smoke_writes_schema(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    native_npz = _write_native_mog5_npz(tmp_path / "random_mog_categorical.npz")
    output_dir = tmp_path / "out"
    gt = np.arange(25, dtype=np.float64).reshape(5, 5)
    gt = gt + gt.T
    np.fill_diagonal(gt, 0.0)

    class FakeModel(torch.nn.Module):
        x_dim = 2

        def forward(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("forward should be mocked out")

    def fake_build_flow_skl_model(**kwargs):
        assert kwargs["velocity_family"] == "nonlinear"
        return FakeModel()

    def fake_estimate_model_symmetric_kl(**kwargs):
        return FlowSKLResult(
            symmetric_kl_matrix=gt + 0.25,
            canonical_metric_matrix=gt + 0.25,
            canonical_metric_name="model_jeffreys_symmetric_kl",
            train_metadata={},
        )

    def fake_sample_model_class(*, theta, n, **kwargs):
        cls = int(np.argmax(theta[0]))
        return np.column_stack(
            [
                np.full(int(n), float(cls), dtype=np.float64),
                np.linspace(0.0, 1.0, int(n), dtype=np.float64),
            ]
        )

    def fake_model_log_prob(*, x, theta, **kwargs):
        cls = int(np.argmax(theta[0]))
        arr = np.asarray(x, dtype=np.float64)
        return -np.sum((arr - float(cls)) ** 2, axis=1)

    monkeypatch.setattr(mod, "require_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(mod, "ensure_dataset", lambda args, dataset_dir: (native_npz, None))
    monkeypatch.setattr(mod, "build_flow_skl_model", fake_build_flow_skl_model)
    monkeypatch.setattr(
        mod,
        "train_flow_skl_model",
        lambda **kwargs: {
            "train_losses": np.array([1.0]),
            "val_losses": np.array([1.5]),
            "val_monitor_losses": np.array([1.5]),
            "best_epoch": 1,
            "best_val_loss": 1.5,
            "stopped_epoch": 1,
            "stopped_early": False,
        },
    )
    monkeypatch.setattr(mod, "estimate_model_symmetric_kl", fake_estimate_model_symmetric_kl)
    monkeypatch.setattr(mod, "sample_model_class", fake_sample_model_class)
    monkeypatch.setattr(mod, "model_log_prob", fake_model_log_prob)

    args = mod.build_parser().parse_args(
        [
            "--n-total",
            "25",
            "--native-x-dim",
            "2",
            "--epochs",
            "1",
            "--endpoint-samples-per-class",
            "8",
            "--logprob-samples-per-class",
            "6",
            "--two-sample-max-points",
            "8",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--skip-plots",
        ]
    )
    paths = mod.run(args)

    assert paths["results_npz"] == output_dir.resolve() / "symmetric_kl_endpoint_diagnostics.npz"
    assert paths["pair_errors_csv"].is_file()
    assert paths["summary_json"].is_file()
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["velocity_family"] == "nonlinear"
    assert summary["exact_logprob_available"] is True

    with np.load(paths["results_npz"], allow_pickle=False) as data:
        assert data["category_mean_error"].shape == (5,)
        assert data["category_diag_variance_error"].shape == (5,)
        assert data["category_covariance_frobenius_error"].shape == (5,)
        assert data["category_classifier_two_sample_auc"].shape == (5,)
        assert data["kl_true_model_by_category"].shape == (5,)
        assert data["kl_model_true_by_category"].shape == (5,)
        assert data["self_jeffreys_by_category"].shape == (5,)
        assert data["pair_indices"].shape == (10, 2)
        assert data["analytic_gt_jeffreys_skl_matrix"].shape == (5, 5)
        assert data["model_sampled_jeffreys_skl_matrix"].shape == (5, 5)
        assert data["true_sampled_model_logratio_jeffreys_skl_matrix"].shape == (5, 5)
