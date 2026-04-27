"""Smoke tests for branch-conditioned theta-flow convergence mode."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.h_matrix import HMatrixEstimator
from fisher.models import ConditionalThetaFlowVelocity
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args
from fisher.trainers import train_conditional_theta_flow_model


def _load_convergence_module():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_branch", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ns(**overrides: object) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_dataset_arguments(p)
    ns = p.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_validate_rejects_branch_without_theta_flow() -> None:
    mod = _load_convergence_module()
    p = mod.build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            str(Path(__file__).resolve().parent.parent / "pyproject.toml"),
            "--theta-flow-branch-conditioned",
            "--theta-field-method",
            "x_flow",
        ]
    )
    with pytest.raises(ValueError, match="branch-conditioned requires --theta-field-method theta_flow"):
        mod._validate_cli(args)


def test_validate_rejects_branch_with_posterior_only() -> None:
    mod = _load_convergence_module()
    p = mod.build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            str(Path(__file__).resolve().parent.parent / "pyproject.toml"),
            "--theta-flow-branch-conditioned",
            "--theta-flow-posterior-only-likelihood",
            "--theta-field-method",
            "theta_flow",
        ]
    )
    with pytest.raises(ValueError, match="requires the branch prior flow"):
        mod._validate_cli(args)


class TestBranchConditionedThetaFlow(unittest.TestCase):
    def test_branch_prior_uses_conditional_theta_flow_trainer(self) -> None:
        torch.manual_seed(0)
        theta = np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(-1, 1)
        branch = np.eye(4, dtype=np.float64)[np.arange(theta.shape[0]) % 4]
        model = ConditionalThetaFlowVelocity(x_dim=4, hidden_dim=12, depth=1, theta_dim=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_model(
            model=model,
            theta_train=theta[:12],
            x_train=branch[:12],
            theta_val=theta[12:],
            x_val=branch[12:],
            epochs=1,
            batch_size=4,
            lr=1e-3,
            device=torch.device("cpu"),
            log_every=1,
        )
        self.assertEqual(len(out["train_losses"]), 1)

    def test_h_matrix_branch_conditioning_smoke(self) -> None:
        torch.manual_seed(1)
        theta = np.linspace(-1.5, 1.5, 6, dtype=np.float64).reshape(-1, 1)
        x_base = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1)
        edges = np.linspace(-1.5, 1.5, 5, dtype=np.float64)
        branch_idx = np.searchsorted(edges[1:-1], theta.reshape(-1), side="right")
        branch_idx = np.clip(branch_idx, 0, 3)
        branch = np.eye(4, dtype=np.float64)[branch_idx]
        x_aug = np.concatenate([x_base, branch], axis=1)
        post = ConditionalThetaFlowVelocity(x_dim=6, hidden_dim=12, depth=1, theta_dim=1)
        prior = ConditionalThetaFlowVelocity(x_dim=4, hidden_dim=12, depth=1, theta_dim=1)
        log_post_branch = np.log(np.full((theta.shape[0], 4), 0.25, dtype=np.float64))
        log_prior_branch = np.log(np.full(4, 0.25, dtype=np.float64))
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=18,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=4,
            theta_flow_branch_edges=edges,
            theta_flow_branch_log_posterior=log_post_branch,
            theta_flow_branch_log_prior=log_prior_branch,
            theta_flow_branch_base_x_dim=2,
        )
        out = est.run(theta=theta, x=x_aug, restore_original_order=True)
        self.assertTrue(np.isfinite(out.c_matrix).all())
        self.assertIsNotNone(out.theta_flow_branch_correction_matrix)
        self.assertEqual(out.theta_flow_branch_index.shape, (theta.shape[0],))

    def test_branch_conditioned_convergence_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        seed = 11
        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=n_total,
            train_frac=0.5,
            seed=seed,
        )
        ds = build_dataset_from_args(ns_ds)
        theta_all, x_all = ds.sample_joint(n_total)
        meta = meta_dict_from_args(ns_ds)
        n_train = int(0.5 * n_total)
        tr = np.arange(0, n_train, dtype=np.int64)
        va = np.arange(n_train, n_total, dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "ds.npz"
            out_dir = tmp_path / "run_out"
            save_shared_dataset_npz(
                ds_path,
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
            cmd = [
                sys.executable,
                str(script),
                "--dataset-npz",
                str(ds_path),
                "--dataset-family",
                "cosine_gaussian_sqrtd",
                "--n-ref",
                str(n_ref),
                "--n-list",
                "80",
                "--num-theta-bins",
                "5",
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "mlp",
                "--theta-flow-branch-conditioned",
                "--theta-flow-branch-bins",
                "4",
                "--flow-epochs",
                "2",
                "--prior-epochs",
                "2",
                "--flow-batch-size",
                "32",
                "--prior-batch-size",
                "32",
                "--flow-hidden-dim",
                "24",
                "--prior-hidden-dim",
                "24",
                "--flow-early-patience",
                "5",
                "--prior-early-patience",
                "5",
                "--h-batch-size",
                "4096",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("theta_flow branch-conditioned mode enabled", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "training_losses" / "n_000080.npz").is_file())


if __name__ == "__main__":
    unittest.main()
