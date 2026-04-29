"""Smoke tests for flow autoencoder methods in ``bin/study_h_decoding_convergence.py``."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import unittest
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


class TestStudyHDecodingConvergenceFlowAutoencoder(unittest.TestCase):
    def _run_smoke(
        self,
        *,
        method_cli: str,
        method_stored: str,
        expected_stdout: str,
        ae_latent_dim: int,
    ) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
        n_list = "60"
        seed = 13

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
        n_train = min(max(n_train, 1), n_total - 1)
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
                n_list,
                "--num-theta-bins",
                str(n_bins),
                "--theta-field-method",
                method_cli,
                "--flow-arch",
                "mlp",
                "--flow-epochs",
                "3",
                "--prior-epochs",
                "3",
                "--flow-batch-size",
                "32",
                "--prior-batch-size",
                "32",
                "--flow-hidden-dim",
                "16",
                "--prior-hidden-dim",
                "16",
                "--flow-depth",
                "1",
                "--prior-depth",
                "1",
                "--flow-early-patience",
                "5",
                "--prior-early-patience",
                "5",
                "--gn-ae-latent-dim",
                str(int(ae_latent_dim)),
                "--gn-ae-epochs",
                "3",
                "--gn-ae-batch-size",
                "32",
                "--gn-ae-hidden-dim",
                "16",
                "--gn-ae-depth",
                "1",
                "--gn-ae-early-patience",
                "5",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn(expected_stdout, r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            loss_path = out_dir / "training_losses" / "n_000060.npz"
            self.assertTrue(loss_path.is_file())
            z = np.load(loss_path, allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), method_stored)
            self.assertIn("ae_train_losses", z.files)
            self.assertGreater(np.asarray(z["ae_train_losses"], dtype=np.float64).size, 0)
            self.assertEqual(int(np.asarray(z["ae_latent_dim"]).reshape(-1)[0]), int(ae_latent_dim))

    def test_theta_flow_autoencoder_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="theta-flow-autoencoder",
            method_stored="theta_flow_autoencoder",
            expected_stdout="theta_flow_autoencoder mode trains x->z autoencoder first",
            ae_latent_dim=2,
        )

    def test_x_flow_autoencoder_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="x-flow-autoencoder",
            method_stored="x_flow_autoencoder",
            expected_stdout="x_flow_autoencoder mode trains x->z autoencoder first",
            ae_latent_dim=1,
        )

    def test_x_flow_pca_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
        seed = 13

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
        n_train = min(max(n_train, 1), n_total - 1)
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
                "60",
                "--num-theta-bins",
                str(n_bins),
                "--theta-field-method",
                "x-flow-pca",
                "--flow-arch",
                "mlp",
                "--flow-epochs",
                "3",
                "--flow-batch-size",
                "32",
                "--flow-hidden-dim",
                "16",
                "--flow-depth",
                "1",
                "--flow-early-patience",
                "5",
                "--flow-pca-dim",
                "1",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("x_flow_pca mode fits PCA from theta-binned train means", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            loss_path = out_dir / "training_losses" / "n_000060.npz"
            self.assertTrue(loss_path.is_file())
            z = np.load(loss_path, allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "x_flow_pca")
            self.assertTrue(bool(np.asarray(z["pca_enabled"]).reshape(-1)[0]))
            self.assertEqual(int(np.asarray(z["flow_pca_dim"]).reshape(-1)[0]), 1)

    def test_x_flow_pca_rejects_impossible_rank(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        spec = importlib.util.spec_from_file_location("study_h_decoding_convergence", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        theta = np.asarray([0.0, 0.1, 1.0, 1.1], dtype=np.float64)
        bins = np.asarray([0, 0, 1, 1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "exceeds available binned-mean PCA rank"):
            mod._fit_binned_mean_pca_projection(
                x_train=x,
                theta_train=theta,
                bin_train=bins,
                x_val=x,
                x_all=x,
                n_bins=2,
                pca_dim=2,
            )


if __name__ == "__main__":
    unittest.main()
