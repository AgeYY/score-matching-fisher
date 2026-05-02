"""Smoke test for gaussian-network support in ``bin/study_h_decoding_convergence.py``."""

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


class TestStudyHDecodingConvergenceGaussianNetwork(unittest.TestCase):
    def _run_smoke(self, *, method_cli: str, method_stored: str, expected_stdout: str) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                method_cli,
                "--gn-epochs",
                "4",
                "--gn-batch-size",
                "32",
                "--gn-hidden-dim",
                "16",
                "--gn-depth",
                "1",
                "--gn-early-patience",
                "5",
                "--gn-pair-batch-size",
                "2048",
                "--gn-low-rank-dim",
                "1",
                "--gn-ae-latent-dim",
                "1",
                "--gn-ae-epochs",
                "4",
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
            self.assertTrue((out_dir / "training_losses" / "n_000060.npz").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), method_stored)
            self.assertFalse(bool(np.asarray(z["prior_enable"]).reshape(-1)[0]))

    def test_gaussian_network_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network",
            method_stored="gaussian_network",
            expected_stdout="gaussian_network mode predicts mean and precision Cholesky factor",
        )

    def test_gaussian_network_diagonal_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network-diagonal",
            method_stored="gaussian_network_diagonal",
            expected_stdout="gaussian_network_diagonal mode predicts mean and diagonal precision Cholesky factor",
        )

    def test_gaussian_network_diagonal_binned_pca_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network-diagonal-binned-pca",
            method_stored="gaussian_network_diagonal_binned_pca",
            expected_stdout="gaussian_network_diagonal_binned_pca mode fits PCA from theta-binned train means",
        )

    def test_gaussian_network_diagonal_binned_pca_rejects_impossible_rank(self) -> None:
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

    def test_gaussian_network_low_rank_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network-low-rank",
            method_stored="gaussian_network_low_rank",
            expected_stdout="gaussian_network_low_rank mode predicts high-dimensional mean and latent covariance Cholesky factor",
        )

    def test_gaussian_network_autoencoder_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network-autoencoder",
            method_stored="gaussian_network_autoencoder",
            expected_stdout="gaussian_network_autoencoder mode trains a plain x autoencoder first",
        )

    def test_gaussian_network_diagonal_autoencoder_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network-diagonal-autoencoder",
            method_stored="gaussian_network_diagonal_autoencoder",
            expected_stdout="gaussian_network_diagonal_autoencoder mode trains a plain x autoencoder first",
        )

    def test_gaussian_network_diagonal_antoencoder_alias_sweep_smoke(self) -> None:
        self._run_smoke(
            method_cli="gaussian-network-diagonal-antoencoder",
            method_stored="gaussian_network_diagonal_autoencoder",
            expected_stdout="gaussian_network_diagonal_autoencoder mode trains a plain x autoencoder first",
        )

    def test_gaussian_x_flow_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                "gaussian-x-flow",
                "--gxf-epochs",
                "4",
                "--gxf-batch-size",
                "32",
                "--gxf-hidden-dim",
                "16",
                "--gxf-depth",
                "1",
                "--gxf-early-patience",
                "5",
                "--gxf-pair-batch-size",
                "2048",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("gaussian_x_flow mode trains", r.stdout)
            self.assertIn("full covariance Cholesky FM", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "gaussian_x_flow")

    def test_gaussian_x_flow_diagonal_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                "gaussian-x-flow-diagonal",
                "--gxf-epochs",
                "4",
                "--gxf-batch-size",
                "32",
                "--gxf-hidden-dim",
                "16",
                "--gxf-depth",
                "1",
                "--gxf-early-patience",
                "5",
                "--gxf-pair-batch-size",
                "2048",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("gaussian_x_flow_diagonal mode trains", r.stdout)
            self.assertIn("diagonal covariance FM", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "gaussian_x_flow_diagonal")

    def test_linear_x_flow_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                "linear-x-flow",
                "--lxf-epochs",
                "4",
                "--lxf-batch-size",
                "32",
                "--lxf-hidden-dim",
                "16",
                "--lxf-depth",
                "1",
                "--lxf-early-patience",
                "5",
                "--lxf-pair-batch-size",
                "2048",
                "--keep-intermediate",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("linear_x_flow mode trains", r.stdout)
            self.assertIn("shared linear A x plus theta-MLP offset FM", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "linear_x_flow")
            h_path = out_dir / "sweep_runs" / "n_000060" / "h_matrix_results_theta_cov.npz"
            hz = np.load(h_path, allow_pickle=True)
            self.assertTrue(np.isfinite(np.asarray(hz["h_sym"], dtype=np.float64)).all())
            self.assertEqual(
                str(np.asarray(hz["h_eval_scalar_name"]).reshape(-1)[0]),
                "linear_x_flow_analytic_gaussian_hellinger",
            )
            self.assertTrue(bool(np.asarray(hz["lxf_analytic_gaussian_hellinger"]).reshape(-1)[0]))
            self.assertNotIn("c_matrix", hz.files)

    def test_linear_x_flow_schedule_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                "linear-x-flow-schedule",
                "--lxfs-path-schedule",
                "cosine",
                "--lxfs-epochs",
                "4",
                "--lxfs-batch-size",
                "32",
                "--lxfs-hidden-dim",
                "16",
                "--lxfs-depth",
                "1",
                "--lxfs-early-patience",
                "5",
                "--lxfs-pair-batch-size",
                "2048",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("linear_x_flow_schedule mode trains", r.stdout)
            self.assertIn("path_schedule=cosine", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "linear_x_flow_schedule")
            self.assertEqual(str(np.asarray(z["lxfs_path_schedule"]).reshape(-1)[0]), "cosine")

    def test_linear_x_flow_restricted_drift_sweep_smokes(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
        n_train = min(max(n_train, 1), n_total - 1)
        tr = np.arange(0, n_train, dtype=np.int64)
        va = np.arange(n_train, n_total, dtype=np.int64)

        for method_cli, method_stored in (
            ("linear-x-flow-scalar", "linear_x_flow_scalar"),
            ("linear-x-flow-diagonal", "linear_x_flow_diagonal"),
            ("linear-x-flow-diagonal-theta", "linear_x_flow_diagonal_theta"),
            ("linear-x-flow-diagonal-theta-spline", "linear_x_flow_diagonal_theta_spline"),
            ("linear-x-flow-low-rank", "linear_x_flow_low_rank"),
            ("linear-x-flow-low-rank-randb", "linear_x_flow_low_rank_randb"),
            ("linear-x-flow-diagonal-t", "linear_x_flow_diagonal_t"),
        ):
            with self.subTest(method=method_cli), tempfile.TemporaryDirectory() as tmp:
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
                    method_cli,
                    "--lxf-epochs",
                    "4",
                    "--lxf-batch-size",
                    "32",
                    "--lxf-hidden-dim",
                    "16",
                    "--lxf-depth",
                    "1",
                    "--lxf-spline-k",
                    "5",
                    "--lxf-low-rank-dim",
                    "1",
                    "--lxf-early-patience",
                    "5",
                    "--lxf-pair-batch-size",
                    "2048",
                    "--lxfs-epochs",
                    "4",
                    "--lxfs-batch-size",
                    "32",
                    "--lxfs-hidden-dim",
                    "16",
                    "--lxfs-depth",
                    "1",
                    "--lxfs-early-patience",
                    "5",
                    "--lxfs-pair-batch-size",
                    "2048",
                    "--lxfs-quadrature-steps",
                    "8",
                    "--output-dir",
                    str(out_dir),
                    "--device",
                    "cpu",
                ]
                r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
                self.assertIn(f"{method_stored} mode trains", r.stdout)
                self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
                z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
                self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), method_stored)

    def test_nf_reduction_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                "nf-reduction",
                "--nfr-latent-dim",
                "1",
                "--nfr-epochs",
                "3",
                "--nfr-batch-size",
                "32",
                "--nfr-hidden-dim",
                "8",
                "--nfr-context-dim",
                "4",
                "--nfr-transforms",
                "1",
                "--nfr-early-patience",
                "5",
                "--nfr-pair-batch-size",
                "2048",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("nf_reduction mode trains", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "nf_reduction")

    def test_linear_theta_flow_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 220
        n_ref = 160
        n_bins = 4
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
                "linear-theta-flow",
                "--ltf-num-components",
                "3",
                "--ltf-epochs",
                "4",
                "--ltf-batch-size",
                "32",
                "--ltf-hidden-dim",
                "16",
                "--ltf-depth",
                "1",
                "--ltf-early-patience",
                "5",
                "--ltf-pair-batch-size",
                "2048",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("linear_theta_flow mode trains pure FM", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            z = np.load(out_dir / "training_losses" / "n_000060.npz", allow_pickle=True)
            self.assertEqual(str(np.asarray(z["theta_field_method"]).reshape(-1)[0]), "linear_theta_flow")
            self.assertEqual(int(np.asarray(z["ltf_num_components"]).reshape(-1)[0]), 3)


if __name__ == "__main__":
    unittest.main()
