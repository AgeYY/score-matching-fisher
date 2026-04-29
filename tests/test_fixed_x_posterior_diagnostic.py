"""Unit test for fixed-x posterior diagnostic writer in ``study_h_decoding_convergence``."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args
from fisher.shared_fisher_est import build_dataset_from_args


def _load_study_module():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
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


class TestFixedXPosteriorDiagnostic(unittest.TestCase):
    def test_training_loss_loader_reads_likelihood_finetune_arrays(self) -> None:
        mod = _load_study_module()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "losses.npz"
            np.savez_compressed(
                p,
                theta_field_method=np.asarray(["theta_flow"], dtype=object),
                prior_enable=np.bool_(True),
                score_train_losses=np.asarray([1.0], dtype=np.float64),
                score_val_losses=np.asarray([1.1], dtype=np.float64),
                score_val_monitor_losses=np.asarray([1.05], dtype=np.float64),
                score_likelihood_finetune_train_losses=np.asarray([0.9, 0.8], dtype=np.float64),
                score_likelihood_finetune_val_losses=np.asarray([0.95, 0.85], dtype=np.float64),
                score_likelihood_finetune_val_monitor_losses=np.asarray([0.95, 0.9], dtype=np.float64),
                prior_train_losses=np.asarray([0.7], dtype=np.float64),
                prior_val_losses=np.asarray([0.8], dtype=np.float64),
                prior_val_monitor_losses=np.asarray([0.75], dtype=np.float64),
                prior_likelihood_finetune_train_losses=np.asarray([0.6], dtype=np.float64),
                prior_likelihood_finetune_val_losses=np.asarray([0.65], dtype=np.float64),
                prior_likelihood_finetune_val_monitor_losses=np.asarray([0.65], dtype=np.float64),
            )
            loaded = mod._load_per_n_training_loss_npz(str(p))
        np.testing.assert_allclose(loaded["score_likelihood_finetune_train_losses"], [0.9, 0.8])
        np.testing.assert_allclose(loaded["prior_likelihood_finetune_train_losses"], [0.6])

    def test_theta_flow_log_weight_selection_prefers_saved_posterior(self) -> None:
        mod = _load_study_module()
        c_row = np.asarray([-2.0, 0.0, 2.0], dtype=np.float64)
        log_prior = np.asarray([3.0, -1.0, 0.5], dtype=np.float64)
        log_post = c_row + log_prior
        with tempfile.TemporaryDirectory() as tmp:
            h_path = Path(tmp) / "h_matrix_results_theta_cov.npz"
            np.savez(
                h_path,
                theta_flow_log_post_matrix=np.vstack([log_post, log_post + 10.0]),
                theta_flow_log_prior_matrix=np.vstack([log_prior, log_prior + 10.0]),
            )
            with np.load(h_path) as z:
                selected, source = mod._model_posterior_log_weights_for_fixed_x(
                    hfm="theta_flow",
                    c_row=c_row,
                    h_npz=z,
                    row=0,
                )
        np.testing.assert_allclose(selected, log_post)
        self.assertEqual(source, "learned posterior log-density")

    def test_theta_flow_old_artifact_falls_back_to_ratio_row(self) -> None:
        mod = _load_study_module()
        c_row = np.asarray([-2.0, 0.0, 2.0], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            h_path = Path(tmp) / "h_matrix_results_theta_cov.npz"
            np.savez(h_path, c_matrix=np.asarray([c_row], dtype=np.float64))
            with np.load(h_path) as z:
                selected, source = mod._model_posterior_log_weights_for_fixed_x(
                    hfm="theta_flow",
                    c_row=c_row,
                    h_npz=z,
                    row=0,
                )
        np.testing.assert_allclose(selected, c_row)
        self.assertEqual(source, "ratio-only fallback")

    def test_writes_png_svg_from_c_matrix(self) -> None:
        mod = _load_study_module()
        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=400,
            train_frac=0.5,
            seed=11,
        )
        ds = build_dataset_from_args(ns_ds)
        meta = meta_dict_from_args(ns_ds)
        n = 32
        rng = np.random.default_rng(0)
        theta_all, x_all = ds.sample_joint(n)
        c = rng.standard_normal((n, n)) * 0.05
        np.fill_diagonal(c, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            out_dir = Path(tmp) / "diag"
            np.savez(
                run_dir / "h_matrix_results_theta_cov.npz",
                c_matrix=c,
                theta_used=np.asarray(theta_all, dtype=np.float64).reshape(-1),
                h_field_method=np.asarray(["theta_flow"], dtype=object),
            )
            out = mod._write_fixed_x_posterior_diagnostic(
                run_dir=str(run_dir),
                persistent_diagnostics_dir=str(out_dir),
                meta=meta,
                perm_seed=7,
                n_subset=32,
                x_aligned=np.asarray(x_all, dtype=np.float64),
            )
            self.assertIsNotNone(out)
            png = out_dir / "theta_flow_single_x_posterior_hist.png"
            svg = out_dir / "theta_flow_single_x_posterior_hist.svg"
            self.assertTrue(png.is_file())
            self.assertTrue(svg.is_file())
            self.assertGreater(png.stat().st_size, 100)
            svg_text = svg.read_text(encoding="utf-8")
            self.assertGreaterEqual(svg_text.count("Fixed-$x$ posterior diagnostics"), 2)

    def test_writes_png_svg_from_theta_flow_log_density_artifact(self) -> None:
        mod = _load_study_module()
        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=400,
            train_frac=0.5,
            seed=17,
        )
        ds = build_dataset_from_args(ns_ds)
        meta = meta_dict_from_args(ns_ds)
        n = 28
        theta_all, x_all = ds.sample_joint(n)
        theta_flat = np.asarray(theta_all, dtype=np.float64).reshape(-1)
        log_prior = -np.log(float(meta["theta_high"]) - float(meta["theta_low"])) * np.ones(n, dtype=np.float64)
        log_post = -0.5 * (theta_flat.reshape(1, -1) - theta_flat.reshape(-1, 1)) ** 2
        c = log_post - log_prior.reshape(1, -1)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            out_dir = Path(tmp) / "diag"
            np.savez(
                run_dir / "h_matrix_results_theta_cov.npz",
                c_matrix=np.asarray(c, dtype=np.float64),
                theta_flow_log_post_matrix=np.asarray(log_post, dtype=np.float64),
                theta_flow_log_prior_matrix=np.repeat(log_prior.reshape(1, -1), repeats=n, axis=0),
                theta_used=theta_flat,
                h_field_method=np.asarray(["theta_flow"], dtype=object),
            )
            out = mod._write_fixed_x_posterior_diagnostic(
                run_dir=str(run_dir),
                persistent_diagnostics_dir=str(out_dir),
                meta=meta,
                perm_seed=3,
                n_subset=n,
                x_aligned=np.asarray(x_all, dtype=np.float64),
            )
            self.assertIsNotNone(out)
            svg = out_dir / "theta_flow_single_x_posterior_hist.svg"
            self.assertTrue(svg.is_file())
            svg_text = svg.read_text(encoding="utf-8")
            self.assertIn("learned posterior log-density", svg_text)

    def test_nf_artifact_with_prior_fields_still_renders(self) -> None:
        mod = _load_study_module()
        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=256,
            train_frac=0.5,
            seed=13,
        )
        ds = build_dataset_from_args(ns_ds)
        meta = meta_dict_from_args(ns_ds)
        n = 24
        rng = np.random.default_rng(1)
        theta_all, x_all = ds.sample_joint(n)
        c_post = rng.standard_normal((n, n)) * 0.03
        log_prior = rng.standard_normal(n) * 0.02
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_nf"
            run_dir.mkdir()
            out_dir = Path(tmp) / "diag_nf"
            np.savez(
                run_dir / "h_matrix_results_theta_cov.npz",
                c_matrix=np.asarray(c_post, dtype=np.float64),
                c_matrix_ratio=np.asarray(c_post - log_prior.reshape(1, -1), dtype=np.float64),
                log_p_theta_prior=np.asarray(log_prior, dtype=np.float64),
                theta_used=np.asarray(theta_all, dtype=np.float64).reshape(-1),
                h_field_method=np.asarray(["nf"], dtype=object),
            )
            out = mod._write_fixed_x_posterior_diagnostic(
                run_dir=str(run_dir),
                persistent_diagnostics_dir=str(out_dir),
                meta=meta,
                perm_seed=5,
                n_subset=n,
                x_aligned=np.asarray(x_all, dtype=np.float64),
            )
            self.assertIsNotNone(out)
            svg = out_dir / "theta_flow_single_x_posterior_hist.svg"
            self.assertTrue(svg.is_file())
            svg_text = svg.read_text(encoding="utf-8")
            self.assertGreaterEqual(svg_text.count("Fixed-$x$ posterior diagnostics"), 2)

    def test_pr_embedded_obs_dim_writes_png_without_gt_posterior_crash(self) -> None:
        """Embedded observation rows (h_dim) vs generative z_dim must not call GT likelihood with wrong shape."""
        mod = _load_study_module()
        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=50,
            n_total=400,
            train_frac=0.5,
            seed=21,
        )
        meta = meta_dict_from_args(ns_ds)
        meta["pr_autoencoder_embedded"] = True
        meta["pr_autoencoder_z_dim"] = 3
        n = 20
        rng = np.random.default_rng(4)
        theta_all = rng.uniform(float(meta["theta_low"]), float(meta["theta_high"]), size=(n, 1)).astype(
            np.float64
        )
        x_all = rng.standard_normal((n, 50))
        c = rng.standard_normal((n, n)) * 0.05
        np.fill_diagonal(c, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_pr"
            run_dir.mkdir()
            out_dir = Path(tmp) / "diag_pr"
            np.savez(
                run_dir / "h_matrix_results_theta_cov.npz",
                c_matrix=c,
                theta_used=np.asarray(theta_all, dtype=np.float64).reshape(-1),
                h_field_method=np.asarray(["theta_flow"], dtype=object),
            )
            out = mod._write_fixed_x_posterior_diagnostic(
                run_dir=str(run_dir),
                persistent_diagnostics_dir=str(out_dir),
                meta=meta,
                perm_seed=9,
                n_subset=n,
                x_aligned=np.asarray(x_all, dtype=np.float64),
            )
            self.assertIsNotNone(out)
            png = out_dir / "theta_flow_single_x_posterior_hist.png"
            self.assertTrue(png.is_file())
            svg_text = (out_dir / "theta_flow_single_x_posterior_hist.svg").read_text(encoding="utf-8")
            self.assertIn("GT posterior n/a: embedded obs. dim", svg_text)


if __name__ == "__main__":
    unittest.main()
