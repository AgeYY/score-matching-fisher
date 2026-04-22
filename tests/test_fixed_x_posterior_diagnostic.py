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
            self.assertIn("Model posterior (approx)", svg_text)
            self.assertIn("GT posterior (approx)", svg_text)

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
            self.assertIn("Model posterior (approx)", svg_text)
            self.assertIn("GT posterior (approx)", svg_text)
            self.assertGreaterEqual(svg_text.count("Fixed-$x$ posterior diagnostics"), 2)


if __name__ == "__main__":
    unittest.main()
