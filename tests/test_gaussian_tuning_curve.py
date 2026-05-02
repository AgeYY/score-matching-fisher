"""Tests for Gaussian tuning curves (direct dataset classes) and dataset-family-only CLI."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.data import (
    ToyConditionalGaussianCosineRandampSqrtdDataset,
    ToyConditionalGaussianDataset,
    ToyConditionalGaussianGridcos2DSqrtdDataset,
    ToyConditionalGaussianRandamp2DSqrtdDataset,
    ToyConditionalGaussianRandampDataset,
    ToyConditionalGaussianRandampSqrtdDataset,
    ToyConditionalGaussianSqrtdDataset,
    _tuning_centers_uniform_theta,
)
from fisher.dataset_family_recipes import (
    assert_no_legacy_dataset_cli_flags,
    family_recipe_dict,
    format_resolved_family_summary,
)
from fisher.pr_autoencoder_embedding import build_randamp_gaussian_sqrtd_pr_autoencoder_dataset
from fisher.shared_dataset_io import load_shared_dataset_npz, meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, build_dataset_from_meta, validate_dataset_sample_args


def _ns(**overrides: object) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)
    ns = parser.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestGaussianTuningCurve(unittest.TestCase):
    def test_uniform_theta_centers_vm_and_gaussian(self) -> None:
        ds = ToyConditionalGaussianDataset(
            tuning_curve_family="gaussian_raw",
            theta_low=-2.0,
            theta_high=4.0,
            x_dim=5,
            seed=0,
        )
        expected = np.linspace(-2.0, 4.0, 5, dtype=np.float64)
        np.testing.assert_allclose(ds._tuning_centers_theta, expected)
        np.testing.assert_allclose(_tuning_centers_uniform_theta(-2.0, 4.0, 5), expected)

    def test_cosine_tuning_unchanged_formula(self) -> None:
        ds = ToyConditionalGaussianDataset(tuning_curve_family="cosine", x_dim=2, seed=0)
        t0 = np.array([[0.0]])
        mu = ds.tuning_curve(t0)
        np.testing.assert_allclose(mu[0], np.array([1.0, -1.0]), rtol=0, atol=1e-15)

    def test_gaussian_peak_at_uniform_centers(self) -> None:
        ds = ToyConditionalGaussianDataset(
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.25,
            gauss_kappa=0.4,
            gauss_omega=0.9,
            theta_low=-6.0,
            theta_high=6.0,
            x_dim=3,
            seed=0,
        )
        for j in range(3):
            t = np.array([[float(ds._tuning_centers_theta[j])]])
            mu = ds.tuning_curve(t)
            self.assertAlmostEqual(float(mu[0, j]), 1.25, places=12)

    def test_gaussian_raw_derivative_matches_finite_difference(self) -> None:
        ds = ToyConditionalGaussianDataset(
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.0,
            gauss_kappa=0.4,
            gauss_omega=1.1,
            seed=2,
        )
        t0 = np.array([[0.37]])
        h = 1e-6
        d_analytic = ds.tuning_curve_derivative(t0)
        mu_p = ds.tuning_curve(t0 + h)
        mu_m = ds.tuning_curve(t0 - h)
        d_fd = (mu_p - mu_m) / (2.0 * h)
        np.testing.assert_allclose(d_analytic, d_fd, rtol=1e-4, atol=1e-5)

    def test_cli_build_dataset_uses_cosine_recipe(self) -> None:
        ns = _ns(dataset_family="cosine_gaussian", n_total=64, train_frac=1.0, seed=0, x_dim=3)
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertEqual(ds.tuning_curve_family, "cosine")
        theta, x = ds.sample_joint(32)
        self.assertEqual(theta.shape, (32, 1))
        self.assertEqual(x.shape, (32, 3))
        self.assertTrue(np.isfinite(x).all())

    def test_cli_build_dataset_cosine_const_noise(self) -> None:
        ns = _ns(dataset_family="cosine_gaussian_const_noise", n_total=64, train_frac=1.0, seed=0, x_dim=3)
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianDataset)
        self.assertAlmostEqual(float(ds.cov_theta_amp1), 0.0, places=12)
        self.assertAlmostEqual(float(ds.cov_theta_amp2), 0.0, places=12)
        theta, x = ds.sample_joint(32)
        self.assertEqual(theta.shape, (32, 1))
        self.assertEqual(x.shape, (32, 3))
        self.assertTrue(np.isfinite(x).all())

    def test_cosine_const_noise_scales_are_theta_independent(self) -> None:
        ns = _ns(dataset_family="cosine_gaussian_const_noise", n_total=64, train_frac=1.0, seed=0, x_dim=4)
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        theta_grid = np.linspace(float(ns.theta_low), float(ns.theta_high), 21, dtype=np.float64).reshape(-1, 1)
        scales = ds.covariance_scales(theta_grid)
        ref = scales[0:1, :]
        np.testing.assert_allclose(scales, np.repeat(ref, scales.shape[0], axis=0), rtol=0, atol=1e-12)

    def test_legacy_cli_flag_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            assert_no_legacy_dataset_cli_flags(["--tuning-curve-family", "gaussian_raw"])
        self.assertIn("Removed CLI option", str(ctx.exception))

    def test_sqrtd_covariance_scales_by_sqrt_dim(self) -> None:
        d = 10
        g = ToyConditionalGaussianDataset(
            x_dim=d,
            seed=0,
            sigma_x1=0.3,
            sigma_x2=0.3,
            tuning_curve_family="gaussian_raw",
        )
        gs = ToyConditionalGaussianSqrtdDataset(
            x_dim=d,
            seed=0,
            sigma_x1=0.3,
            sigma_x2=0.3,
            tuning_curve_family="gaussian_raw",
        )
        t0 = np.array([[0.0]])
        sg = g.covariance_scales(t0)
        sgs = gs.covariance_scales(t0)
        np.testing.assert_allclose(sgs / sg, float(np.sqrt(d)), rtol=1e-7)

    def test_gaussian_randamp_same_seed_same_amps(self) -> None:
        d1 = ToyConditionalGaussianRandampDataset(x_dim=5, seed=123)
        d2 = ToyConditionalGaussianRandampDataset(x_dim=5, seed=123)
        np.testing.assert_allclose(d1._randamp_amp, d2._randamp_amp)

    def test_build_gaussian_randamp_family_samples(self) -> None:
        ns = _ns(
            dataset_family="randamp_gaussian",
            x_dim=4,
            n_total=64,
            train_frac=1.0,
            seed=0,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianRandampDataset)
        theta, x = ds.sample_joint(32)
        self.assertEqual(theta.shape, (32, 1))
        self.assertEqual(x.shape, (32, 4))
        self.assertTrue(np.isfinite(x).all())

    def test_build_gaussian_randamp_sqrtd_family_samples(self) -> None:
        ns = _ns(
            dataset_family="randamp_gaussian_sqrtd",
            x_dim=4,
            n_total=32,
            train_frac=1.0,
            seed=1,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianRandampSqrtdDataset)
        theta, x = ds.sample_joint(16)
        self.assertEqual(x.shape, (16, 4))
        self.assertTrue(np.isfinite(x).all())

    def test_build_gaussian_randamp_sqrtd_low_x_dim(self) -> None:
        ns = _ns(
            dataset_family="randamp_gaussian_sqrtd",
            x_dim=2,
            n_total=32,
            train_frac=1.0,
            seed=1,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianRandampSqrtdDataset)
        self.assertEqual(int(ds.x_dim), 2)

    def test_pr_autoencoder_embedding_shape_and_seed_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ns = _ns(
                dataset_family="randamp_gaussian_sqrtd",
                x_dim=12,
                n_total=48,
                train_frac=1.0,
                seed=9,
            )
            ns.pr_autoencoder_z_dim = 2
            ns.device = "cpu"
            ns.pr_autoencoder_train_samples = 512
            ns.pr_autoencoder_train_epochs = 6
            ns.pr_autoencoder_train_batch_size = 128
            ns.pr_autoencoder_cache_dir = tmp
            validate_dataset_sample_args(ns)
            out1 = build_randamp_gaussian_sqrtd_pr_autoencoder_dataset(ns)
            out2 = build_randamp_gaussian_sqrtd_pr_autoencoder_dataset(ns)
            self.assertEqual(out1.theta_all.shape, (48, 1))
            self.assertEqual(out1.x_embed_all.shape, (48, 12))
            self.assertTrue(np.isfinite(out1.x_embed_all).all())
            np.testing.assert_allclose(out1.theta_all, out2.theta_all, rtol=0, atol=0)
            np.testing.assert_allclose(out1.x_embed_all, out2.x_embed_all, rtol=0, atol=0)

    def test_meta_roundtrip_gaussian_randamp_sqrtd_pr_embedded_uses_zdim(self) -> None:
        ns_low = _ns(
            dataset_family="randamp_gaussian_sqrtd",
            x_dim=2,
            n_total=10,
            train_frac=1.0,
            seed=11,
        )
        validate_dataset_sample_args(ns_low)
        ds0 = build_dataset_from_args(ns_low)
        ns_meta = _ns(
            dataset_family="randamp_gaussian_sqrtd",
            x_dim=16,
            n_total=10,
            train_frac=1.0,
            seed=11,
        )
        validate_dataset_sample_args(ns_meta)
        meta = meta_dict_from_args(ns_meta)
        meta["pr_autoencoder_enabled"] = True
        meta["pr_autoencoder_embedded"] = True
        meta["pr_autoencoder_z_dim"] = 2
        meta["pr_autoencoder_hidden1"] = 100
        meta["pr_autoencoder_hidden2"] = 200
        meta["pr_autoencoder_train_samples"] = 12000
        meta["pr_autoencoder_train_epochs"] = 200
        meta["pr_autoencoder_train_batch_size"] = 512
        meta["pr_autoencoder_train_lr"] = 1e-3
        meta["pr_autoencoder_lambda_pr"] = 1e-2
        meta["pr_autoencoder_pr_eps"] = 1e-8
        meta["pr_autoencoder_seed"] = int(ns_meta.seed)
        meta["pr_autoencoder_cache_key"] = "pr_ae_dummy"
        meta["randamp_mu_amp_per_dim"] = ds0._randamp_amp.tolist()
        ds = build_dataset_from_meta(meta)
        self.assertIsInstance(ds, ToyConditionalGaussianRandampSqrtdDataset)
        self.assertEqual(int(ds.x_dim), 2)

    def test_meta_roundtrip_cosine_gaussian_sqrtd_pr_embedded_uses_zdim(self) -> None:
        ns_low = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=10,
            train_frac=1.0,
            seed=11,
        )
        validate_dataset_sample_args(ns_low)
        ds0 = build_dataset_from_args(ns_low)
        ns_meta = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=50,
            n_total=10,
            train_frac=1.0,
            seed=11,
        )
        validate_dataset_sample_args(ns_meta)
        meta = meta_dict_from_args(ns_meta)
        meta["pr_autoencoder_enabled"] = True
        meta["pr_autoencoder_embedded"] = True
        meta["pr_autoencoder_z_dim"] = 2
        ds = build_dataset_from_meta(meta)
        self.assertIsInstance(ds, ToyConditionalGaussianSqrtdDataset)
        self.assertEqual(int(ds.x_dim), 2)
        t0 = np.array([[0.1]])
        np.testing.assert_allclose(ds.tuning_curve(t0), ds0.tuning_curve(t0), rtol=0, atol=1e-15)

    def test_removed_pr_autoencoder_dataset_family_meta_raises(self) -> None:
        ns = _ns(dataset_family="randamp_gaussian_sqrtd", x_dim=2, n_total=10, train_frac=1.0, seed=0)
        meta = meta_dict_from_args(ns)
        meta["dataset_family"] = "randamp_gaussian_sqrtd_pr_autoencoder"
        with self.assertRaises(ValueError) as ctx:
            build_dataset_from_meta(meta)
        self.assertIn("no longer supported", str(ctx.exception))

    def test_meta_roundtrip_gaussian_randamp(self) -> None:
        ns = _ns(
            dataset_family="randamp_gaussian",
            x_dim=4,
            n_total=10,
            train_frac=1.0,
            seed=42,
        )
        validate_dataset_sample_args(ns)
        ds0 = build_dataset_from_args(ns)
        meta = meta_dict_from_args(ns)
        meta["randamp_mu_amp_per_dim"] = ds0._randamp_amp.tolist()
        ds = build_dataset_from_meta(meta)
        self.assertIsInstance(ds, ToyConditionalGaussianRandampDataset)
        t0 = np.array([[0.1]])
        np.testing.assert_allclose(ds.tuning_curve(t0), ds0.tuning_curve(t0), rtol=0, atol=1e-15)

    def test_build_gaussian_sqrtd_family_samples(self) -> None:
        ns = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=4,
            n_total=64,
            train_frac=1.0,
            seed=0,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianSqrtdDataset)
        theta, x = ds.sample_joint(32)
        self.assertEqual(theta.shape, (32, 1))
        self.assertEqual(x.shape, (32, int(ns.x_dim)))
        self.assertTrue(np.isfinite(x).all())

    def test_family_recipe_cosine_sqrtd_rand_tune_uses_randamp_gain_bounds(self) -> None:
        r = family_recipe_dict("cosine_gaussian_sqrtd_rand_tune")
        self.assertAlmostEqual(float(r["cosine_tune_amp_low"]), 0.2, places=12)
        self.assertAlmostEqual(float(r["cosine_tune_amp_high"]), 2.0, places=12)
        self.assertAlmostEqual(float(r["cov_theta_amp1"]), 0.70, places=12)
        self.assertAlmostEqual(float(r["cov_theta_amp2"]), 0.60, places=12)
        self.assertEqual(r["cosine_sqrtd_obs_var_mu_law"], "legacy_multiplicative_sqrtd")
        r_add = family_recipe_dict("cosine_gaussian_sqrtd_rand_tune_additive")
        self.assertAlmostEqual(float(r_add["cov_theta_amp1"]), 0.70, places=12)
        self.assertAlmostEqual(float(r_add["cov_theta_amp2"]), 0.60, places=12)

    def test_format_resolved_family_summary_rand_tune_includes_gain_range(self) -> None:
        ns = _ns(dataset_family="cosine_gaussian_sqrtd_rand_tune")
        validate_dataset_sample_args(ns)
        summary = format_resolved_family_summary(ns)
        self.assertIn("0.2", summary)
        self.assertIn("2.0", summary)
        self.assertIn("randamp-style", summary)
        self.assertIn("legacy multiplicative", summary)
        self.assertIn("cosine_tune_amp_scale=1.0", summary)

    def test_family_recipe_cosine_sqrtd_rand_tune_variance_laws(self) -> None:
        r_leg = family_recipe_dict("cosine_gaussian_sqrtd_rand_tune")
        r_add = family_recipe_dict("cosine_gaussian_sqrtd_rand_tune_additive")
        self.assertEqual(r_leg["cosine_sqrtd_obs_var_mu_law"], "legacy_multiplicative_sqrtd")
        self.assertEqual(r_add["cosine_sqrtd_obs_var_mu_law"], "additive_abs_mu")

    def test_cosine_rand_tune_additive_vs_legacy_variance_diagonal(self) -> None:
        from fisher.data import RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE, RANDAMP_SQRTD_VAR_MU_LAW_LEGACY

        ns_a = _ns(
            dataset_family="cosine_gaussian_sqrtd_rand_tune_additive",
            x_dim=4,
            n_total=8,
            train_frac=1.0,
            seed=41,
        )
        validate_dataset_sample_args(ns_a)
        meta_a = meta_dict_from_args(ns_a)
        meta_a["cosine_tune_amp_per_dim"] = [1.0, 1.0, 1.0, 1.0]
        ds_a = build_dataset_from_meta(meta_a)
        self.assertEqual(ds_a.cosine_sqrtd_obs_var_mu_law, RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE)

        ns_l = _ns(
            dataset_family="cosine_gaussian_sqrtd_rand_tune",
            x_dim=4,
            n_total=8,
            train_frac=1.0,
            seed=41,
        )
        validate_dataset_sample_args(ns_l)
        meta_l = meta_dict_from_args(ns_l)
        meta_l["cosine_tune_amp_per_dim"] = [1.0, 1.0, 1.0, 1.0]
        ds_l = build_dataset_from_meta(meta_l)
        self.assertEqual(ds_l.cosine_sqrtd_obs_var_mu_law, RANDAMP_SQRTD_VAR_MU_LAW_LEGACY)

        t0 = np.array([[0.3]])
        mu_a = ds_a.tuning_curve(t0)
        mu_l = ds_l.tuning_curve(t0)
        np.testing.assert_allclose(mu_a, mu_l, rtol=0, atol=1e-15)
        v_a = ds_a._variance_diag_from_mu(mu_a)
        v_l = ds_l._variance_diag_from_mu(mu_l)
        self.assertFalse(np.allclose(v_a, v_l, rtol=0, atol=1e-9))

    def test_cosine_sqrtd_rand_tune_gains_in_range_and_reproducible(self) -> None:
        ns = _ns(
            dataset_family="cosine_gaussian_sqrtd_rand_tune",
            x_dim=6,
            n_total=32,
            train_frac=1.0,
            seed=202,
        )
        validate_dataset_sample_args(ns)
        ds1 = build_dataset_from_args(ns)
        self.assertIsInstance(ds1, ToyConditionalGaussianCosineRandampSqrtdDataset)
        amps = ds1._cosine_tune_amp
        self.assertEqual(amps.shape, (6,))
        self.assertTrue(np.all(amps >= 0.2 - 1e-12))
        self.assertTrue(np.all(amps <= 2.0 + 1e-12))
        ds2 = build_dataset_from_args(ns)
        np.testing.assert_allclose(ds1._cosine_tune_amp, ds2._cosine_tune_amp, rtol=0, atol=0)

    def test_cosine_tune_amp_scale_doubles_drawn_gains(self) -> None:
        base = dict(
            dataset_family="cosine_gaussian_sqrtd_rand_tune_additive",
            x_dim=5,
            n_total=8,
            train_frac=1.0,
            seed=99,
        )
        ns1 = _ns(**base)
        setattr(ns1, "cosine_tune_amp_scale", 1.0)
        validate_dataset_sample_args(ns1)
        ds1 = build_dataset_from_args(ns1)
        ns2 = _ns(**base)
        setattr(ns2, "cosine_tune_amp_scale", 2.0)
        validate_dataset_sample_args(ns2)
        ds2 = build_dataset_from_args(ns2)
        np.testing.assert_allclose(ds2._cosine_tune_amp, 2.0 * ds1._cosine_tune_amp, rtol=0, atol=0)

    def test_cosine_gaussian_sqrtd_not_random_gain_variant(self) -> None:
        ns = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=3,
            n_total=16,
            train_frac=1.0,
            seed=7,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianSqrtdDataset)
        self.assertNotIsInstance(ds, ToyConditionalGaussianCosineRandampSqrtdDataset)

    def test_meta_roundtrip_gaussian_sqrtd(self) -> None:
        ns = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=4,
            n_total=10,
            train_frac=1.0,
            seed=42,
        )
        meta = meta_dict_from_args(ns)
        self.assertEqual(meta["dataset_family"], "cosine_gaussian_sqrtd")
        ds = build_dataset_from_meta(meta)
        self.assertIsInstance(ds, ToyConditionalGaussianSqrtdDataset)
        t0 = np.array([[0.1]])
        ds2 = build_dataset_from_args(ns)
        np.testing.assert_allclose(ds.tuning_curve(t0), ds2.tuning_curve(t0), rtol=0, atol=1e-15)

    def test_npz_save_load_gaussian_sqrtd(self) -> None:
        ns = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=4,
            n_total=20,
            train_frac=1.0,
            seed=7,
        )
        ds = build_dataset_from_args(ns)
        theta_all, x_all = ds.sample_joint(20)
        meta = meta_dict_from_args(ns)
        tr = np.arange(20, dtype=np.int64)
        ev = np.array([], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.npz"
            save_shared_dataset_npz(
                path,
                meta=meta,
                theta_all=theta_all,
                x_all=x_all,
                train_idx=tr,
                validation_idx=ev,
                theta_train=theta_all,
                x_train=x_all,
                theta_validation=np.empty((0, 1), dtype=np.float64),
                x_validation=np.empty((0, 4), dtype=np.float64),
            )
            bundle = load_shared_dataset_npz(path)
        self.assertEqual(bundle.meta["dataset_family"], "cosine_gaussian_sqrtd")
        ds2 = build_dataset_from_meta(bundle.meta)
        self.assertIsInstance(ds2, ToyConditionalGaussianSqrtdDataset)
        np.testing.assert_allclose(bundle.x_all, x_all)

    def test_npz_save_load_gaussian_randamp(self) -> None:
        ns = _ns(
            dataset_family="randamp_gaussian",
            x_dim=4,
            n_total=20,
            train_frac=1.0,
            seed=7,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        theta_all, x_all = ds.sample_joint(20)
        meta = meta_dict_from_args(ns)
        meta["randamp_mu_amp_per_dim"] = ds._randamp_amp.tolist()
        tr = np.arange(20, dtype=np.int64)
        ev = np.array([], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.npz"
            save_shared_dataset_npz(
                path,
                meta=meta,
                theta_all=theta_all,
                x_all=x_all,
                train_idx=tr,
                validation_idx=ev,
                theta_train=theta_all,
                x_train=x_all,
                theta_validation=np.empty((0, 1), dtype=np.float64),
                x_validation=np.empty((0, 4), dtype=np.float64),
            )
            bundle = load_shared_dataset_npz(path)
        self.assertEqual(bundle.meta["dataset_family"], "randamp_gaussian")
        ds2 = build_dataset_from_meta(bundle.meta)
        self.assertIsInstance(ds2, ToyConditionalGaussianRandampDataset)
        np.testing.assert_allclose(bundle.x_all, x_all)

    def test_legacy_dataset_family_meta_rejected(self) -> None:
        ns = _ns(dataset_family="cosine_gaussian", n_total=64, train_frac=1.0, seed=0, x_dim=3)
        meta = meta_dict_from_args(ns)
        meta["dataset_family"] = "gaussian"
        with self.assertRaises(ValueError) as ctx:
            build_dataset_from_meta(meta)
        self.assertIn("cosine_gaussian", str(ctx.exception))

    def test_randamp_gaussian2d_sqrtd_samples_gradients_and_roundtrip(self) -> None:
        ns = _ns(dataset_family="randamp_gaussian2d_sqrtd", x_dim=5, n_total=32, train_frac=1.0, seed=123)
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianRandamp2DSqrtdDataset)
        theta, x = ds.sample_joint(16)
        self.assertEqual(theta.shape, (16, 2))
        self.assertEqual(x.shape, (16, 5))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(ds._variance_diag_from_mu(ds.tuning_curve(theta)) > 0.0))

        t0 = np.array([[0.4, -0.7]], dtype=np.float64)
        h = 1e-6
        grad = ds.tuning_curve_derivative(t0)
        for k in range(2):
            step = np.zeros_like(t0)
            step[0, k] = h
            fd = (ds.tuning_curve(t0 + step) - ds.tuning_curve(t0 - step)) / (2.0 * h)
            np.testing.assert_allclose(grad[:, :, k], fd, rtol=1e-4, atol=1e-5)

        meta = meta_dict_from_args(ns)
        meta["randamp_mu_amp_per_dim"] = ds._randamp_amp.tolist()
        meta["randamp_center_per_dim"] = ds._randamp_centers_2d.tolist()
        ds2 = build_dataset_from_meta(meta)
        self.assertEqual(int(meta["theta_dim"]), 2)
        self.assertIsInstance(ds2, ToyConditionalGaussianRandamp2DSqrtdDataset)
        np.testing.assert_allclose(ds2._randamp_amp, ds._randamp_amp)
        np.testing.assert_allclose(ds2._randamp_centers_2d, ds._randamp_centers_2d)

    def test_gridcos_gaussian2d_sqrtd_samples_gradients_and_roundtrip(self) -> None:
        ns = _ns(
            dataset_family="gridcos_gaussian2d_sqrtd_rand_tune_additive",
            x_dim=5,
            n_total=32,
            train_frac=1.0,
            seed=321,
            obs_noise_scale=0.5,
            cov_theta_amp_scale=2.0,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        self.assertIsInstance(ds, ToyConditionalGaussianGridcos2DSqrtdDataset)
        theta, x = ds.sample_joint(16)
        self.assertEqual(theta.shape, (16, 2))
        self.assertEqual(x.shape, (16, 5))
        self.assertTrue(np.isfinite(ds.covariance_scales(theta)).all())

        t0 = np.array([[0.2, 0.9]], dtype=np.float64)
        h = 1e-6
        grad = ds.tuning_curve_derivative(t0)
        for k in range(2):
            step = np.zeros_like(t0)
            step[0, k] = h
            fd = (ds.tuning_curve(t0 + step) - ds.tuning_curve(t0 - step)) / (2.0 * h)
            np.testing.assert_allclose(grad[:, :, k], fd, rtol=1e-4, atol=1e-5)

        meta = meta_dict_from_args(ns)
        meta["cosine_tune_amp_per_dim"] = ds._cosine_tune_amp.tolist()
        meta["gridcos_orientation_per_dim"] = ds._gridcos_orientation.tolist()
        meta["gridcos_phase_per_dim"] = ds._gridcos_phase.tolist()
        meta["gridcos_omega_per_dim"] = ds._gridcos_omega.tolist()
        ds2 = build_dataset_from_meta(meta)
        self.assertEqual(int(meta["theta_dim"]), 2)
        self.assertIsInstance(ds2, ToyConditionalGaussianGridcos2DSqrtdDataset)
        np.testing.assert_allclose(ds2._cosine_tune_amp, ds._cosine_tune_amp)
        np.testing.assert_allclose(ds2._gridcos_phase, ds._gridcos_phase)


if __name__ == "__main__":
    unittest.main()
