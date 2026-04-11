"""Tests for gaussian_raw vs von_mises_raw tuning (separate --gauss-* vs --vm-* flags)."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.data import (
    ToyConditionalGaussianDataset,
    ToyConditionalGaussianRandampDataset,
    ToyConditionalGaussianRandampSqrtdDataset,
    ToyConditionalGaussianSqrtdDataset,
    _tuning_centers_uniform_theta,
)
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
        # phi_j = 2*pi*j/d => [0, pi]; cos(0+0)=1, cos(0+pi)=-1
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

    def test_validate_accepts_gaussian_raw(self) -> None:
        ns = _ns(tuning_curve_family="gaussian_raw", gauss_kappa=0.5, gauss_mu_amp=1.0)
        validate_dataset_sample_args(ns)

    def test_validate_rejects_negative_gauss_kappa(self) -> None:
        ns = _ns(tuning_curve_family="gaussian_raw", gauss_kappa=-0.1, gauss_mu_amp=1.0)
        with self.assertRaisesRegex(ValueError, "gauss-kappa"):
            validate_dataset_sample_args(ns)

    def test_validate_accepts_von_mises_raw(self) -> None:
        ns = _ns(tuning_curve_family="von_mises_raw", vm_kappa=0.5, vm_mu_amp=1.0)
        validate_dataset_sample_args(ns)

    def test_build_gaussian_family_gaussian_raw_samples(self) -> None:
        ns = _ns(
            dataset_family="gaussian",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.2,
            gauss_kappa=0.3,
            gauss_omega=0.8,
            n_total=64,
            train_frac=1.0,
            seed=0,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        theta, x = ds.sample_joint(32)
        self.assertEqual(theta.shape, (32, 1))
        self.assertEqual(x.shape, (32, int(ns.x_dim)))
        self.assertTrue(np.isfinite(x).all())

    def test_build_gmm_family_gaussian_raw_samples(self) -> None:
        ns = _ns(
            dataset_family="gmm_non_gauss",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.0,
            gauss_kappa=0.5,
            gauss_omega=1.0,
            n_total=64,
            train_frac=1.0,
            seed=1,
        )
        validate_dataset_sample_args(ns)
        ds = build_dataset_from_args(ns)
        theta, x = ds.sample_joint(20)
        self.assertEqual(theta.shape, (20, 1))
        self.assertEqual(x.shape, (20, int(ns.x_dim)))
        self.assertTrue(np.isfinite(x).all())

    def test_gaussian_raw_ignores_vm_flags(self) -> None:
        t0 = np.array([[0.2]])
        ns1 = _ns(
            dataset_family="gaussian",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.0,
            gauss_kappa=0.4,
            gauss_omega=1.1,
            vm_mu_amp=999.0,
            vm_kappa=50.0,
            vm_omega=-3.0,
            seed=2,
        )
        ns2 = _ns(
            dataset_family="gaussian",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.0,
            gauss_kappa=0.4,
            gauss_omega=1.1,
            vm_mu_amp=1e-6,
            vm_kappa=0.0,
            vm_omega=0.0,
            seed=2,
        )
        d1 = build_dataset_from_args(ns1)
        d2 = build_dataset_from_args(ns2)
        np.testing.assert_allclose(d1.tuning_curve(t0), d2.tuning_curve(t0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(d1.tuning_curve_derivative(t0), d2.tuning_curve_derivative(t0), rtol=0, atol=1e-15)

    def test_von_mises_raw_ignores_gauss_flags(self) -> None:
        t0 = np.array([[-0.15]])
        ns1 = _ns(
            dataset_family="gaussian",
            tuning_curve_family="von_mises_raw",
            vm_mu_amp=1.0,
            vm_kappa=0.7,
            vm_omega=1.0,
            gauss_mu_amp=999.0,
            gauss_kappa=888.0,
            gauss_omega=-2.0,
            seed=3,
        )
        ns2 = _ns(
            dataset_family="gaussian",
            tuning_curve_family="von_mises_raw",
            vm_mu_amp=1.0,
            vm_kappa=0.7,
            vm_omega=1.0,
            gauss_mu_amp=0.01,
            gauss_kappa=0.01,
            gauss_omega=0.01,
            seed=3,
        )
        d1 = build_dataset_from_args(ns1)
        d2 = build_dataset_from_args(ns2)
        np.testing.assert_allclose(d1.tuning_curve(t0), d2.tuning_curve(t0), rtol=0, atol=1e-15)

    def test_gaussian_raw_derivative_matches_finite_difference(self) -> None:
        ns = _ns(
            dataset_family="gaussian",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.0,
            gauss_kappa=0.4,
            gauss_omega=1.1,
            seed=2,
        )
        ds = build_dataset_from_args(ns)
        t0 = np.array([[0.37]])
        h = 1e-6
        d_analytic = ds.tuning_curve_derivative(t0)
        mu_p = ds.tuning_curve(t0 + h)
        mu_m = ds.tuning_curve(t0 - h)
        d_fd = (mu_p - mu_m) / (2.0 * h)
        np.testing.assert_allclose(d_analytic, d_fd, rtol=1e-4, atol=1e-5)

    def test_meta_roundtrip_gaussian_raw(self) -> None:
        ns = _ns(
            dataset_family="gaussian",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.1,
            gauss_kappa=0.25,
            gauss_omega=0.9,
            n_total=10,
            train_frac=1.0,
            seed=42,
        )
        meta = meta_dict_from_args(ns)
        ds = build_dataset_from_meta(meta)
        t0 = np.array([[0.1]])
        ds2 = build_dataset_from_args(ns)
        np.testing.assert_allclose(ds.tuning_curve(t0), ds2.tuning_curve(t0), rtol=0, atol=1e-15)

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
            dataset_family="gaussian_randamp",
            x_dim=4,
            n_total=64,
            train_frac=1.0,
            seed=0,
            randamp_kappa=0.3,
            randamp_omega=0.8,
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
            dataset_family="gaussian_randamp_sqrtd",
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

    def test_meta_roundtrip_gaussian_randamp(self) -> None:
        ns = _ns(
            dataset_family="gaussian_randamp",
            x_dim=4,
            n_total=10,
            train_frac=1.0,
            seed=42,
            randamp_kappa=0.25,
            randamp_omega=0.9,
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
            dataset_family="gaussian_sqrtd",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.2,
            gauss_kappa=0.3,
            gauss_omega=0.8,
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

    def test_meta_roundtrip_gaussian_sqrtd(self) -> None:
        ns = _ns(
            dataset_family="gaussian_sqrtd",
            tuning_curve_family="gaussian_raw",
            gauss_mu_amp=1.1,
            gauss_kappa=0.25,
            gauss_omega=0.9,
            n_total=10,
            train_frac=1.0,
            seed=42,
        )
        meta = meta_dict_from_args(ns)
        self.assertEqual(meta["dataset_family"], "gaussian_sqrtd")
        ds = build_dataset_from_meta(meta)
        self.assertIsInstance(ds, ToyConditionalGaussianSqrtdDataset)
        t0 = np.array([[0.1]])
        ds2 = build_dataset_from_args(ns)
        np.testing.assert_allclose(ds.tuning_curve(t0), ds2.tuning_curve(t0), rtol=0, atol=1e-15)

    def test_npz_save_load_gaussian_sqrtd(self) -> None:
        ns = _ns(
            dataset_family="gaussian_sqrtd",
            tuning_curve_family="cosine",
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
                eval_idx=ev,
                theta_train=theta_all,
                x_train=x_all,
                theta_eval=np.empty((0, 1), dtype=np.float64),
                x_eval=np.empty((0, 4), dtype=np.float64),
            )
            bundle = load_shared_dataset_npz(path)
        self.assertEqual(bundle.meta["dataset_family"], "gaussian_sqrtd")
        ds2 = build_dataset_from_meta(bundle.meta)
        self.assertIsInstance(ds2, ToyConditionalGaussianSqrtdDataset)
        np.testing.assert_allclose(bundle.x_all, x_all)

    def test_npz_save_load_gaussian_randamp(self) -> None:
        ns = _ns(
            dataset_family="gaussian_randamp",
            tuning_curve_family="cosine",
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
                eval_idx=ev,
                theta_train=theta_all,
                x_train=x_all,
                theta_eval=np.empty((0, 1), dtype=np.float64),
                x_eval=np.empty((0, 4), dtype=np.float64),
            )
            bundle = load_shared_dataset_npz(path)
        self.assertEqual(bundle.meta["dataset_family"], "gaussian_randamp")
        ds2 = build_dataset_from_meta(bundle.meta)
        self.assertIsInstance(ds2, ToyConditionalGaussianRandampDataset)
        np.testing.assert_allclose(bundle.x_all, x_all)


if __name__ == "__main__":
    unittest.main()
