from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from fisher.binned_theta_flow import (
    assign_theta_bins,
    assemble_log_pi_pairwise_ls,
    compute_h_from_c_matrix,
    empirical_bin_priors,
    expected_calibration_error,
    make_equal_width_theta_bins,
    make_soft_theta_bins,
    mixture_log_density_matrix,
    multiclass_metrics_from_log_pi,
    normalize_theta_in_bins,
    soft_theta_responsibilities,
    unordered_bin_pairs,
)


class BinnedThetaFlowHelpersTest(unittest.TestCase):
    def test_equal_width_bins_and_local_normalization(self) -> None:
        theta = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        spec = make_equal_width_theta_bins(theta, 2)
        np.testing.assert_allclose(spec.edges, [-2.0, 0.0, 2.0])
        labels = assign_theta_bins(theta, spec)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1])
        u = normalize_theta_in_bins(theta, labels, spec).reshape(-1)
        np.testing.assert_allclose(u, [0.0, 0.5, 0.0, 0.5, 1.0])

    def test_classifier_ece_shape_and_perfect_confidence(self) -> None:
        probs = np.asarray(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        self.assertAlmostEqual(expected_calibration_error(probs, labels, n_bins=5), 0.15)

    def test_h_from_c_matrix_has_zero_diagonal_and_is_symmetric(self) -> None:
        c = np.asarray(
            [
                [0.0, 1.0, -0.5],
                [0.3, 0.0, 0.7],
                [-0.2, 0.4, 0.0],
            ],
            dtype=np.float64,
        )
        delta_l, h_sym = compute_h_from_c_matrix(c)
        np.testing.assert_allclose(np.diag(delta_l), 0.0)
        np.testing.assert_allclose(np.diag(h_sym), 0.0)
        np.testing.assert_allclose(h_sym, h_sym.T)
        self.assertTrue(np.all(h_sym >= 0.0))

    def test_soft_theta_responsibilities_rows_sum_to_one(self) -> None:
        th = np.linspace(-1.0, 2.0, 12)
        centers = np.array([-1.0, 2.0], dtype=np.float64)
        r = soft_theta_responsibilities(th, centers, 0.4)
        np.testing.assert_allclose(r.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(r >= 0.0))

    def test_make_soft_theta_bins_default_sigma(self) -> None:
        theta = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        spec = make_soft_theta_bins(theta, 2, center_mode="uniform", sigma=None)
        self.assertEqual(spec.n_experts, 2)
        np.testing.assert_allclose(spec.centers, [-1.0, 1.0])
        np.testing.assert_allclose(spec.sigma, 0.5)

    def test_mixture_log_density_matrix_matches_manual_logsumexp(self) -> None:
        log_pi = np.log(np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64))
        q0 = np.array([[0.0, -2.0], [-3.0, 0.0]], dtype=np.float64)
        q1 = np.array([[-1.0, 0.0], [0.0, -4.0]], dtype=np.float64)
        c = mixture_log_density_matrix(log_pi=log_pi, log_q_experts=[q0, q1])
        stack = np.stack(
            [log_pi[:, k][:, None] + [q0, q1][k] for k in range(2)],
            axis=0,
        )
        m = np.max(stack, axis=0)
        manual = m + np.log(np.sum(np.exp(stack - m), axis=0))
        np.testing.assert_allclose(c, manual)

    def test_soft_cross_entropy_reduces_with_perfect_soft_targets(self) -> None:
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        self.assertLess(float(loss.item()), 1e-3)

    def test_unordered_bin_pairs_count(self) -> None:
        self.assertEqual(len(unordered_bin_pairs(2)), 1)
        self.assertEqual(len(unordered_bin_pairs(4)), 6)

    def test_assemble_log_pi_pairwise_ls_k3_noise_free(self) -> None:
        # True logits (gauge ℓ_0 = 0): [0, 1, -0.5]
        # Pair order (0,1), (0,2), (1,2): s = ℓ_a - ℓ_b
        s_row = np.asarray([[-1.0, 0.5, 1.5]], dtype=np.float64)
        log_pi = assemble_log_pi_pairwise_ls(s_row, K=3, ridge=1e-12, pair_weights=None)
        self.assertEqual(log_pi.shape, (1, 3))
        pr = np.exp(log_pi - np.max(log_pi, axis=1, keepdims=True))
        pr = pr / np.sum(pr, axis=1, keepdims=True)
        ell = np.log(pr + 1e-300)
        ell = ell - ell[:, :1]
        np.testing.assert_allclose(ell[0], [0.0, 1.0, -0.5], rtol=1e-5, atol=1e-5)

    def test_multiclass_metrics_from_log_pi_perfect(self) -> None:
        raw = np.asarray([[2.0, 0.0, -10.0], [-10.0, 2.0, -10.0]], dtype=np.float64)
        m = np.max(raw, axis=1, keepdims=True)
        lp = raw - m - np.log(np.sum(np.exp(raw - m), axis=1, keepdims=True))
        labels = np.asarray([0, 1], dtype=np.int64)
        met = multiclass_metrics_from_log_pi(lp, labels)
        self.assertGreater(met["val_accuracy"], 0.99)

    def test_empirical_bin_priors(self) -> None:
        p = empirical_bin_priors(np.asarray([0, 0, 1], dtype=np.int64), K=2)
        np.testing.assert_allclose(p, [2.0 / 3.0, 1.0 / 3.0])


if __name__ == "__main__":
    unittest.main()
