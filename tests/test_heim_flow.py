from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

import fisher.heim_flow as heim_flow_mod
from fisher.heim_flow import (
    HeimFlowConfig,
    HeimIterationInput,
    HeimIterationOutput,
    initialize_h2_classifier,
    initialize_h2_euclidean,
    run_heim_flow,
)


class TestHeimFlow(unittest.TestCase):
    def test_initializers_return_square_symmetric_h2(self) -> None:
        rng = np.random.default_rng(123)
        n_bins = 4
        n_per_bin = 20
        x_dim = 3
        x_parts = []
        b_parts = []
        for b in range(n_bins):
            mu = np.array([0.4 * b, -0.2 * b, 0.1 * b], dtype=np.float64)
            x_b = rng.normal(loc=mu, scale=0.2, size=(n_per_bin, x_dim))
            x_parts.append(x_b)
            b_parts.append(np.full(n_per_bin, b, dtype=np.int64))
        x_all = np.vstack(x_parts)
        bin_all = np.concatenate(b_parts)
        n_train = int(0.7 * x_all.shape[0])
        x_train = np.asarray(x_all[:n_train], dtype=np.float64)
        bin_train = np.asarray(bin_all[:n_train], dtype=np.int64)

        h2_e = initialize_h2_euclidean(
            x_all=x_all,
            bin_all=bin_all,
            n_bins=n_bins,
            min_bin_count=5,
        )
        h2_c = initialize_h2_classifier(
            x_train=x_train,
            bin_train=bin_train,
            x_eval=x_all,
            bin_eval=bin_all,
            n_bins=n_bins,
            min_class_count=5,
            random_state=7,
        )
        for h2 in (h2_e, h2_c):
            self.assertEqual(h2.shape, (n_bins, n_bins))
            self.assertTrue(np.allclose(h2, h2.T, equal_nan=True))
            self.assertTrue(np.allclose(np.diag(h2), 0.0))
            finite = np.isfinite(h2)
            self.assertTrue(np.all((h2[finite] >= 0.0) & (h2[finite] <= 1.0)))

    def test_run_heim_flow_updates_h2_and_returns_history(self) -> None:
        rng = np.random.default_rng(7)
        n_bins = 3
        n_per_bin = 12
        x_dim = 2
        x_parts = []
        b_parts = []
        for b in range(n_bins):
            mu = np.array([float(b), -0.5 * float(b)], dtype=np.float64)
            x_b = rng.normal(loc=mu, scale=0.15, size=(n_per_bin, x_dim))
            x_parts.append(x_b)
            b_parts.append(np.full(n_per_bin, b, dtype=np.int64))
        x_all = np.vstack(x_parts)
        bin_all = np.concatenate(b_parts)
        n_train = int(0.7 * x_all.shape[0])
        x_train = np.asarray(x_all[:n_train], dtype=np.float64)
        x_val = np.asarray(x_all[n_train:], dtype=np.float64)
        bin_train = np.asarray(bin_all[:n_train], dtype=np.int64)
        bin_val = np.asarray(bin_all[n_train:], dtype=np.int64)

        def _fake_estimate(payload: HeimIterationInput) -> HeimIterationOutput:
            n = int(payload.theta_state_all.shape[0])
            d = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if int(bin_all[i]) != int(bin_all[j]):
                        d[i, j] = 1.0 + 0.2 * float(payload.iteration)
            return HeimIterationOutput(
                h2_binned=None,
                delta_l_matrix=d,
                h_sym_matrix=None,
                metadata={"iter_dir": payload.output_dir},
            )

        cfg = HeimFlowConfig(
            n_bins=n_bins,
            n_iters=3,
            mds_dim=2,
            init_mode="euclidean",
            min_bin_count=5,
            min_class_count=5,
            convergence_tol=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_heim_flow(
                x_all=x_all,
                x_train=x_train,
                x_validation=x_val,
                bin_all=bin_all,
                bin_train=bin_train,
                bin_validation=bin_val,
                output_root=td,
                cfg=cfg,
                estimate_callback=_fake_estimate,
            )
        self.assertEqual(out.final_h2.shape, (n_bins, n_bins))
        self.assertEqual(out.final_d.shape, (n_bins, n_bins))
        self.assertEqual(out.final_embedding.shape[1], 2)
        self.assertTrue(np.allclose(np.diag(out.final_h2), 0.0))
        self.assertEqual(len(out.history_h2), len(out.history_d))
        self.assertGreaterEqual(len(out.history_h2), 2)
        self.assertEqual(len(out.rel_change_history), len(out.history_embedding))
        self.assertTrue(np.all(np.isfinite(out.final_embedding)))
        self.assertTrue(all(md.get("embedding_method") == "metric_mds" for md in out.iteration_metadata))
        self.assertEqual([bool(md.get("embedding_warm_start")) for md in out.iteration_metadata], [False, True, True])

    def test_run_heim_flow_stops_early_with_positive_convergence_tol(self) -> None:
        rng = np.random.default_rng(9)
        n_bins = 3
        n_per_bin = 10
        x_dim = 2
        x_parts = []
        b_parts = []
        for b in range(n_bins):
            mu = np.array([0.6 * float(b), -0.2 * float(b)], dtype=np.float64)
            x_b = rng.normal(loc=mu, scale=0.2, size=(n_per_bin, x_dim))
            x_parts.append(x_b)
            b_parts.append(np.full(n_per_bin, b, dtype=np.int64))
        x_all = np.vstack(x_parts)
        bin_all = np.concatenate(b_parts)
        n_train = int(0.7 * x_all.shape[0])
        x_train = np.asarray(x_all[:n_train], dtype=np.float64)
        x_val = np.asarray(x_all[n_train:], dtype=np.float64)
        bin_train = np.asarray(bin_all[:n_train], dtype=np.int64)
        bin_val = np.asarray(bin_all[n_train:], dtype=np.int64)

        def _stationary_estimate(payload: HeimIterationInput) -> HeimIterationOutput:
            n = int(payload.theta_state_all.shape[0])
            d = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if int(bin_all[i]) != int(bin_all[j]):
                        d[i, j] = 1.0
            return HeimIterationOutput(
                h2_binned=None,
                delta_l_matrix=d,
                h_sym_matrix=None,
                metadata={"iter_dir": payload.output_dir},
            )

        cfg = HeimFlowConfig(
            n_bins=n_bins,
            n_iters=8,
            mds_dim=2,
            init_mode="euclidean",
            min_bin_count=5,
            min_class_count=5,
            convergence_tol=1e-12,
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_heim_flow(
                x_all=x_all,
                x_train=x_train,
                x_validation=x_val,
                bin_all=bin_all,
                bin_train=bin_train,
                bin_validation=bin_val,
                output_root=td,
                cfg=cfg,
                estimate_callback=_stationary_estimate,
            )
        self.assertLess(len(out.rel_change_history), int(cfg.n_iters))
        self.assertGreaterEqual(len(out.rel_change_history), 2)
        self.assertLessEqual(float(out.rel_change_history[-1]), float(cfg.convergence_tol) + 1e-15)

    def test_metric_mds_warm_starts_from_previous_embedding(self) -> None:
        rng = np.random.default_rng(11)
        n_bins = 3
        n_per_bin = 9
        x_parts = []
        b_parts = []
        for b in range(n_bins):
            x_b = rng.normal(loc=float(b), scale=0.1, size=(n_per_bin, 2))
            x_parts.append(x_b)
            b_parts.append(np.full(n_per_bin, b, dtype=np.int64))
        x_all = np.vstack(x_parts)
        bin_all = np.concatenate(b_parts)
        n_train = int(0.7 * x_all.shape[0])
        x_train = np.asarray(x_all[:n_train], dtype=np.float64)
        x_val = np.asarray(x_all[n_train:], dtype=np.float64)
        bin_train = np.asarray(bin_all[:n_train], dtype=np.int64)
        bin_val = np.asarray(bin_all[n_train:], dtype=np.int64)

        def _stationary_estimate(payload: HeimIterationInput) -> HeimIterationOutput:
            n = int(payload.theta_state_all.shape[0])
            d = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    if int(bin_all[i]) != int(bin_all[j]):
                        d[i, j] = 1.0
            return HeimIterationOutput(h2_binned=None, delta_l_matrix=d, h_sym_matrix=None, metadata={})

        cfg = HeimFlowConfig(
            n_bins=n_bins,
            n_iters=3,
            mds_dim=2,
            init_mode="euclidean",
            min_bin_count=3,
            min_class_count=3,
            convergence_tol=0.0,
        )
        init_inputs: list[np.ndarray] = []

        def _fake_metric_mds(distance: np.ndarray, *, n_components: int, init_embedding: np.ndarray | None = None) -> np.ndarray:
            self.assertIsNotNone(init_embedding)
            init_arr = np.asarray(init_embedding, dtype=np.float64)
            init_inputs.append(np.asarray(init_arr, dtype=np.float64))
            return np.asarray(init_arr + 0.01, dtype=np.float64)

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            heim_flow_mod, "_metric_mds_embedding_from_distance", side_effect=_fake_metric_mds
        ):
            out = run_heim_flow(
                x_all=x_all,
                x_train=x_train,
                x_validation=x_val,
                bin_all=bin_all,
                bin_train=bin_train,
                bin_validation=bin_val,
                output_root=td,
                cfg=cfg,
                estimate_callback=_stationary_estimate,
            )

        self.assertEqual(len(init_inputs), int(cfg.n_iters))
        self.assertTrue(np.allclose(init_inputs[1], init_inputs[0] + 0.01))
        self.assertTrue(np.allclose(init_inputs[2], init_inputs[1] + 0.01))
        self.assertTrue(np.allclose(out.history_embedding[0], init_inputs[0] + 0.01))


if __name__ == "__main__":
    unittest.main()
