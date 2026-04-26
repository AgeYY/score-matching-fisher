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
    initialize_mean_mahalanobis_distance,
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

    def test_initialize_mean_mahalanobis_distance(self) -> None:
        """Well-separated bin means => unbounded Mahalanobis distance can exceed 1 off-diagonal."""
        rng = np.random.default_rng(4)
        n_bins = 4
        n_per_bin = 25
        x_dim = 2
        centers = np.asarray(
            [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [20.0, 0.0]], dtype=np.float64
        )
        x_parts = []
        b_parts = []
        for b in range(n_bins):
            x_b = rng.normal(loc=centers[b], scale=0.1, size=(n_per_bin, x_dim))
            x_parts.append(x_b)
            b_parts.append(np.full(n_per_bin, b, dtype=np.int64))
        x_all = np.vstack(x_parts)
        bin_all = np.concatenate(b_parts)
        d = initialize_mean_mahalanobis_distance(
            x_all=x_all,
            bin_all=bin_all,
            n_bins=n_bins,
            min_bin_count=5,
        )
        self.assertEqual(d.shape, (n_bins, n_bins))
        self.assertTrue(np.allclose(d, d.T, equal_nan=True))
        self.assertTrue(np.allclose(np.diag(d), 0.0))
        fin = np.isfinite(d) & (np.arange(n_bins)[:, None] != np.arange(n_bins))
        self.assertTrue(np.all(d[fin] >= 0.0))
        self.assertTrue(np.all(np.isfinite(d[fin])))
        # Furthest means should yield distance > 1
        self.assertGreater(float(d[0, 3]), 1.0)

    def test_run_heim_flow_mean_mahalanobis_passes_raw_distance_to_metric_mds(self) -> None:
        """First metric MDS dissimilarity must be the unbounded init matrix, not H^2-derived distance."""
        rng = np.random.default_rng(31)
        n_bins = 3
        n_per_bin = 15
        x_dim = 2
        x_parts = []
        b_parts = []
        for b in range(n_bins):
            mu = np.array([4.0 * b, 0.0], dtype=np.float64)
            x_b = rng.normal(loc=mu, scale=0.1, size=(n_per_bin, x_dim))
            x_parts.append(x_b)
            b_parts.append(np.full(n_per_bin, b, dtype=np.int64))
        x_all = np.vstack(x_parts)
        bin_all = np.concatenate(b_parts)
        n_train = int(0.6 * x_all.shape[0])
        x_train = np.asarray(x_all[:n_train], dtype=np.float64)
        x_val = np.asarray(x_all[n_train:], dtype=np.float64)
        bin_train = np.asarray(bin_all[:n_train], dtype=np.int64)
        bin_val = np.asarray(bin_all[n_train:], dtype=np.int64)

        expected_d = initialize_mean_mahalanobis_distance(
            x_all=x_all,
            bin_all=bin_all,
            n_bins=n_bins,
            min_bin_count=4,
        )

        seen_metric: list[np.ndarray] = []

        def _zero_classical_seed(distance: np.ndarray, *, n_components: int) -> np.ndarray:
            n = int(np.asarray(distance).shape[0])
            return np.zeros((n, int(n_components)), dtype=np.float64)

        def _fake_metric_mds(
            distance: np.ndarray, *, n_components: int, init_embedding: np.ndarray | None = None
        ) -> np.ndarray:
            seen_metric.append(np.asarray(distance, dtype=np.float64))
            n = int(np.asarray(distance).shape[0])
            return np.zeros((n, int(n_components)), dtype=np.float64)

        h2_init = np.asarray(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.0, 0.15],
                [0.2, 0.15, 0.0],
            ],
            dtype=np.float64,
        )

        def _fake_estimate(_payload: HeimIterationInput) -> HeimIterationOutput:
            return HeimIterationOutput(
                h2_binned=np.asarray(h2_init, dtype=np.float64), delta_l_matrix=None, h_sym_matrix=None, metadata={}
            )

        cfg = HeimFlowConfig(
            n_bins=n_bins,
            n_iters=1,
            mds_dim=2,
            init_mode="mean_mahalanobis",
            distance_transform="hellinger",
            min_bin_count=4,
            min_class_count=4,
            convergence_tol=0.0,
        )
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            heim_flow_mod, "_mds_embedding_from_distance", side_effect=_zero_classical_seed
        ), mock.patch.object(heim_flow_mod, "_metric_mds_embedding_from_distance", side_effect=_fake_metric_mds):
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

        self.assertGreaterEqual(len(seen_metric), 1)
        np.testing.assert_allclose(seen_metric[0], expected_d, rtol=0.0, atol=1e-10, equal_nan=True)
        # history_d[0] is the same raw init matrix used for MDS
        np.testing.assert_allclose(out.history_d[0], expected_d, rtol=0.0, atol=1e-10, equal_nan=True)

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

    def test_bhattacharyya_distance_transform_and_mds_input(self) -> None:
        h2 = np.asarray(
            [
                [0.0, 0.25, 1.0],
                [0.25, 0.0, 0.75],
                [1.0, 0.75, 0.0],
            ],
            dtype=np.float64,
        )
        d = heim_flow_mod._distance_from_h2(h2, transform="bhattacharyya")
        expected = -np.log1p(-np.clip(h2, 0.0, np.nextafter(1.0, 0.0)))
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(d, expected, rtol=0.0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(d)))
        self.assertTrue(np.allclose(np.diag(d), 0.0))

        x_all = np.zeros((6, 2), dtype=np.float64)
        bin_all = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        x_train = x_all[:3]
        x_val = x_all[3:]
        bin_train = bin_all[:3]
        bin_val = bin_all[3:]
        h2_init = np.asarray(
            [
                [0.0, 0.25, 0.75],
                [0.25, 0.0, 0.5],
                [0.75, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        expected_init_d = heim_flow_mod._distance_from_h2(h2_init, transform="bhattacharyya")
        seen_metric_distances: list[np.ndarray] = []

        def _fake_estimate(payload: HeimIterationInput) -> HeimIterationOutput:
            return HeimIterationOutput(h2_binned=h2_init, delta_l_matrix=None, h_sym_matrix=None, metadata={})

        def _fake_classical_mds(distance: np.ndarray, *, n_components: int) -> np.ndarray:
            np.testing.assert_allclose(distance, expected_init_d, rtol=0.0, atol=1e-12)
            return np.zeros((3, int(n_components)), dtype=np.float64)

        def _fake_metric_mds(
            distance: np.ndarray,
            *,
            n_components: int,
            init_embedding: np.ndarray | None = None,
        ) -> np.ndarray:
            seen_metric_distances.append(np.asarray(distance, dtype=np.float64))
            return np.zeros((3, int(n_components)), dtype=np.float64)

        cfg = HeimFlowConfig(
            n_bins=3,
            n_iters=1,
            mds_dim=2,
            init_mode="euclidean",
            distance_transform="bhattacharyya",
            min_bin_count=1,
            min_class_count=1,
            convergence_tol=0.0,
        )
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            heim_flow_mod, "initialize_h2_euclidean", return_value=h2_init
        ), mock.patch.object(
            heim_flow_mod, "_mds_embedding_from_distance", side_effect=_fake_classical_mds
        ), mock.patch.object(
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
                estimate_callback=_fake_estimate,
            )

        self.assertEqual(len(seen_metric_distances), 1)
        np.testing.assert_allclose(seen_metric_distances[0], expected_init_d, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(out.final_d, expected_init_d, rtol=0.0, atol=1e-12)
        self.assertEqual(out.iteration_metadata[0].get("distance_transform"), "bhattacharyya")

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
