from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_BIN = _REPO / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import study_h_decoding_single_n_heim_theta_flow as single_theta_heim  # noqa: E402


class TestSingleNHeimThetaFlowScript(unittest.TestCase):
    def test_parser_requires_n_and_has_expected_defaults(self) -> None:
        parser = single_theta_heim.build_parser()
        args = parser.parse_args(["--dataset-npz", "dummy.npz", "--n", "200"])
        self.assertEqual(int(args.n), 200)
        self.assertEqual(int(args.heim_flow_max_iters), 20)
        self.assertAlmostEqual(float(args.heim_flow_convergence_tol), 0.02, places=12)
        self.assertEqual(single_theta_heim.HEIM_DISTANCE_TRANSFORM, "bhattacharyya")
        self.assertEqual(str(args.heim_flow_distance_transform), "hellinger")

    def test_validation_binned_matrix_uses_validation_block_only(self) -> None:
        mat = np.asarray(
            [
                [99.0, 99.0, 99.0, 99.0],
                [99.0, 99.0, 99.0, 99.0],
                [99.0, 99.0, 1.0, 2.0],
                [99.0, 99.0, 3.0, 4.0],
            ],
            dtype=np.float64,
        )
        out = single_theta_heim._validation_binned_matrix(
            mat,
            n_train=2,
            bin_validation=np.asarray([0, 1], dtype=np.int64),
            n_bins=2,
        )
        expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)

    def test_bayes_iteration_visualization_writes_csv_and_figures(self) -> None:
        n_bins = 2
        n = 4
        n_train = 2
        bin_validation = np.asarray([0, 1], dtype=np.int64)
        ratio = np.arange(16, dtype=np.float64).reshape(4, 4)
        delta_l = ratio - np.diag(ratio).reshape(-1, 1)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            np.savez_compressed(
                out_dir / "heim_flow_iterations.npz",
                rel_change_history=np.asarray([0.2], dtype=np.float64),
                heim_iters_completed=np.int64(1),
            )
            iter_dir = out_dir / "heim_flow" / "iter_000"
            iter_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                iter_dir / "h_matrix_results_theta_cov.npz",
                c_matrix_ratio=ratio,
                delta_l_matrix=delta_l,
            )

            rows = single_theta_heim._render_bayes_iteration_visualizations(
                output_dir=str(out_dir),
                n=n,
                dataset_family="cosine_gaussian_sqrtd",
                n_bins=n_bins,
                n_train=n_train,
                bin_validation=bin_validation,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ratio_key"], "c_matrix_ratio")
            self.assertTrue((out_dir / "heim_flow_bayes_iteration_metrics.csv").is_file())
            self.assertTrue(
                (out_dir / "heim_flow_iter_viz" / "bayes_theta_flow_heim_flow_iter_000_n_000004.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
