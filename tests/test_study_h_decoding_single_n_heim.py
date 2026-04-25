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

import study_h_decoding_single_n_heim as single_heim  # noqa: E402


def _expand_hsym_from_binned(h_binned: np.ndarray, bin_all: np.ndarray) -> np.ndarray:
    bi = np.asarray(bin_all, dtype=np.int64).reshape(-1)
    hb = np.asarray(h_binned, dtype=np.float64)
    out = np.zeros((bi.size, bi.size), dtype=np.float64)
    for i in range(bi.size):
        for j in range(bi.size):
            out[i, j] = float(hb[int(bi[i]), int(bi[j])])
    return out


class TestSingleNHeimScript(unittest.TestCase):
    def test_parser_requires_n_and_has_expected_defaults(self) -> None:
        parser = single_heim.build_parser()
        args = parser.parse_args(["--dataset-npz", "dummy.npz", "--n", "200"])
        self.assertEqual(int(args.n), 200)
        self.assertEqual(int(args.heim_flow_max_iters), 20)
        self.assertAlmostEqual(float(args.heim_flow_convergence_tol), 0.02, places=12)

    def test_iteration_visualization_writes_init_and_each_iter(self) -> None:
        n_bins = 3
        n = 6
        bin_all = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        n_train = 2
        bin_validation = np.asarray(bin_all[n_train:], dtype=np.int64)
        h_gt_sqrt = np.asarray(
            [
                [0.0, 0.6, 0.7],
                [0.6, 0.0, 0.5],
                [0.7, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        init_h2 = np.asarray(
            [
                [0.0, 0.16, 0.25],
                [0.16, 0.0, 0.09],
                [0.25, 0.09, 0.0],
            ],
            dtype=np.float64,
        )
        h2_k0 = np.asarray(
            [
                [0.0, 0.20, 0.30],
                [0.20, 0.0, 0.15],
                [0.30, 0.15, 0.0],
            ],
            dtype=np.float64,
        )
        h2_k1 = np.asarray(
            [
                [0.0, 0.22, 0.28],
                [0.22, 0.0, 0.18],
                [0.28, 0.18, 0.0],
            ],
            dtype=np.float64,
        )
        rel_hist = np.asarray([0.12, 0.07], dtype=np.float64)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            np.savez_compressed(
                out_dir / "heim_flow_iterations.npz",
                init_h2=init_h2,
                rel_change_history=rel_hist,
                heim_iters_completed=np.int64(2),
            )
            heim_root = out_dir / "heim_flow"
            (heim_root / "iter_000").mkdir(parents=True, exist_ok=True)
            (heim_root / "iter_001").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                heim_root / "iter_000" / "h_matrix_results_theta_cov.npz",
                h_sym=_expand_hsym_from_binned(h2_k0, bin_all),
            )
            np.savez_compressed(
                heim_root / "iter_001" / "h_matrix_results_theta_cov.npz",
                h_sym=_expand_hsym_from_binned(h2_k1, bin_all),
            )

            rows = single_heim._render_iteration_visualizations(
                output_dir=str(out_dir),
                n=n,
                dataset_family="cosine_gaussian_sqrtd",
                n_bins=n_bins,
                n_train=n_train,
                bin_validation=bin_validation,
                h_gt_sqrt=h_gt_sqrt,
            )

            self.assertEqual(len(rows), 3)  # init + iter_000 + iter_001
            labels = [r["iter_label"] for r in rows]
            self.assertEqual(labels, ["init", "iter_000", "iter_001"])
            self.assertTrue((out_dir / "heim_flow_iteration_metrics.csv").is_file())
            self.assertTrue((out_dir / "heim_flow_iter_viz" / "h_x_flow_heim_flow_iter_init_n_000006.png").is_file())
            self.assertTrue((out_dir / "heim_flow_iter_viz" / "h_x_flow_heim_flow_iter_000_n_000006.png").is_file())
            self.assertTrue((out_dir / "heim_flow_iter_viz" / "h_x_flow_heim_flow_iter_001_n_000006.png").is_file())
            self.assertEqual(rows[0]["fro_rel_change"], "")
            self.assertAlmostEqual(float(rows[1]["fro_rel_change"]), float(rel_hist[0]), places=12)
            self.assertAlmostEqual(float(rows[2]["fro_rel_change"]), float(rel_hist[1]), places=12)

    def test_h_sqrt_from_iter_run_uses_validation_block_only(self) -> None:
        n_bins = 2
        n_train = 2
        bin_validation = np.asarray([0, 1], dtype=np.int64)
        # Build a 4x4 h_sym where train block has large values that must be ignored.
        h_sym = np.asarray(
            [
                [0.0, 1.0, 0.9, 0.9],
                [1.0, 0.0, 0.9, 0.9],
                [0.9, 0.9, 0.0, 0.25],
                [0.9, 0.9, 0.25, 0.0],
            ],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td)
            np.savez_compressed(iter_dir / "h_matrix_results_theta_cov.npz", h_sym=h_sym)
            out = single_heim._h_sqrt_from_iter_run(
                iter_dir=str(iter_dir),
                dataset_family="cosine_gaussian_sqrtd",
                n_train=n_train,
                bin_validation=bin_validation,
                n_bins=n_bins,
            )
        expected_h2 = np.asarray([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
        expected = np.sqrt(expected_h2)
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
