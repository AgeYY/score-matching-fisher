"""Smoke test for ``--visualization-only`` in ``bin/study_h_decoding_convergence.py``."""

from __future__ import annotations

import argparse
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


class TestStudyHDecodingConvergenceVizOnly(unittest.TestCase):
    def test_visualization_only_regenerates_artifacts(self) -> None:
        """Cached results NPZ + training_losses/*.npz -> figures without training."""
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 2000
        n_ref = 1000
        ns_list = [80, 160]
        n_bins = 3
        seed = 7

        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=4,
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
            out_dir = tmp_path / "run_out"
            out_dir.mkdir()
            edges = np.linspace(-1.0, 1.0, n_bins + 1, dtype=np.float64)
            centers = 0.5 * (edges[:-1] + edges[1:])
            n_mc = n_ref // n_bins
            h_cols = np.full((len(ns_list) + 1, n_bins, n_bins), 0.5, dtype=np.float64)
            clf_cols = np.full((len(ns_list) + 1, n_bins, n_bins), 0.6, dtype=np.float64)
            corr_h = np.array([0.9, 0.91], dtype=np.float64)
            corr_clf = np.array([0.8, 0.82], dtype=np.float64)
            corr_llr = np.array([0.4, 0.41], dtype=np.float64)
            wall_s = np.array([1.0, 2.0], dtype=np.float64)
            llr_gt = np.full((n_bins, n_bins), 0.12, dtype=np.float64)
            llr_cols = np.full((len(ns_list) + 1, n_bins, n_bins), 0.12, dtype=np.float64)
            results_npz = out_dir / "h_decoding_convergence_results.npz"
            np.savez_compressed(
                results_npz,
                n=np.asarray(ns_list, dtype=np.int64),
                corr_h_binned_vs_gt_mc=corr_h,
                corr_clf_vs_ref=corr_clf,
                corr_llr_binned_vs_gt_mc=corr_llr,
                wall_seconds=wall_s,
                n_ref=np.int64(n_ref),
                perm_seed=np.int64(seed),
                convergence_base_seed=np.int64(seed),
                dataset_meta_seed=np.int64(seed),
                theta_bin_edges=edges,
                theta_bin_centers=centers,
                hellinger_gt_sq_mc=np.full((n_bins, n_bins), 0.5, dtype=np.float64),
                gt_hellinger_n_mc=np.int64(n_mc),
                gt_hellinger_n_ref_budget=np.int64(n_ref),
                gt_hellinger_seed=np.int64(seed),
                gt_hellinger_symmetrize=np.int32(0),
                h_binned_ref_is_gt_mc=np.int32(1),
                h_binned_columns=h_cols,
                clf_acc_columns=clf_cols,
                column_n=np.asarray(ns_list + [n_ref], dtype=np.int64),
                gt_mean_llr_one_sided_mc=llr_gt,
                llr_binned_columns=llr_cols,
            )
            loss_dir = out_dir / "training_losses"
            loss_dir.mkdir()
            for n in ns_list:
                p = loss_dir / f"n_{int(n):06d}.npz"
                np.savez_compressed(
                    p,
                    theta_field_method=np.array("theta_flow"),
                    prior_enable=np.array(True),
                    score_train_losses=np.array([0.1], dtype=np.float64),
                    score_val_losses=np.array([0.2], dtype=np.float64),
                    score_val_monitor_losses=np.array([0.15], dtype=np.float64),
                    prior_train_losses=np.array([0.05], dtype=np.float64),
                    prior_val_losses=np.array([0.06], dtype=np.float64),
                    prior_val_monitor_losses=np.array([0.055], dtype=np.float64),
                )
            perm = np.random.default_rng(seed).permutation(n_total)
            for n in ns_list:
                sub = perm[: int(n)]
                run_dir = out_dir / "sweep_runs" / f"n_{int(n):06d}"
                run_dir.mkdir(parents=True)
                theta_used = np.asarray(theta_all[sub], dtype=np.float64).reshape(-1)
                c = np.zeros((int(n), int(n)), dtype=np.float64)
                log_prior = np.zeros(int(n), dtype=np.float64)
                np.savez(
                    run_dir / "h_matrix_results_theta_cov.npz",
                    c_matrix=c,
                    theta_flow_log_post_matrix=c + log_prior.reshape(1, -1),
                    theta_flow_log_prior_matrix=np.repeat(log_prior.reshape(1, -1), repeats=int(n), axis=0),
                    theta_used=theta_used,
                    h_field_method=np.asarray(["theta_flow"], dtype=object),
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
                "80,160",
                "--num-theta-bins",
                str(n_bins),
                "--output-dir",
                str(out_dir),
                "--visualization-only",
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertTrue(
                "visualization-only" in r.stdout.lower() or "Saved (visualization-only)" in r.stdout
            )
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_matrices_panel.png").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence_combined.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            summary_path = out_dir / "h_decoding_convergence_summary.txt"
            self.assertTrue(summary_path.is_file())
            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertIn("embedded_fixed_x_diagnostic_pngs", summary_text)
            for n in ns_list:
                diag_png = (
                    out_dir
                    / "sweep_runs"
                    / f"n_{int(n):06d}"
                    / "diagnostics"
                    / "theta_flow_single_x_posterior_hist.png"
                )
                self.assertTrue(diag_png.is_file())
                self.assertIn(f"n={int(n)}:", summary_text)


if __name__ == "__main__":
    unittest.main()
