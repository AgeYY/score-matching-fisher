"""Smoke test for ``bin/study_h_decoding_twofig.py``."""

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


class TestStudyHDecodingTwoFigSmoke(unittest.TestCase):
    def _make_dataset(self, ds_path: Path, *, n_total: int, seed: int) -> None:
        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
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

    def test_twofig_smoke_multi_method_outputs_and_shapes(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_twofig.py"
        n_total = 160
        n_ref = 120
        n_bins = 5
        seed = 9

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "ds.npz"
            out_dir = tmp_path / "run_out"
            self._make_dataset(ds_path, n_total=n_total, seed=seed)
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
                "40,60",
                "--num-theta-bins",
                str(n_bins),
                "--theta-field-methods",
                "theta_flow,x_flow",
                "--theta-field-method",
                "ctsm_v",
                "--flow-arch",
                "mlp",
                "--flow-epochs",
                "2",
                "--prior-epochs",
                "2",
                "--flow-batch-size",
                "32",
                "--prior-batch-size",
                "32",
                "--flow-hidden-dim",
                "32",
                "--prior-hidden-dim",
                "32",
                "--flow-early-patience",
                "5",
                "--prior-early-patience",
                "5",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))

            self.assertTrue((out_dir / "h_decoding_twofig_sweep.svg").is_file())
            self.assertTrue((out_dir / "h_decoding_twofig_gt.svg").is_file())
            self.assertTrue((out_dir / "h_decoding_twofig_corr_vs_n.svg").is_file())
            self.assertTrue((out_dir / "h_decoding_twofig_training_losses_panel.svg").is_file())
            self.assertTrue((out_dir / "h_decoding_twofig_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_twofig_summary.txt").is_file())
            self.assertTrue((out_dir / "training_losses" / "theta_flow" / "n_000040.npz").is_file())
            self.assertTrue((out_dir / "training_losses" / "theta_flow" / "n_000060.npz").is_file())
            self.assertTrue((out_dir / "training_losses" / "x_flow" / "n_000040.npz").is_file())
            self.assertTrue((out_dir / "training_losses" / "x_flow" / "n_000060.npz").is_file())

            z = np.load(out_dir / "h_decoding_twofig_results.npz", allow_pickle=True)
            self.assertEqual(tuple(z["h_binned_sweep"].shape), (2, 2, n_bins, n_bins))
            self.assertEqual(tuple(z["decode_sweep"].shape), (2, n_bins, n_bins))
            self.assertEqual(tuple(z["corr_h_binned_vs_gt_mc"].shape), (2, 2))
            self.assertEqual(tuple(z["corr_decode_vs_ref_shared"].shape), (2,))
            self.assertEqual(tuple(z["wall_seconds"].shape), (2, 2))
            self.assertEqual(tuple(z["h_gt_sqrt"].shape), (n_bins, n_bins))
            self.assertEqual(tuple(z["decode_ref"].shape), (n_bins, n_bins))
            np.testing.assert_array_equal(z["n"], np.asarray([40, 60], dtype=np.int64))
            np.testing.assert_array_equal(
                z["theta_field_methods"],
                np.asarray(["theta_flow", "x_flow"], dtype=np.str_),
            )
            self.assertIn("training_losses_root", z.files)
            self.assertIn("training_losses_panel_svg", z.files)
            self.assertIn("corr_curve_svg", z.files)

    def test_twofig_backward_compat_single_method_shape(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_twofig.py"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "ds.npz"
            out_dir = tmp_path / "run_out"
            self._make_dataset(ds_path, n_total=160, seed=17)
            cmd = [
                sys.executable,
                str(script),
                "--dataset-npz",
                str(ds_path),
                "--dataset-family",
                "cosine_gaussian_sqrtd",
                "--n-ref",
                "120",
                "--n-list",
                "40,60",
                "--num-theta-bins",
                "5",
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "mlp",
                "--flow-epochs",
                "2",
                "--prior-epochs",
                "2",
                "--flow-batch-size",
                "32",
                "--prior-batch-size",
                "32",
                "--flow-hidden-dim",
                "32",
                "--prior-hidden-dim",
                "32",
                "--flow-early-patience",
                "3",
                "--prior-early-patience",
                "3",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))

            z = np.load(out_dir / "h_decoding_twofig_results.npz", allow_pickle=True)
            self.assertEqual(tuple(z["h_binned_sweep"].shape), (1, 2, 5, 5))
            self.assertEqual(tuple(z["decode_sweep"].shape), (2, 5, 5))
            self.assertEqual(tuple(z["wall_seconds"].shape), (1, 2))
            np.testing.assert_array_equal(z["theta_field_methods"], np.asarray(["theta_flow"], dtype=np.str_))

    def test_twofig_theta_field_rows_supports_method_arch_rows(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_twofig.py"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "ds.npz"
            out_dir = tmp_path / "run_out"
            self._make_dataset(ds_path, n_total=140, seed=21)
            cmd = [
                sys.executable,
                str(script),
                "--dataset-npz",
                str(ds_path),
                "--dataset-family",
                "cosine_gaussian_sqrtd",
                "--n-ref",
                "100",
                "--n-list",
                "40,60",
                "--num-theta-bins",
                "5",
                "--theta-field-rows",
                "theta_flow:mlp,theta_flow:film",
                "--theta-field-method",
                "ctsm_v",
                "--theta-field-methods",
                "x_flow",
                "--flow-epochs",
                "2",
                "--prior-epochs",
                "2",
                "--flow-batch-size",
                "32",
                "--prior-batch-size",
                "32",
                "--flow-hidden-dim",
                "32",
                "--prior-hidden-dim",
                "32",
                "--flow-early-patience",
                "3",
                "--prior-early-patience",
                "3",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))

            self.assertTrue((out_dir / "h_decoding_twofig_sweep.svg").is_file())
            self.assertTrue((out_dir / "h_decoding_twofig_results.npz").is_file())
            z = np.load(out_dir / "h_decoding_twofig_results.npz", allow_pickle=True)
            self.assertEqual(tuple(z["h_binned_sweep"].shape), (2, 2, 5, 5))
            self.assertEqual(tuple(z["decode_sweep"].shape), (2, 5, 5))
            np.testing.assert_array_equal(
                z["theta_field_rows"], np.asarray(["theta_flow:mlp", "theta_flow:film"], dtype=np.str_)
            )
            np.testing.assert_array_equal(
                z["theta_field_row_methods"], np.asarray(["theta_flow", "theta_flow"], dtype=np.str_)
            )
            np.testing.assert_array_equal(
                z["theta_field_row_arches"], np.asarray(["mlp", "film"], dtype=np.str_)
            )

    def test_twofig_loss_panel_mixed_theta_and_x_flow_rows(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_twofig.py"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "ds.npz"
            out_dir = tmp_path / "run_out"
            self._make_dataset(ds_path, n_total=120, seed=31)
            cmd = [
                sys.executable,
                str(script),
                "--dataset-npz",
                str(ds_path),
                "--dataset-family",
                "cosine_gaussian_sqrtd",
                "--n-ref",
                "90",
                "--n-list",
                "40",
                "--num-theta-bins",
                "5",
                "--theta-field-rows",
                "theta_flow:mlp,x_flow:film",
                "--flow-epochs",
                "2",
                "--prior-epochs",
                "2",
                "--flow-batch-size",
                "32",
                "--prior-batch-size",
                "32",
                "--flow-hidden-dim",
                "32",
                "--prior-hidden-dim",
                "32",
                "--flow-early-patience",
                "3",
                "--prior-early-patience",
                "3",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))

            self.assertTrue((out_dir / "h_decoding_twofig_training_losses_panel.svg").is_file())
            tf_loss = out_dir / "training_losses" / "theta_flow_mlp" / "n_000040.npz"
            xf_loss = out_dir / "training_losses" / "x_flow_film" / "n_000040.npz"
            self.assertTrue(tf_loss.is_file())
            self.assertTrue(xf_loss.is_file())

            z_tf = np.load(tf_loss, allow_pickle=True)
            z_xf = np.load(xf_loss, allow_pickle=True)
            self.assertIn("prior_enable", z_tf.files)
            self.assertIn("prior_enable", z_xf.files)
            self.assertTrue(bool(np.asarray(z_tf["prior_enable"]).reshape(-1)[0]))
            self.assertFalse(bool(np.asarray(z_xf["prior_enable"]).reshape(-1)[0]))


if __name__ == "__main__":
    unittest.main()
