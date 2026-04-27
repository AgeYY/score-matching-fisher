"""Smoke tests for pairwise binary theta-flow mode in ``study_h_decoding_convergence.py``."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_BIN = _REPO / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import study_h_decoding_convergence as conv  # noqa: E402
from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import SharedDatasetBundle, meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args


def _ns(**overrides: object) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_dataset_arguments(p)
    ns = p.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestStudyHDecodingConvergencePairwiseBinThetaFlow(unittest.TestCase):
    def test_pairwise_binary_bundle_labels_and_alignment(self) -> None:
        meta = {"train_frac": 0.5}
        bundle = SharedDatasetBundle(
            meta=meta,
            theta_all=np.zeros((7, 1), dtype=np.float64),
            x_all=np.zeros((7, 2), dtype=np.float64),
            train_idx=np.arange(4, dtype=np.int64),
            validation_idx=np.arange(4, 7, dtype=np.int64),
            theta_train=np.zeros((4, 1), dtype=np.float64),
            x_train=np.asarray([[10, 0], [11, 0], [12, 0], [13, 0]], dtype=np.float64),
            theta_validation=np.zeros((3, 1), dtype=np.float64),
            x_validation=np.asarray([[20, 0], [21, 0], [22, 0]], dtype=np.float64),
        )
        subset = conv.SweepSubset(
            bundle=bundle,
            bin_all=np.asarray([0, 1, 2, 1, 2, 0, 1], dtype=np.int64),
            bin_train=np.asarray([0, 1, 2, 1], dtype=np.int64),
            bin_validation=np.asarray([2, 0, 1], dtype=np.int64),
        )

        pair_bundle, pair_bins = conv._pairwise_binary_bundle(subset, 1, 2)

        np.testing.assert_allclose(pair_bundle.theta_train.reshape(-1), [-1.0, 1.0, -1.0])
        np.testing.assert_allclose(pair_bundle.theta_validation.reshape(-1), [1.0, -1.0])
        np.testing.assert_allclose(pair_bundle.x_all[:, 0], [11.0, 12.0, 13.0, 20.0, 22.0])
        np.testing.assert_array_equal(pair_bins, [1, 2, 1, 2, 1])
        np.testing.assert_array_equal(pair_bundle.train_idx, [0, 1, 2])
        np.testing.assert_array_equal(pair_bundle.validation_idx, [3, 4])

    def test_pairwise_bin_theta_flow_smoke(self) -> None:
        script = _REPO / "bin" / "study_h_decoding_convergence.py"
        n_total = 180
        n_ref = 120
        n_bins = 3
        ns_list = "60"
        seed = 7

        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=n_total,
            train_frac=0.6,
            seed=seed,
        )
        ds = build_dataset_from_args(ns_ds)
        theta_all, x_all = ds.sample_joint(n_total)
        meta = meta_dict_from_args(ns_ds)
        n_train = int(0.6 * n_total)
        n_train = min(max(n_train, 1), n_total - 1)
        tr = np.arange(0, n_train, dtype=np.int64)
        va = np.arange(n_train, n_total, dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "ds.npz"
            out_dir = tmp_path / "run_out"
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
                ns_list,
                "--num-theta-bins",
                str(n_bins),
                "--theta-field-method",
                "pairwise_bin_theta_flow",
                "--pairwise-bin-min-class-count",
                "2",
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
                "24",
                "--prior-hidden-dim",
                "24",
                "--flow-depth",
                "1",
                "--prior-depth",
                "1",
                "--flow-early-patience",
                "4",
                "--prior-early-patience",
                "4",
                "--keep-intermediate",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("pairwise_bin_theta_flow maps each bin pair to labels -1/+1", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            self.assertTrue((out_dir / "training_losses" / "n_000060.npz").is_file())
            pair_npz = out_dir / "sweep_runs" / "n_000060" / "pairwise_bin_theta_flow_results.npz"
            self.assertTrue(pair_npz.is_file())
            z = np.load(pair_npz, allow_pickle=True)
            self.assertEqual(np.asarray(z["h_sym_binned"]).shape, (n_bins, n_bins))
            self.assertEqual(np.asarray(z["delta_l_binned"]).shape, (n_bins, n_bins))


if __name__ == "__main__":
    unittest.main()
