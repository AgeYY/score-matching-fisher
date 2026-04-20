"""Smoke tests for Fourier theta-flow state in ``bin/study_h_decoding_convergence.py``."""

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


class TestStudyHDecodingConvergenceFourierThetaFlowSmoke(unittest.TestCase):
    def test_theta_flow_fourier_state_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 260
        n_ref = 200
        n_bins = 6
        ns_list = "80"
        seed = 11

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
                "theta_flow",
                "--flow-arch",
                "mlp",
                "--theta-flow-fourier-state",
                "--theta-flow-fourier-k",
                "3",
                "--theta-flow-fourier-period-mult",
                "2.0",
                "--flow-epochs",
                "4",
                "--prior-epochs",
                "4",
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
            self.assertIn("theta_flow Fourier state enabled", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            self.assertTrue((out_dir / "training_losses" / "n_000080.npz").is_file())

    def test_theta_flow_fourier_state_rejects_non_mlp(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "missing.npz"
            cmd = [
                sys.executable,
                str(script),
                "--dataset-npz",
                str(ds_path),
                "--dataset-family",
                "cosine_gaussian_sqrtd",
                "--flow-arch",
                "iid_soft",
                "--theta-flow-fourier-state",
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--theta-flow-fourier-state currently supports --flow-arch mlp only", r.stderr)


if __name__ == "__main__":
    unittest.main()

