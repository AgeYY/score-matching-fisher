"""Smoke test for CTSM-v support in ``bin/study_h_decoding_convergence.py``."""

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


class TestStudyHDecodingConvergenceCtsmVSmoke(unittest.TestCase):
    def test_ctsm_v_sweep_smoke(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "study_h_decoding_convergence.py"
        n_total = 260
        n_ref = 200
        n_bins = 4
        ns_list = "80"
        seed = 7

        ns_ds = _ns(
            dataset_family="cosine_gaussian_sqrtd",
            x_dim=2,
            n_total=n_total,
            train_frac=1.0,
            seed=seed,
        )
        ds = build_dataset_from_args(ns_ds)
        theta_all, x_all = ds.sample_joint(n_total)
        meta = meta_dict_from_args(ns_ds)
        tr = np.arange(n_total, dtype=np.int64)
        ev = np.array([], dtype=np.int64)

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
                eval_idx=ev,
                theta_train=theta_all,
                x_train=x_all,
                theta_eval=np.empty((0, 1), dtype=np.float64),
                x_eval=np.empty((0, int(ns_ds.x_dim)), dtype=np.float64),
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
                "ctsm_v",
                "--ctsm-epochs",
                "4",
                "--ctsm-batch-size",
                "32",
                "--ctsm-hidden-dim",
                "32",
                "--ctsm-int-n-time",
                "24",
                "--flow-early-patience",
                "5",
                "--flow-early-min-delta",
                "1e-4",
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=(r.stdout, r.stderr))
            self.assertIn("ctsm_v mode uses pair-conditioned CTSM-v", r.stdout)
            self.assertTrue((out_dir / "h_decoding_convergence_results.npz").is_file())
            self.assertTrue((out_dir / "h_decoding_convergence.png").is_file())
            self.assertTrue((out_dir / "h_decoding_training_losses_panel.png").is_file())
            self.assertTrue((out_dir / "training_losses" / "n_000080.npz").is_file())


if __name__ == "__main__":
    unittest.main()
