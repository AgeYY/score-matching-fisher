"""Integration tests for ``bin/project_dataset_pr_autoencoder.py``."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import load_shared_dataset_npz, meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, validate_dataset_sample_args


def _ns(**overrides: object):
    import argparse

    p = argparse.ArgumentParser()
    add_dataset_arguments(p)
    ns = p.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestProjectDatasetPrAutoencoder(unittest.TestCase):
    def test_cli_outputs_embedded_npz_and_meta(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        script = repo / "bin" / "project_dataset_pr_autoencoder.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            low_npz = tmp_path / "low.npz"
            ns = _ns(
                dataset_family="randamp_gaussian_sqrtd",
                x_dim=2,
                n_total=40,
                train_frac=0.75,
                seed=3,
            )
            validate_dataset_sample_args(ns)
            ds = build_dataset_from_args(ns)
            theta_all, x_all = ds.sample_joint(int(ns.n_total))
            rng = np.random.default_rng(int(ns.seed))
            perm = rng.permutation(int(ns.n_total))
            n_train = int(0.75 * int(ns.n_total))
            tr = perm[:n_train]
            ev = perm[n_train:]
            meta = meta_dict_from_args(ns)
            meta["randamp_mu_amp_per_dim"] = ds._randamp_amp.tolist()
            save_shared_dataset_npz(
                low_npz,
                meta=meta,
                theta_all=theta_all,
                x_all=x_all,
                train_idx=tr.astype(np.int64),
                validation_idx=ev.astype(np.int64),
                theta_train=theta_all[tr],
                x_train=x_all[tr],
                theta_validation=theta_all[ev],
                x_validation=x_all[ev],
            )
            out_npz = tmp_path / "high.npz"
            cmd = [
                sys.executable,
                str(script),
                "--input-npz",
                str(low_npz),
                "--output-npz",
                str(out_npz),
                "--h-dim",
                "8",
                "--device",
                "cpu",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--pr-train-samples",
                "128",
                "--pr-train-epochs",
                "2",
                "--pr-train-batch-size",
                "32",
                "--skip-viz",
            ]
            subprocess.check_call(cmd, cwd=str(repo))

            bundle = load_shared_dataset_npz(out_npz)
            self.assertEqual(bundle.meta["dataset_family"], "randamp_gaussian_sqrtd")
            self.assertEqual(int(bundle.meta["x_dim"]), 8)
            self.assertTrue(bundle.meta.get("pr_autoencoder_embedded"))
            self.assertEqual(int(bundle.meta["pr_autoencoder_z_dim"]), 2)
            self.assertEqual(bundle.x_all.shape, (40, 8))
            self.assertTrue(np.isfinite(bundle.x_all).all())
            self.assertIn("pr_autoencoder_source_sha256", bundle.meta)


if __name__ == "__main__":
    unittest.main()
