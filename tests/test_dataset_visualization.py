from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.data import ToyConditionalGaussianDataset
from fisher.dataset_visualization import _project_covariances_to_basis, plot_joint_and_tuning


def test_project_covariances_to_basis_matches_matrix_formula() -> None:
    basis = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    ) / np.sqrt(2.0)
    covariances = np.asarray(
        [
            [[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 0.7]],
            [[1.0, -0.1, 0.4], [-0.1, 3.0, 0.5], [0.4, 0.5, 2.0]],
        ],
        dtype=np.float64,
    )

    got = _project_covariances_to_basis(covariances, basis)
    expected = np.stack([basis.T @ cov @ basis for cov in covariances], axis=0)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_plot_joint_and_tuning_scalar_continuous_writes_png_and_svg(tmp_path: Path) -> None:
    dataset = ToyConditionalGaussianDataset(theta_low=-2.0, theta_high=2.0, x_dim=4, seed=123)
    theta, x = dataset.sample_joint(160)
    out_png = tmp_path / "joint_tuning.png"

    plot_joint_and_tuning(theta, x, dataset, str(out_png), scatter_max_points=80)

    out_svg = out_png.with_suffix(".svg")
    assert out_png.exists()
    assert out_png.stat().st_size > 0
    assert out_svg.exists()
    assert out_svg.stat().st_size > 0
