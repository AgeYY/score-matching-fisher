import numpy as np

from fisher.lxf_bin_likelihood_hellinger import lxf_bin_likelihood_hellinger


def test_lxf_bin_likelihood_hellinger_zeros_same_bin_pairs() -> None:
    c_matrix = np.asarray(
        [
            [1.0, 0.8, -0.4, -0.2],
            [0.9, 1.1, -0.3, -0.1],
            [-0.5, -0.4, 1.2, 1.0],
            [-0.6, -0.2, 0.7, 1.3],
        ],
        dtype=np.float64,
    )
    bin_all = np.asarray([0, 0, 1, 1], dtype=np.int64)

    out = lxf_bin_likelihood_hellinger(c_matrix, bin_all, n_bins=2)
    h_sym = np.asarray(out["h_sym"], dtype=np.float64)

    assert h_sym.shape == (4, 4)
    assert np.all(h_sym[bin_all[:, None] == bin_all[None, :]] == 0.0)
    assert np.all(np.isfinite(h_sym[bin_all[:, None] != bin_all[None, :]]))
    assert np.any(h_sym[bin_all[:, None] != bin_all[None, :]] > 0.0)
