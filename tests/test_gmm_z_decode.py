from __future__ import annotations

import numpy as np
import torch

from fisher.gmm_z_decode import (
    GMMZDecodeModel,
    compute_gmm_z_decode_c_matrix,
    train_gmm_z_decode,
)
from fisher.nf_hellinger import compute_delta_l


def test_gmm_z_decode_log_prob_and_c_matrix_are_finite() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 32
    theta = rng.normal(size=(n, 1))
    x = np.concatenate([theta, theta**2, rng.normal(scale=0.1, size=(n, 1))], axis=1)
    model = GMMZDecodeModel(x_dim=3, latent_dim=2, components=3, hidden_dim=16, depth=1)
    with torch.no_grad():
        lp = model.log_prob_theta_given_x(
            torch.from_numpy(theta[:5].astype(np.float32)),
            torch.from_numpy(x[:5].astype(np.float32)),
        )
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()

    out = train_gmm_z_decode(
        model=model,
        theta_train=theta[:24],
        x_train=x[:24],
        theta_val=theta[24:],
        x_val=x[24:],
        device=torch.device("cpu"),
        epochs=2,
        batch_size=8,
        lr=1e-3,
        patience=0,
        log_every=10,
    )
    c, z = compute_gmm_z_decode_c_matrix(
        model=model,
        theta_all=theta,
        x_all=x,
        device=torch.device("cpu"),
        x_mean=out["x_mean"],
        x_std=out["x_std"],
        theta_mean=out["theta_mean"],
        theta_std=out["theta_std"],
        pair_batch_size=128,
    )
    assert c.shape == (n, n)
    assert z.shape == (n, 2)
    assert np.all(np.isfinite(c))
    d = compute_delta_l(c)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)
