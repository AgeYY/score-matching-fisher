from __future__ import annotations

import numpy as np
import torch

from fisher.nf_hellinger import compute_delta_l, require_zuko_for_nf
from fisher.pi_nf import PiNFModel, compute_pi_nf_c_matrix, pi_nf_diagnostics, train_pi_nf


def test_pi_nf_log_prob_training_and_c_matrix_are_finite() -> None:
    require_zuko_for_nf()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 28
    theta = rng.normal(size=(n, 1))
    x = np.concatenate(
        [
            theta,
            np.sin(theta),
            rng.normal(scale=0.2, size=(n, 1)),
        ],
        axis=1,
    )
    model = PiNFModel(theta_dim=1, x_dim=3, latent_dim=1, hidden_dim=8, transforms=1, min_std=1e-3)
    x_t = torch.from_numpy(x[:4].astype(np.float32))
    with torch.no_grad():
        lp = model.log_prob_normalized_x_given_theta(
            x_t,
            torch.from_numpy(theta[:4].astype(np.float32)),
        )
        z0, r0, _ = model.encode_normalized(x_t)
        x_roundtrip = model.decode_normalized(z0, r0)
        recon_mse = model.reconstruction_mse_with_sampled_residual(x_t)
    assert lp.shape == (4,)
    assert torch.isfinite(lp).all()
    assert torch.allclose(x_roundtrip, x_t, atol=1e-5)
    assert recon_mse.ndim == 0
    assert torch.isfinite(recon_mse)

    out = train_pi_nf(
        model=model,
        theta_train=theta[:20],
        x_train=x[:20],
        theta_val=theta[20:],
        x_val=x[20:],
        device=torch.device("cpu"),
        epochs=2,
        batch_size=8,
        lr=1e-3,
        recon_weight=1.0,
        patience=0,
        log_every=10,
    )
    assert out["train_total_losses"].shape == (2,)
    assert out["train_nll_losses"].shape == (2,)
    assert out["train_recon_losses"].shape == (2,)
    assert out["val_total_losses"].shape == (2,)
    assert out["val_nll_losses"].shape == (2,)
    assert out["val_recon_losses"].shape == (2,)
    c, z, r = compute_pi_nf_c_matrix(
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
    assert z.shape == (n, 1)
    assert r.shape == (n, 2)
    assert np.all(np.isfinite(c))
    d = compute_delta_l(c)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)
    diag = pi_nf_diagnostics(z_all=z, r_all=r, theta_all=theta)
    assert "pinf_z_to_theta_r2" in diag
    assert "pinf_r_to_theta_r2" in diag
