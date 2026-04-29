from __future__ import annotations

import math

import numpy as np
import torch

from fisher.gaussian_network import (
    ConditionalDiagonalGaussianPrecisionMLP,
    ConditionalGaussianPrecisionMLP,
    ConditionalLowRankGaussianCovarianceMLP,
    ObservationAutoencoder,
    compute_gaussian_network_c_matrix,
    encode_observations,
    train_observation_autoencoder,
)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(float(y)))


def test_gaussian_network_log_prob_matches_standard_normal() -> None:
    model = ConditionalGaussianPrecisionMLP(theta_dim=1, x_dim=2, hidden_dim=4, depth=1, diag_floor=1e-4)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    raw_diag = _inv_softplus(1.0 - model.diag_floor)
    final = model.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias[:] = torch.tensor([0.0, 0.0, raw_diag, 0.0, raw_diag])

    x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
    theta = torch.tensor([[0.0], [0.5]], dtype=torch.float32)
    got = model.log_prob(x, theta).detach().numpy()
    expected = np.asarray(
        [
            -math.log(2.0 * math.pi),
            -0.5 * (2.0 * math.log(2.0 * math.pi) + 5.0),
        ],
        dtype=np.float64,
    )
    assert np.allclose(got, expected, atol=1e-5)


def test_diagonal_gaussian_network_log_prob_matches_standard_normal() -> None:
    model = ConditionalDiagonalGaussianPrecisionMLP(theta_dim=1, x_dim=2, hidden_dim=4, depth=1, diag_floor=1e-4)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    raw_diag = _inv_softplus(1.0 - model.diag_floor)
    final = model.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias[:] = torch.tensor([0.0, 0.0, raw_diag, raw_diag])

    x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
    theta = torch.tensor([[0.0], [0.5]], dtype=torch.float32)
    got = model.log_prob(x, theta).detach().numpy()
    expected = np.asarray(
        [
            -math.log(2.0 * math.pi),
            -0.5 * (2.0 * math.log(2.0 * math.pi) + 5.0),
        ],
        dtype=np.float64,
    )
    assert np.allclose(got, expected, atol=1e-5)


def test_low_rank_gaussian_network_log_prob_matches_full_covariance() -> None:
    model = ConditionalLowRankGaussianCovarianceMLP(
        theta_dim=1,
        x_dim=3,
        rank=2,
        hidden_dim=4,
        depth=1,
        diag_floor=1e-4,
        psi_floor=1e-6,
    )
    for p in model.net.parameters():
        torch.nn.init.zeros_(p)
    a = torch.tensor([[1.0, 0.2], [0.1, 0.8], [-0.3, 0.5]], dtype=torch.float32)
    psi = torch.tensor([0.7, 1.1, 0.9], dtype=torch.float32)
    mu_x = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float32)
    l10 = torch.tensor(0.15, dtype=torch.float32)
    ldiag = torch.tensor([1.2, 0.6], dtype=torch.float32)
    final = model.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        model.A.copy_(a)
        model.raw_psi.copy_(torch.log(torch.expm1(psi - model.psi_floor)))
        final.bias[:] = torch.tensor(
            [
                float(mu_x[0]),
                float(mu_x[1]),
                float(mu_x[2]),
                _inv_softplus(float(ldiag[0] - model.diag_floor)),
                float(l10),
                _inv_softplus(float(ldiag[1] - model.diag_floor)),
            ],
            dtype=torch.float32,
        )

    x = torch.tensor([[0.5, -0.4, 1.0], [-0.2, 0.3, 0.1]], dtype=torch.float32)
    theta = torch.zeros((2, 1), dtype=torch.float32)
    got = model.log_prob(x, theta).detach().numpy()

    a_np = a.numpy()
    mu_x_np = mu_x.numpy()
    l_np = np.asarray([[float(ldiag[0]), 0.0], [float(l10), float(ldiag[1])]], dtype=np.float64)
    cov = a_np @ (l_np @ l_np.T) @ a_np.T + np.diag(psi.numpy())
    inv_cov = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    expected = []
    for row in x.numpy():
        y = row - mu_x_np
        expected.append(-0.5 * (float(y @ inv_cov @ y) + float(logdet) + 3.0 * math.log(2.0 * math.pi)))
    assert np.allclose(got, np.asarray(expected, dtype=np.float64), atol=2e-5)


def test_gaussian_network_c_matrix_shapes_and_delta_contract() -> None:
    model = ConditionalGaussianPrecisionMLP(theta_dim=1, x_dim=1, hidden_dim=8, depth=1, diag_floor=1e-4)
    theta = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    x = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float64)
    c = compute_gaussian_network_c_matrix(
        model=model,
        theta_all=theta,
        x_all=x,
        device=torch.device("cpu"),
        pair_batch_size=4,
    )
    assert c.shape == (3, 3)
    assert np.isfinite(c).all()
    delta = c - np.diag(c).reshape(-1, 1)
    assert np.allclose(np.diag(delta), 0.0)


def test_observation_autoencoder_shapes_and_encoding() -> None:
    model = ObservationAutoencoder(x_dim=5, latent_dim=2, hidden_dim=8, depth=1)
    x = torch.randn(7, 5)
    z, x_hat = model(x)
    assert z.shape == (7, 2)
    assert x_hat.shape == (7, 5)

    encoded = encode_observations(model=model, x=x.numpy(), device=torch.device("cpu"), batch_size=3)
    assert encoded.shape == (7, 2)
    assert np.isfinite(encoded).all()


def test_train_observation_autoencoder_returns_finite_losses() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 4)).astype(np.float64)
    x_val = rng.normal(size=(12, 4)).astype(np.float64)
    model = ObservationAutoencoder(x_dim=4, latent_dim=2, hidden_dim=8, depth=1)
    out = train_observation_autoencoder(
        model=model,
        x_train=x_train,
        x_val=x_val,
        device=torch.device("cpu"),
        epochs=3,
        batch_size=10,
        lr=1e-3,
        patience=5,
        log_every=10,
    )
    assert len(out["train_losses"]) == 3
    assert len(out["val_losses"]) == 3
    assert np.isfinite(np.asarray(out["train_losses"], dtype=np.float64)).all()
    assert np.isfinite(np.asarray(out["val_losses"], dtype=np.float64)).all()
