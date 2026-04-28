"""Tests for theta_flow_discrete_scaffold_q0 H-matrix path (numpy q0 log-prob only)."""

from __future__ import annotations

import numpy as np
import torch

from fisher.h_matrix import HMatrixEstimator
from fisher.theta_gaussian_scaffold import ThetaDiscreteScaffold


def test_discrete_q0_ratio_matrix_matches_log_prob_np() -> None:
    rng = np.random.default_rng(42)
    n = 6
    x_dim = 3
    theta_raw = rng.normal(scale=2.0, size=(n, 1))
    x_all = rng.normal(scale=1.0, size=(n, x_dim))

    scaffold = ThetaDiscreteScaffold.fit(
        theta_train=theta_raw,
        x_train=x_all,
        n_bins=5,
        variance_floor=1e-6,
        source_eps=0.0,
        theta_low=float(np.min(theta_raw)),
        theta_high=float(np.max(theta_raw)),
    )

    theta_disc = scaffold.discretize_theta_np(theta_raw).reshape(-1)

    estimator = HMatrixEstimator(
        model_post=None,
        model_prior=None,
        sigma_eval=0.8,
        device=torch.device("cpu"),
        pair_batch_size=65536,
        field_method="theta_flow_discrete_scaffold_q0",
        flow_ode_steps=64,
        theta_gaussian_scaffold=scaffold,
    )
    out = estimator.run(theta_disc, x_all, restore_original_order=False)

    perm = out.perm
    theta_s = theta_disc[perm].reshape(-1)
    x_s = x_all[perm]

    manual = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            th1 = np.array([float(theta_s[j])], dtype=np.float64).reshape(-1)
            xi = np.asarray(x_s[i], dtype=np.float64).reshape(1, -1)
            manual[i, j] = float(scaffold.log_prob_np(th1, xi)[0])

    c_manual = manual
    delta_manual = estimator.compute_delta_l(c_manual)
    h_dir_manual = estimator.compute_h_directed(delta_manual)
    h_sym_manual = estimator.symmetrize(h_dir_manual)

    np.testing.assert_allclose(out.c_matrix, c_manual, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(out.delta_l_matrix, delta_manual, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(out.h_sym, h_sym_manual, rtol=0.0, atol=1e-10)


def test_h_estimator_rejects_non_none_model_for_q0() -> None:
    from fisher.models import ConditionalThetaFlowVelocity

    rng = np.random.default_rng(1)
    th = rng.normal(size=(6, 1))
    x = rng.normal(size=(6, 2))
    sc = ThetaDiscreteScaffold.fit(
        theta_train=th,
        x_train=x,
        n_bins=3,
        variance_floor=1e-6,
        theta_low=float(np.min(th)),
        theta_high=float(np.max(th)),
    )
    m = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=8, depth=2, use_logit_time=True, theta_dim=1)
    try:
        HMatrixEstimator(
            model_post=m,
            model_prior=None,
            sigma_eval=0.5,
            device=torch.device("cpu"),
            field_method="theta_flow_discrete_scaffold_q0",
            theta_gaussian_scaffold=sc,
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "model_post=None" in str(e)
