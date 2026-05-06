"""CLI and helper coverage for contrastive-soft Fourier via --theta-flow-fourier-* flags."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from fisher.h_decoding_convergence_cli import _validate_cli, _validate_contrastive_cli, build_parser
from fisher.h_decoding_convergence_methods import contrastive_soft_fourier_settings_from_theta_flow_args


def test_helper_returns_zero_k_without_theta_flow_fourier_state() -> None:
    ns = argparse.Namespace(
        theta_flow_fourier_state=False,
        theta_flow_fourier_k=4,
        theta_flow_fourier_period_mult=2.0,
        theta_flow_fourier_include_linear=True,
    )
    k, pm, inc = contrastive_soft_fourier_settings_from_theta_flow_args(ns)
    assert k == 0
    assert pm == 2.0
    assert inc is False


def test_helper_maps_theta_flow_fourier_state() -> None:
    ns = argparse.Namespace(
        theta_flow_fourier_state=True,
        theta_flow_fourier_k=3,
        theta_flow_fourier_period_mult=1.5,
        theta_flow_fourier_include_linear=True,
    )
    k, pm, inc = contrastive_soft_fourier_settings_from_theta_flow_args(ns)
    assert k == 3
    assert pm == 1.5
    assert inc is True


def test_hard_contrastive_rejects_theta_flow_fourier_state() -> None:
    args = argparse.Namespace(
        theta_field_method="contrastive",
        theta_flow_fourier_state=True,
        theta_flow_fourier_k=2,
        theta_flow_fourier_period_mult=2.0,
        theta_flow_fourier_include_linear=False,
    )
    with pytest.raises(ValueError, match="hard contrastive"):
        _validate_contrastive_cli(args)


def test_contrastive_soft_mlp_arch_rejects_fourier_state() -> None:
    args = argparse.Namespace(
        theta_field_method="contrastive_soft",
        theta_flow_fourier_state=True,
        theta_flow_fourier_k=2,
        theta_flow_fourier_period_mult=2.0,
        theta_flow_fourier_include_linear=False,
        contrastive_epochs=100,
        contrastive_batch_size=64,
        contrastive_lr=1e-3,
        contrastive_hidden_dim=32,
        contrastive_depth=2,
        contrastive_weight_decay=0.0,
        contrastive_early_patience=10,
        contrastive_early_min_delta=1e-4,
        contrastive_early_ema_alpha=0.05,
        contrastive_max_grad_norm=10.0,
        contrastive_pair_batch_size=1024,
        contrastive_theta_encoding="one_hot_bin",
        contrastive_soft_score_arch="mlp",
        contrastive_soft_dot_dim=8,
        contrastive_soft_coordinate_embed_dim=16,
        contrastive_soft_gaussian_logvar_min=-8.0,
        contrastive_soft_gaussian_logvar_max=5.0,
        contrastive_soft_bandwidth_bins=10,
        contrastive_soft_bandwidth_start=0.0,
        contrastive_soft_bandwidth_end=0.0,
        contrastive_soft_period=6.28318,
    )
    with pytest.raises(ValueError, match="theta-flow-fourier-state"):
        _validate_contrastive_cli(args)


@pytest.mark.parametrize(
    "bad_flag,extra",
    [
        ("--contrastive-theta-fourier-k", ["4"]),
        ("--contrastive-theta-fourier-period-mult", ["2.0"]),
        ("--contrastive-theta-fourier-include-linear", []),
    ],
)
def test_removed_contrastive_fourier_flags_rejected(bad_flag: str, extra: list[str]) -> None:
    p = build_parser()
    argv = [
        "--dataset-npz",
        "dummy.npz",
        "--theta-field-method",
        "contrastive_soft",
        bad_flag,
        *extra,
    ]
    with pytest.raises(SystemExit):
        p.parse_args(argv)


def test_validate_cli_allows_contrastive_soft_with_theta_flow_fourier_state() -> None:
    p = build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            "contrastive_soft",
            "--theta-flow-fourier-state",
        ]
    )
    _validate_cli(args)


def test_removed_contrastive_soft_bandwidth_k_rejected() -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(
            [
                "--dataset-npz",
                "dummy.npz",
                "--theta-field-method",
                "contrastive_soft",
                "--contrastive-soft-bandwidth-k",
                "5",
            ]
        )


def test_contrastive_soft_bandwidth_from_train_bins() -> None:
    from fisher.contrastive_llr import contrastive_soft_normalization_and_bandwidth_from_train

    th_tr = np.array([[0.0], [2.0]], dtype=np.float64)
    x_tr = np.zeros((2, 3), dtype=np.float64)
    nb = contrastive_soft_normalization_and_bandwidth_from_train(
        th_tr=th_tr,
        x_tr=x_tr,
        bandwidth_bins=10,
        bandwidth_start=0.0,
        bandwidth_end=0.0,
        periodic=False,
        period=2.0 * np.pi,
    )
    theta_scale = float(nb["theta_scale"])
    h_norm = float(nb["h_start_norm"])
    assert abs(float(2.0 / 10.0) / theta_scale - h_norm) < 1e-12
    assert abs(h_norm * theta_scale - 0.2) < 1e-12
