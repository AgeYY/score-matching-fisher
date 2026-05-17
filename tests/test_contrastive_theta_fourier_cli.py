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


@pytest.mark.parametrize("arch", ["normalized_dot", "additive_independent"])
def test_contrastive_soft_supported_arches_allow_fourier_state(arch: str) -> None:
    p = build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            "contrastive_soft",
            "--theta-flow-fourier-state",
            "--contrastive-soft-score-arch",
            arch,
        ]
    )
    _validate_cli(args)


@pytest.mark.parametrize(
    "arch",
    [
        "mlp",
        "independent" + "_gaussian",
        "gaussian",
        "independent" + "_dot" + "_product",
        "independent" + "_dot",
        "dot" + "_independent",
        "norm" + "_dot",
        "dot",
        "additive",
        "additive" + "_independent" + "_feature",
        "independent",
    ],
)
def test_removed_contrastive_soft_score_arches_rejected(arch: str) -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(
            [
                "--dataset-npz",
                "dummy.npz",
                "--theta-field-method",
                "contrastive_soft",
                "--contrastive-soft-score-arch",
                arch,
            ]
        )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--contrastive-soft-coordinate" + "-embed-dim", "16"),
        ("--contrastive-soft-gaussian" + "-logvar-min", "-8.0"),
        ("--contrastive-soft-gaussian" + "-logvar-max", "5.0"),
    ],
)
def test_removed_contrastive_soft_arch_specific_flags_rejected(flag: str, value: str) -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(
            [
                "--dataset-npz",
                "dummy.npz",
                "--theta-field-method",
                "contrastive_soft",
                flag,
                value,
            ]
        )


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


def test_validate_cli_allows_contrastive_soft_categorical_alias() -> None:
    p = build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            "contrastive-soft-categorical",
        ]
    )
    _validate_cli(args)
    assert args.theta_field_method == "contrastive_soft_categorical"


def test_validate_cli_rejects_negative_contrastive_soft_categorical_beta() -> None:
    p = build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            "contrastive_soft_categorical",
            "--contrastive-soft-categorical-beta",
            "-0.1",
        ]
    )
    with pytest.raises(ValueError, match="categorical-beta"):
        _validate_cli(args)


@pytest.mark.parametrize(
    "method",
    [
        "bidir" + "_contrastive_soft",
        "bidir" + "-contrastive-soft",
        "bidirectional" + "-contrastive-soft",
        "bidirectional" + "_contrastive_soft",
        "bidir" + "-contrasive-soft",
        "contrastive_soft" + "_gaussian_net",
        "contrastive-soft" + "-gaussian-net",
        "contrasive-soft" + "-gaussian-net",
        "contrastive_soft" + "_gaussian_net_no_finetune",
        "contrastive-soft" + "-gaussian-net-no-finetune",
        "contrasive-soft" + "-gaussian-net-no-finetune",
        "contrastive" + "_theta" + "_flow",
        "contrastive" + "-thetaflow",
        "contrastive" + "_thetaflow",
        "contrastive" + "_x" + "_flow",
        "contrastive" + "-xflow",
        "contrastive" + "_xflow",
        "contrastive",
        "contrasive",
        "shuffled-contrastive",
        "shuffled_contrastive",
    ],
)
def test_removed_contrastive_method_aliases_rejected(method: str) -> None:
    p = build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            method,
        ]
    )
    with pytest.raises(ValueError):
        _validate_cli(args)


def test_removed_hard_contrastive_encoding_flag_rejected() -> None:
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(
            [
                "--dataset-npz",
                "dummy.npz",
                "--theta-field-method",
                "contrastive",
                "--contrastive" + "-theta" + "-encoding",
                "integer" + "_bin",
            ]
        )


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
        periodic=False,
        period=2.0 * np.pi,
    )
    theta_scale = float(nb["theta_scale"])
    h_norm = float(nb["h_norm"])
    assert abs(float(2.0 / (2.0 * 10.0)) / theta_scale - h_norm) < 1e-12
    assert abs(h_norm * theta_scale - 0.1) < 1e-12
