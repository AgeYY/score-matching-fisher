from __future__ import annotations

import argparse

import pytest

from fisher.cli_shared_fisher import parse_estimate_only_args
from fisher.shared_fisher_est import normalize_theta_field_method, validate_gmm_z_decode_args


def test_normalize_gmm_z_decode_aliases() -> None:
    assert normalize_theta_field_method("gmm-z-decode") == "gmm_z_decode"
    assert normalize_theta_field_method("gmm_z_decode") == "gmm_z_decode"


def test_estimate_parser_has_gmm_z_decode_flags() -> None:
    args = parse_estimate_only_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            "gmm-z-decode",
            "--gzd-latent-dim",
            "4",
            "--gzd-components",
            "7",
        ]
    )
    assert args.theta_field_method == "gmm-z-decode"
    assert args.gzd_latent_dim == 4
    assert args.gzd_components == 7


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("gzd_latent_dim", 0, "--gzd-latent-dim"),
        ("gzd_components", 0, "--gzd-components"),
        ("gzd_lr", 0.0, "--gzd-lr"),
        ("gzd_min_std", 0.0, "--gzd-min-std"),
        ("gzd_early_ema_alpha", 1.5, "--gzd-early-ema-alpha"),
    ],
)
def test_validate_rejects_bad_gmm_z_decode_flags(field: str, value: object, match: str) -> None:
    args = argparse.Namespace(
        gzd_latent_dim=2,
        gzd_components=5,
        gzd_epochs=2,
        gzd_batch_size=8,
        gzd_lr=1e-3,
        gzd_hidden_dim=16,
        gzd_depth=1,
        gzd_weight_decay=0.0,
        gzd_min_std=1e-3,
        gzd_early_patience=0,
        gzd_early_min_delta=0.0,
        gzd_early_ema_alpha=0.1,
        gzd_max_grad_norm=1.0,
        gzd_pair_batch_size=128,
    )
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        validate_gmm_z_decode_args(args)
