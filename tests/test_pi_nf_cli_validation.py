from __future__ import annotations

import pytest

import bin.study_h_decoding_convergence as conv


def test_normalize_pi_nf_aliases() -> None:
    assert conv._normalize_pi_nf_method("pi-nf") == "pi_nf"
    assert conv._normalize_pi_nf_method("pi_nf") == "pi_nf"


def test_convergence_parser_has_pi_nf_flags() -> None:
    args = conv.build_parser().parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-method",
            "pi-nf",
            "--pinf-latent-dim",
            "3",
            "--pinf-transforms",
            "2",
            "--pinf-recon-weight",
            "0.25",
        ]
    )
    assert args.theta_field_method == "pi-nf"
    assert args.pinf_latent_dim == 3
    assert args.pinf_transforms == 2
    assert args.pinf_recon_weight == 0.25


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--pinf-latent-dim", "0", "--pinf-latent-dim"),
        ("--pinf-epochs", "0", "--pinf-epochs"),
        ("--pinf-lr", "0", "--pinf-lr"),
        ("--pinf-min-std", "0", "--pinf-min-std"),
        ("--pinf-recon-weight", "-1", "--pinf-recon-weight"),
        ("--pinf-early-ema-alpha", "1.5", "--pinf-early-ema-alpha"),
    ],
)
def test_validate_rejects_bad_pi_nf_flags(flag: str, value: str, match: str) -> None:
    args = conv.build_parser().parse_args(["--dataset-npz", "dummy.npz", "--theta-field-method", "pi-nf", flag, value])
    with pytest.raises(ValueError, match=match):
        conv._validate_cli(args)
