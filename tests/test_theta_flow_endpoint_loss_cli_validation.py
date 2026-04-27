from __future__ import annotations

import sys
import types

import pytest

from fisher.cli_shared_fisher import parse_full_args

if "fisher.ot_cfm_core" not in sys.modules:
    _stub = types.ModuleType("fisher.ot_cfm_core")

    def _train_conditional_x_ot_cfm_model(*_args, **_kwargs):
        raise RuntimeError("stub should not be called in CLI validation tests")

    _stub.train_conditional_x_ot_cfm_model = _train_conditional_x_ot_cfm_model
    sys.modules["fisher.ot_cfm_core"] = _stub

from fisher.shared_fisher_est import validate_estimation_args


def test_validate_accepts_theta_flow_endpoint_defaults() -> None:
    args = parse_full_args([])
    validate_estimation_args(args)
    assert float(args.flow_endpoint_loss_weight) == pytest.approx(0.0)
    assert int(args.flow_endpoint_steps) == 20


def test_validate_rejects_negative_theta_flow_endpoint_weight() -> None:
    args = parse_full_args(["--flow-endpoint-loss-weight", "-0.01"])
    with pytest.raises(ValueError, match="--flow-endpoint-loss-weight must be non-negative."):
        validate_estimation_args(args)


def test_validate_rejects_non_positive_theta_flow_endpoint_steps() -> None:
    args = parse_full_args(["--flow-endpoint-steps", "0"])
    with pytest.raises(ValueError, match="--flow-endpoint-steps must be >= 1."):
        validate_estimation_args(args)


def test_validate_accepts_theta_flow_likelihood_finetune_flags() -> None:
    args = parse_full_args(
        [
            "--theta-field-method",
            "theta_flow",
            "--flow-likelihood-finetune-epochs",
            "2",
            "--flow-likelihood-finetune-lr",
            "1e-4",
            "--flow-likelihood-finetune-ode-steps",
            "4",
        ]
    )
    validate_estimation_args(args)
    assert int(args.flow_likelihood_finetune_epochs) == 2


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--flow-likelihood-finetune-epochs", "-1", "--flow-likelihood-finetune-epochs must be >= 0."),
        ("--flow-likelihood-finetune-epochs", "2001", "--flow-likelihood-finetune-epochs must be <= 2000."),
        ("--flow-likelihood-finetune-lr", "0", "--flow-likelihood-finetune-lr must be positive."),
        ("--flow-likelihood-finetune-ode-steps", "0", "--flow-likelihood-finetune-ode-steps must be >= 1."),
    ],
)
def test_validate_rejects_invalid_theta_flow_likelihood_finetune_flags(flag: str, value: str, match: str) -> None:
    args = parse_full_args([flag, value])
    with pytest.raises(ValueError, match=match):
        validate_estimation_args(args)


def test_validate_rejects_likelihood_finetune_for_x_flow() -> None:
    args = parse_full_args(
        [
            "--theta-field-method",
            "x_flow",
            "--flow-likelihood-finetune-epochs",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="only supported with --theta-field-method theta_flow"):
        validate_estimation_args(args)
