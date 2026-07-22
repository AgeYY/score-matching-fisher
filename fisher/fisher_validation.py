"""Validation utilities for linear Fisher estimators without population truth.

The functions here evaluate fitted Fisher *directions* on held-out responses and
calibrate scalar estimates with a conditionally independent probe channel.  They
do not alter the Flow Matching or GKR training objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class StratifiedSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    stratum: np.ndarray


@dataclass(frozen=True)
class DecoderEvaluation:
    achieved_fisher_raw: np.ndarray
    achieved_fisher_display: np.ndarray
    roc_auc: np.ndarray
    projected_mean_left: np.ndarray
    projected_mean_right: np.ndarray
    projected_variance_left: np.ndarray
    projected_variance_right: np.ndarray
    n_left: np.ndarray
    n_right: np.ndarray


@dataclass(frozen=True)
class LinearThresholdEvaluation:
    balanced_error: np.ndarray
    false_positive_rate: np.ndarray
    false_negative_rate: np.ndarray
    threshold: np.ndarray
    orientation: np.ndarray


def stratified_disjoint_subset_indices(
    theta: np.ndarray,
    subset_size: int,
    *,
    n_subsets: int,
    n_strata: int,
    seed: int,
    period: float | None = None,
) -> list[np.ndarray]:
    """Return equal-size, disjoint, condition-stratified subsets.

    A random ordering is generated within each stratum. Observations are then
    interleaved according to their within-stratum quantile and divided into
    consecutive blocks. This makes each block follow the full condition
    histogram while ensuring sampling without replacement across blocks.
    """

    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    size = int(subset_size)
    count = int(n_subsets)
    if values.size < 1:
        raise ValueError("theta must contain at least one observation.")
    if size < 1:
        raise ValueError("subset_size must be positive.")
    if count < 1:
        raise ValueError("n_subsets must be positive.")
    if size * count > values.size:
        raise ValueError("subset_size * n_subsets must not exceed len(theta).")
    if int(n_strata) < 1:
        raise ValueError("n_strata must be positive.")

    if period is None:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if not upper > lower:
            raise ValueError("theta must vary.")
        scaled = (values - lower) / (upper - lower)
    else:
        if not float(period) > 0.0:
            raise ValueError("period must be positive.")
        scaled = np.mod(values, float(period)) / float(period)
    stratum = np.minimum(
        (scaled * int(n_strata)).astype(np.int64), int(n_strata) - 1
    )

    rng = np.random.default_rng(int(seed))
    ordered_index: list[np.ndarray] = []
    ordered_priority: list[np.ndarray] = []
    for bin_index in range(int(n_strata)):
        index = np.flatnonzero(stratum == bin_index)
        if index.size == 0:
            raise ValueError(f"Stratum {bin_index} has no observations.")
        index = rng.permutation(index)
        priority = (np.arange(index.size, dtype=np.float64) + 0.5) / index.size
        priority += rng.uniform(-1e-12, 1e-12, size=index.size)
        ordered_index.append(index)
        ordered_priority.append(priority)
    all_index = np.concatenate(ordered_index)
    all_priority = np.concatenate(ordered_priority)
    order = all_index[np.argsort(all_priority, kind="mergesort")]
    return [
        np.sort(order[subset_index * size : (subset_index + 1) * size]).astype(np.int64)
        for subset_index in range(count)
    ]


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
) -> tuple[float, float, float]:
    train = float(train_fraction)
    validation = float(validation_fraction)
    test = 1.0 - train - validation
    if min(train, validation, test) <= 0.0:
        raise ValueError("train, validation, and test fractions must all be positive.")
    return train, validation, test


def stratified_train_validation_test_split(
    theta: np.ndarray,
    *,
    n_strata: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    period: float | None = None,
    fixed_partition: Literal["test", "validation"] = "test",
) -> StratifiedSplit:
    """Split observations within equal-width condition strata.

    The split is deterministic for a seed and keeps every observation in exactly
    one partition. For circular variables, values are wrapped before binning.
    ``fixed_partition`` controls which held-out partition remains unchanged when
    the other fraction varies across repeated calls with the same seed.
    """

    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.size < 3:
        raise ValueError("At least three observations are required.")
    if int(n_strata) < 1:
        raise ValueError("n_strata must be positive.")
    if fixed_partition not in ("test", "validation"):
        raise ValueError("fixed_partition must be 'test' or 'validation'.")
    train_fraction, validation_fraction, test_fraction = _validate_split_fractions(
        train_fraction, validation_fraction
    )

    if period is None:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if not upper > lower:
            raise ValueError("theta must vary.")
        scaled = (values - lower) / (upper - lower)
    else:
        if not float(period) > 0.0:
            raise ValueError("period must be positive.")
        scaled = np.mod(values, float(period)) / float(period)
    stratum = np.minimum((scaled * int(n_strata)).astype(np.int64), int(n_strata) - 1)

    rng = np.random.default_rng(int(seed))
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for bin_index in range(int(n_strata)):
        index = np.flatnonzero(stratum == bin_index)
        if index.size < 3:
            raise ValueError(f"Stratum {bin_index} has fewer than three observations.")
        index = rng.permutation(index)
        n_test = max(1, int(np.floor(test_fraction * index.size + 1e-9)))
        n_validation = max(1, int(np.floor(validation_fraction * index.size + 1e-9)))
        if n_test + n_validation >= index.size:
            n_validation = 1
            n_test = 1
        n_train = index.size - n_validation - n_test
        train.append(index[:n_train])
        if fixed_partition == "test":
            validation.append(index[n_train : n_train + n_validation])
            test.append(index[-n_test:])
        else:
            test.append(index[n_train : n_train + n_test])
            validation.append(index[-n_validation:])
    return StratifiedSplit(
        train=np.sort(np.concatenate(train)).astype(np.int64),
        validation=np.sort(np.concatenate(validation)).astype(np.int64),
        test=np.sort(np.concatenate(test)).astype(np.int64),
        stratum=stratum,
    )


def decoder_directions(
    signal: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Return unit-norm, deterministically oriented Fisher decoder directions."""

    signal = np.asarray(signal, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if signal.ndim != 2 or covariance.shape != (signal.shape[0], signal.shape[1], signal.shape[1]):
        raise ValueError("Expected signal [k,d] and covariance [k,d,d].")
    eye = np.eye(signal.shape[1], dtype=np.float64)
    directions = np.linalg.solve(covariance + float(ridge) * eye[None], signal[..., None])[..., 0]
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0) or np.any(~np.isfinite(norms)):
        raise ValueError("All decoder directions must have finite nonzero norm.")
    directions = directions / norms[:, None]
    sign = np.sign(np.einsum("kd,kd->k", directions, signal))
    sign[sign == 0.0] = 1.0
    return directions * sign[:, None]


def finite_endpoint_oracle(
    mean_left: np.ndarray,
    mean_right: np.ndarray,
    covariance_left: np.ndarray,
    covariance_right: np.ndarray,
    dtheta: np.ndarray,
    *,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the optimal direction and information for finite endpoint pairs.

    The returned direction maximizes the same pooled-endpoint objective used by
    :func:`evaluate_endpoint_decoders` at the population level.
    """

    left_mean = np.asarray(mean_left, dtype=np.float64)
    right_mean = np.asarray(mean_right, dtype=np.float64)
    left_covariance = np.asarray(covariance_left, dtype=np.float64)
    right_covariance = np.asarray(covariance_right, dtype=np.float64)
    spacing = np.asarray(dtheta, dtype=np.float64).reshape(-1)
    if left_mean.shape != right_mean.shape or left_mean.ndim != 2:
        raise ValueError("Endpoint means must have matching shape [k,d].")
    expected_covariance_shape = (
        left_mean.shape[0],
        left_mean.shape[1],
        left_mean.shape[1],
    )
    if (
        left_covariance.shape != expected_covariance_shape
        or right_covariance.shape != expected_covariance_shape
    ):
        raise ValueError("Endpoint covariances must have matching shape [k,d,d].")
    if spacing.shape != (left_mean.shape[0],) or np.any(spacing <= 0.0):
        raise ValueError("dtheta must contain one positive spacing per endpoint pair.")

    signal = right_mean - left_mean
    pooled_covariance = 0.5 * (left_covariance + right_covariance)
    directions = decoder_directions(signal, pooled_covariance, ridge=ridge)
    eye = np.eye(left_mean.shape[1], dtype=np.float64)
    solved_signal = np.linalg.solve(
        pooled_covariance + float(ridge) * eye[None],
        signal[..., None],
    )[..., 0]
    information = np.einsum("kd,kd->k", signal, solved_signal) / spacing**2
    return directions, information


def evaluate_endpoint_decoders(
    directions: np.ndarray,
    x_left: np.ndarray,
    x_right: np.ndarray,
    dtheta: np.ndarray,
) -> DecoderEvaluation:
    """Evaluate fixed local decoders on independent endpoint observations.

    ``x_left`` and ``x_right`` are arrays of shape ``[k, n, d]``.  The raw
    achieved information subtracts the finite-sample variance of the two sample
    means.  Negative values are retained in ``achieved_fisher_raw`` and clipped
    only in ``achieved_fisher_display``.
    """

    directions = np.asarray(directions, dtype=np.float64)
    x_left = np.asarray(x_left, dtype=np.float64)
    x_right = np.asarray(x_right, dtype=np.float64)
    spacing = np.asarray(dtheta, dtype=np.float64).reshape(-1)
    if x_left.ndim != 3 or x_right.ndim != 3 or x_left.shape != x_right.shape:
        raise ValueError("x_left and x_right must have equal shape [k,n,d].")
    if directions.shape != (x_left.shape[0], x_left.shape[2]) or spacing.shape[0] != x_left.shape[0]:
        raise ValueError("Direction or spacing shape does not match endpoint samples.")
    if np.any(spacing <= 0.0):
        raise ValueError("dtheta must be positive.")

    left_projection = np.einsum("knd,kd->kn", x_left, directions)
    right_projection = np.einsum("knd,kd->kn", x_right, directions)
    mean_left = np.mean(left_projection, axis=1)
    mean_right = np.mean(right_projection, axis=1)
    var_left = np.var(left_projection, axis=1, ddof=1)
    var_right = np.var(right_projection, axis=1, ddof=1)
    n_left = np.full(x_left.shape[0], x_left.shape[1], dtype=np.int64)
    n_right = np.full(x_right.shape[0], x_right.shape[1], dtype=np.int64)
    numerator = (mean_right - mean_left) ** 2 - var_right / n_right - var_left / n_left
    pooled_variance = 0.5 * (var_left + var_right)
    raw = numerator / (spacing**2 * np.maximum(pooled_variance, 1e-12))

    auc = np.empty(x_left.shape[0], dtype=np.float64)
    labels = np.concatenate(
        [np.zeros(x_left.shape[1], dtype=np.int64), np.ones(x_right.shape[1], dtype=np.int64)]
    )
    for index in range(x_left.shape[0]):
        score = np.concatenate([left_projection[index], right_projection[index]])
        auc[index] = max(roc_auc_score(labels, score), 1.0 - roc_auc_score(labels, score))
    return DecoderEvaluation(
        achieved_fisher_raw=raw,
        achieved_fisher_display=np.maximum(raw, 0.0),
        roc_auc=auc,
        projected_mean_left=mean_left,
        projected_mean_right=mean_right,
        projected_variance_left=var_left,
        projected_variance_right=var_right,
        n_left=n_left,
        n_right=n_right,
    )


def _linear_threshold_statistics(
    direction: np.ndarray,
    calibration_left: np.ndarray,
    calibration_right: np.ndarray,
    test_left: np.ndarray,
    test_right: np.ndarray,
) -> tuple[float, float, float, float, float]:
    calibration_left_score = np.asarray(calibration_left, dtype=np.float64) @ direction
    calibration_right_score = np.asarray(calibration_right, dtype=np.float64) @ direction
    test_left_score = np.asarray(test_left, dtype=np.float64) @ direction
    test_right_score = np.asarray(test_right, dtype=np.float64) @ direction
    if min(
        calibration_left_score.size,
        calibration_right_score.size,
        test_left_score.size,
        test_right_score.size,
    ) < 2:
        raise ValueError("Each calibration and test endpoint requires at least two observations.")

    calibration_mean_left = float(np.mean(calibration_left_score))
    calibration_mean_right = float(np.mean(calibration_right_score))
    orientation = 1.0 if calibration_mean_right >= calibration_mean_left else -1.0
    threshold = 0.5 * (calibration_mean_left + calibration_mean_right)
    false_positive = float(np.mean(orientation * (test_left_score - threshold) >= 0.0))
    false_negative = float(np.mean(orientation * (test_right_score - threshold) < 0.0))
    return (
        0.5 * (false_positive + false_negative),
        false_positive,
        false_negative,
        threshold,
        orientation,
    )


def evaluate_linear_threshold_decoders(
    directions: np.ndarray,
    calibration_left: np.ndarray,
    calibration_right: np.ndarray,
    test_left: np.ndarray,
    test_right: np.ndarray,
) -> LinearThresholdEvaluation:
    """Fit affine thresholds on calibration scores and evaluate held-out error.

    All endpoint arrays have shape ``[k,n,d]``. Calibration and test sample
    counts may differ, but the number of pairs and response dimensions must
    agree. Only the scalar score ``w.T @ x`` and one affine threshold are used.
    """

    direction = np.asarray(directions, dtype=np.float64)
    cal_left = np.asarray(calibration_left, dtype=np.float64)
    cal_right = np.asarray(calibration_right, dtype=np.float64)
    left = np.asarray(test_left, dtype=np.float64)
    right = np.asarray(test_right, dtype=np.float64)
    arrays = (cal_left, cal_right, left, right)
    if any(value.ndim != 3 for value in arrays):
        raise ValueError("Endpoint arrays must have shape [k,n,d].")
    if cal_left.shape != cal_right.shape or left.shape != right.shape:
        raise ValueError("Left and right arrays must match within each split.")
    if cal_left.shape[0] != left.shape[0] or cal_left.shape[2] != left.shape[2]:
        raise ValueError("Calibration and test pair or response dimensions do not match.")
    if direction.shape != (left.shape[0], left.shape[2]):
        raise ValueError("Direction shape does not match endpoint arrays.")

    records = [
        _linear_threshold_statistics(
            direction[index],
            cal_left[index],
            cal_right[index],
            left[index],
            right[index],
        )
        for index in range(direction.shape[0])
    ]
    values = np.asarray(records, dtype=np.float64)
    return LinearThresholdEvaluation(
        balanced_error=values[:, 0],
        false_positive_rate=values[:, 1],
        false_negative_rate=values[:, 2],
        threshold=values[:, 3],
        orientation=values[:, 4],
    )


def fisher_predicted_linear_error(fisher: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """Local equal-variance Gaussian error predicted by linear Fisher."""

    information = np.asarray(fisher, dtype=np.float64).reshape(-1)
    spacing = np.asarray(dtheta, dtype=np.float64).reshape(-1)
    if information.shape != spacing.shape:
        raise ValueError("fisher and dtheta must have equal shape.")
    if np.any(spacing <= 0.0):
        raise ValueError("dtheta must be positive.")
    return ndtr(-0.5 * spacing * np.sqrt(np.maximum(information, 0.0)))


def evaluate_windowed_decoders(
    directions: np.ndarray,
    responses: np.ndarray,
    theta: np.ndarray,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    *,
    half_width: float,
    period: float,
) -> DecoderEvaluation:
    """Evaluate decoders with circular held-out endpoint windows."""

    x = np.asarray(responses, dtype=np.float64)
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    left = np.asarray(theta_left, dtype=np.float64).reshape(-1)
    right = np.asarray(theta_right, dtype=np.float64).reshape(-1)
    direction = np.asarray(directions, dtype=np.float64)
    if direction.shape != (left.size, x.shape[1]) or right.size != left.size:
        raise ValueError("Endpoint and direction shapes do not agree.")

    def distance(center: float) -> np.ndarray:
        delta = np.abs(values - center)
        return np.minimum(delta, float(period) - delta)

    records: list[DecoderEvaluation] = []
    for index, (center_left, center_right) in enumerate(zip(left, right, strict=True)):
        left_x = x[distance(float(center_left)) <= float(half_width)]
        right_x = x[distance(float(center_right)) <= float(half_width)]
        if min(left_x.shape[0], right_x.shape[0]) < 2:
            raise ValueError(f"Insufficient held-out samples for endpoint pair {index}.")
        projection_left = left_x @ direction[index]
        projection_right = right_x @ direction[index]
        mean_left = float(np.mean(projection_left))
        mean_right = float(np.mean(projection_right))
        var_left = float(np.var(projection_left, ddof=1))
        var_right = float(np.var(projection_right, ddof=1))
        spacing = (float(center_right) - float(center_left)) % float(period)
        numerator = (mean_right - mean_left) ** 2 - var_left / left_x.shape[0] - var_right / right_x.shape[0]
        raw = numerator / (spacing**2 * max(0.5 * (var_left + var_right), 1e-12))
        labels = np.concatenate([np.zeros(left_x.shape[0]), np.ones(right_x.shape[0])])
        scores = np.concatenate([projection_left, projection_right])
        auc = roc_auc_score(labels, scores)
        records.append(
            DecoderEvaluation(
                achieved_fisher_raw=np.asarray([raw]),
                achieved_fisher_display=np.asarray([max(raw, 0.0)]),
                roc_auc=np.asarray([max(auc, 1.0 - auc)]),
                projected_mean_left=np.asarray([mean_left]),
                projected_mean_right=np.asarray([mean_right]),
                projected_variance_left=np.asarray([var_left]),
                projected_variance_right=np.asarray([var_right]),
                n_left=np.asarray([left_x.shape[0]]),
                n_right=np.asarray([right_x.shape[0]]),
            )
        )
    fields = DecoderEvaluation.__dataclass_fields__
    return DecoderEvaluation(**{name: np.concatenate([getattr(record, name) for record in records]) for name in fields})


def evaluate_windowed_linear_threshold_decoders(
    directions: np.ndarray,
    calibration_responses: np.ndarray,
    calibration_theta: np.ndarray,
    test_responses: np.ndarray,
    test_theta: np.ndarray,
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    *,
    half_width: float,
    period: float,
) -> LinearThresholdEvaluation:
    """Evaluate matched linear thresholds in circular endpoint windows."""

    direction = np.asarray(directions, dtype=np.float64)
    calibration_x = np.asarray(calibration_responses, dtype=np.float64)
    calibration_values = np.asarray(calibration_theta, dtype=np.float64).reshape(-1)
    test_x = np.asarray(test_responses, dtype=np.float64)
    test_values = np.asarray(test_theta, dtype=np.float64).reshape(-1)
    left = np.asarray(theta_left, dtype=np.float64).reshape(-1)
    right = np.asarray(theta_right, dtype=np.float64).reshape(-1)
    if calibration_x.shape[0] != calibration_values.size or test_x.shape[0] != test_values.size:
        raise ValueError("Responses and theta must have equal lengths within each split.")
    if calibration_x.ndim != 2 or test_x.ndim != 2 or calibration_x.shape[1] != test_x.shape[1]:
        raise ValueError("Calibration and test responses must have compatible shape [n,d].")
    if direction.shape != (left.size, test_x.shape[1]) or right.size != left.size:
        raise ValueError("Endpoint and direction shapes do not agree.")

    def select(values: np.ndarray, center: float) -> np.ndarray:
        delta = np.abs(values - center)
        return np.minimum(delta, float(period) - delta) <= float(half_width)

    records = []
    for index, (center_left, center_right) in enumerate(zip(left, right, strict=True)):
        records.append(
            _linear_threshold_statistics(
                direction[index],
                calibration_x[select(calibration_values, float(center_left))],
                calibration_x[select(calibration_values, float(center_right))],
                test_x[select(test_values, float(center_left))],
                test_x[select(test_values, float(center_right))],
            )
        )
    values = np.asarray(records, dtype=np.float64)
    return LinearThresholdEvaluation(
        balanced_error=values[:, 0],
        false_positive_rate=values[:, 1],
        false_negative_rate=values[:, 2],
        threshold=values[:, 3],
        orientation=values[:, 4],
    )


def probe_mean(theta: np.ndarray, *, peak_fisher: float, omega: float, phase: float = 0.0) -> np.ndarray:
    """Mean of a unit-noise sinusoidal probe with the requested peak Fisher."""

    if float(peak_fisher) < 0.0 or float(omega) == 0.0:
        raise ValueError("peak_fisher must be nonnegative and omega must be nonzero.")
    amplitude = np.sqrt(float(peak_fisher)) / abs(float(omega))
    return amplitude * np.sin(float(omega) * np.asarray(theta, dtype=np.float64) + float(phase))


def append_paired_probe(
    responses: np.ndarray,
    theta: np.ndarray,
    *,
    peak_fisher: float,
    omega: float,
    phase: float,
    seed: int,
    noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append paired control and signal probe channels sharing identical noise."""

    x = np.asarray(responses, dtype=np.float64)
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if x.shape[0] != values.size:
        raise ValueError("responses and theta must have equal length.")
    eps = (
        np.random.default_rng(int(seed)).standard_normal(values.size)
        if noise is None
        else np.asarray(noise, dtype=np.float64).reshape(-1)
    )
    if eps.size != values.size:
        raise ValueError("noise must have one value per observation.")
    signal = probe_mean(values, peak_fisher=peak_fisher, omega=omega, phase=phase)
    control = np.column_stack([x, eps])
    probe = np.column_stack([x, signal + eps])
    return control, probe, eps


def finite_grid_probe_increment(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    *,
    peak_fisher: float,
    omega: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Exact probe increment for the finite adjacent-pair Fisher definition."""

    left = np.asarray(theta_left, dtype=np.float64).reshape(-1)
    right = np.asarray(theta_right, dtype=np.float64).reshape(-1)
    spacing = right - left
    if left.shape != right.shape or np.any(spacing == 0.0):
        raise ValueError("theta endpoints must have equal shape and nonzero spacing.")
    mean_left = probe_mean(left, peak_fisher=peak_fisher, omega=omega, phase=phase)
    mean_right = probe_mean(right, peak_fisher=peak_fisher, omega=omega, phase=phase)
    return ((mean_right - mean_left) / spacing) ** 2


def calibration_metrics(estimated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return signed calibration diagnostics against a known increment."""

    estimate = np.asarray(estimated, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if estimate.shape != truth.shape or estimate.size < 2:
        raise ValueError("estimated and target must have equal nontrivial shape.")
    slope, intercept = np.polyfit(truth, estimate, deg=1)
    residual = estimate - truth
    ss_total = float(np.sum((estimate - np.mean(estimate)) ** 2))
    ss_residual = float(np.sum((estimate - (slope * truth + intercept)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - ss_residual / ss_total) if ss_total > 0.0 else float("nan"),
    }


def gaussian_mixture_moments(dataset: Any, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact total mean, mean derivative, and covariance for the toy two-Gaussian mixture."""

    values = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    probability, probability_derivative = dataset._mix_weight(values)
    mean1, mean2 = dataset.component_means(values)
    covariance1, covariance2, _, _ = dataset.component_covariances(values)
    base_derivative = dataset.tuning_curve_derivative(values)
    _, separation_derivative = dataset._separation(values)
    derivative1 = base_derivative + separation_derivative
    derivative2 = base_derivative - separation_derivative
    p = probability[:, None]
    mean = p * mean1 + (1.0 - p) * mean2
    mean_derivative = (
        probability_derivative[:, None] * (mean1 - mean2)
        + p * derivative1
        + (1.0 - p) * derivative2
    )
    delta1 = mean1 - mean
    delta2 = mean2 - mean
    covariance = (
        probability[:, None, None] * (covariance1 + np.einsum("ni,nj->nij", delta1, delta1))
        + (1.0 - probability)[:, None, None]
        * (covariance2 + np.einsum("ni,nj->nij", delta2, delta2))
    )
    return mean, mean_derivative, covariance


def population_linear_moments(dataset: Any, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return population mean, mean derivative, and covariance for supported toys."""

    if hasattr(dataset, "_mix_weight") and hasattr(dataset, "component_means"):
        return gaussian_mixture_moments(dataset, theta)
    values = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    return (
        np.asarray(dataset.tuning_curve(values), dtype=np.float64),
        np.asarray(dataset.tuning_curve_derivative(values), dtype=np.float64),
        np.asarray(dataset.covariance(values), dtype=np.float64),
    )


def fit_flow_direction_estimator(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    theta_grid: np.ndarray,
    device: Any,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    depth: int,
    ode_steps: int,
    condition_train: np.ndarray | None = None,
    condition_validation: np.ndarray | None = None,
    condition_grid: np.ndarray | None = None,
    theta_rbf_num_centers: int = 8,
    theta_rbf_bandwidth: float | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], np.ndarray]:
    """Fit the established affine Flow estimator and return decoder directions."""

    import torch

    from fisher.flow_matching_skl import (
        build_flow_skl_model,
        estimate_affine_mixed_symmetric_kl_fisher,
        train_flow_skl_model,
    )
    from fisher.stringer_session_identification import (
        estimate_affine_mixed_symmetric_kl_fisher_for_conditions,
    )

    scalar_train = np.asarray(theta_train, dtype=np.float64).reshape(-1, 1)
    scalar_validation = np.asarray(theta_validation, dtype=np.float64).reshape(-1, 1)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1, 1)
    flow_train = scalar_train if condition_train is None else np.asarray(condition_train, dtype=np.float64)
    flow_validation = (
        scalar_validation if condition_validation is None else np.asarray(condition_validation, dtype=np.float64)
    )
    flow_grid = grid if condition_grid is None else np.asarray(condition_grid, dtype=np.float64)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=int(flow_train.shape[1]),
        x_dim=int(np.asarray(x_train).shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        quadrature_steps=64,
        path_schedule="cosine",
        divergence_estimator="exact",
        theta_embedding="gaussian_rbf" if condition_train is None else "identity",
        theta_rbf_num_centers=int(theta_rbf_num_centers),
        theta_rbf_lower=float(np.min(grid)),
        theta_rbf_upper=float(np.max(grid)),
        theta_rbf_bandwidth=theta_rbf_bandwidth,
    ).to(device)
    training = train_flow_skl_model(
        model=model,
        theta_train=flow_train,
        x_train=np.asarray(x_train, dtype=np.float64),
        theta_val=flow_validation,
        x_val=np.asarray(x_validation, dtype=np.float64),
        device=device,
        velocity_family="condition_affine",
        path_schedule="cosine",
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(learning_rate),
        lr_schedule="constant",
        weight_decay=0.0,
        t_eps=5e-4,
        patience=int(patience),
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        log_every=50,
        checkpoint_selection="best",
        best_checkpoint_metric="flow_matching",
        fixed_validation=True,
        fixed_validation_paths=10,
        validation_seed=int(seed) + 10_000,
    )
    if condition_grid is None:
        estimate = estimate_affine_mixed_symmetric_kl_fisher(
            model=model,
            theta_all=grid,
            device=device,
            ridge=1e-6,
            ode_steps=int(ode_steps),
        )
    else:
        estimate = estimate_affine_mixed_symmetric_kl_fisher_for_conditions(
            model=model,
            theta_all=grid,
            condition_all=flow_grid,
            device=device,
            ridge=1e-6,
            ode_steps=int(ode_steps),
        )
    directions = decoder_directions(estimate["delta_mu"], estimate["mixed_covariance"])
    return model, training, estimate, directions


def fit_gkr_direction_estimator(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_grid: np.ndarray,
    device: Any,
    seed: int,
    circular_period: float | None = None,
) -> tuple[Any, Any, np.ndarray]:
    """Fit GKR and return finite-difference midpoint decoder directions."""

    import torch

    from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher

    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1, 1)
    midpoint = 0.5 * (grid[:-1] + grid[1:])
    spacing = np.diff(grid[:, 0])
    model = TorchGKR(
        n_input=1,
        n_output=int(np.asarray(x_train).shape[1]),
        circular_period=circular_period,
        config=GKRConfig(),
        dtype=torch.float64,
        device=device,
        seed=int(seed),
    )
    model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(theta_train, dtype=np.float64).reshape(-1, 1))
    estimate = estimate_gkr_linear_fisher(
        model,
        midpoint,
        finite_difference_step=spacing[:, None],
        solve_jitter=1e-6,
    )
    signal = estimate.mean_jacobian[:, :, 0] * spacing[:, None]
    directions = decoder_directions(signal, estimate.covariance)
    return model, estimate, directions


def fit_cross_fitted_ole_direction_estimator(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_grid: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    window_radius: float | None = None,
    min_endpoint_samples: int = 8,
    period: float | None = None,
) -> tuple[Any, np.ndarray]:
    """Fit cross-fitted OLE folds and return their averaged decoder direction."""

    from fisher.optimal_linear_estimator import cross_fitted_ole_linear_fisher

    result = cross_fitted_ole_linear_fisher(
        theta_train,
        x_train,
        theta_grid,
        n_splits=int(n_splits),
        seed=int(seed),
        window_radius=window_radius,
        min_endpoint_samples=int(min_endpoint_samples),
        period=period,
    )
    fold_directions = np.asarray(result.fold_weights, dtype=np.float64)
    fold_norms = np.linalg.norm(fold_directions, axis=2, keepdims=True)
    if np.any(~np.isfinite(fold_norms)) or np.any(fold_norms <= 0.0):
        raise ValueError("Cross-fitted OLE produced a non-finite or zero fold direction.")
    # Held-out information is invariant to decoder scale. Average unit
    # directions so folds with small fitted information do not receive an
    # artificially large ensemble weight through OLE's calibration scale.
    directions = np.mean(fold_directions / fold_norms, axis=1)
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Cross-fitted OLE produced a non-finite or zero ensemble direction.")
    return result, directions / norms[:, None]


def gkr_checkpoint(model: Any) -> dict[str, Any]:
    """Return a portable state bundle for a fitted :class:`TorchGKR`."""

    return {
        "mean_model": None if model.mean_model is None else model.mean_model.state_dict(),
        "mean_likelihood": (
            None if model.mean_likelihood is None else model.mean_likelihood.state_dict()
        ),
        "covariance_model": model.covariance_model.state_dict(),
        "output_mean": model.output_mean.detach().cpu(),
        "output_std": model.output_std.detach().cpu(),
        "mean_loss": np.asarray(model.mean_loss_history, dtype=np.float64),
        "covariance_loss": np.asarray(model.covariance_loss_history, dtype=np.float64),
        "config": dict(vars(model.config)),
        "n_input": int(model.n_input),
        "n_output": int(model.n_output),
        "circular_period": model.circular_period,
        "seed": int(model.seed),
    }
