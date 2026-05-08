"""Hard-separated mean sampling for ``ToyCategoricalRandomMoGDataset``."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.shared_dataset_io import meta_dict_from_args
from fisher.shared_fisher_est import build_dataset_from_meta


def test_explicit_means_load_unchanged_even_if_close() -> None:
    means = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]], dtype=np.float64)
    gains = np.ones_like(means)
    ds = ToyCategoricalRandomMoGDataset(
        x_dim=2,
        num_categories=3,
        mog_component_means=means,
        mog_component_gains=gains,
        mog_mean_min_dist=1.0e6,
        mog_mean_max_attempts=3,
        seed=0,
    )
    assert np.allclose(ds._mog_means, means)


def test_impossible_min_dist_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="could not sample"):
        ToyCategoricalRandomMoGDataset(
            x_dim=2,
            num_categories=5,
            mog_mean_min_dist=1.0e6,
            mog_mean_max_attempts=50,
            seed=123,
        )


def test_default_separation_pairwise_ge_threshold() -> None:
    ds = ToyCategoricalRandomMoGDataset(
        x_dim=2,
        num_categories=5,
        seed=999,
        mog_mean_max_attempts=200_000,
    )
    mu = ds._mog_means
    min_d = 0.5 * math.sqrt(2.0)
    for i in range(mu.shape[0]):
        for j in range(i + 1, mu.shape[0]):
            assert float(np.linalg.norm(mu[i] - mu[j])) + 1e-9 >= min_d


def test_meta_roundtrip_preserves_separation_and_meta_keys() -> None:
    p = argparse.ArgumentParser()
    add_dataset_arguments(p)
    ns = p.parse_args(
        [
            "--dataset-family",
            "random_mog_categorical",
            "--x-dim",
            "2",
            "--num-categories",
            "5",
            "--n-total",
            "200",
            "--train-frac",
            "0.7",
            "--seed",
            "0",
        ]
    )
    meta = meta_dict_from_args(ns)
    assert meta["mog_mean_min_dist"] is not None
    assert meta["mog_mean_max_attempts"] == 10_000
    ds = build_dataset_from_meta(meta)
    assert isinstance(ds, ToyCategoricalRandomMoGDataset)
    thr = float(meta["mog_mean_min_dist"])
    mu = ds._mog_means
    for i in range(mu.shape[0]):
        for j in range(i + 1, mu.shape[0]):
            assert float(np.linalg.norm(mu[i] - mu[j])) + 1e-9 >= thr
