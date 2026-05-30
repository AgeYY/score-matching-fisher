"""VAE latent feature helpers for categorical/continuous H-decoding estimators."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from fisher.ctsm_models import ToyBinaryTimeScoreNet
from fisher.gaussian_network import (
    ObservationVariationalAutoencoder,
    encode_observation_vae_means,
    sample_observation_vae_latents,
    train_observation_variational_autoencoder,
)
from fisher.shared_fisher_est import estimate_binary_ctsm_v_log_ratio, train_binary_ctsm_v_model


def logmeanexp_rows(values: np.ndarray, *, n_rows: int, n_samples: int) -> np.ndarray:
    """Stable row-wise log(mean(exp(.))) for row-major MC samples."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    nr = int(n_rows)
    ns = int(n_samples)
    if nr < 1 or ns < 1:
        raise ValueError("n_rows and n_samples must be >= 1.")
    if arr.size != nr * ns:
        raise ValueError(f"values length {arr.size} must equal n_rows*n_samples={nr * ns}.")
    mat = arr.reshape(nr, ns)
    maxv = np.max(mat, axis=1)
    return maxv + np.log(np.mean(np.exp(mat - maxv[:, None]), axis=1))


def _as_2d_float64(arr: np.ndarray, *, name: str, method_name: str = "vae_ctsm_v") -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{method_name} expects {name} to be 2D.")
    return out


def _vae_latent_dim(args: argparse.Namespace, x_dim: int) -> int:
    raw = int(getattr(args, "vae_latent_dim", 0))
    latent_dim = min(5, int(x_dim)) if raw <= 0 else raw
    if latent_dim < 1 or latent_dim > int(x_dim):
        raise ValueError(f"--vae-latent-dim must be 0 or in [1, {int(x_dim)}]; got {raw}.")
    return int(latent_dim)


def prepare_vae_latent_features(
    args: argparse.Namespace,
    *,
    device: torch.device,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_eval: np.ndarray,
) -> dict[str, Any]:
    """Train an observation VAE and return sampled latent features for CTSM-v."""
    train = _as_2d_float64(x_train, name="x_train")
    val = _as_2d_float64(x_val, name="x_val")
    ev = _as_2d_float64(x_eval, name="x_eval")
    x_dim = int(train.shape[1])
    if val.shape[1] != x_dim or ev.shape[1] != x_dim:
        raise ValueError("vae_ctsm_v x dimension mismatch across train/validation/eval.")
    if int(getattr(args, "vae_n_samples", 16)) < 1:
        raise ValueError("--vae-n-samples must be >= 1.")
    if int(getattr(args, "latent_n_mc_eval", 16)) < 1:
        raise ValueError("--latent-n-mc-eval must be >= 1.")

    latent_dim = _vae_latent_dim(args, x_dim)
    model = ObservationVariationalAutoencoder(
        x_dim=x_dim,
        latent_dim=latent_dim,
        hidden_dim=int(getattr(args, "vae_hidden_dim", 128)),
        depth=int(getattr(args, "vae_depth", 4)),
    ).to(device)
    train_out = train_observation_variational_autoencoder(
        model=model,
        x_train=train,
        x_val=val,
        device=device,
        epochs=int(getattr(args, "vae_epochs", 5000)),
        batch_size=int(getattr(args, "vae_batch_size", 256)),
        lr=float(getattr(args, "vae_lr", 1e-3)),
        beta=float(getattr(args, "vae_kl_weight", 0.01)),
        weight_decay=float(getattr(args, "vae_weight_decay", 0.0)),
        patience=int(getattr(args, "vae_early_patience", 500)),
        min_delta=float(getattr(args, "vae_early_min_delta", 1e-4)),
        ema_alpha=float(getattr(args, "vae_early_ema_alpha", 0.05)),
        log_every=max(1, int(getattr(args, "log_every", 50))),
        restore_best=True,
    )
    n_samples = int(getattr(args, "vae_n_samples", 16))
    n_mc_eval = int(getattr(args, "latent_n_mc_eval", 16))
    batch_size = int(getattr(args, "vae_batch_size", 256))
    return {
        "x_train": sample_observation_vae_latents(
            model=model,
            x=train,
            device=device,
            n_samples=n_samples,
            batch_size=batch_size,
        ),
        "x_validation": sample_observation_vae_latents(
            model=model,
            x=val,
            device=device,
            n_samples=n_samples,
            batch_size=batch_size,
        ),
        "x_eval_samples": sample_observation_vae_latents(
            model=model,
            x=ev,
            device=device,
            n_samples=n_mc_eval,
            batch_size=batch_size,
        ),
        "x_train_mean": encode_observation_vae_means(model=model, x=train, device=device, batch_size=batch_size),
        "x_validation_mean": encode_observation_vae_means(model=model, x=val, device=device, batch_size=batch_size),
        "x_eval_mean": encode_observation_vae_means(model=model, x=ev, device=device, batch_size=batch_size),
        "x_dim": latent_dim,
        "input_x_dim": x_dim,
        "train_out": train_out,
        "n_samples": n_samples,
        "n_mc_eval": n_mc_eval,
    }


def prepare_vae_mean_features(
    args: argparse.Namespace,
    *,
    device: torch.device,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_eval: np.ndarray,
) -> dict[str, Any]:
    """Train an observation VAE and return deterministic posterior-mean features."""
    train = _as_2d_float64(x_train, name="x_train", method_name="vae_preprocess")
    val = _as_2d_float64(x_val, name="x_val", method_name="vae_preprocess")
    ev = _as_2d_float64(x_eval, name="x_eval", method_name="vae_preprocess")
    x_dim = int(train.shape[1])
    if val.shape[1] != x_dim or ev.shape[1] != x_dim:
        raise ValueError("vae_preprocess x dimension mismatch across train/validation/eval.")

    latent_dim = _vae_latent_dim(args, x_dim)
    model = ObservationVariationalAutoencoder(
        x_dim=x_dim,
        latent_dim=latent_dim,
        hidden_dim=int(getattr(args, "vae_hidden_dim", 128)),
        depth=int(getattr(args, "vae_depth", 4)),
    ).to(device)
    train_out = train_observation_variational_autoencoder(
        model=model,
        x_train=train,
        x_val=val,
        device=device,
        epochs=int(getattr(args, "vae_epochs", 5000)),
        batch_size=int(getattr(args, "vae_batch_size", 256)),
        lr=float(getattr(args, "vae_lr", 1e-3)),
        beta=float(getattr(args, "vae_kl_weight", 0.01)),
        weight_decay=float(getattr(args, "vae_weight_decay", 0.0)),
        patience=int(getattr(args, "vae_early_patience", 500)),
        min_delta=float(getattr(args, "vae_early_min_delta", 1e-4)),
        ema_alpha=float(getattr(args, "vae_early_ema_alpha", 0.05)),
        log_every=max(1, int(getattr(args, "log_every", 50))),
        restore_best=True,
    )
    batch_size = int(getattr(args, "vae_batch_size", 256))
    return {
        "x_train": encode_observation_vae_means(model=model, x=train, device=device, batch_size=batch_size),
        "x_validation": encode_observation_vae_means(model=model, x=val, device=device, batch_size=batch_size),
        "x_eval": encode_observation_vae_means(model=model, x=ev, device=device, batch_size=batch_size),
        "x_train_mean": encode_observation_vae_means(model=model, x=train, device=device, batch_size=batch_size),
        "x_validation_mean": encode_observation_vae_means(model=model, x=val, device=device, batch_size=batch_size),
        "x_eval_mean": encode_observation_vae_means(model=model, x=ev, device=device, batch_size=batch_size),
        "x_dim": latent_dim,
        "input_x_dim": x_dim,
        "train_out": train_out,
        "n_samples": 1,
        "n_mc_eval": 1,
    }


def binary_llr_1_minus_0_to_delta_l(llr_1_minus_0: np.ndarray, bins_all: np.ndarray) -> np.ndarray:
    r = np.asarray(llr_1_minus_0, dtype=np.float64).reshape(-1)
    bins = np.asarray(bins_all, dtype=np.int64).reshape(-1)
    if bins.shape[0] != r.shape[0]:
        raise ValueError("vae_ctsm_v LLR vector length must match bins_all.")
    if np.any((bins != 0) & (bins != 1)):
        raise ValueError("vae_ctsm_v delta transform expects binary bins 0/1.")
    target_sign = (bins == 1).astype(np.float64)
    delta = r.reshape(-1, 1) * (target_sign.reshape(1, -1) - target_sign.reshape(-1, 1))
    np.fill_diagonal(delta, 0.0)
    return delta.astype(np.float64, copy=False)


def _vae_payload(vae: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_out = dict(vae.get("train_out", {}))
    return {
        "enabled": True,
        "latent_dim": int(vae["x_dim"]),
        "input_x_dim": int(vae["input_x_dim"]),
        "x_train_mean": np.asarray(vae["x_train_mean"], dtype=np.float64),
        "x_validation_mean": np.asarray(vae["x_validation_mean"], dtype=np.float64),
        "x_eval_mean": np.asarray(vae["x_eval_mean"], dtype=np.float64),
        "train_losses": np.asarray(train_out.get("train_losses", []), dtype=np.float64),
        "train_recon_losses": np.asarray(train_out.get("train_recon_losses", []), dtype=np.float64),
        "train_kl_losses": np.asarray(train_out.get("train_kl_losses", []), dtype=np.float64),
        "val_losses": np.asarray(train_out.get("val_losses", []), dtype=np.float64),
        "val_recon_losses": np.asarray(train_out.get("val_recon_losses", []), dtype=np.float64),
        "val_kl_losses": np.asarray(train_out.get("val_kl_losses", []), dtype=np.float64),
        "val_monitor_losses": np.asarray(train_out.get("val_monitor_losses", []), dtype=np.float64),
        "best_val_loss": float(train_out.get("best_val_loss", float("nan"))),
        "best_epoch": int(train_out.get("best_epoch", 0)),
        "stopped_epoch": int(train_out.get("stopped_epoch", 0)),
        "stopped_early": bool(train_out.get("stopped_early", False)),
        "kl_weight": float(getattr(args, "vae_kl_weight", 0.01)),
        "n_samples": int(vae["n_samples"]),
        "n_mc_eval": int(vae["n_mc_eval"]),
    }


def train_vae_ctsm_v_binary_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_val: np.ndarray,
    bins_val: np.ndarray,
    x_all: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> dict[str, Any]:
    """Train VAE latents + binary CTSM-v and return categorical delta_l."""
    if int(k_cat) != 2:
        raise ValueError("vae_ctsm_v currently supports exactly two categories.")
    bins_train = np.asarray(bins_train, dtype=np.int64).reshape(-1)
    bins_val = np.asarray(bins_val, dtype=np.int64).reshape(-1)
    bins_all = np.asarray(bins_all, dtype=np.int64).reshape(-1)
    x_train = _as_2d_float64(x_train, name="x_train")
    x_val = _as_2d_float64(x_val, name="x_val")
    x_all = _as_2d_float64(x_all, name="x_all")
    if x_train.shape[0] != bins_train.size or x_val.shape[0] != bins_val.size or x_all.shape[0] != bins_all.size:
        raise ValueError("vae_ctsm_v bin arrays must match x rows.")
    if x_train.shape[1] != x_val.shape[1] or x_train.shape[1] != x_all.shape[1]:
        raise ValueError("vae_ctsm_v x dimension mismatch across train/validation/eval.")

    vae = prepare_vae_latent_features(args, device=dev, x_train=x_train, x_val=x_val, x_eval=x_all)
    z_train = np.asarray(vae["x_train"], dtype=np.float64)
    z_val = np.asarray(vae["x_validation"], dtype=np.float64)
    z_eval_samples = np.asarray(vae["x_eval_samples"], dtype=np.float64)
    n_vae_samples = int(vae["n_samples"])
    n_mc_eval = int(vae["n_mc_eval"])
    bins_train_rep = np.repeat(bins_train, n_vae_samples)
    bins_val_rep = np.repeat(bins_val, n_vae_samples)
    train0 = np.flatnonzero(bins_train_rep == 0)
    train1 = np.flatnonzero(bins_train_rep == 1)
    val0 = np.flatnonzero(bins_val_rep == 0)
    val1 = np.flatnonzero(bins_val_rep == 1)
    min_count = int(getattr(args, "clf_min_class_count", 5))
    if train0.size < min_count or train1.size < min_count:
        raise ValueError(
            f"vae_ctsm_v requires at least {min_count} sampled latent training rows in each class; "
            f"got {train0.size} and {train1.size}."
        )
    x0_val = z_val[val0] if val0.size >= 1 and val1.size >= 1 else None
    x1_val = z_val[val1] if val0.size >= 1 and val1.size >= 1 else None
    print(
        f"[cat-twofig] training vae_ctsm_v: class0={train0.size} class1={train1.size} "
        f"val0={val0.size} val1={val1.size} eval={x_all.shape[0]} "
        f"x_dim={x_all.shape[1]} latent_dim={int(vae['x_dim'])}",
        flush=True,
    )
    model = ToyBinaryTimeScoreNet(
        dim=int(vae["x_dim"]),
        hidden_dim=int(getattr(args, "ctsm_hidden_dim", 256)),
    ).to(dev)
    train_out = train_binary_ctsm_v_model(
        model=model,
        x0_train=z_train[train0],
        x1_train=z_train[train1],
        epochs=int(getattr(args, "ctsm_binary_epochs", 50000)),
        batch_size=int(getattr(args, "ctsm_batch_size", 512)),
        lr=float(getattr(args, "ctsm_lr", 2e-3)),
        weight_decay=float(getattr(args, "ctsm_weight_decay", 0.0)),
        device=dev,
        log_every=max(1, int(getattr(args, "log_every", 50))),
        two_sb_var=float(getattr(args, "ctsm_two_sb_var", 2.0)),
        path_schedule=str(getattr(args, "ctsm_path_schedule", "linear")),
        path_eps=float(getattr(args, "ctsm_path_eps", 1e-12)),
        factor=float(getattr(args, "ctsm_factor", 1.0)),
        t_eps=float(getattr(args, "ctsm_t_eps", 1e-5)),
        x0_val=x0_val,
        x1_val=x1_val,
        early_stopping_patience=int(getattr(args, "flow_early_patience", 1000)),
        early_stopping_min_delta=float(getattr(args, "flow_early_min_delta", 1e-4)),
        early_stopping_ema_alpha=float(getattr(args, "flow_early_ema_alpha", 0.05)),
        restore_best=bool(getattr(args, "flow_restore_best", True)),
    )
    latent_llr_samples = estimate_binary_ctsm_v_log_ratio(
        model,
        z_eval_samples,
        device=dev,
        batch_size=int(getattr(args, "h_batch_size", 65536)),
        eps1=float(getattr(args, "ctsm_t_eps", 1e-5)),
        eps2=float(getattr(args, "ctsm_t_eps", 1e-5)),
        n_time=int(getattr(args, "ctsm_int_n_time", 300)),
    )
    llr_1_minus_0 = logmeanexp_rows(
        latent_llr_samples,
        n_rows=int(x_all.shape[0]),
        n_samples=n_mc_eval,
    )
    return {
        "c_matrix": None,
        "delta_l": binary_llr_1_minus_0_to_delta_l(llr_1_minus_0, bins_all),
        "ctsm_binary_llr_1_minus_0": np.asarray(llr_1_minus_0, dtype=np.float64),
        "vae_ctsm_binary_llr_1_minus_0": np.asarray(llr_1_minus_0, dtype=np.float64),
        "train_out": train_out,
        "vae_train_out": vae.get("train_out", {}),
        "vae_payload": _vae_payload(vae, args),
        "ctsm_theta_encoding": np.asarray(["none_binary_vae_ctsm_v"], dtype=object),
    }
