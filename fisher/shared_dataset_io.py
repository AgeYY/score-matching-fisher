"""Save/load shared (theta, x) splits for Fisher estimation pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from fisher.dataset_family_recipes import apply_family_recipe_to_namespace

SHARED_DATASET_NPZ_VERSION = 1


def apply_sigma_defaults_for_dataset_family(ns: Any) -> None:
    """Deprecated: sigma defaults are fixed per ``dataset_family`` via ``apply_family_recipe_to_namespace``.

    Kept as a no-op for backwards compatibility with callers that still invoke it.
    """
    apply_family_recipe_to_namespace(ns)


@dataclass(frozen=True)
class SharedDatasetBundle:
    meta: dict[str, Any]
    theta_all: np.ndarray
    x_all: np.ndarray
    train_idx: np.ndarray
    eval_idx: np.ndarray
    theta_train: np.ndarray
    x_train: np.ndarray
    theta_eval: np.ndarray
    x_eval: np.ndarray


def meta_dict_from_args(ns: Any) -> dict[str, Any]:
    """Build JSON-serializable metadata from an argparse-like namespace (dataset fields)."""
    apply_family_recipe_to_namespace(ns)
    out: dict[str, Any] = {
        "version": SHARED_DATASET_NPZ_VERSION,
        "dataset_family": str(ns.dataset_family),
        "tuning_curve_family": str(ns.tuning_curve_family),
        "vm_mu_amp": float(ns.vm_mu_amp),
        "vm_kappa": float(ns.vm_kappa),
        "vm_omega": float(ns.vm_omega),
        "gauss_mu_amp": float(getattr(ns, "gauss_mu_amp", 1.0)),
        "gauss_kappa": float(getattr(ns, "gauss_kappa", 0.2)),
        "gauss_omega": float(getattr(ns, "gauss_omega", 1.0)),
        "randamp_mu_low": float(getattr(ns, "randamp_mu_low", 0.5)),
        "randamp_mu_high": float(getattr(ns, "randamp_mu_high", 1.5)),
        "randamp_kappa": float(getattr(ns, "randamp_kappa", 0.2)),
        "randamp_omega": float(getattr(ns, "randamp_omega", 1.0)),
        "seed": int(ns.seed),
        "theta_low": float(ns.theta_low),
        "theta_high": float(ns.theta_high),
        "x_dim": int(ns.x_dim),
        "sigma_x1": float(ns.sigma_x1),
        "sigma_x2": float(ns.sigma_x2),
        "rho": float(ns.rho),
        "rho_clip": float(ns.rho_clip),
        "cov_theta_amp1": float(ns.cov_theta_amp1),
        "cov_theta_amp2": float(ns.cov_theta_amp2),
        "cov_theta_amp_rho": float(ns.cov_theta_amp_rho),
        "cov_theta_freq1": float(ns.cov_theta_freq1),
        "cov_theta_freq2": float(ns.cov_theta_freq2),
        "cov_theta_freq_rho": float(ns.cov_theta_freq_rho),
        "cov_theta_phase1": float(ns.cov_theta_phase1),
        "cov_theta_phase2": float(ns.cov_theta_phase2),
        "cov_theta_phase_rho": float(ns.cov_theta_phase_rho),
        "gmm_sep_scale": float(ns.gmm_sep_scale),
        "gmm_sep_freq": float(ns.gmm_sep_freq),
        "gmm_sep_phase": float(ns.gmm_sep_phase),
        "gmm_mix_logit_scale": float(ns.gmm_mix_logit_scale),
        "gmm_mix_bias": float(ns.gmm_mix_bias),
        "gmm_mix_freq": float(ns.gmm_mix_freq),
        "gmm_mix_phase": float(ns.gmm_mix_phase),
        "sigma_piecewise_low": float(ns.sigma_piecewise_low),
        "sigma_piecewise_high": float(ns.sigma_piecewise_high),
        "linear_k": float(getattr(ns, "linear_k", 1.0)),
        "linear_sigma_schedule": str(getattr(ns, "linear_sigma_schedule", "linear")),
        "linear_sigma_sigmoid_center": float(getattr(ns, "linear_sigma_sigmoid_center", 0.0)),
        "linear_sigma_sigmoid_steepness": float(getattr(ns, "linear_sigma_sigmoid_steepness", 2.0)),
        "theta_zero_to_low": bool(ns.theta_zero_to_low),
        "n_total": int(ns.n_total),
        "train_frac": float(ns.train_frac),
    }
    _ra = getattr(ns, "randamp_mu_amp_per_dim", None)
    if _ra is not None:
        out["randamp_mu_amp_per_dim"] = np.asarray(_ra, dtype=np.float64).reshape(-1).tolist()
    else:
        out["randamp_mu_amp_per_dim"] = None
    return out


def _shared_dataset_meta_keys() -> frozenset[str]:
    """Keys stored in shared dataset NPZ ``meta_json_utf8`` (excluding ``version``).

    Used by ``merge_meta_into_args`` so dataset metadata does not overwrite unrelated
    CLI fields (e.g. ``theta_field_method``).
    """
    dummy = SimpleNamespace(
        dataset_family="cosine_gaussian",
        seed=0,
        theta_low=-6.0,
        theta_high=6.0,
        x_dim=2,
        n_total=100,
        train_frac=1.0,
    )
    keys = set(meta_dict_from_args(dummy).keys())
    keys.discard("version")
    return frozenset(keys)


SHARED_DATASET_META_KEYS = _shared_dataset_meta_keys()


def save_shared_dataset_npz(
    path: str | Path,
    *,
    meta: dict[str, Any],
    theta_all: np.ndarray,
    x_all: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, sort_keys=True)
    meta_utf8 = meta_json.encode("utf-8")
    np.savez_compressed(
        path,
        meta_json_utf8=np.frombuffer(meta_utf8, dtype=np.uint8),
        theta_all=theta_all.astype(np.float64, copy=False),
        x_all=x_all.astype(np.float64, copy=False),
        train_idx=train_idx.astype(np.int64, copy=False),
        eval_idx=eval_idx.astype(np.int64, copy=False),
        theta_train=theta_train.astype(np.float64, copy=False),
        x_train=x_train.astype(np.float64, copy=False),
        theta_eval=theta_eval.astype(np.float64, copy=False),
        x_eval=x_eval.astype(np.float64, copy=False),
    )


def load_shared_dataset_npz(path: str | Path) -> SharedDatasetBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta_utf8 = np.asarray(data["meta_json_utf8"], dtype=np.uint8)
        meta_json = meta_utf8.tobytes().decode("utf-8")
        meta = json.loads(meta_json)
        ver = int(meta.get("version", 0))
        if ver != SHARED_DATASET_NPZ_VERSION:
            raise ValueError(
                f"Unsupported shared dataset npz version: {ver} (expected {SHARED_DATASET_NPZ_VERSION})"
            )
        bundle = SharedDatasetBundle(
            meta=meta,
            theta_all=np.asarray(data["theta_all"], dtype=np.float64),
            x_all=np.asarray(data["x_all"], dtype=np.float64),
            train_idx=np.asarray(data["train_idx"], dtype=np.int64),
            eval_idx=np.asarray(data["eval_idx"], dtype=np.int64),
            theta_train=np.asarray(data["theta_train"], dtype=np.float64),
            x_train=np.asarray(data["x_train"], dtype=np.float64),
            theta_eval=np.asarray(data["theta_eval"], dtype=np.float64),
            x_eval=np.asarray(data["x_eval"], dtype=np.float64),
        )
    return bundle
