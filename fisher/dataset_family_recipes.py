"""Fixed per-`dataset_family` tuning + noise hyperparameters (not user-configurable via CLI).

`make_dataset` and `meta_dict_from_args` apply these recipes so `--dataset-family` is the only
behavior selector for dataset generation.
"""

from __future__ import annotations

import math
from typing import Any, Final

# Breaking rename (2026): old meta/CLI strings -> new canonical token (for migration errors only).
LEGACY_DATASET_FAMILY_TO_CANONICAL: Final[dict[str, str]] = {
    "gaussian": "cosine_gaussian",
    "gaussian_sqrtd": "cosine_gaussian_sqrtd",
    "gaussian_randamp": "randamp_gaussian",
    "gaussian_randamp_sqrtd": "randamp_gaussian_sqrtd",
    "gmm_non_gauss": "cosine_gmm",
    "cos_sin_piecewise_noise": "cos_sin_piecewise",
    "linear_piecewise_noise": "linear_piecewise",
}


def raise_if_legacy_dataset_family(family: str) -> None:
    """Raise ``ValueError`` if ``family`` is a pre-rename token (NPZ meta or manual namespace)."""
    fam = str(family)
    if fam in LEGACY_DATASET_FAMILY_TO_CANONICAL:
        new = LEGACY_DATASET_FAMILY_TO_CANONICAL[fam]
        raise ValueError(
            f"dataset_family={fam!r} was renamed; use {new!r} instead. "
            "Regenerate the archive with bin/make_dataset.py --dataset-family <new name>, "
            "or update --dataset-family / meta to the new token."
        )


# CLI flags removed from `add_dataset_arguments`; if present on argv, fail with a clear message.
LEGACY_DATASET_CLI_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--tuning-curve-family",
        "--vm-mu-amp",
        "--vm-kappa",
        "--vm-omega",
        "--gauss-mu-amp",
        "--gauss-kappa",
        "--gauss-omega",
        "--randamp-mu-low",
        "--randamp-mu-high",
        "--randamp-kappa",
        "--randamp-omega",
        "--sigma-x1",
        "--sigma-x2",
        "--rho",
        "--cov-theta-amp1",
        "--cov-theta-amp2",
        "--cov-theta-amp-rho",
        "--cov-theta-freq1",
        "--cov-theta-freq2",
        "--cov-theta-freq-rho",
        "--cov-theta-phase1",
        "--cov-theta-phase2",
        "--cov-theta-phase-rho",
        "--rho-clip",
        "--gmm-sep-scale",
        "--gmm-sep-freq",
        "--gmm-sep-phase",
        "--gmm-mix-logit-scale",
        "--gmm-mix-bias",
        "--gmm-mix-freq",
        "--gmm-mix-phase",
        "--sigma-piecewise-low",
        "--sigma-piecewise-high",
        "--linear-k",
        "--linear-sigma-schedule",
        "--linear-sigma-sigmoid-center",
        "--linear-sigma-sigmoid-steepness",
        "--theta-zero-to-low",
        "--no-theta-zero-to-low",
    }
)


def _base_gaussian_like() -> dict[str, Any]:
    """Defaults shared by Gaussian-branch families (matches former argparse defaults)."""
    return {
        "tuning_curve_family": "cosine",
        "vm_mu_amp": 1.0,
        "vm_kappa": 1.0,
        "vm_omega": 1.0,
        "gauss_mu_amp": 1.0,
        "gauss_kappa": 0.2,
        "gauss_omega": 1.0,
        "rho": 0.15,
        "rho_clip": 0.85,
        "cov_theta_amp1": 0.35,
        "cov_theta_amp2": 0.30,
        "cov_theta_amp_rho": 0.30,
        "cov_theta_freq1": 0.90,
        "cov_theta_freq2": 0.75,
        "cov_theta_freq_rho": 1.10,
        "cov_theta_phase1": 0.20,
        "cov_theta_phase2": -0.35,
        "cov_theta_phase_rho": 0.40,
        "randamp_mu_low": 0.5,
        "randamp_mu_high": 1.5,
        "randamp_kappa": 0.2,
        "randamp_omega": 1.0,
        "gmm_sep_scale": 1.10,
        "gmm_sep_freq": 0.85,
        "gmm_sep_phase": 0.35,
        "gmm_mix_logit_scale": 1.40,
        "gmm_mix_bias": 0.00,
        "gmm_mix_freq": 0.95,
        "gmm_mix_phase": -0.20,
        "sigma_piecewise_low": 0.1,
        "sigma_piecewise_high": 0.1,
        "linear_k": 1.0,
        "linear_sigma_schedule": "linear",
        "linear_sigma_sigmoid_center": 0.0,
        "linear_sigma_sigmoid_steepness": 2.0,
        "theta_zero_to_low": True,
    }


def family_recipe_dict(family: str) -> dict[str, Any]:
    """Return fixed hyperparameters for `family` (excluding seed/theta bounds/x_dim/n_total/train_frac)."""
    fam = str(family)
    raise_if_legacy_dataset_family(fam)
    base = _base_gaussian_like()
    if fam == "cosine_gaussian":
        out = {**base, "sigma_x1": 0.50, "sigma_x2": 0.50}
        return out
    if fam == "cosine_gaussian_const_noise":
        return {
            **base,
            "sigma_x1": 0.50,
            "sigma_x2": 0.50,
            "cov_theta_amp1": 0.0,
            "cov_theta_amp2": 0.0,
            "cov_theta_amp_rho": 0.0,
        }
    if fam == "cosine_gaussian_sqrtd":
        return {**base, "sigma_x1": 0.50, "sigma_x2": 0.50}
    if fam == "cosine_gaussian_sqrtd_rand_tune":
        return {
            **base,
            "sigma_x1": 0.50,
            "sigma_x2": 0.50,
            "cosine_tune_amp_low": 0.5,
            "cosine_tune_amp_high": 1.5,
        }
    if fam == "randamp_gaussian":
        return {**base, "sigma_x1": 0.30, "sigma_x2": 0.30}
    if fam == "randamp_gaussian_sqrtd":
        return {**base, "sigma_x1": 0.20, "sigma_x2": 0.20}
    if fam == "randamp_gaussian_sqrtd_realnvp":
        return {**base, "sigma_x1": 0.20, "sigma_x2": 0.20}
    if fam == "cosine_gmm":
        return {**base, "sigma_x1": 0.30, "sigma_x2": 0.30}
    if fam == "cos_sin_piecewise":
        return {**base, "sigma_x1": 0.30, "sigma_x2": 0.30}
    if fam == "linear_piecewise":
        return {**base, "sigma_x1": 0.30, "sigma_x2": 0.30}
    raise ValueError(f"Unknown dataset_family for recipe: {family!r}")


def apply_family_recipe_to_namespace(ns: Any) -> None:
    """Mutate argparse-like namespace with the fixed recipe for ``ns.dataset_family``."""
    fam = str(getattr(ns, "dataset_family", "cosine_gaussian"))
    recipe = family_recipe_dict(fam)
    for k, v in recipe.items():
        setattr(ns, k, v)
    scale = float(getattr(ns, "obs_noise_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("--obs-noise-scale must be a finite positive number.")
    if scale != 1.0:
        setattr(ns, "sigma_x1", float(getattr(ns, "sigma_x1")) * scale)
        setattr(ns, "sigma_x2", float(getattr(ns, "sigma_x2")) * scale)


def assert_no_legacy_dataset_cli_flags(argv: list[str]) -> None:
    """Raise ``ValueError`` if argv contains removed dataset composition flags."""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            break
        if tok.startswith("--"):
            key = tok.split("=", 1)[0]
            if key in LEGACY_DATASET_CLI_FLAGS:
                raise ValueError(
                    f"Removed CLI option {key!r} is no longer supported. "
                    "Dataset tuning and noise are fixed per --dataset-family. "
                    "Pass only generic options (e.g. --seed, --n-total, --theta-low, --x-dim, --output-npz) "
                    "plus --dataset-family. "
                    "See fisher.dataset_family_recipes.family_recipe_dict for the internal defaults."
                )
        i += 1


def format_resolved_family_summary(ns: Any) -> str:
    """Human-readable summary of the fixed recipe (for logging)."""
    fam = str(getattr(ns, "dataset_family", "cosine_gaussian"))
    r = family_recipe_dict(fam)
    scale = float(getattr(ns, "obs_noise_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    sx1 = float(r["sigma_x1"]) * scale
    sx2 = float(r["sigma_x2"]) * scale
    lines = [
        f"dataset_family={fam} (fixed internal recipe; not configurable via CLI)",
        f"  tuning_curve_family={r['tuning_curve_family']!r}",
        f"  observation noise: sigma_x1={sx1}, sigma_x2={sx2}, rho={r['rho']}  (obs_noise_scale={scale})",
    ]
    if fam in ("randamp_gaussian", "randamp_gaussian_sqrtd", "randamp_gaussian_sqrtd_realnvp"):
        lines.append(
            f"  randamp bumps: low={r['randamp_mu_low']}, high={r['randamp_mu_high']}, "
            f"kappa={r['randamp_kappa']}, omega={r['randamp_omega']}"
        )
    if fam == "randamp_gaussian_sqrtd_realnvp":
        lines.append(
            "  realnvp embedding: fixed untrained map with z_dim=2, n_transforms=6, hidden_width=128"
        )
    if fam == "cosine_gaussian_sqrtd_rand_tune":
        lines.append(
            f"  cosine per-dim amp: Uniform({r['cosine_tune_amp_low']}, {r['cosine_tune_amp_high']}) "
            "(fixed across samples; stored in NPZ meta)"
        )
    if fam == "cosine_gaussian_const_noise":
        lines.append("  constant noise: cov_theta_amp1=cov_theta_amp2=0 (no activity-coupled variance modulation)")
    if fam == "cosine_gmm":
        lines.append(
            f"  gmm: sep_scale={r['gmm_sep_scale']}, mix_logit_scale={r['gmm_mix_logit_scale']}, ..."
        )
    if fam in ("cos_sin_piecewise", "linear_piecewise"):
        lines.append(
            f"  piecewise: sigma_low={r['sigma_piecewise_low']}, sigma_high={r['sigma_piecewise_high']}, "
            f"theta_zero_to_low={r['theta_zero_to_low']}"
        )
    if fam == "linear_piecewise":
        lines.append(
            f"  linear_piecewise: linear_k={r['linear_k']}, "
            f"linear_sigma_schedule={r['linear_sigma_schedule']!r}"
        )
    return "\n".join(lines)
