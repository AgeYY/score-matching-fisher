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

# Dataset tokens removed from the CLI; archives using them must be regenerated.
REMOVED_DATASET_FAMILIES: Final[dict[str, str]] = {
    "randamp_gaussian_sqrtd_pr_autoencoder": (
        "Use a two-step workflow: (1) `python bin/make_dataset.py --dataset-family randamp_gaussian_sqrtd ...` "
        "for low-dimensional x, then (2) `python bin/project_dataset_pr_autoencoder.py --input-npz ... "
        "--output-npz ... --h-dim ...` to embed into high-dimensional observation space."
    ),
}


def raise_if_removed_dataset_family(family: str) -> None:
    fam = str(family)
    if fam in REMOVED_DATASET_FAMILIES:
        raise ValueError(f"dataset_family={fam!r} is no longer supported. {REMOVED_DATASET_FAMILIES[fam]}")


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
        "randamp_mu_low": 0.2,
        "randamp_mu_high": 2.0,
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
    raise_if_removed_dataset_family(fam)
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
            # Stronger activity coupling (alpha = 0.5*(amp1+amp2)) vs base Gaussian recipe,
            # aligned with randamp_gaussian_sqrtd for comparable |mu|-driven noise modulation.
            "cov_theta_amp1": 0.70,
            "cov_theta_amp2": 0.60,
            "cosine_tune_amp_low": 0.2,
            "cosine_tune_amp_high": 2.0,
            "cosine_sqrtd_obs_var_mu_law": "legacy_multiplicative_sqrtd",
        }
    if fam == "cosine_gaussian_sqrtd_rand_tune_additive":
        return {
            **base,
            "sigma_x1": 0.50,
            "sigma_x2": 0.50,
            "cov_theta_amp1": 0.70,
            "cov_theta_amp2": 0.60,
            "cosine_tune_amp_low": 0.2,
            "cosine_tune_amp_high": 2.0,
            "cosine_sqrtd_obs_var_mu_law": "additive_abs_mu",
        }
    if fam == "randamp_gaussian":
        return {**base, "sigma_x1": 0.30, "sigma_x2": 0.30}
    if fam == "randamp_gaussian_sqrtd":
        # Half baseline variance (sigma_base^2) vs legacy 0.20; double activity alpha
        # (alpha = 0.5 * (cov_theta_amp1 + cov_theta_amp2)) vs base 0.35/0.30.
        _s = 0.20 / (2.0**0.5)
        return {
            **base,
            "sigma_x1": _s,
            "sigma_x2": _s,
            "cov_theta_amp1": 0.70,
            "cov_theta_amp2": 0.60,
        }
    if fam == "randamp_gaussian2d_sqrtd":
        _s = 0.20 / math.sqrt(2.0)
        return {
            **base,
            "sigma_x1": _s,
            "sigma_x2": _s,
            "cov_theta_amp1": 0.70,
            "cov_theta_amp2": 0.60,
        }
    if fam == "gridcos_gaussian2d_sqrtd_rand_tune_additive":
        return {
            **base,
            "sigma_x1": 0.50,
            "sigma_x2": 0.50,
            "cov_theta_amp1": 0.70,
            "cov_theta_amp2": 0.60,
            "cosine_tune_amp_low": 0.2,
            "cosine_tune_amp_high": 2.0,
            "cosine_sqrtd_obs_var_mu_law": "additive_abs_mu",
        }
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
    amp_scale = float(getattr(ns, "cov_theta_amp_scale", 1.0))
    if not math.isfinite(amp_scale) or amp_scale <= 0.0:
        raise ValueError("--cov-theta-amp-scale must be a finite positive number.")
    if amp_scale != 1.0:
        setattr(ns, "cov_theta_amp1", float(getattr(ns, "cov_theta_amp1")) * amp_scale)
        setattr(ns, "cov_theta_amp2", float(getattr(ns, "cov_theta_amp2")) * amp_scale)


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
    amp_scale = float(getattr(ns, "cov_theta_amp_scale", 1.0))
    if not math.isfinite(amp_scale) or amp_scale <= 0.0:
        amp_scale = 1.0
    eff_amp1 = float(r["cov_theta_amp1"]) * amp_scale
    eff_amp2 = float(r["cov_theta_amp2"]) * amp_scale
    alpha_mean = 0.5 * (eff_amp1 + eff_amp2)
    lines = [
        f"dataset_family={fam} (fixed internal recipe; not configurable via CLI)",
        f"  tuning_curve_family={r['tuning_curve_family']!r}",
        f"  observation noise: sigma_x1={sx1}, sigma_x2={sx2}, rho={r['rho']}  (obs_noise_scale={scale})",
        f"  theta-variance coupling: cov_theta_amp1={eff_amp1}, cov_theta_amp2={eff_amp2} "
        f"(recipe * cov_theta_amp_scale={amp_scale}); alpha_mean_activity={alpha_mean}",
    ]
    if fam in (
        "randamp_gaussian",
        "randamp_gaussian_sqrtd",
        "randamp_gaussian2d_sqrtd",
    ):
        lines.append(
            f"  randamp bumps: low={r['randamp_mu_low']}, high={r['randamp_mu_high']}, "
            f"kappa={r['randamp_kappa']}, omega={r['randamp_omega']}"
        )
    if fam in (
        "cosine_gaussian_sqrtd_rand_tune",
        "cosine_gaussian_sqrtd_rand_tune_additive",
        "gridcos_gaussian2d_sqrtd_rand_tune_additive",
    ):
        law_note = (
            "additive |mu| term (same law token as randamp_gaussian_sqrtd additive)"
            if fam in ("cosine_gaussian_sqrtd_rand_tune_additive", "gridcos_gaussian2d_sqrtd_rand_tune_additive")
            else "legacy multiplicative (1+alpha|mu|) inside d*sigma^2 term"
        )
        cta_scale = float(getattr(ns, "cosine_tune_amp_scale", 1.0))
        if not math.isfinite(cta_scale) or cta_scale <= 0.0:
            cta_scale = 1.0
        lines.append(
            f"  cosine per-dim amp (randamp-style range): Uniform({r['cosine_tune_amp_low']}, "
            f"{r['cosine_tune_amp_high']}) * cosine_tune_amp_scale={cta_scale} (after draw; fixed across samples; "
            f"stored in NPZ meta as cosine_tune_amp_per_dim); sqrt-d variance law: {law_note}"
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
