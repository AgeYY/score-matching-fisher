"""Argparse helpers for shared-dataset Fisher scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from global_setting import DATA_DIR


def add_dataset_arguments(p: argparse.ArgumentParser) -> None:
    """Public dataset CLI: only `--dataset-family` plus generic sampling/shape controls.

    Tuning-curve shape, noise covariance details, randamp bounds, GMM/piecewise hyperparameters, etc.
    are fixed per family in ``fisher.dataset_family_recipes`` (not user-configurable).
    """
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Integer seed for NumPy RNG (joint sampling and train/validation split permutation).",
    )
    p.add_argument(
        "--dataset-family",
        type=str,
        default="cosine_gaussian",
        choices=[
            "cosine_gaussian",
            "cosine_gaussian_const_noise",
            "cosine_gaussian_sqrtd",
            "cosine_gaussian_sqrtd_rand_tune",
            "randamp_gaussian",
            "randamp_gaussian_sqrtd",
            "cosine_gmm",
            "cos_sin_piecewise",
            "linear_piecewise",
        ],
        help=(
            "Generative family (selects fixed tuning + noise internally). Options: "
            "'cosine_gaussian' (theta-modulated Gaussian obs. noise, cosine means); "
            "'cosine_gaussian_const_noise' (cosine means + constant Gaussian obs. noise); "
            "'cosine_gaussian_sqrtd' (same means; obs. noise std scales by sqrt(x_dim)); "
            "'cosine_gaussian_sqrtd_rand_tune' (like cosine_gaussian_sqrtd but per-dim cosine "
            "amplitudes drawn once from Uniform(0.5, 1.5)); "
            "'randamp_gaussian' (random-amplitude Gaussian bumps + Gaussian obs. noise); "
            "'randamp_gaussian_sqrtd' (same as randamp_gaussian with sqrt(x_dim) noise scaling). "
            "For PR-autoencoder embedding into higher-dimensional x, generate this family first, then run "
            "`bin/project_dataset_pr_autoencoder.py`; "
            "'cosine_gmm' (theta-dependent 2-component mixture); "
            "'cos_sin_piecewise' (cos/sin means + piecewise obs. std vs theta sign); "
            "'linear_piecewise' (linear means + piecewise obs. std vs theta)."
        ),
    )
    p.add_argument(
        "--theta-low",
        type=float,
        default=-6.0,
        help="Lower bound of theta; samples are drawn uniformly on [theta-low, theta-high].",
    )
    p.add_argument(
        "--theta-high",
        type=float,
        default=6.0,
        help="Upper bound of theta (see --theta-low).",
    )
    p.add_argument("--x-dim", type=int, default=2, help="Observation dimension (length of x).")
    p.add_argument(
        "--n-total",
        "--num-samples",
        type=int,
        default=3000,
        metavar="N",
        help=(
            "Total number of data points: joint (theta, x) samples drawn before the train/validation split. "
            "Same as --num-samples."
        ),
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help=(
            "Fraction of n_total in train_idx; remainder is validation_idx. "
            "Use a value in (0, 1) so score training, H-matrix evaluation, and pairwise CLF "
            "(train-fit / val-eval) share the same split. 1.0 leaves validation empty (not supported for shared Fisher)."
        ),
    )
    p.add_argument(
        "--obs-noise-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplies the family-fixed baseline observation-noise scales sigma_x1 and sigma_x2 "
            "after applying --dataset-family (default 1.0). Example: 0.5 halves Gaussian observation noise."
        ),
    )


def add_estimation_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gt-mc-samples-per-bin", type=int, default=6000)
    p.add_argument(
        "--theta-field-method",
        type=str,
        default="theta_flow",
        help=(
            "Likelihood-ratio field method: theta_flow (theta-space flow ODE log-likelihood Bayes ratios "
            "log p(theta|x)-log p(theta)), theta_path_integral (velocity-to-score plus trapezoid integral "
            "along sorted theta), x_flow (conditional x-space flow ODE log p(x|theta)), or "
            "ctsm_v (pair-conditioned CTSM-v time-score integration)."
        ),
    )

    p.add_argument("--score-epochs", type=int, default=10000)
    p.add_argument("--score-batch-size", type=int, default=1024)
    p.add_argument("--score-lr", type=float, default=1e-3)
    p.add_argument("--score-hidden-dim", type=int, default=128)
    p.add_argument("--score-depth", type=int, default=3)
    p.add_argument(
        "--dsm-stability-preset",
        type=str,
        default="stable_v1",
        choices=["legacy", "stable_v1"],
        help="Preset for DSM stability knobs (optimizer/scheduler/clipping/loss). Default stable_v1.",
    )
    p.add_argument(
        "--score-arch",
        type=str,
        default="film",
        choices=["mlp", "film"],
        help=(
            "Posterior DSM architecture: mlp concatenates [theta, x, sigma]; "
            "film uses an x-input trunk with residual FiLM blocks conditioned on (theta_tilde, sigma). "
            "Default: film."
        ),
    )
    p.add_argument(
        "--score-sigma-feature-mode",
        type=str,
        default="auto",
        choices=["auto", "log", "linear"],
        help="Feature transform for sigma in posterior DSM input. auto: log for continuous noise, linear otherwise.",
    )
    p.add_argument(
        "--prior-sigma-feature-mode",
        type=str,
        default="auto",
        choices=["auto", "log", "linear"],
        help="Feature transform for sigma in prior DSM input. auto: log for continuous noise, linear otherwise.",
    )
    p.add_argument("--score-use-layer-norm", action="store_true", default=False)
    p.add_argument("--prior-use-layer-norm", action="store_true", default=False)
    p.add_argument("--score-gated-film", action="store_true", default=False)
    p.add_argument("--prior-gated-film", action="store_true", default=False)
    p.add_argument("--score-zero-out-init", action="store_true", default=False)
    p.add_argument("--prior-zero-out-init", action="store_true", default=False)
    p.add_argument(
        "--prior-score-arch",
        type=str,
        default="mlp",
        choices=["mlp", "film"],
        help=(
            "Prior DSM architecture: mlp concatenates [theta, sigma]; "
            "film uses a theta_tilde trunk with residual FiLM blocks conditioned on (theta_tilde, sigma). "
            "Default: mlp."
        ),
    )
    p.add_argument("--score-early-patience", type=int, default=1500)
    p.add_argument("--score-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--score-early-ema-alpha",
        type=float,
        default=0.05,
        help="EMA smoothing factor α in (0,1] for validation loss monitor used by score early stopping.",
    )
    p.add_argument(
        "--score-early-ema-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Score early stopping: for epochs 1..N use raw validation loss as the monitor (no EMA); "
            "EMA accumulation starts after epoch N. Default 0 (no warmup)."
        ),
    )
    p.add_argument("--score-restore-best", action="store_true", default=True)
    p.add_argument("--no-score-restore-best", action="store_false", dest="score_restore_best")
    p.add_argument("--score-optimizer", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--prior-optimizer", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--score-weight-decay", type=float, default=0.0)
    p.add_argument("--prior-weight-decay", type=float, default=0.0)
    p.add_argument(
        "--score-lr-scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
    )
    p.add_argument(
        "--prior-lr-scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
    )
    p.add_argument("--score-lr-warmup-frac", type=float, default=0.0)
    p.add_argument("--prior-lr-warmup-frac", type=float, default=0.0)
    p.add_argument("--score-max-grad-norm", type=float, default=0.0)
    p.add_argument("--prior-max-grad-norm", type=float, default=0.0)
    p.add_argument("--score-abort-on-nonfinite", action="store_true", default=False)
    p.add_argument("--no-score-abort-on-nonfinite", action="store_false", dest="score_abort_on_nonfinite")
    p.add_argument("--prior-abort-on-nonfinite", action="store_true", default=False)
    p.add_argument("--no-prior-abort-on-nonfinite", action="store_false", dest="prior_abort_on_nonfinite")
    p.add_argument("--score-loss-type", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--prior-loss-type", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--score-huber-delta", type=float, default=1.0)
    p.add_argument("--prior-huber-delta", type=float, default=1.0)
    p.add_argument("--score-normalize-by-sigma", action="store_true", default=False)
    p.add_argument("--prior-normalize-by-sigma", action="store_true", default=False)
    p.add_argument(
        "--score-sigma-sample-mode",
        type=str,
        default="uniform_log",
        choices=["uniform_log", "beta_log"],
        help="Continuous DSM sigma sampling mode in log-space.",
    )
    p.add_argument("--score-sigma-sample-beta", type=float, default=2.0)
    p.add_argument("--score-noise-mode", type=str, default="continuous", choices=["discrete", "continuous"])
    p.add_argument(
        "--score-sigma-scale-mode",
        type=str,
        default="theta_std",
        choices=["theta_std", "posterior_proxy", "fixed"],
    )
    p.add_argument("--score-sigma-alpha-list", type=float, nargs="+", default=[0.08, 0.06, 0.045, 0.03, 0.02])
    p.add_argument(
        "--score-sigma-min-alpha",
        type=float,
        default=0.05,
        help="With --score-sigma-scale-mode theta_std: sigma_min = this × std(theta on score fit). Default 0.05 (5%%).",
    )
    p.add_argument(
        "--score-sigma-max-alpha",
        type=float,
        default=0.25,
        help="With --score-sigma-scale-mode theta_std: sigma_max = this × std(theta on score fit). Default 0.25 (25%%).",
    )
    p.add_argument("--score-eval-sigmas", type=int, default=12)
    p.add_argument("--score-proxy-l2", type=float, default=1e-3)
    p.add_argument("--score-proxy-min-mult", type=float, default=0.1)
    p.add_argument("--score-proxy-max-mult", type=float, default=2.0)
    p.add_argument("--score-fixed-sigma", type=float, default=0.02)
    p.add_argument(
        "--flow-epochs",
        type=int,
        default=10000,
        help=(
            "FM velocity pretraining epochs for flow-based fields. "
            "theta_flow: use 0 to skip pretraining and rely on --flow-likelihood-finetune-epochs (NLL) only."
        ),
    )
    p.add_argument("--flow-batch-size", type=int, default=256)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument(
        "--flow-likelihood-finetune-epochs",
        type=int,
        default=0,
        help=(
            "theta_flow only: optional second-stage fine-tune epochs that minimize ODE "
            "negative log likelihood after flow-matching pretraining. Default 0 disables. "
            "Maximum allowed value is 2000."
        ),
    )
    p.add_argument(
        "--flow-likelihood-finetune-lr",
        type=float,
        default=1e-4,
        help="theta_flow NLL fine-tune learning rate.",
    )
    p.add_argument(
        "--flow-likelihood-finetune-batch-size",
        type=int,
        default=0,
        help="theta_flow NLL fine-tune batch size; <=0 inherits --flow-batch-size.",
    )
    p.add_argument(
        "--flow-likelihood-finetune-ode-steps",
        type=int,
        default=64,
        help="theta_flow NLL fine-tune midpoint ODE steps for likelihood integration.",
    )
    p.add_argument(
        "--flow-likelihood-finetune-exact-divergence",
        action="store_true",
        default=True,
        help="theta_flow NLL fine-tune uses exact divergence (currently required and enabled by default).",
    )
    p.add_argument(
        "--no-flow-likelihood-finetune-exact-divergence",
        action="store_false",
        dest="flow_likelihood_finetune_exact_divergence",
        help="Disable exact divergence for theta_flow NLL fine-tune (currently unsupported; validation rejects).",
    )
    p.add_argument(
        "--flow-likelihood-exact-divergence",
        action="store_true",
        default=False,
        help=(
            "theta_flow / x_flow H-matrix evaluation: use exact_divergence=True in "
            "flow_matching ODESolver.compute_likelihood (per-dim Jacobian trace; slower than Hutchinson). "
            "Default False keeps Hutchinson for ODE log-density blocks."
        ),
    )
    p.add_argument("--flow-likelihood-finetune-patience", type=int, default=100)
    p.add_argument("--flow-likelihood-finetune-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-likelihood-finetune-ema-alpha", type=float, default=0.05)
    p.add_argument(
        "--flow-endpoint-loss-weight",
        type=float,
        default=0.0,
        help=(
            "theta_flow only: auxiliary conditional likelihood weight lambda for "
            "loss = flow_matching + lambda * (-mean log p(theta|x)). "
            "Default 0 disables; set >0 to enable."
        ),
    )
    p.add_argument(
        "--flow-endpoint-steps",
        type=int,
        default=20,
        help=(
            "theta_flow only: ODE integration steps for the auxiliary "
            "-mean log p(theta|x) likelihood term."
        ),
    )
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument("--flow-scheduler", type=str, default="cosine", choices=["cosine", "vp", "linear_vp"])
    p.add_argument(
        "--flow-eval-t",
        type=float,
        default=0.8,
        help="Fixed time t used to evaluate theta-flow velocity field for H-matrix (flow mode).",
    )
    p.add_argument(
        "--theta-flow-posterior-only-likelihood",
        action="store_true",
        default=False,
        help=(
            "theta_flow only (H-matrix): use posterior ODE log-density only for the likelihood matrix "
            "(c_ij = log p(theta_j|x_i)), instead of subtracting the learned prior flow log p(theta_j). "
            "Skips prior flow training, NLL fine-tune, checkpoint, and prior ODE likelihood; "
            "theta_flow_log_prior_matrix is omitted (None)."
        ),
    )
    p.add_argument("--flow-early-patience", type=int, default=1000)
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--flow-early-ema-alpha",
        type=float,
        default=0.05,
        help="EMA smoothing factor α in (0,1] for flow validation monitor used by early stopping.",
    )
    p.add_argument("--flow-restore-best", action="store_true", default=True)
    p.add_argument("--no-flow-restore-best", action="store_false", dest="flow_restore_best")
    p.add_argument(
        "--flow-x-two-stage-mean-theta-pretrain",
        action="store_true",
        default=False,
        help=(
            "x_flow only: split --flow-epochs 50/50 — stage 1 trains with theta fixed at "
            "mean(theta on score-fit split) (unconditional-like); stage 2 finetunes with true per-sample theta. "
            "Odd E: stage1=floor(E/2), stage2=E-stage1 (extra epoch to stage 2). Requires --flow-epochs >= 2."
        ),
    )
    p.add_argument(
        "--flow-arch",
        type=str,
        default="mlp",
        choices=["mlp", "soft_moe", "film", "film_fourier"],
        help=(
            "Flow architecture shared by theta_flow, theta_path_integral, and x_flow: "
            "mlp, soft_moe (dense soft gating over MLP experts), "
            "film (FiLM blocks with embedded raw theta), or "
            "film_fourier (FiLM blocks with Fourier theta features)."
        ),
    )
    p.add_argument(
        "--flow-moe-num-experts",
        type=int,
        default=4,
        help="Posterior soft_moe only: number of experts in dense softmax routing.",
    )
    p.add_argument(
        "--flow-moe-router-temperature",
        type=float,
        default=1.0,
        help="Posterior soft_moe only: router softmax temperature (>0).",
    )
    p.add_argument(
        "--flow-score-arch",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--flow-prior-arch",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--flow-gated-film",
        action="store_true",
        default=False,
        help="Posterior FiLM (flow-arch=film): use bounded multiplicative FiLM (tanh-gated gamma).",
    )
    p.add_argument(
        "--flow-prior-gated-film",
        action="store_true",
        default=False,
        help="Theta-flow FiLM prior: use bounded multiplicative FiLM (tanh-gated gamma).",
    )
    p.add_argument(
        "--flow-use-layer-norm",
        action="store_true",
        default=False,
        help="Posterior FiLM (flow-arch=film): LayerNorm on trunk and FiLM block outputs.",
    )
    p.add_argument(
        "--flow-prior-use-layer-norm",
        action="store_true",
        default=False,
        help="Theta-flow FiLM prior: LayerNorm on trunk and FiLM block outputs.",
    )
    p.add_argument(
        "--flow-zero-out-init",
        action="store_true",
        default=False,
        help="Posterior FiLM (flow-arch=film): zero-initialize final linear head.",
    )
    p.add_argument(
        "--flow-prior-zero-out-init",
        action="store_true",
        default=False,
        help="Theta-flow FiLM prior: zero-initialize final linear head.",
    )
    p.add_argument(
        "--flow-cond-embed-dim",
        type=int,
        default=16,
        help=(
            "Posterior FiLM (flow-arch=film): per-channel embedding width for theta_t and for logit(t); "
            "FiLM cond is concat(theta_embed, time_embed) with total dim 2×this. Default: 16."
        ),
    )
    p.add_argument(
        "--flow-cond-embed-depth",
        type=int,
        default=1,
        help="Posterior FiLM (flow-arch=film): number of linear layers in each scalar embedding MLP (theta and t). Default: 1.",
    )
    p.add_argument(
        "--flow-cond-embed-act",
        type=str,
        default="silu",
        choices=["silu", "relu", "tanh"],
        help="Posterior FiLM (flow-arch=film): activation between layers in each scalar embedding MLP (not after last layer).",
    )
    p.add_argument(
        "--flow-prior-cond-embed-dim",
        type=int,
        default=16,
        help=(
            "Theta-flow FiLM prior only: per-channel embedding width for theta_t and for logit(t); "
            "FiLM cond is concat(theta_embed, time_embed) with total dim 2×this. Default: 16."
        ),
    )
    p.add_argument(
        "--flow-prior-cond-embed-depth",
        type=int,
        default=1,
        help="Theta-flow FiLM prior: number of linear layers in each scalar embedding MLP (theta and t). Default: 1.",
    )
    p.add_argument(
        "--flow-prior-cond-embed-act",
        type=str,
        default="silu",
        choices=["silu", "relu", "tanh"],
        help="Theta-flow FiLM prior: activation between layers in each scalar embedding MLP (not after last layer).",
    )

    p.add_argument(
        "--flow-theta-fourier-k",
        type=int,
        default=4,
        help=(
            "theta_flow / theta_path_integral + --flow-arch film_fourier: number of harmonic pairs "
            "(sin, cos) for theta encoding in the posterior theta-flow. Ignored for other arch/methods."
        ),
    )
    p.add_argument(
        "--flow-theta-fourier-omega-mode",
        type=str,
        default="theta_range",
        choices=["theta_range", "fixed"],
        help=(
            "Posterior theta_flow film_fourier: how to set omega in sin(k*omega*theta). "
            "theta_range (default): omega_eff = (2*pi / (theta_high-theta_low)) * mult, where mult is "
            "--flow-theta-fourier-omega. "
            "fixed: use --flow-theta-fourier-omega as omega directly (e.g. 1 gives period 2*pi for k=1)."
        ),
    )
    p.add_argument(
        "--flow-theta-fourier-omega",
        type=float,
        default=1.0,
        help=(
            "Posterior theta_flow film_fourier: with --flow-theta-fourier-omega-mode fixed, this is omega. "
            "With theta_range, this is a positive multiplier mult on (2*pi/span) (default 1 => fundamental period "
            "equals theta_high-theta_low)."
        ),
    )
    p.add_argument(
        "--flow-theta-fourier-no-linear",
        action="store_true",
        default=False,
        help="Posterior theta_flow film_fourier: drop raw theta from the theta feature vector.",
    )
    p.add_argument(
        "--flow-theta-fourier-no-bias",
        action="store_true",
        default=False,
        help="Posterior theta_flow film_fourier: drop constant 1 from the theta feature vector.",
    )
    p.add_argument(
        "--flow-prior-theta-fourier-k",
        type=int,
        default=4,
        help=(
            "theta_flow / theta_path_integral + --flow-prior-arch theta_fourier_mlp: number of harmonic pairs "
            "(sin, cos) for theta encoding in the prior theta-flow."
        ),
    )
    p.add_argument(
        "--flow-prior-theta-fourier-omega-mode",
        type=str,
        default="theta_range",
        choices=["theta_range", "fixed"],
        help=(
            "Prior theta_flow film_fourier: how to set omega in sin(k*omega*theta). "
            "Same semantics as --flow-theta-fourier-omega-mode, but for the prior network."
        ),
    )
    p.add_argument(
        "--flow-prior-theta-fourier-omega",
        type=float,
        default=1.0,
        help=(
            "Prior theta_flow film_fourier: omega or theta_range multiplier (see "
            "--flow-prior-theta-fourier-omega-mode)."
        ),
    )
    p.add_argument(
        "--flow-prior-theta-fourier-no-linear",
        action="store_true",
        default=False,
        help="Prior theta_flow film_fourier: drop raw theta from the theta feature vector.",
    )
    p.add_argument(
        "--flow-prior-theta-fourier-no-bias",
        action="store_true",
        default=False,
        help="Prior theta_flow film_fourier: drop constant 1 from the theta feature vector.",
    )

    p.add_argument(
        "--flow-x-theta-fourier-k",
        type=int,
        default=4,
        help=(
            "x_flow + --flow-arch film_fourier: number of harmonic pairs "
            "(sin, cos) for theta encoding. Ignored for other methods."
        ),
    )
    p.add_argument(
        "--flow-x-theta-fourier-omega-mode",
        type=str,
        default="theta_range",
        choices=["theta_range", "fixed"],
        help=(
            "x_flow + film_fourier: how to set omega in sin(k*omega*theta). "
            "theta_range (default): omega_eff = (2*pi / (theta_high-theta_low)) * mult, where mult is "
            "--flow-x-theta-fourier-omega, so the k=1 harmonic has period (theta_high-theta_low)/mult. "
            "fixed: use --flow-x-theta-fourier-omega as omega directly (e.g. 1 gives period 2*pi)."
        ),
    )
    p.add_argument(
        "--flow-x-theta-fourier-omega",
        type=float,
        default=1.0,
        help=(
            "x_flow + film_fourier: with --flow-x-theta-fourier-omega-mode fixed, this is omega. "
            "With theta_range, this is a positive multiplier mult on (2*pi/span) (default 1 => fundamental period "
            "equals theta_high-theta_low)."
        ),
    )
    p.add_argument(
        "--flow-x-theta-fourier-no-linear",
        action="store_true",
        default=False,
        help="x_flow + film_fourier: drop raw theta from the theta feature vector.",
    )
    p.add_argument(
        "--flow-x-theta-fourier-no-bias",
        action="store_true",
        default=False,
        help="x_flow + film_fourier: drop constant 1 from the theta feature vector.",
    )
    p.add_argument(
        "--ctsm-epochs",
        type=int,
        default=8000,
        help="ctsm_v: training epochs for pair-conditioned CTSM-v model.",
    )
    p.add_argument(
        "--ctsm-batch-size",
        type=int,
        default=512,
        help="ctsm_v: batch size of ordered sample pairs (x0,a),(x1,b).",
    )
    p.add_argument(
        "--ctsm-lr",
        type=float,
        default=2e-3,
        help="ctsm_v: learning rate.",
    )
    p.add_argument(
        "--ctsm-hidden-dim",
        type=int,
        default=256,
        help="ctsm_v: hidden width of pair-conditioned time-score MLP.",
    )
    p.add_argument(
        "--ctsm-two-sb-var",
        type=float,
        default=2.0,
        help="ctsm_v: TwoSB bridge variance parameter (var = sigma^2).",
    )
    p.add_argument(
        "--ctsm-path-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="ctsm_v: schedule for the two-endpoint bridge clock u=s(t).",
    )
    p.add_argument(
        "--ctsm-path-eps",
        type=float,
        default=1e-12,
        help="ctsm_v: numerical epsilon for path denominators.",
    )
    p.add_argument(
        "--ctsm-factor",
        type=float,
        default=1.0,
        help="ctsm_v: CTSM-v weighting factor in the closed-form target.",
    )
    p.add_argument(
        "--ctsm-t-eps",
        type=float,
        default=1e-5,
        help="ctsm_v: time sampling clamp t in [t_eps, 1-t_eps] for stability.",
    )
    p.add_argument(
        "--ctsm-int-n-time",
        type=int,
        default=300,
        help="ctsm_v: number of trapezoid time points for DeltaL integration.",
    )
    p.add_argument(
        "--ctsm-m-scale",
        type=float,
        default=1.0,
        help="ctsm_v: multiplicative scale for midpoint conditioning m.",
    )
    p.add_argument(
        "--ctsm-delta-scale",
        type=float,
        default=0.5,
        help="ctsm_v: multiplicative scale for offset conditioning Delta.",
    )
    p.add_argument(
        "--ctsm-arch",
        type=str,
        default="film",
        choices=["mlp", "film"],
        help=(
            "ctsm_v: network style. 'mlp' concatenates [x,t,m,Delta] into a 4-layer MLP. "
            "'film' uses an x-trunk with per-layer FiLM from (logit(t), m, Delta), "
            "matching the flow FiLM style in models.py."
        ),
    )
    p.add_argument(
        "--ctsm-film-depth",
        type=int,
        default=3,
        help="ctsm_v + FiLM: number of residual FiLM blocks after silu(in_proj(x)). Ignored for --ctsm-arch mlp.",
    )
    p.add_argument(
        "--ctsm-gated-film",
        action="store_true",
        default=False,
        help="ctsm_v + FiLM: use tanh-gated multiplicative FiLM (gamma) for stability.",
    )
    p.add_argument(
        "--ctsm-raw-time",
        action="store_true",
        default=False,
        help="ctsm_v + FiLM: use raw t in FiLM cond instead of logit(t).",
    )
    p.add_argument(
        "--ctsm-weight-decay",
        type=float,
        default=0.0,
        help="ctsm_v: AdamW weight decay (L2 penalty on weights). Default 0 preserves prior behavior.",
    )
    p.add_argument("--gzd-latent-dim", type=int, default=2, help="gmm-z-decode only: z bottleneck dimension.")
    p.add_argument("--gzd-components", type=int, default=5, help="gmm-z-decode only: GMM component count.")
    p.add_argument("--gzd-epochs", type=int, default=2000, help="gmm-z-decode only: training epochs.")
    p.add_argument("--gzd-batch-size", type=int, default=256, help="gmm-z-decode only: training batch size.")
    p.add_argument("--gzd-lr", type=float, default=1e-3, help="gmm-z-decode only: learning rate.")
    p.add_argument("--gzd-hidden-dim", type=int, default=128, help="gmm-z-decode only: MLP hidden width.")
    p.add_argument("--gzd-depth", type=int, default=2, help="gmm-z-decode only: encoder MLP depth.")
    p.add_argument("--gzd-weight-decay", type=float, default=0.0, help="gmm-z-decode only: AdamW weight decay.")
    p.add_argument("--gzd-min-std", type=float, default=1e-3, help="gmm-z-decode only: minimum normalized-theta std.")
    p.add_argument(
        "--gzd-early-patience",
        type=int,
        default=300,
        help="gmm-z-decode only: early-stop patience; 0 disables early stopping.",
    )
    p.add_argument(
        "--gzd-early-min-delta",
        type=float,
        default=1e-4,
        help="gmm-z-decode only: early-stop min delta.",
    )
    p.add_argument(
        "--gzd-early-ema-alpha",
        type=float,
        default=0.05,
        help="gmm-z-decode only: EMA alpha for validation NLL monitor.",
    )
    p.add_argument(
        "--gzd-max-grad-norm",
        type=float,
        default=10.0,
        help="gmm-z-decode only: gradient clipping max norm; <=0 disables clipping.",
    )
    p.add_argument(
        "--gzd-pair-batch-size",
        type=int,
        default=65536,
        help="gmm-z-decode only: approximate pair budget per C-matrix block (rows*cols).",
    )

    p.add_argument(
        "--no-prior-score",
        action="store_false",
        dest="prior_enable",
        default=True,
        help="Disable unconditional prior score DSM; use posterior-only Fisher (legacy behavior).",
    )
    p.add_argument(
        "--fisher-score-mode",
        type=str,
        default="posterior_minus_prior",
        choices=["posterior_only", "posterior_minus_prior"],
        help="When prior score is trained: which curve to treat as primary vs GT (combined uses s_post - s_prior).",
    )
    p.add_argument("--prior-epochs", type=int, default=10000)
    p.add_argument("--prior-batch-size", type=int, default=1024)
    p.add_argument("--prior-lr", type=float, default=1e-3)
    p.add_argument("--prior-hidden-dim", type=int, default=128)
    p.add_argument("--prior-depth", type=int, default=3)
    p.add_argument("--prior-early-patience", type=int, default=1000)
    p.add_argument("--prior-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--prior-early-ema-alpha",
        type=float,
        default=0.05,
        help="EMA α for prior model validation monitor (early stopping).",
    )
    p.add_argument(
        "--prior-early-ema-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Prior early stopping: for epochs 1..N use raw validation loss as the monitor (no EMA); "
            "EMA starts after epoch N. Default 0."
        ),
    )
    p.add_argument("--prior-restore-best", action="store_true", default=True)
    p.add_argument("--no-prior-restore-best", action="store_false", dest="prior_restore_best")

    p.add_argument("--n-bins", type=int, default=35)
    p.add_argument("--eval-margin", type=float, default=0.30)
    p.add_argument("--score-min-bin-count", type=int, default=10)
    p.add_argument("--fd-delta", type=float, default=0.03)

    p.add_argument("--decoder-epsilon", type=float, default=0.12)
    p.add_argument("--decoder-bandwidth", type=float, default=0.10)
    p.add_argument("--decoder-epochs", type=int, default=80)
    p.add_argument("--decoder-batch-size", type=int, default=256)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--decoder-hidden-dim", type=int, default=64)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument(
        "--decoder-min-class-count",
        type=int,
        default=5,
        help="Minimum balanced samples per class in train AND eval windows; limited by eval mass near theta±epsilon/2.",
    )
    p.add_argument("--decoder-train-cap", type=int, default=1200)
    p.add_argument("--decoder-eval-cap", type=int, default=1200)
    p.add_argument("--decoder-val-frac", type=float, default=0.15)
    p.add_argument(
        "--decoder-min-val-class-size",
        type=int,
        default=10,
        help="Minimum validation samples per class after holdout; lower values ease nfit>=min_class_count when ntr is modest.",
    )
    p.add_argument("--decoder-early-patience", type=int, default=100)
    p.add_argument("--decoder-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--decoder-early-ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor α in (0,1] for validation loss monitor used by decoder early stopping.",
    )
    p.add_argument("--decoder-restore-best", action="store_true", default=True)
    p.add_argument("--no-decoder-restore-best", action="store_false", dest="decoder_restore_best")
    p.add_argument(
        "--decoder-debug-bins",
        action="store_true",
        default=False,
        help="Log per-center decoder skip details (counts and reasons) to stdout.",
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(DATA_DIR) / "outputs_step6_shared_dataset"),
    )
    p.add_argument(
        "--compute-h-matrix",
        action="store_true",
        default=False,
        help="Estimate the sample-wise H matrix from trained posterior/prior score models.",
    )
    p.add_argument(
        "--h-sigma-eval",
        type=float,
        default=-1.0,
        help="Sigma used for h-matrix score evaluation; <=0 means use min(eval sigma grid).",
    )
    p.add_argument(
        "--h-batch-size",
        type=int,
        default=65536,
        help="Max pair evaluations per h-matrix score forward chunk.",
    )
    p.add_argument(
        "--h-restore-original-order",
        action="store_true",
        default=False,
        help="Return H matrices in original dataset order instead of theta-sorted order.",
    )
    p.add_argument(
        "--h-save-intermediates",
        action="store_true",
        default=False,
        help="Save intermediate matrices G, C, and DeltaL to h-matrix npz output.",
    )
    p.add_argument(
        "--skip-shared-fisher-gt-compare",
        action="store_true",
        default=False,
        help=(
            "DSM only: after training score and prior, estimate and save the H-matrix only; "
            "skip binned Fisher evaluation, decoder training, analytic/MC GT Fisher, and "
            "Fisher-vs-GT comparison artifacts."
        ),
    )


def parse_full_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-dataset score-vs-decoder comparison with analytic GT.")
    add_dataset_arguments(p)
    add_estimation_arguments(p)
    return p.parse_args(argv)


def parse_dataset_only_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and save a shared (theta, x) train/eval dataset.")
    add_dataset_arguments(p)
    p.add_argument(
        "--output-npz",
        type=str,
        default=str(Path(DATA_DIR) / "shared_fisher_dataset.npz"),
        help="Path to write the shared dataset .npz (under DATAROOT recommended).",
    )
    return p.parse_args(argv)


def parse_estimate_only_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisher estimation from a saved shared dataset .npz.")
    add_estimation_arguments(p)
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to a shared dataset .npz produced by fisher_make_dataset.py.",
    )
    return p.parse_args(argv)
