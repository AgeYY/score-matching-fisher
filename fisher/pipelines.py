from __future__ import annotations

import os

import numpy as np
import torch

from fisher.config import DatasetConfig, DecoderRunConfig, ScoreRunConfig
from fisher.data import make_local_decoder_data, make_theta_grid, set_seed, ToyConditionalGaussianDataset
from fisher.evaluation import (
    BinnedFisher,
    compute_curve_metrics,
    evaluate_local_decoder,
    evaluate_score_fisher,
    parse_sigma_alpha_list,
)
from fisher.models import ConditionalScore1D, LocalDecoderLogit
from fisher.plots import (
    plot_decoder_calibration_examples,
    plot_decoder_loss_examples,
    plot_extrapolation_diagnostics,
    plot_fisher_curve,
    plot_training_loss,
)
from fisher.trainers import train_local_decoder, train_score_model


def _device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(name)


def run_score_pipeline(dataset_cfg: DatasetConfig, run_cfg: ScoreRunConfig) -> dict[str, str]:
    os.makedirs(run_cfg.output_dir, exist_ok=True)
    set_seed(dataset_cfg.seed)
    torch.manual_seed(dataset_cfg.seed)
    device = _device_from_name(run_cfg.device)

    dataset = ToyConditionalGaussianDataset(
        theta_low=dataset_cfg.theta_low,
        theta_high=dataset_cfg.theta_high,
        sigma_x1=dataset_cfg.sigma_x1,
        sigma_x2=dataset_cfg.sigma_x2,
        rho=dataset_cfg.rho,
        seed=dataset_cfg.seed,
    )
    theta_train, x_train = dataset.sample_joint(run_cfg.n_train)
    theta_eval, x_eval = dataset.sample_joint(run_cfg.n_eval)

    theta_std = float(np.std(theta_train))
    sigma_alpha = parse_sigma_alpha_list(run_cfg.sigma_alpha_list)
    sigma_values = sigma_alpha * theta_std
    print(f"[sigma] theta_std={theta_std:.6f}")
    print(f"[sigma] alpha grid={sigma_alpha.tolist()}")
    print(f"[sigma] absolute grid={sigma_values.tolist()}")

    model = ConditionalScore1D(hidden_dim=run_cfg.hidden_dim, depth=run_cfg.depth).to(device)
    losses = train_score_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        sigma_values=sigma_values,
        epochs=run_cfg.epochs,
        batch_size=run_cfg.batch_size,
        lr=run_cfg.lr,
        device=device,
        log_every=run_cfg.log_every,
    )

    eval_low = dataset_cfg.theta_low + run_cfg.eval_margin
    eval_high = dataset_cfg.theta_high - run_cfg.eval_margin
    result = evaluate_score_fisher(
        model=model,
        theta_eval=theta_eval,
        x_eval=x_eval,
        dataset=dataset,
        sigma_values=sigma_values,
        fd_delta=run_cfg.fd_delta,
        n_bins=run_cfg.n_bins,
        min_bin_count=run_cfg.min_bin_count,
        eval_low=eval_low,
        eval_high=eval_high,
        device=device,
    )
    mean_r2 = float(np.nanmean(result.r2[result.curves.valid])) if np.any(result.curves.valid) else float("nan")

    loss_path = os.path.join(run_cfg.output_dir, "training_loss.png")
    fisher_path = os.path.join(run_cfg.output_dir, "fisher_curve_extrapolated_vs_fd.png")
    diag_path = os.path.join(run_cfg.output_dir, "extrapolation_diagnostics.png")
    metrics_path = os.path.join(run_cfg.output_dir, "metrics_extrapolated.txt")
    npz_path = os.path.join(run_cfg.output_dir, "binned_fisher_multi_sigma.npz")

    plot_training_loss(losses, loss_path, title="Score Model Training Loss (Multi-Sigma DSM)")
    plot_fisher_curve(
        result.curves,
        fisher_path,
        model_label=r"Extrapolated $\hat I_{0}(\theta)$",
        title=r"Fisher Curve: Extrapolated $\sigma \to 0$ vs Finite-Difference",
    )
    plot_extrapolation_diagnostics(
        centers=result.curves.centers,
        sigma_values=result.sigma_values,
        fisher_per_sigma=result.fisher_per_sigma,
        intercept=result.curves.fisher_model,
        slope=result.slope,
        valid_bins=result.curves.valid,
        out_path=diag_path,
    )

    np.savez(
        npz_path,
        centers=result.curves.centers,
        sigma_values=result.sigma_values,
        fisher_per_sigma=result.fisher_per_sigma,
        se_per_sigma=result.se_per_sigma,
        fisher_extrapolated=result.curves.fisher_model,
        fisher_fd=result.curves.fisher_fd,
        se_fd=result.curves.se_fd,
        slope=result.slope,
        r2=result.r2,
        counts=result.curves.counts,
        valid=result.curves.valid.astype(np.int32),
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Multi-sigma score-to-Fisher evaluation (no kernel averaging)\n")
        f.write("Extrapolation model: I_sigma(theta)=a(theta)+b(theta)*sigma^2\n")
        f.write(f"sigma_alpha: {sigma_alpha.tolist()}\n")
        f.write(f"sigma_values: {sigma_values.tolist()}\n")
        f.write(f"n_valid_bins: {result.metrics['n_valid_bins']:.0f}\n")
        f.write(f"rmse: {result.metrics['rmse']:.6f}\n")
        f.write(f"mae: {result.metrics['mae']:.6f}\n")
        f.write(f"relative_rmse: {result.metrics['relative_rmse']:.6f}\n")
        f.write(f"corr: {result.metrics['corr']:.6f}\n")
        f.write(f"mean_extrapolation_r2: {mean_r2:.6f}\n")

    print("[evaluation]")
    print(f"  valid bins: {int(result.metrics['n_valid_bins'])}/{run_cfg.n_bins}")
    print(f"  rmse: {result.metrics['rmse']:.6f}")
    print(f"  mae: {result.metrics['mae']:.6f}")
    print(f"  relative_rmse: {result.metrics['relative_rmse']:.6f}")
    print(f"  corr: {result.metrics['corr']:.6f}")
    print(f"  mean extrapolation r2: {mean_r2:.6f}")

    return {
        "loss_path": loss_path,
        "fisher_path": fisher_path,
        "diag_path": diag_path,
        "metrics_path": metrics_path,
        "npz_path": npz_path,
    }


def _as_curves(centers: np.ndarray, fisher_decoder: np.ndarray, se_decoder: np.ndarray, fisher_fd: np.ndarray) -> BinnedFisher:
    valid = np.isfinite(fisher_decoder) & np.isfinite(fisher_fd)
    return BinnedFisher(
        centers=centers,
        fisher_model=fisher_decoder,
        fisher_fd=fisher_fd,
        se_model=se_decoder,
        se_fd=np.full_like(se_decoder, np.nan),
        counts=np.zeros_like(centers, dtype=np.int64),
        valid=valid,
    )


def run_decoder_pipeline(dataset_cfg: DatasetConfig, run_cfg: DecoderRunConfig) -> dict[str, str]:
    os.makedirs(run_cfg.output_dir, exist_ok=True)
    set_seed(dataset_cfg.seed)
    torch.manual_seed(dataset_cfg.seed)
    device = _device_from_name(run_cfg.device)

    dataset = ToyConditionalGaussianDataset(
        theta_low=dataset_cfg.theta_low,
        theta_high=dataset_cfg.theta_high,
        sigma_x1=dataset_cfg.sigma_x1,
        sigma_x2=dataset_cfg.sigma_x2,
        rho=dataset_cfg.rho,
        seed=dataset_cfg.seed,
    )

    centers = make_theta_grid(dataset_cfg.theta_low, dataset_cfg.theta_high, run_cfg.eval_margin, run_cfg.n_bins)
    if run_cfg.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if (centers.min() - 0.5 * run_cfg.epsilon) < dataset_cfg.theta_low or (
        centers.max() + 0.5 * run_cfg.epsilon
    ) > dataset_cfg.theta_high:
        raise ValueError("epsilon too large for chosen eval range; theta0 +/- epsilon/2 leaves support.")

    fisher_decoder = np.full(run_cfg.n_bins, np.nan, dtype=np.float64)
    se_decoder = np.full(run_cfg.n_bins, np.nan, dtype=np.float64)
    fisher_fd = np.full(run_cfg.n_bins, np.nan, dtype=np.float64)

    rep_count = min(4, run_cfg.n_bins)
    rep_idx = np.linspace(0, run_cfg.n_bins - 1, rep_count).round().astype(int).tolist()
    rep_losses: dict[int, list[float]] = {}
    rep_logits_pos: dict[int, np.ndarray] = {}
    rep_logits_neg: dict[int, np.ndarray] = {}

    for i, theta0 in enumerate(centers):
        local_data = make_local_decoder_data(
            dataset=dataset,
            theta0=float(theta0),
            epsilon=run_cfg.epsilon,
            n_train_local=run_cfg.n_train_local,
            n_eval_local=run_cfg.n_eval_local,
        )
        model = LocalDecoderLogit(hidden_dim=run_cfg.hidden_dim, depth=run_cfg.depth).to(device)
        loss_trace = train_local_decoder(
            model=model,
            x_train=local_data["x_train"],
            y_train=local_data["y_train"],
            epochs=run_cfg.epochs,
            batch_size=run_cfg.batch_size,
            lr=run_cfg.lr,
            device=device,
        )
        out = evaluate_local_decoder(
            model=model,
            x_eval_pos=local_data["x_eval_pos"],
            x_eval_neg=local_data["x_eval_neg"],
            epsilon=run_cfg.epsilon,
            dataset=dataset,
            theta0=float(theta0),
            n_eval_local=run_cfg.n_eval_local,
            fd_delta=run_cfg.fd_delta,
            device=device,
        )
        fisher_decoder[i] = out.fisher_decoder
        se_decoder[i] = out.se_decoder
        fisher_fd[i] = out.fisher_fd

        if i in rep_idx:
            rep_losses[i] = loss_trace
            rep_logits_pos[i] = out.logits_pos
            rep_logits_neg[i] = out.logits_neg
        if i == 0 or (i + 1) % run_cfg.log_every == 0 or (i + 1) == run_cfg.n_bins:
            print(
                f"[theta {i+1:3d}/{run_cfg.n_bins}] theta0={theta0:+.3f} "
                f"fisher_decoder={out.fisher_decoder:.4f} fisher_fd={out.fisher_fd:.4f}"
            )

    curves = _as_curves(centers, fisher_decoder, se_decoder, fisher_fd)
    metrics = compute_curve_metrics(curves.fisher_model, curves.fisher_fd, curves.valid)

    curve_path = os.path.join(run_cfg.output_dir, "fisher_curve_decoder_vs_fd.png")
    calib_path = os.path.join(run_cfg.output_dir, "decoder_calibration_examples.png")
    loss_path = os.path.join(run_cfg.output_dir, "training_loss_examples.png")
    metrics_path = os.path.join(run_cfg.output_dir, "metrics.txt")
    npz_path = os.path.join(run_cfg.output_dir, "fisher_curve_data.npz")

    plot_fisher_curve(
        curves,
        curve_path,
        model_label="Decoder estimate",
        title="Decoder Fisher vs Finite-Difference Baseline",
    )
    plot_decoder_calibration_examples(centers, rep_logits_pos, rep_logits_neg, calib_path)
    plot_decoder_loss_examples(centers, rep_losses, loss_path)

    np.savez(
        npz_path,
        centers=centers,
        fisher_decoder=fisher_decoder,
        se_decoder=se_decoder,
        fisher_fd=fisher_fd,
        valid=curves.valid.astype(np.int32),
        epsilon=np.array([run_cfg.epsilon], dtype=np.float64),
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Decoder-based Fisher via local classification\n")
        f.write(f"epsilon: {run_cfg.epsilon:.6f}\n")
        f.write(f"n_bins: {run_cfg.n_bins}\n")
        f.write(f"valid_bins: {int(metrics['n_valid_bins'])}/{run_cfg.n_bins}\n")
        f.write(f"rmse: {metrics['rmse']:.6f}\n")
        f.write(f"mae: {metrics['mae']:.6f}\n")
        f.write(f"relative_rmse: {metrics['relative_rmse']:.6f}\n")
        f.write(f"corr: {metrics['corr']:.6f}\n")

    print("[evaluation]")
    print(f"  valid bins: {int(metrics['n_valid_bins'])}/{run_cfg.n_bins}")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    print(f"  relative_rmse: {metrics['relative_rmse']:.6f}")
    print(f"  corr: {metrics['corr']:.6f}")

    return {
        "curve_path": curve_path,
        "calib_path": calib_path,
        "loss_path": loss_path,
        "metrics_path": metrics_path,
        "npz_path": npz_path,
    }
