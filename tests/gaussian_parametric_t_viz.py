#!/usr/bin/env python3
"""Parametric 1D Gaussian p_t(x)=N(mu(t),1), dataset generation, and CTSM-v ratio estimation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.ctsm_models import ToyPairConditionedTimeScoreNetFiLM
from fisher.ctsm_objectives import ctsm_v_pair_conditioned_loss, estimate_log_ratio_trapz_pair
from fisher.ctsm_paths import TwoSB
from global_setting import DATAROOT


DEFAULT_VIZ_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_parametric_t_viz.png"
DEFAULT_DATASET_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_parametric_t_dataset.csv"
DEFAULT_RATIO_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_parametric_t_ratio_eval.csv"
DEFAULT_CTSM_LOSS_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_parametric_t_ctsm_loss.png"
DEFAULT_CTSM_SCATTER_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_parametric_t_ctsm_scatter.png"
DEFAULT_CTSM_SUMMARY_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_parametric_t_ctsm_training_summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate (x,t) from p_t(x)=N(2t-1,1) and estimate held-out ratios with CTSM-v.")
    p.add_argument("--n-dataset", type=int, default=4000, help="Number of (x,t) samples to generate for dataset export/visualization.")
    p.add_argument("--n-ratio", type=int, default=1000, help="Number of held-out (x,t_i,t_j) triples for ratio evaluation.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device for sampling/training ('cuda' or 'cpu'). Default is cuda.",
    )
    p.add_argument(
        "--viz-output",
        type=str,
        default=str(DEFAULT_VIZ_OUTPUT),
        help="Output PNG path for dataset visualization.",
    )
    p.add_argument(
        "--dataset-output",
        type=str,
        default=str(DEFAULT_DATASET_OUTPUT),
        help="Output CSV path for sampled (x,t) dataset.",
    )
    p.add_argument(
        "--ratio-output",
        type=str,
        default=str(DEFAULT_RATIO_OUTPUT),
        help="Output CSV path for held-out ratio evaluation.",
    )
    p.add_argument(
        "--disable-ctsm",
        action="store_true",
        help="If set, skip CTSM-v training/estimation and only export analytic held-out ratios.",
    )
    p.add_argument("--ctsm-max-epochs", type=int, default=2000, help="CTSM-v maximum training epochs.")
    p.add_argument("--ctsm-batch-size", type=int, default=512, help="CTSM-v batch size.")
    p.add_argument("--ctsm-hidden-dim", type=int, default=128, help="CTSM-v hidden dimension.")
    p.add_argument("--ctsm-lr", type=float, default=2e-3, help="CTSM-v learning rate.")
    p.add_argument("--ctsm-two-sb-var", type=float, default=2.0, help="CTSM-v TwoSB bridge variance.")
    p.add_argument("--ctsm-factor", type=float, default=1.0, help="CTSM-v target factor.")
    p.add_argument("--ctsm-t-eps", type=float, default=1e-5, help="CTSM-v sampled time clamp.")
    p.add_argument("--ctsm-n-time", type=int, default=200, help="Trapezoid points for CTSM-v ratio integration.")
    p.add_argument("--ctsm-val-pool-size", type=int, default=4096, help="Held-out validation pool size.")
    p.add_argument("--ctsm-val-batches-per-epoch", type=int, default=8, help="Validation mini-batches per epoch.")
    p.add_argument("--ctsm-early-patience", type=int, default=250, help="Early stopping patience on EMA monitor.")
    p.add_argument("--ctsm-early-min-delta", type=float, default=1e-4, help="Minimum EMA improvement to reset patience.")
    p.add_argument("--ctsm-early-ema-alpha", type=float, default=0.05, help="EMA alpha for validation monitor.")
    p.add_argument(
        "--ctsm-early-ema-warmup-epochs",
        type=int,
        default=0,
        help="Warmup epochs that use raw val loss as monitor before EMA updates.",
    )
    p.add_argument(
        "--ctsm-no-restore-best",
        action="store_true",
        help="If set, do not restore the best EMA checkpoint at the end.",
    )
    p.add_argument("--ctsm-film-depth", type=int, default=3, help="FiLM network depth.")
    p.add_argument("--ctsm-film-gated", action="store_true", help="Use tanh-gated FiLM modulation.")
    p.add_argument("--ctsm-film-use-raw-time", action="store_true", help="Use raw t (not logit(t)) in FiLM conditioning.")
    p.add_argument("--ctsm-m-scale", type=float, default=1.0, help="Scale for m=(a+b)/2 conditioning.")
    p.add_argument("--ctsm-delta-scale", type=float, default=0.5, help="Scale for delta=(b-a) conditioning.")
    p.add_argument("--ctsm-log-every", type=int, default=100, help="Progress logging interval.")
    p.add_argument(
        "--ctsm-loss-output",
        type=str,
        default=str(DEFAULT_CTSM_LOSS_OUTPUT),
        help="Output PNG path for CTSM-v training loss curve.",
    )
    p.add_argument(
        "--ctsm-scatter-output",
        type=str,
        default=str(DEFAULT_CTSM_SCATTER_OUTPUT),
        help="Output PNG path for analytic-vs-CTSM ratio scatter.",
    )
    p.add_argument(
        "--ctsm-summary-output",
        type=str,
        default=str(DEFAULT_CTSM_SUMMARY_OUTPUT),
        help="Output JSON path for CTSM-v early-stopping/training summary.",
    )
    return p.parse_args()


def mu_of_t(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * t - 1.0


def sample_x_given_t(t: torch.Tensor) -> torch.Tensor:
    return mu_of_t(t) + torch.randn_like(t)


def gaussian_log_prob_unit_var(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * (x - mean) ** 2


def format_via_data_symlink(abs_path: Path) -> Path:
    display_path = abs_path
    data_symlink = _REPO_ROOT / "data"
    if data_symlink.exists():
        try:
            data_real = data_symlink.resolve()
            out_real = abs_path.resolve()
            if str(out_real).startswith(str(data_real) + "/"):
                display_path = data_symlink / out_real.relative_to(data_real)
        except OSError:
            pass
    return display_path


def compute_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    if torch.std(x, unbiased=False) == 0 or torch.std(y, unbiased=False) == 0:
        return float("nan")
    corr_tensor = torch.corrcoef(torch.stack([x, y]))
    return float(corr_tensor[0, 1].item())


def main() -> None:
    args = parse_args()
    ctsm_enabled = not bool(args.disable_ctsm)
    ctsm_restore_best = not bool(args.ctsm_no_restore_best)

    if args.n_dataset <= 0:
        raise ValueError("--n-dataset must be > 0.")
    if args.n_ratio <= 0:
        raise ValueError("--n-ratio must be > 0.")
    if args.ctsm_max_epochs <= 0:
        raise ValueError("--ctsm-max-epochs must be > 0.")
    if args.ctsm_batch_size <= 0:
        raise ValueError("--ctsm-batch-size must be > 0.")
    if args.ctsm_hidden_dim <= 0:
        raise ValueError("--ctsm-hidden-dim must be > 0.")
    if args.ctsm_n_time < 2:
        raise ValueError("--ctsm-n-time must be >= 2.")
    if args.ctsm_log_every <= 0:
        raise ValueError("--ctsm-log-every must be > 0.")
    if args.ctsm_val_pool_size < 2:
        raise ValueError("--ctsm-val-pool-size must be >= 2.")
    if args.ctsm_val_batches_per_epoch <= 0:
        raise ValueError("--ctsm-val-batches-per-epoch must be > 0.")
    if args.ctsm_early_patience <= 0:
        raise ValueError("--ctsm-early-patience must be > 0.")
    if args.ctsm_early_ema_warmup_epochs < 0:
        raise ValueError("--ctsm-early-ema-warmup-epochs must be >= 0.")
    if not (0.0 < float(args.ctsm_early_ema_alpha) <= 1.0):
        raise ValueError("--ctsm-early-ema-alpha must be in (0, 1].")
    if not (0.0 <= args.ctsm_t_eps < 0.5):
        raise ValueError("--ctsm-t-eps must be in [0, 0.5).")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    viz_out_path = Path(args.viz_output).resolve()
    dataset_out_path = Path(args.dataset_output).resolve()
    ratio_out_path = Path(args.ratio_output).resolve()
    ctsm_loss_out_path = Path(args.ctsm_loss_output).resolve()
    ctsm_scatter_out_path = Path(args.ctsm_scatter_output).resolve()
    ctsm_summary_out_path = Path(args.ctsm_summary_output).resolve()
    for p in [viz_out_path, dataset_out_path, ratio_out_path, ctsm_loss_out_path, ctsm_scatter_out_path, ctsm_summary_out_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # Dataset: sample t~U[0,1], x~N(mu(t),1)
    t_data = torch.rand(args.n_dataset, 1, device=device)
    x_data = sample_x_given_t(t_data)
    mu_data = mu_of_t(t_data)
    x_data_cpu = x_data.detach().cpu().squeeze(-1)
    t_data_cpu = t_data.detach().cpu().squeeze(-1)
    mu_data_cpu = mu_data.detach().cpu().squeeze(-1)

    with dataset_out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "t", "mu_t"])
        for x_i, t_i, m_i in zip(x_data_cpu.tolist(), t_data_cpu.tolist(), mu_data_cpu.tolist()):
            writer.writerow([f"{x_i:.10f}", f"{t_i:.10f}", f"{m_i:.10f}"])

    plt.figure(figsize=(7, 4.2))
    plt.scatter(t_data_cpu.numpy(), x_data_cpu.numpy(), s=8, alpha=0.30, label="samples")
    t_line = torch.linspace(0.0, 1.0, 256)
    mu_line = mu_of_t(t_line)
    plt.plot(t_line.numpy(), mu_line.numpy(), color="black", linewidth=1.8, label=r"$\mu(t)=2t-1$")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Parametric Gaussian Dataset: $p_t(x)=\\mathcal{N}(2t-1,1)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_out_path, dpi=150)
    plt.close()

    # Held-out ratio eval: x ~ p_{t_i}, compare p_{t_j}(x) vs p_{t_i}(x)
    t_i_eval = torch.rand(args.n_ratio, 1, device=device)
    t_j_eval = torch.rand(args.n_ratio, 1, device=device)
    x_eval = sample_x_given_t(t_i_eval)
    mu_i_eval = mu_of_t(t_i_eval)
    mu_j_eval = mu_of_t(t_j_eval)
    gt_log_ratio = gaussian_log_prob_unit_var(x_eval, mu_j_eval) - gaussian_log_prob_unit_var(x_eval, mu_i_eval)
    gt_log_ratio = gt_log_ratio.squeeze(-1)

    ctsm_ratio_cpu = torch.full_like(gt_log_ratio.detach().cpu(), float("nan"))
    ctsm_loss_history: list[float] = []
    ctsm_val_losses: list[float] = []
    ctsm_val_monitor_losses: list[float] = []
    ctsm_metrics: dict[str, float] = {}
    ctsm_training_summary: dict[str, float | int | bool] = {}

    if ctsm_enabled:
        prob_path = TwoSB(dim=1, var=float(args.ctsm_two_sb_var))
        ctsm_model = ToyPairConditionedTimeScoreNetFiLM(
            dim=1,
            hidden_dim=int(args.ctsm_hidden_dim),
            depth=int(args.ctsm_film_depth),
            m_scale=float(args.ctsm_m_scale),
            delta_scale=float(args.ctsm_delta_scale),
            use_logit_time=not bool(args.ctsm_film_use_raw_time),
            gated_film=bool(args.ctsm_film_gated),
        ).to(device)
        optimizer = torch.optim.Adam(ctsm_model.parameters(), lr=float(args.ctsm_lr))

        # Fixed held-out validation pool of pair-conditioned samples.
        t0_val = torch.rand(int(args.ctsm_val_pool_size), 1, device=device)
        t1_val = torch.rand(int(args.ctsm_val_pool_size), 1, device=device)
        x0_val = sample_x_given_t(t0_val)
        x1_val = sample_x_given_t(t1_val)

        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_monitor = float("inf")
        ema_monitor: float | None = None
        patience_bad = 0
        stopped_early = False

        progress = tqdm(range(int(args.ctsm_max_epochs)), desc="pair-conditioned CTSM-v (FiLM)")
        for step in progress:
            t0 = torch.rand(int(args.ctsm_batch_size), 1, device=device)
            t1 = torch.rand(int(args.ctsm_batch_size), 1, device=device)
            x0 = sample_x_given_t(t0)
            x1 = sample_x_given_t(t1)
            loss = ctsm_v_pair_conditioned_loss(
                model=ctsm_model,
                prob_path=prob_path,
                x0=x0,
                x1=x1,
                a=t0,
                b=t1,
                factor=float(args.ctsm_factor),
                t_eps=float(args.ctsm_t_eps),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            ctsm_loss_history.append(loss_value)
            ctsm_model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for _ in range(int(args.ctsm_val_batches_per_epoch)):
                    idx = torch.randint(0, int(args.ctsm_val_pool_size), (int(args.ctsm_batch_size),), device=device)
                    v_loss = ctsm_v_pair_conditioned_loss(
                        model=ctsm_model,
                        prob_path=prob_path,
                        x0=x0_val[idx],
                        x1=x1_val[idx],
                        a=t0_val[idx],
                        b=t1_val[idx],
                        factor=float(args.ctsm_factor),
                        t_eps=float(args.ctsm_t_eps),
                    )
                    val_epoch_losses.append(float(v_loss.item()))
            val_loss_value = float(sum(val_epoch_losses) / max(1, len(val_epoch_losses)))
            ctsm_val_losses.append(val_loss_value)

            epoch = step + 1
            if ema_monitor is None:
                ema_monitor = val_loss_value
            elif epoch <= int(args.ctsm_early_ema_warmup_epochs):
                ema_monitor = val_loss_value
            else:
                alpha = float(args.ctsm_early_ema_alpha)
                ema_monitor = alpha * val_loss_value + (1.0 - alpha) * ema_monitor
            ctsm_val_monitor_losses.append(float(ema_monitor))

            improved = float(ema_monitor) < (best_monitor - float(args.ctsm_early_min_delta))
            if improved:
                best_monitor = float(ema_monitor)
                best_epoch = int(epoch)
                patience_bad = 0
                if ctsm_restore_best:
                    best_state = deepcopy(ctsm_model.state_dict())
            else:
                patience_bad += 1

            if step % int(args.ctsm_log_every) == 0:
                progress.set_postfix(train=f"{loss_value:.4f}", val=f"{val_loss_value:.4f}", ema=f"{float(ema_monitor):.4f}")
            if patience_bad >= int(args.ctsm_early_patience):
                stopped_early = True
                break

        stopped_epoch = int(len(ctsm_loss_history))
        if ctsm_restore_best and best_state is not None:
            ctsm_model.load_state_dict(best_state)
        if best_epoch <= 0:
            best_epoch = stopped_epoch
            best_monitor = float(ctsm_val_monitor_losses[-1]) if ctsm_val_monitor_losses else float("nan")

        ctsm_ratio = estimate_log_ratio_trapz_pair(
            ctsm_model,
            x_eval,
            t_i_eval,
            t_j_eval,
            n_time=int(args.ctsm_n_time),
            eps1=float(args.ctsm_t_eps),
            eps2=float(args.ctsm_t_eps),
        )
        ctsm_ratio_cpu = ctsm_ratio.detach().cpu()
        gt_cpu = gt_log_ratio.detach().cpu()
        abs_error = torch.abs(ctsm_ratio_cpu - gt_cpu)

        ctsm_metrics = {
            "mse": float(torch.mean((ctsm_ratio_cpu - gt_cpu) ** 2).item()),
            "mae": float(abs_error.mean().item()),
            "corr": float(compute_corr(gt_cpu, ctsm_ratio_cpu)),
            "final_loss": float(ctsm_loss_history[-1]),
        }
        ctsm_training_summary = {
            "max_epochs": int(args.ctsm_max_epochs),
            "stopped_epoch": int(stopped_epoch),
            "best_epoch": int(best_epoch),
            "stopped_early": bool(stopped_early),
            "restore_best": bool(ctsm_restore_best),
            "best_val_ema": float(best_monitor),
            "final_train_loss": float(ctsm_loss_history[-1]),
            "final_val_loss": float(ctsm_val_losses[-1]),
            "final_val_ema": float(ctsm_val_monitor_losses[-1]),
            "early_patience": int(args.ctsm_early_patience),
            "early_min_delta": float(args.ctsm_early_min_delta),
            "early_ema_alpha": float(args.ctsm_early_ema_alpha),
            "early_ema_warmup_epochs": int(args.ctsm_early_ema_warmup_epochs),
            "mse": ctsm_metrics["mse"],
            "mae": ctsm_metrics["mae"],
            "corr": ctsm_metrics["corr"],
        }
        with ctsm_summary_out_path.open("w", encoding="utf-8") as f:
            json.dump(ctsm_training_summary, f, indent=2, sort_keys=True)

        plt.figure(figsize=(6.5, 3.6))
        plt.plot(ctsm_loss_history, label="train")
        plt.plot(ctsm_val_losses, label="val")
        plt.plot(ctsm_val_monitor_losses, linestyle="--", label=f"val EMA (alpha={args.ctsm_early_ema_alpha:g})")
        plt.xlabel("epoch")
        plt.ylabel("CTSM-v loss")
        plt.title("Pair-conditioned FiLM CTSM-v (parametric Gaussian)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ctsm_loss_out_path, dpi=150)
        plt.close()

        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(
            gt_cpu.numpy(),
            ctsm_ratio_cpu.numpy(),
            s=10,
            alpha=0.55,
        )
        lo = min(float(gt_cpu.min().item()), float(ctsm_ratio_cpu.min().item()))
        hi = max(float(gt_cpu.max().item()), float(ctsm_ratio_cpu.max().item()))
        plt.plot([lo, hi], [lo, hi], "--", color="gray")
        plt.xlabel(r"GT $\log p_{t_j}(x)-\log p_{t_i}(x)$")
        plt.ylabel("CTSM-v estimate")
        plt.title("Held-out ratio: estimate vs GT")
        plt.tight_layout()
        plt.savefig(ctsm_scatter_out_path, dpi=150)
        plt.close()

    t_i_cpu = t_i_eval.detach().cpu().squeeze(-1)
    t_j_cpu = t_j_eval.detach().cpu().squeeze(-1)
    x_eval_cpu = x_eval.detach().cpu().squeeze(-1)
    gt_cpu = gt_log_ratio.detach().cpu()
    with ratio_out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if ctsm_enabled:
            writer.writerow(["x", "t_i", "t_j", "gt_log_ratio", "ctsm_log_ratio", "abs_error"])
            abs_error = torch.abs(ctsm_ratio_cpu - gt_cpu)
            for x_i, ti_i, tj_i, gt_i, c_i, e_i in zip(
                x_eval_cpu.tolist(),
                t_i_cpu.tolist(),
                t_j_cpu.tolist(),
                gt_cpu.tolist(),
                ctsm_ratio_cpu.tolist(),
                abs_error.tolist(),
            ):
                writer.writerow(
                    [f"{x_i:.10f}", f"{ti_i:.10f}", f"{tj_i:.10f}", f"{gt_i:.10f}", f"{c_i:.10f}", f"{e_i:.10f}"]
                )
        else:
            writer.writerow(["x", "t_i", "t_j", "gt_log_ratio"])
            for x_i, ti_i, tj_i, gt_i in zip(x_eval_cpu.tolist(), t_i_cpu.tolist(), t_j_cpu.tolist(), gt_cpu.tolist()):
                writer.writerow([f"{x_i:.10f}", f"{ti_i:.10f}", f"{tj_i:.10f}", f"{gt_i:.10f}"])

    display_viz_output = format_via_data_symlink(viz_out_path)
    display_dataset_output = format_via_data_symlink(dataset_out_path)
    display_ratio_output = format_via_data_symlink(ratio_out_path)
    display_ctsm_loss_output = format_via_data_symlink(ctsm_loss_out_path)
    display_ctsm_scatter_output = format_via_data_symlink(ctsm_scatter_out_path)
    display_ctsm_summary_output = format_via_data_symlink(ctsm_summary_out_path)

    print(f"device={device}")
    print("distribution=p_t(x)=N(mu(t),1), mu(t)=2t-1")
    print(f"n_dataset={args.n_dataset}")
    print(f"n_ratio={args.n_ratio}")
    print("ratio_convention=log p_{t_j}(x) - log p_{t_i}(x), x~p_{t_i}")
    print(f"viz_output={display_viz_output}")
    print(f"viz_output_abs={viz_out_path}")
    print(f"dataset_output={display_dataset_output}")
    print(f"dataset_output_abs={dataset_out_path}")
    print(f"ratio_output={display_ratio_output}")
    print(f"ratio_output_abs={ratio_out_path}")
    print(
        "dataset_stats: mean_t={:.6f}, mean_x={:.6f}, var_x={:.6f}".format(
            float(t_data_cpu.mean().item()),
            float(x_data_cpu.mean().item()),
            float(x_data_cpu.var(unbiased=False).item()),
        )
    )
    print(
        "gt_ratio_stats: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
            float(gt_cpu.mean().item()),
            float(gt_cpu.std(unbiased=False).item()),
            float(gt_cpu.min().item()),
            float(gt_cpu.max().item()),
        )
    )
    print("first_10_heldout:")
    for idx, (x_i, ti_i, tj_i, gt_i) in enumerate(
        zip(x_eval_cpu[:10].tolist(), t_i_cpu[:10].tolist(), t_j_cpu[:10].tolist(), gt_cpu[:10].tolist())
    ):
        print(f"  {idx:02d}: x={x_i:.6f}, t_i={ti_i:.6f}, t_j={tj_i:.6f}, gt={gt_i:.6f}")

    if ctsm_enabled:
        print("ctsm_enabled=True")
        print("ctsm_model=ToyPairConditionedTimeScoreNetFiLM")
        print(
            "ctsm_config: max_epochs={} batch_size={} hidden_dim={} lr={} two_sb_var={} n_time={} t_eps={}".format(
                args.ctsm_max_epochs,
                args.ctsm_batch_size,
                args.ctsm_hidden_dim,
                args.ctsm_lr,
                args.ctsm_two_sb_var,
                args.ctsm_n_time,
                args.ctsm_t_eps,
            )
        )
        print(
            "ctsm_early_stop: stopped_early={} stopped_epoch={} best_epoch={} best_val_ema={:.6f} restore_best={}".format(
                bool(ctsm_training_summary.get("stopped_early", False)),
                int(ctsm_training_summary.get("stopped_epoch", 0)),
                int(ctsm_training_summary.get("best_epoch", 0)),
                float(ctsm_training_summary.get("best_val_ema", float("nan"))),
                bool(ctsm_training_summary.get("restore_best", ctsm_restore_best)),
            )
        )
        print(
            "ctsm_metrics: final_loss={:.6f}, mse={:.6f}, mae={:.6f}, corr={:.6f}".format(
                ctsm_metrics["final_loss"],
                ctsm_metrics["mse"],
                ctsm_metrics["mae"],
                ctsm_metrics["corr"],
            )
        )
        print(f"ctsm_loss_output={display_ctsm_loss_output}")
        print(f"ctsm_loss_output_abs={ctsm_loss_out_path}")
        print(f"ctsm_scatter_output={display_ctsm_scatter_output}")
        print(f"ctsm_scatter_output_abs={ctsm_scatter_out_path}")
        print(f"ctsm_summary_output={display_ctsm_summary_output}")
        print(f"ctsm_summary_output_abs={ctsm_summary_out_path}")
    else:
        print("ctsm_enabled=False")


if __name__ == "__main__":
    main()
