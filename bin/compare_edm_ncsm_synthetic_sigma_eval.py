#!/usr/bin/env python3
"""Reproduce synthetic EDM vs NCSM benchmark; sweep sigma_eval for predict_score."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher.models import ConditionalScore1D, ConditionalThetaEDM
from fisher.trainers import train_score_model_ncsm_continuous, train_theta_edm_model


def _metrics(pred: np.ndarray, score_true: np.ndarray) -> dict[str, float]:
    err = pred - score_true
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "corr": float(np.corrcoef(pred.reshape(-1), score_true.reshape(-1))[0, 1]),
    }


def _scatter_one(
    out_path: Path,
    score_true: np.ndarray,
    pred: np.ndarray,
    *,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    lo = float(np.min(score_true))
    hi = float(np.max(score_true))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(score_true.reshape(-1), pred.reshape(-1), s=8, alpha=0.25, color=color)
    ax.plot([lo, hi], [lo, hi], color="k", linewidth=1.0)
    ax.set_xlabel("True posterior score")
    ax.set_ylabel("Predicted score")
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _default_sigma_evals(ncsm_sigma_min: float, grid: list[float]) -> list[float]:
    return sorted({float(ncsm_sigma_min), *[float(x) for x in grid]})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--ncsm-sigma-min",
        type=float,
        default=0.03,
        help="NCSM continuous training sigma_min (also included in default eval sweep).",
    )
    p.add_argument(
        "--ncsm-sigma-max",
        type=float,
        default=0.35,
        help="NCSM continuous training sigma_max.",
    )
    p.add_argument(
        "--sigma-eval-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
        help="Extra sigma_eval values for predict_score (default includes common grid; sigma_min is merged in).",
    )
    p.add_argument(
        "--sigma-evals",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Explicit sigma_eval list (optional). If set, overrides --sigma-eval-grid / --ncsm-sigma-min merge."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output directory (default: journal/notes/figs/"
            "2026-04-11-edm-vs-ncsm-synthetic-sigma-sweep-full under repo root)."
        ),
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "journal" / "notes" / "figs" / "2026-04-11-edm-vs-ncsm-synthetic-sigma-sweep-full"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ncsm_sigma_min = float(args.ncsm_sigma_min)
    ncsm_sigma_max = float(args.ncsm_sigma_max)
    if args.sigma_evals is not None and len(args.sigma_evals) > 0:
        sigma_evals = sorted({float(s) for s in args.sigma_evals})
    else:
        sigma_evals = _default_sigma_evals(ncsm_sigma_min, list(args.sigma_eval_grid))

    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested per repo policy; unavailable.")
    device = torch.device("cuda")

    n_train, n_val, n_eval = 4096, 512, 2048
    noise_std = 0.35
    noise_var = noise_std**2

    theta_all = rng.normal(size=(n_train + n_val, 1)).astype(np.float32)
    x1 = theta_all + noise_std * rng.normal(size=(n_train + n_val, 1)).astype(np.float32)
    x2 = rng.normal(size=(n_train + n_val, 1)).astype(np.float32)
    x_all = np.concatenate([x1, x2], axis=1)
    theta_train = theta_all[:n_train]
    x_train = x_all[:n_train]
    theta_val = theta_all[n_train:]
    x_val = x_all[n_train:]

    theta_eval = rng.normal(size=(n_eval, 1)).astype(np.float32)
    x1_eval = theta_eval + noise_std * rng.normal(size=(n_eval, 1)).astype(np.float32)
    x2_eval = rng.normal(size=(n_eval, 1)).astype(np.float32)
    x_eval = np.concatenate([x1_eval, x2_eval], axis=1)

    var_post = 1.0 / (1.0 + 1.0 / noise_var)
    mu_post = var_post * (x_eval[:, :1] / noise_var)
    score_true = -(theta_eval - mu_post) / var_post

    ncsm = ConditionalScore1D(x_dim=2, hidden_dim=64, depth=3, use_log_sigma=True).to(device)
    out_ncsm = train_score_model_ncsm_continuous(
        model=ncsm,
        theta_train=theta_train,
        x_train=x_train,
        sigma_min=ncsm_sigma_min,
        sigma_max=ncsm_sigma_max,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        device=device,
        log_every=40,
        theta_val=theta_val,
        x_val=x_val,
        optimizer_name="adamw",
        weight_decay=1e-4,
        max_grad_norm=1.0,
        lr_scheduler="cosine",
        lr_warmup_frac=0.05,
        loss_type="huber",
        huber_delta=1.0,
        normalize_by_sigma=False,
        abort_on_nonfinite=True,
    )

    edm_backbone = ConditionalScore1D(x_dim=2, hidden_dim=64, depth=3, use_log_sigma=False).to(device)
    edm = ConditionalThetaEDM(backbone=edm_backbone, sigma_data=0.5).to(device)
    out_edm = train_theta_edm_model(
        model=edm,
        theta_train=theta_train,
        x_train=x_train,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        device=device,
        log_every=40,
        theta_val=theta_val,
        x_val=x_val,
        optimizer_name="adamw",
        weight_decay=1e-4,
        max_grad_norm=1.0,
        lr_scheduler="cosine",
        lr_warmup_frac=0.05,
        loss_type="mse",
        abort_on_nonfinite=True,
        p_mean=-1.2,
        p_std=1.2,
        sigma_data=0.5,
    )

    t = torch.from_numpy(theta_eval).to(device)
    x = torch.from_numpy(x_eval).to(device)

    results: dict[str, object] = {
        "seed": seed,
        "sigma_evals": [float(s) for s in sigma_evals],
        "noise_std_likelihood": noise_std,
        "ncsm_train": {"sigma_min": ncsm_sigma_min, "sigma_max": ncsm_sigma_max},
        "edm_train": {"p_mean": -1.2, "p_std": 1.2, "sigma_data": 0.5},
        "per_sigma": {},
    }

    rows_csv: list[dict[str, float]] = []

    for sigma_eval in sigma_evals:
        se = float(sigma_eval)
        with torch.no_grad():
            pred_ncsm = ncsm.predict_score(t, x, sigma_eval=se).cpu().numpy()
            pred_edm = edm.predict_score(t, x, sigma_eval=se).cpu().numpy()

        key = f"sigma_eval_{se:g}".replace(".", "p")
        m_ncsm = _metrics(pred_ncsm, score_true)
        m_edm = _metrics(pred_edm, score_true)
        results["per_sigma"][key] = {
            "sigma_eval": se,
            "ncsm": m_ncsm,
            "edm": m_edm,
        }
        rows_csv.append(
            {
                "sigma_eval": se,
                "ncsm_mse": m_ncsm["mse"],
                "ncsm_mae": m_ncsm["mae"],
                "ncsm_corr": m_ncsm["corr"],
                "edm_mse": m_edm["mse"],
                "edm_mae": m_edm["mae"],
                "edm_corr": m_edm["corr"],
            }
        )

        safe = f"{se:.4f}".replace(".", "p")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(score_true.reshape(-1), pred_ncsm.reshape(-1), s=6, alpha=0.2, label="NCSM")
        ax.scatter(score_true.reshape(-1), pred_edm.reshape(-1), s=6, alpha=0.2, label="EDM")
        lo = float(np.min(score_true))
        hi = float(np.max(score_true))
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=1.0)
        ax.set_xlabel("True posterior score")
        ax.set_ylabel("Predicted score")
        ax.set_title(f"sigma_eval={se:g} — NCSM vs EDM (overlay)")
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_combined_sigma_{safe}.png", dpi=180)
        fig.savefig(out_dir / f"edm_vs_ncsm_score_scatter_sigma_{safe}.png", dpi=180)
        plt.close(fig)

        _scatter_one(
            out_dir / f"scatter_ncsm_only_sigma_{safe}.png",
            score_true,
            pred_ncsm,
            title=f"NCSM — sigma_eval={se:g}",
            subtitle=f"MSE={m_ncsm['mse']:.4f}  MAE={m_ncsm['mae']:.4f}  corr={m_ncsm['corr']:.4f}",
            color="#1f77b4",
        )
        _scatter_one(
            out_dir / f"scatter_edm_only_sigma_{safe}.png",
            score_true,
            pred_edm,
            title=f"EDM — sigma_eval={se:g}",
            subtitle=f"MSE={m_edm['mse']:.4f}  MAE={m_edm['mae']:.4f}  corr={m_edm['corr']:.4f}",
            color="#d62728",
        )

    csv_path = out_dir / "edm_vs_ncsm_metrics_vs_sigma_eval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sigma_eval",
                "ncsm_mse",
                "ncsm_mae",
                "ncsm_corr",
                "edm_mse",
                "edm_mae",
                "edm_corr",
            ],
        )
        w.writeheader()
        for row in rows_csv:
            w.writerow(row)

    sig_arr = np.asarray([r["sigma_eval"] for r in rows_csv], dtype=np.float64)
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ax2.plot(sig_arr, [r["ncsm_mse"] for r in rows_csv], "o-", label="NCSM MSE", color="#1f77b4")
    ax2.plot(sig_arr, [r["edm_mse"] for r in rows_csv], "s-", label="EDM MSE", color="#d62728")
    ax2.set_xlabel("sigma_eval (predict_score)")
    ax2.set_ylabel("MSE vs analytic posterior score")
    ax2.set_title("Synthetic toy: score error vs sigma_eval")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "edm_vs_ncsm_mse_vs_sigma_eval.png", dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))
    ax3.plot(sig_arr, [r["ncsm_corr"] for r in rows_csv], "o-", label="NCSM corr", color="#1f77b4")
    ax3.plot(sig_arr, [r["edm_corr"] for r in rows_csv], "s-", label="EDM corr", color="#d62728")
    ax3.set_xlabel("sigma_eval (predict_score)")
    ax3.set_ylabel("Correlation vs analytic posterior score")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Synthetic toy: correlation vs sigma_eval")
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "edm_vs_ncsm_corr_vs_sigma_eval.png", dpi=180)
    plt.close(fig3)

    np.savez(
        out_dir / "edm_vs_ncsm_metrics_sigma_sweep.npz",
        theta_eval=theta_eval,
        score_true=score_true,
        ncsm_train=np.asarray(out_ncsm["train_losses"], dtype=np.float64),
        ncsm_val=np.asarray(out_ncsm["val_losses"], dtype=np.float64),
        edm_train=np.asarray(out_edm["train_losses"], dtype=np.float64),
        edm_val=np.asarray(out_edm["val_losses"], dtype=np.float64),
    )

    (out_dir / "edm_vs_ncsm_sigma_sweep_summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}", flush=True)
    print(json.dumps(results, indent=2))
    print(f"saved_dir={out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
