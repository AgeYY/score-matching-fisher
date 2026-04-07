#!/usr/bin/env python3
"""Load a shared dataset, estimate H-matrix via score/prior training, then Classical MDS embedding.

Computes classical MDS from H^sym, plus Euclidean/PCA/Isomap/UMAP-on-x baselines.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

try:
    import umap
except ImportError as e:
    raise ImportError(
        "visualize_h_matrix_mds.py requires the umap-learn package. "
        "Install with: pip install umap-learn"
    ) from e

from global_setting import DATAROOT
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    run_shared_fisher_estimation,
    validate_estimation_args,
)

# Dense N×N arrays in h_mds_embedding.npz are omitted when n exceeds this (saves disk / quota).
N_FULL_MATRIX_SAVE_MAX = 5000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, run Fisher + H-matrix estimation, then 2D Classical MDS "
            "from H_sym distance matrix, plus baseline embeddings."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from fisher_make_dataset.py.",
    )
    p.add_argument(
        "--b-sym-eps",
        type=float,
        default=1e-15,
        help="Lower clip for affinity (1-H^sym) when computing B^sym; avoids -log(0).",
    )
    p.add_argument(
        "--distance-transform",
        type=str,
        default="sqrt",
        choices=["identity", "sqrt"],
        help="Build distance D from H_sym: identity (D=L) or sqrt (D=sqrt(max(L,0))).",
    )
    p.add_argument(
        "--isomap-n-neighbors",
        type=int,
        default=100,
        help="Isomap n_neighbors on x (capped at n_samples - 1).",
    )
    p.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=100,
        help="UMAP n_neighbors for UMAP on x (capped at n_samples - 1).",
    )
    p.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist for UMAP on x.",
    )
    p.add_argument(
        "--umap-random-state",
        type=int,
        default=-1,
        help="Random seed for UMAP; -1 uses the dataset seed from the .npz meta.",
    )
    p.add_argument(
        "--pca-random-state",
        type=int,
        default=0,
        help="Random seed for PCA (sklearn).",
    )
    p.add_argument(
        "--mds-only",
        action="store_true",
        default=False,
        help="Skip score/prior/decoder training; load h_matrix_results*.npz from output-dir and only build MDS/embeddings.",
    )
    p.add_argument(
        "--omit-npz-full-matrices",
        action="store_true",
        default=False,
        help="Do not store N×N distance / H / B arrays in h_mds_embedding.npz (much smaller file; avoids disk quota issues).",
    )
    add_estimation_arguments(p)
    # Prefer a dedicated default output directory for this workflow (overrides estimation default).
    p.set_defaults(output_dir=str(Path(DATAROOT) / "outputs_h_matrix_mds"))
    return p.parse_args()


def distance_matrix_from_symmetric_loss(loss_sym: np.ndarray, transform: str) -> np.ndarray:
    """Build a symmetric distance matrix from nonnegative symmetric loss entries."""
    m = np.asarray(loss_sym, dtype=np.float64)
    if transform == "identity":
        d = np.clip(m, 0.0, None)
    elif transform == "sqrt":
        d = np.sqrt(np.clip(m, 0.0, None))
    else:
        raise ValueError(f"Unknown distance transform: {transform}")
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def compute_b_sym_from_h_sym(h_sym: np.ndarray, eps: float) -> tuple[np.ndarray, dict[str, float | int]]:
    """B^sym_ij = -log(clip(1 - H^sym_ij, eps, 1)); symmetric average; diagonal forced to 0."""
    h = np.asarray(h_sym, dtype=np.float64)
    n = h.shape[0]
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h_sym must be square.")
    raw_aff = 1.0 - h
    off = ~np.eye(n, dtype=bool)
    n_off = int(np.sum(off))
    clipped_mask = (raw_aff < eps) & off
    n_clipped = int(np.sum(clipped_mask))
    affinity = np.clip(raw_aff, eps, 1.0)
    b = -np.log(affinity)
    b = 0.5 * (b + b.T)
    np.fill_diagonal(b, 0.0)
    diag = {
        "b_sym_eps": float(eps),
        "b_sym_n_offdiag": n_off,
        "b_sym_n_affinity_clipped": n_clipped,
        "b_sym_frac_affinity_clipped": float(n_clipped / max(n_off, 1)),
    }
    return b, diag


def compute_b_sym_from_directed(h_directed: np.ndarray, eps: float) -> np.ndarray:
    """B^sym = -log(1 - 0.5 H^→ - 0.5 H^→T) with affinity clipped."""
    hd = np.asarray(h_directed, dtype=np.float64)
    raw_aff = 1.0 - 0.5 * hd - 0.5 * hd.T
    affinity = np.clip(raw_aff, eps, 1.0)
    b = -np.log(affinity)
    b = 0.5 * (b + b.T)
    np.fill_diagonal(b, 0.0)
    return b


def validate_symmetric_loss_matrix(mat: np.ndarray, name: str) -> None:
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} must be square.")
    asym = float(np.max(np.abs(m - m.T)))
    if asym > 1e-8:
        raise ValueError(f"{name} is not symmetric within tolerance: max|m-m.T|={asym}")
    n = m.shape[0]
    off = m[~np.eye(n, dtype=bool)]
    if np.any(off < -1e-10):
        raise ValueError(f"{name} has negative off-diagonal entries (min={float(np.min(off))}).")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} contains non-finite values.")


def classical_mds_from_distances(
    d: np.ndarray, n_components: int = 2, eig_tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Double-center D^2 and eigendecompose B = -0.5 J D^2 J.

    Returns
    -------
    embedding : (N, n_components)
    eigenvalues_used : (n_components,)
    eigenvalues_all : (N,)
    strain_rel : relative Frobenius strain ||B - B_hat||_F / ||B||_F
    """
    n = int(d.shape[0])
    if d.shape != (n, n):
        raise ValueError("Distance matrix must be square.")
    j = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    d2 = d * d
    b = -0.5 * j @ d2 @ j
    b = 0.5 * (b + b.T)
    evals, evecs = np.linalg.eigh(b)
    # ascending order: largest last
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    pos = evals > eig_tol
    n_pos = int(np.sum(pos))
    if n_pos < n_components:
        print(
            f"[mds] WARNING: only {n_pos} positive eigenvalue(s) > {eig_tol}; "
            f"embedding may be degenerate."
        )

    lam = np.maximum(evals[:n_components], 0.0)
    v = evecs[:, :n_components]
    embedding = v * np.sqrt(lam.reshape(1, -1))

    # Strain: relative reconstruction error of B using top n_components
    b_hat = (v * lam.reshape(1, -1)) @ v.T
    fn = float(np.linalg.norm(b, ord="fro"))
    strain_rel = float(np.linalg.norm(b - b_hat, ord="fro") / (fn + 1e-12))

    return embedding, evals[:n_components], evals, strain_rel


def euclidean_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be 2D for Euclidean distances.")
    n = x.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    d = squareform(pdist(x, metric="euclidean"))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def x_for_fisher_theta_alignment(bundle: SharedDatasetBundle, full_args: SimpleNamespace) -> np.ndarray:
    """Same rows as theta_score_fisher_eval / x_score_fisher_eval in shared_fisher_est."""
    mode = str(getattr(full_args, "score_fisher_eval_data", "full"))
    if mode == "full":
        return np.asarray(bundle.x_all, dtype=np.float64)
    if mode == "score_eval":
        return np.asarray(bundle.x_eval, dtype=np.float64)
    raise ValueError(f"Unknown score_fisher_eval_data: {mode}")


def theta_for_fisher_alignment(bundle: SharedDatasetBundle, full_args: SimpleNamespace) -> np.ndarray:
    mode = str(getattr(full_args, "score_fisher_eval_data", "full"))
    if mode == "full":
        return np.asarray(bundle.theta_all, dtype=np.float64).reshape(-1)
    if mode == "score_eval":
        return np.asarray(bundle.theta_eval, dtype=np.float64).reshape(-1)
    raise ValueError(f"Unknown score_fisher_eval_data: {mode}")


def fit_pca_2d(x: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    pca = PCA(n_components=2, random_state=int(random_state))
    emb = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_
    return emb.astype(np.float64), evr.astype(np.float64)


def fit_isomap_2d(x: np.ndarray, n_neighbors: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    k = int(n_neighbors)
    k = max(2, min(k, n - 1))
    iso = Isomap(n_components=2, n_neighbors=k)
    return iso.fit_transform(x).astype(np.float64)


def fit_umap_2d(
    x: np.ndarray, n_neighbors: int, min_dist: float, random_state: int
) -> np.ndarray:
    """UMAP on feature vectors x (Euclidean metric)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    k = int(n_neighbors)
    k = max(2, min(k, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=k,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    return reducer.fit_transform(x).astype(np.float64)


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz
    distance_transform = args.distance_transform
    b_sym_eps = float(args.b_sym_eps)

    validate_estimation_args(args)
    if b_sym_eps <= 0.0:
        raise ValueError("--b-sym-eps must be positive.")

    bundle = load_shared_dataset_npz(dataset_npz)
    meta = bundle.meta
    full_args = merge_meta_into_args(meta, args)

    # Force H-matrix in original row order (aligned with score_fisher_eval rows).
    setattr(full_args, "compute_h_matrix", True)
    setattr(full_args, "h_restore_original_order", True)

    np.random.seed(int(meta["seed"]))
    torch.manual_seed(int(meta["seed"]))
    rng = np.random.default_rng(int(meta["seed"]))

    _ = require_device(str(full_args.device))

    dataset = build_dataset_from_meta(meta)
    os.makedirs(full_args.output_dir, exist_ok=True)

    print(
        f"[h_mds] dataset_npz={dataset_npz} output_dir={full_args.output_dir} "
        f"distance_transform={distance_transform} mds_only={bool(getattr(args, 'mds_only', False))} "
        f"omit_npz_full_matrices={bool(getattr(args, 'omit_npz_full_matrices', False))}"
    )

    if not bool(getattr(args, "mds_only", False)):
        run_shared_fisher_estimation(
            full_args,
            dataset,
            theta_all=bundle.theta_all,
            x_all=bundle.x_all,
            theta_train=bundle.theta_train,
            x_train=bundle.x_train,
            theta_eval=bundle.theta_eval,
            x_eval=bundle.x_eval,
            rng=rng,
        )
    else:
        print("[h_mds] --mds-only: skipping Fisher training; using existing h_matrix_results*.npz.")

    suffix = "_non_gauss" if full_args.dataset_family == "gmm_non_gauss" else "_theta_cov"
    h_path = os.path.join(full_args.output_dir, f"h_matrix_results{suffix}.npz")
    if not os.path.isfile(h_path):
        raise FileNotFoundError(f"Expected H-matrix file not found: {h_path}")

    h_npz = np.load(h_path, allow_pickle=True)
    h_sym = np.asarray(h_npz["h_sym"], dtype=np.float64)
    theta_used = np.asarray(h_npz["theta_used"], dtype=np.float64).reshape(-1)
    h_field_method = str(h_npz["h_field_method"][0]) if "h_field_method" in h_npz.files else "dsm"
    h_eval_scalar_name = str(h_npz["h_eval_scalar_name"][0]) if "h_eval_scalar_name" in h_npz.files else "sigma_eval"
    h_eval_scalar_value = (
        float(np.asarray(h_npz["sigma_eval"], dtype=np.float64).reshape(-1)[0])
        if "sigma_eval" in h_npz.files
        else float("nan")
    )
    print(f"[h_mds] source_field={h_field_method} {h_eval_scalar_name}={h_eval_scalar_value:.6f}")

    x_aligned = x_for_fisher_theta_alignment(bundle, full_args)
    theta_chk = theta_for_fisher_alignment(bundle, full_args)
    if x_aligned.shape[0] != theta_used.shape[0]:
        raise ValueError(
            f"x/H row mismatch: x_aligned={x_aligned.shape[0]} theta_used={theta_used.shape[0]}"
        )
    if not np.allclose(theta_chk, theta_used, rtol=0.0, atol=1e-5):
        raise ValueError("theta_used from H-matrix npz does not match dataset theta for score_fisher_eval_data split.")

    validate_symmetric_loss_matrix(h_sym, "h_sym")
    b_sym, b_sym_diag = compute_b_sym_from_h_sym(h_sym, b_sym_eps)
    validate_symmetric_loss_matrix(b_sym, "b_sym")
    if "h_directed" in h_npz.files:
        b_from_dir = compute_b_sym_from_directed(
            np.asarray(h_npz["h_directed"], dtype=np.float64), b_sym_eps
        )
        max_diff = float(np.max(np.abs(b_sym - b_from_dir)))
        b_sym_diag["b_sym_max_abs_diff_hdir_route"] = max_diff
        if max_diff > 1e-5:
            print(f"[h_mds] WARNING: B_sym from H_sym vs h_directed differ max|Δ|={max_diff:.3e}")

    d_mat_h = distance_matrix_from_symmetric_loss(h_sym, distance_transform)
    emb_h_mds, lam_h_used, lam_h_all, strain_h = classical_mds_from_distances(d_mat_h, n_components=2)

    d_euclid = euclidean_distance_matrix(x_aligned)
    emb_euclid_mds, lam_euclid_used, lam_euclid_all, strain_euclid = classical_mds_from_distances(
        d_euclid, n_components=2
    )

    emb_pca, pca_evr = fit_pca_2d(x_aligned, random_state=int(args.pca_random_state))
    isomap_k = int(args.isomap_n_neighbors)
    emb_isomap = fit_isomap_2d(x_aligned, n_neighbors=isomap_k)

    umap_k = int(args.umap_n_neighbors)
    umap_min = float(args.umap_min_dist)
    umap_rs = int(meta["seed"]) if int(args.umap_random_state) < 0 else int(args.umap_random_state)
    emb_x_umap = fit_umap_2d(
        x_aligned, n_neighbors=umap_k, min_dist=umap_min, random_state=umap_rs
    )

    out_npz = os.path.join(full_args.output_dir, "h_mds_embedding.npz")
    n_samples = int(h_sym.shape[0])
    save_full_dense = (n_samples <= N_FULL_MATRIX_SAVE_MAX) and (not bool(getattr(args, "omit_npz_full_matrices", False)))

    npz_payload: dict[str, object] = {
        "embedding_2d_h_mds": emb_h_mds,
        "embedding_2d_selected_mds": emb_h_mds,
        "embedding_2d_euclidean_mds": emb_euclid_mds,
        "embedding_2d_pca": emb_pca,
        "embedding_2d_isomap": emb_isomap,
        "embedding_2d_x_umap": emb_x_umap,
        "embedding_2d": emb_h_mds,
        "theta": theta_used,
        "x_aligned": x_aligned,
        "distance_transform": np.asarray([distance_transform], dtype=object),
        "eigenvalues_mds_h": lam_h_used,
        "eigenvalues_all_h": lam_h_all,
        "strain_relative_h": np.asarray([strain_h], dtype=np.float64),
        "eigenvalues_mds_euclidean": lam_euclid_used,
        "eigenvalues_all_euclidean": lam_euclid_all,
        "strain_relative_euclidean": np.asarray([strain_euclid], dtype=np.float64),
        "pca_explained_variance_ratio": pca_evr,
        "isomap_n_neighbors": np.asarray([isomap_k], dtype=np.int64),
        "umap_n_neighbors": np.asarray([umap_k], dtype=np.int64),
        "umap_min_dist": np.asarray([umap_min], dtype=np.float64),
        "umap_random_state": np.asarray([umap_rs], dtype=np.int64),
        "dataset_npz": np.asarray([os.path.abspath(dataset_npz)], dtype=object),
        "h_field_method": np.asarray([h_field_method], dtype=object),
        "h_eval_scalar_name": np.asarray([h_eval_scalar_name], dtype=object),
        "h_eval_scalar_value": np.asarray([h_eval_scalar_value], dtype=np.float64),
        "b_sym_eps": np.asarray([b_sym_eps], dtype=np.float64),
        "npz_omits_full_matrices": np.asarray([not save_full_dense], dtype=bool),
        "npz_full_matrix_save_max_n": np.asarray([N_FULL_MATRIX_SAVE_MAX], dtype=np.int64),
    }
    if save_full_dense:
        npz_payload["distance_matrix_h"] = d_mat_h
        npz_payload["distance_matrix_selected"] = d_mat_h
        npz_payload["distance_matrix_euclidean"] = d_euclid
        npz_payload["h_sym"] = h_sym
        npz_payload["b_sym"] = b_sym
        npz_payload["base_matrix_selected"] = h_sym
    for k, v in b_sym_diag.items():
        if k == "b_sym_eps":
            continue
        if isinstance(v, float):
            npz_payload[k] = np.asarray([v], dtype=np.float64)
        else:
            npz_payload[k] = np.asarray([v], dtype=np.int64)
    np.savez_compressed(out_npz, **npz_payload)

    fig_path = os.path.join(full_args.output_dir, "h_mds_embedding_theta_color.png")
    k_used = max(2, min(isomap_k, theta_used.size - 1))
    umap_k_used = max(2, min(umap_k, theta_used.size - 1))
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.0), sharex=False, sharey=False, layout="constrained")
    ax_flat = np.asarray(axes).ravel()
    umap_title_suffix = (
        r"$k$=" + str(umap_k_used) + r", min\_dist=" + f"{umap_min:.3g}" + r", seed=" + str(umap_rs)
    )
    panels: list[tuple[np.ndarray, str, str, str]] = [
        (emb_h_mds, r"$H^{\mathrm{sym}}$ → Classical MDS", "MDS 1", "MDS 2"),
        (emb_euclid_mds, r"Euclidean $\rightarrow$ Classical MDS", "MDS 1", "MDS 2"),
        (emb_pca, "PCA (2 components)", "PC 1", "PC 2"),
        (emb_isomap, f"Isomap on $x$ ($k$={k_used})", "Isomap 1", "Isomap 2"),
        (
            emb_x_umap,
            r"UMAP on $x$ (" + umap_title_suffix + ")",
            "UMAP 1",
            "UMAP 2",
        ),
    ]
    for ax, (emb, title, xl, yl) in zip(ax_flat[: len(panels)], panels):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=theta_used, s=12, alpha=0.65, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    for ax in ax_flat[len(panels) :]:
        ax.set_visible(False)
    fig.suptitle("2D embedding of rep", fontsize=13)
    fig.colorbar(sc, ax=ax_flat[: len(panels)].tolist(), label=r"$\theta$", shrink=0.58, aspect=30)
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    summary_path = os.path.join(full_args.output_dir, "h_mds_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("H-matrix + baseline embeddings summary\n")
        f.write(f"dataset_npz: {dataset_npz}\n")
        f.write(f"output_dir: {full_args.output_dir}\n")
        f.write(f"n_samples: {h_sym.shape[0]}\n")
        f.write(f"score_fisher_eval_data: {getattr(full_args, 'score_fisher_eval_data', '')}\n")
        f.write(f"h_field_method: {h_field_method}\n")
        f.write(f"{h_eval_scalar_name}: {h_eval_scalar_value}\n")
        f.write(f"distance_transform (loss→D): {distance_transform}\n")
        f.write(f"npz_omits_full_matrices: {not save_full_dense} (threshold n<={N_FULL_MATRIX_SAVE_MAX})\n")
        f.write(f"h_sym min: {float(np.min(h_sym))} max: {float(np.max(h_sym))}\n")
        f.write(f"b_sym min: {float(np.min(b_sym))} max: {float(np.max(b_sym))}\n")
        f.write(f"b_sym_eps: {b_sym_eps}\n")
        f.write(f"b_sym_n_affinity_clipped: {b_sym_diag.get('b_sym_n_affinity_clipped', '')}\n")
        f.write(f"b_sym_frac_affinity_clipped: {b_sym_diag.get('b_sym_frac_affinity_clipped', '')}\n")
        if "b_sym_max_abs_diff_hdir_route" in b_sym_diag:
            f.write(f"b_sym_max_abs_diff_hdir_route: {b_sym_diag['b_sym_max_abs_diff_hdir_route']}\n")
        f.write(f"H MDS eigenvalues_used: {lam_h_used.tolist()}\n")
        f.write(f"H MDS strain_relative_frobenius: {strain_h}\n")
        f.write(f"Euclidean MDS eigenvalues_used: {lam_euclid_used.tolist()}\n")
        f.write(f"Euclidean MDS strain_relative_frobenius: {strain_euclid}\n")
        f.write(f"PCA explained_variance_ratio: {pca_evr.tolist()}\n")
        f.write(f"Isomap n_neighbors (requested): {isomap_k}\n")
        f.write(f"UMAP n_neighbors: {umap_k} min_dist: {umap_min} random_state: {umap_rs}\n")
        f.write("(UMAP on x: embedding_2d_x_umap.)\n")
        f.write(f"min_eigenvalue_all (H Gram): {float(np.min(lam_h_all))}\n")
        f.write(f"max_eigenvalue_all (H Gram): {float(np.max(lam_h_all))}\n")
        f.write(f"artifacts:\n  {out_npz}\n  {fig_path}\n")

    print("[h_mds] Saved:")
    print(f"  - {out_npz}")
    print(f"  - {fig_path}")
    print(f"  - {summary_path}")
    print(
        f"[h_mds] H_mds strain={strain_h:.6f} "
        f"Euclid_mds strain={strain_euclid:.6f} pca_evr={pca_evr} isomap_k={isomap_k} "
        f"umap_k={umap_k} umap_min_dist={umap_min}"
    )


if __name__ == "__main__":
    main()
