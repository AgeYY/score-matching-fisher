#!/usr/bin/env python3
"""Load a shared dataset, estimate H-matrix via score/prior training, then Classical MDS embedding."""

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

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    run_shared_fisher_estimation,
    validate_estimation_args,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, run Fisher + H-matrix estimation, then 2D Classical MDS "
            "from a distance matrix derived from H_sym."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from fisher_make_dataset.py.",
    )
    p.add_argument(
        "--distance-transform",
        type=str,
        default="sqrt",
        choices=["identity", "sqrt"],
        help="Build distance D from H_sym: identity (D=H) or sqrt (D=sqrt(max(H,0))).",
    )
    p.add_argument(
        "--isomap-n-neighbors",
        type=int,
        default=15,
        help="Isomap n_neighbors (capped at n_samples - 1).",
    )
    p.add_argument(
        "--pca-random-state",
        type=int,
        default=0,
        help="Random seed for PCA (sklearn).",
    )
    add_estimation_arguments(p)
    # Prefer a dedicated default output directory for this workflow (overrides estimation default).
    p.set_defaults(output_dir="data/outputs_h_matrix_mds")
    return p.parse_args()


def distance_matrix_from_h_sym(h_sym: np.ndarray, transform: str) -> np.ndarray:
    h = np.asarray(h_sym, dtype=np.float64)
    if transform == "identity":
        d = np.clip(h, 0.0, None)
    elif transform == "sqrt":
        d = np.sqrt(np.clip(h, 0.0, None))
    else:
        raise ValueError(f"Unknown distance transform: {transform}")
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


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


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz
    distance_transform = args.distance_transform

    validate_estimation_args(args)

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
        f"distance_transform={distance_transform}"
    )

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

    suffix = "_non_gauss" if full_args.dataset_family == "gmm_non_gauss" else "_theta_cov"
    h_path = os.path.join(full_args.output_dir, f"h_matrix_results{suffix}.npz")
    if not os.path.isfile(h_path):
        raise FileNotFoundError(f"Expected H-matrix file not found: {h_path}")

    h_npz = np.load(h_path, allow_pickle=True)
    h_sym = np.asarray(h_npz["h_sym"], dtype=np.float64)
    theta_used = np.asarray(h_npz["theta_used"], dtype=np.float64).reshape(-1)

    x_aligned = x_for_fisher_theta_alignment(bundle, full_args)
    theta_chk = theta_for_fisher_alignment(bundle, full_args)
    if x_aligned.shape[0] != theta_used.shape[0]:
        raise ValueError(
            f"x/H row mismatch: x_aligned={x_aligned.shape[0]} theta_used={theta_used.shape[0]}"
        )
    if not np.allclose(theta_chk, theta_used, rtol=0.0, atol=1e-5):
        raise ValueError("theta_used from H-matrix npz does not match dataset theta for score_fisher_eval_data split.")

    d_mat = distance_matrix_from_h_sym(h_sym, distance_transform)
    emb_h, lam_used, lam_all, strain_rel = classical_mds_from_distances(d_mat, n_components=2)

    d_euclid = euclidean_distance_matrix(x_aligned)
    emb_euclid_mds, lam_euclid_used, lam_euclid_all, strain_euclid = classical_mds_from_distances(
        d_euclid, n_components=2
    )

    emb_pca, pca_evr = fit_pca_2d(x_aligned, random_state=int(args.pca_random_state))
    isomap_k = int(args.isomap_n_neighbors)
    emb_isomap = fit_isomap_2d(x_aligned, n_neighbors=isomap_k)

    out_npz = os.path.join(full_args.output_dir, "h_mds_embedding.npz")
    np.savez(
        out_npz,
        embedding_2d_h_mds=emb_h,
        embedding_2d_euclidean_mds=emb_euclid_mds,
        embedding_2d_pca=emb_pca,
        embedding_2d_isomap=emb_isomap,
        embedding_2d=emb_h,
        theta=theta_used,
        x_aligned=x_aligned,
        distance_matrix_h=d_mat,
        distance_matrix_euclidean=d_euclid,
        h_sym=h_sym,
        distance_transform=np.asarray([distance_transform], dtype=object),
        eigenvalues_mds_h=lam_used,
        eigenvalues_all_h=lam_all,
        strain_relative_h=np.asarray([strain_rel], dtype=np.float64),
        eigenvalues_mds_euclidean=lam_euclid_used,
        eigenvalues_all_euclidean=lam_euclid_all,
        strain_relative_euclidean=np.asarray([strain_euclid], dtype=np.float64),
        pca_explained_variance_ratio=pca_evr,
        isomap_n_neighbors=np.asarray([isomap_k], dtype=np.int64),
        dataset_npz=np.asarray([os.path.abspath(dataset_npz)], dtype=object),
    )

    fig_path = os.path.join(full_args.output_dir, "h_mds_embedding_theta_color.png")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), sharex=False, sharey=False, layout="constrained")
    panels: list[tuple[np.ndarray, str, str, str]] = [
        (emb_h, "H → Classical MDS", "MDS 1", "MDS 2"),
        (emb_euclid_mds, r"Euclidean $\rightarrow$ Classical MDS", "MDS 1", "MDS 2"),
        (emb_pca, "PCA (2 components)", "PC 1", "PC 2"),
        (emb_isomap, f"Isomap ($k$={max(2, min(isomap_k, theta_used.size - 1))})", "Isomap 1", "Isomap 2"),
    ]
    for ax, (emb, title, xl, yl) in zip(np.asarray(axes).ravel(), panels):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=theta_used, s=12, alpha=0.65, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.suptitle(r"2D embeddings colored by $\theta$", fontsize=13)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label=r"$\theta$", shrink=0.58, aspect=30)
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    summary_path = os.path.join(full_args.output_dir, "h_mds_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("H-matrix + baseline embeddings summary\n")
        f.write(f"dataset_npz: {dataset_npz}\n")
        f.write(f"output_dir: {full_args.output_dir}\n")
        f.write(f"n_samples: {h_sym.shape[0]}\n")
        f.write(f"score_fisher_eval_data: {getattr(full_args, 'score_fisher_eval_data', '')}\n")
        f.write(f"distance_transform (H→D): {distance_transform}\n")
        f.write(f"H MDS eigenvalues_used: {lam_used.tolist()}\n")
        f.write(f"H MDS strain_relative_frobenius: {strain_rel}\n")
        f.write(f"Euclidean MDS eigenvalues_used: {lam_euclid_used.tolist()}\n")
        f.write(f"Euclidean MDS strain_relative_frobenius: {strain_euclid}\n")
        f.write(f"PCA explained_variance_ratio: {pca_evr.tolist()}\n")
        f.write(f"Isomap n_neighbors (requested): {isomap_k}\n")
        f.write(f"min_eigenvalue_all (H Gram): {float(np.min(lam_all))}\n")
        f.write(f"max_eigenvalue_all (H Gram): {float(np.max(lam_all))}\n")
        f.write(f"artifacts:\n  {out_npz}\n  {fig_path}\n")

    print("[h_mds] Saved:")
    print(f"  - {out_npz}")
    print(f"  - {fig_path}")
    print(f"  - {summary_path}")
    print(
        f"[h_mds] H_mds strain={strain_rel:.6f} Euclid_mds strain={strain_euclid:.6f} "
        f"pca_evr={pca_evr} isomap_k={isomap_k}"
    )


if __name__ == "__main__":
    main()
