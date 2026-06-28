"""Project the configured example Stringer session onto neural PC1/PC2.

Run from the repo root:
    mamba run -n geo_diffusion python tests/visualize_stringer_example_pca.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.stringer_dataset import load_stringer_session


def _default_output_png() -> Path:
    return _REPO_ROOT / "tests" / "stringer_example_pc1_pc2.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-png", type=Path, default=_default_output_png())
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-neurons", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    session = load_stringer_session()
    responses = np.asarray(session.neural_responses, dtype=np.float32)
    orientation = np.asarray(session.grating_orientation, dtype=np.float64)

    rng = np.random.default_rng(int(args.seed))
    trial_idx = np.arange(responses.shape[0])
    neuron_idx = np.arange(responses.shape[1])
    if args.max_trials is not None and int(args.max_trials) < trial_idx.size:
        trial_idx = np.sort(rng.choice(trial_idx, size=int(args.max_trials), replace=False))
    if args.max_neurons is not None and int(args.max_neurons) < neuron_idx.size:
        neuron_idx = np.sort(rng.choice(neuron_idx, size=int(args.max_neurons), replace=False))

    x = responses[np.ix_(trial_idx, neuron_idx)]
    theta = orientation[trial_idx]

    pca = PCA(n_components=2, svd_solver="randomized", random_state=int(args.seed))
    pcs = pca.fit_transform(x)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg = args.output_svg
    if output_svg is None:
        output_svg = args.output_png.with_suffix(".svg")
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    scatter = ax.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=theta,
        s=9,
        alpha=0.75,
        cmap="hsv",
        linewidths=0,
    )
    date = str(session.meta.get("date", "")).replace("_", "-")
    title = f"{session.session_stimuli_type}: {session.meta.get('mouse_name', '')} {date} block {session.meta.get('block', '')}"
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("grating orientation (radians)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(args.output_png, dpi=220)
    fig.savefig(output_svg)
    plt.close(fig)

    print(f"session_file: {session.session_file}")
    print(f"session_stimuli_type: {session.session_stimuli_type}")
    print(f"raw_neural_responses_shape: {session.neural_responses.shape}")
    print(f"pca_input_shape: {x.shape}")
    print(f"explained_variance_ratio: {pca.explained_variance_ratio_.tolist()}")
    print(f"output_png: {args.output_png}")
    print(f"output_svg: {output_svg}")


if __name__ == "__main__":
    main()
