from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fisher.autoencoder_embedding import PRAutoencoderConfig, config_cache_key, train_or_load_pr_autoencoder


def _load_project_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "project_dataset_pr_autoencoder.py"
    spec = importlib.util.spec_from_file_location("project_dataset_pr_autoencoder", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_categorical_labels_from_one_hot_theta() -> None:
    mod = _load_project_module()
    theta = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    got = mod._categorical_labels_from_theta(theta, num_categories=3)

    np.testing.assert_array_equal(got, np.array([0, 2, 1], dtype=np.int64))


def test_categorical_labels_from_integer_theta() -> None:
    mod = _load_project_module()
    theta = np.array([[0], [2], [1]])

    got = mod._categorical_labels_from_theta(theta, num_categories=3)

    np.testing.assert_array_equal(got, np.array([0, 2, 1], dtype=np.int64))


def test_categorical_labels_reject_non_integer_theta() -> None:
    mod = _load_project_module()

    with pytest.raises(ValueError, match="integer categorical"):
        mod._categorical_labels_from_theta(np.array([0.0, 0.5, 1.0]), num_categories=2)


def test_categorical_empirical_embedded_mean_includes_last_category() -> None:
    mod = _load_project_module()
    k = 5
    labels = np.array([0, 1, 2, 3, 4, 4, 4], dtype=np.float64)
    x = np.stack(
        [
            labels,
            labels + 10.0,
        ],
        axis=1,
    )

    t_emp, mu_emp = mod._categorical_empirical_embedded_mean(labels, x, k)

    np.testing.assert_allclose(t_emp.ravel(), np.arange(k, dtype=np.float64))
    np.testing.assert_allclose(mu_emp[:, 0], np.arange(k, dtype=np.float64))
    np.testing.assert_allclose(mu_emp[4, 0], 4.0)
    np.testing.assert_allclose(mu_emp[4, 1], 14.0)


def test_project_parser_adversarial_defaults() -> None:
    mod = _load_project_module()

    args = mod.parse_args(["--input-npz", "in.npz", "--output-npz", "out.npz", "--h-dim", "5"])

    assert args.pr_adversarial_categorical is False
    assert args.pr_lambda_adv == 0.1
    assert args.pr_adv_warmup_epochs == 0
    assert args.pr_adv_ramp_epochs is None
    assert args.pr_adv_steps == 1
    assert args.pr_adv_train_samples == 0


def test_project_h_dim_validation_allows_native_dim() -> None:
    mod = _load_project_module()

    mod.validate_h_dim(h_dim=2, z_dim=2)

    with pytest.raises(ValueError, match="--h-dim must be >= latent z_dim=2"):
        mod.validate_h_dim(h_dim=1, z_dim=2)


def test_adversarial_cache_key_depends_on_source_sha() -> None:
    base = PRAutoencoderConfig(z_dim=2, h_dim=4, train_epochs=2)
    assert config_cache_key(base, seed=7) == config_cache_key(PRAutoencoderConfig(z_dim=2, h_dim=4, train_epochs=2), seed=7)

    adv_a = PRAutoencoderConfig(
        z_dim=2,
        h_dim=4,
        train_epochs=2,
        adversarial_categorical=True,
        adv_num_classes=2,
        adv_source_sha256="a",
    )
    adv_b = PRAutoencoderConfig(
        z_dim=2,
        h_dim=4,
        train_epochs=2,
        adversarial_categorical=True,
        adv_num_classes=2,
        adv_source_sha256="b",
    )

    assert config_cache_key(adv_a, seed=7) != config_cache_key(adv_b, seed=7)


def test_train_adversarial_pr_autoencoder_smoke_cpu(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    z0 = rng.normal(loc=-1.0, scale=0.1, size=(8, 2))
    z1 = rng.normal(loc=1.0, scale=0.1, size=(8, 2))
    train_z = np.vstack([z0, z1]).astype(np.float32)
    train_y = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    cfg = PRAutoencoderConfig(
        z_dim=2,
        h_dim=3,
        hidden1=8,
        hidden2=8,
        train_epochs=2,
        train_batch_size=4,
        train_lr=1e-3,
        adversarial_categorical=True,
        lambda_adv=0.1,
        adv_warmup_epochs=0,
        adv_ramp_epochs=1,
        adv_steps=1,
        adv_num_classes=2,
        adv_source_sha256="unit-test",
    )

    got = train_or_load_pr_autoencoder(
        config=cfg,
        seed=3,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
        force_retrain=True,
        train_z=train_z,
        train_y=train_y,
        logger=None,
    )

    assert got.loaded_from_cache is False
    assert set(["loss", "recon", "pr", "adv_ce", "adv_acc", "lambda_adv_eff"]).issubset(got.metrics)
    assert got.metrics["adv_acc"].shape == (2,)
    assert np.all(np.isfinite(got.metrics["adv_ce"]))


def test_non_adversarial_training_does_not_require_labels_cpu(tmp_path: Path) -> None:
    cfg = PRAutoencoderConfig(z_dim=2, h_dim=3, hidden1=8, hidden2=8, train_epochs=1, train_samples=8)

    got = train_or_load_pr_autoencoder(
        config=cfg,
        seed=3,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
        force_retrain=True,
        logger=None,
    )

    assert set(got.metrics) == {"loss", "recon", "pr"}


def test_native_dim_pr_autoencoder_trains_and_preserves_shapes_cpu(tmp_path: Path) -> None:
    cfg = PRAutoencoderConfig(
        z_dim=2,
        h_dim=2,
        hidden1=8,
        hidden2=8,
        train_epochs=1,
        train_samples=8,
        train_batch_size=4,
    )

    got = train_or_load_pr_autoencoder(
        config=cfg,
        seed=5,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
        force_retrain=True,
        logger=None,
    )

    x = torch.randn(5, 2)
    with torch.no_grad():
        h, z_hat = got.model(x)

    assert got.loaded_from_cache is False
    assert h.shape == (5, 2)
    assert z_hat.shape == (5, 2)
    assert got.metrics["loss"].shape == (1,)
    assert np.all(np.isfinite(got.metrics["loss"]))


def _balanced_binary_gaussian_data(
    *,
    n_per_class: int,
    seed: int,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    z0 = rng.normal(loc=(-1.0, -0.35), scale=(0.20, 0.55), size=(int(n_per_class), 2))
    z1 = rng.normal(loc=(1.0, 0.35), scale=(0.20, 0.55), size=(int(n_per_class), 2))
    z = np.vstack([z0, z1]).astype(np.float32)
    y = np.array([0] * int(n_per_class) + [1] * int(n_per_class), dtype=np.int64)

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = max(1, int(round(float(test_frac) * int(idx.shape[0]))))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return z[train_idx], y[train_idx], z[test_idx], y[test_idx]


def _encode_with_pr_model(model: torch.nn.Module, x: np.ndarray, *, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        h, _ = model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device))
    return h.detach().cpu().numpy().astype(np.float64, copy=False)


def _logistic_accuracy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(solver="lbfgs", random_state=0, max_iter=2000)
    clf.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.int64))
    return float(clf.score(np.asarray(x_test, dtype=np.float64), np.asarray(y_test, dtype=np.int64)))


def _plot_adversarial_pr_training_metrics(
    metrics: dict[str, np.ndarray],
    out_dir: Path,
    *,
    stem: str = "pr_adv_training_curves",
) -> Path:
    """Save loss / component / adversary curves vs epoch (Agg backend, no display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    loss = np.asarray(metrics.get("loss", []), dtype=np.float64).reshape(-1)
    if loss.size < 1:
        raise ValueError("metrics must contain a non-empty per-epoch 'loss' array")

    n_epochs = int(loss.shape[0])
    epochs = np.arange(1, n_epochs + 1, dtype=np.float64)

    def _series(key: str) -> np.ndarray | None:
        raw = metrics.get(key)
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        return arr if arr.shape[0] == n_epochs else None

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), layout="constrained")
    ax_loss, ax_recon, ax_adv, ax_acc = axes.ravel()

    ax_loss.plot(epochs, loss, color="#4c78a8", linewidth=1.2, label="total loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("PR-autoencoder total loss")
    ax_loss.grid(True, alpha=0.25, linewidth=0.6)

    recon = _series("recon")
    pr = _series("pr")
    if recon is not None:
        ax_recon.plot(epochs, recon, color="#4c78a8", linewidth=1.1, label="recon")
    if pr is not None:
        ax_recon.plot(epochs, pr, color="#54a24b", linewidth=1.1, label="PR penalty")
    ax_recon.set_xlabel("epoch")
    ax_recon.set_ylabel("value")
    ax_recon.set_title("Reconstruction and PR penalty")
    if recon is not None or pr is not None:
        ax_recon.legend(frameon=False, fontsize=8)
    ax_recon.grid(True, alpha=0.25, linewidth=0.6)

    adv_ce = _series("adv_ce")
    lam_eff = _series("lambda_adv_eff")
    if adv_ce is not None:
        ax_adv.plot(epochs, adv_ce, color="#e45756", linewidth=1.1, label="adv CE")
    if lam_eff is not None:
        ax_adv.plot(epochs, lam_eff, color="#b279a2", linewidth=1.1, label=r"$\lambda_{\mathrm{adv}}$ eff")
    ax_adv.set_xlabel("epoch")
    ax_adv.set_ylabel("value")
    ax_adv.set_title("Adversary CE and effective weight")
    if adv_ce is not None or lam_eff is not None:
        ax_adv.legend(frameon=False, fontsize=8)
    ax_adv.grid(True, alpha=0.25, linewidth=0.6)

    adv_acc = _series("adv_acc")
    if adv_acc is not None:
        ax_acc.plot(epochs, adv_acc, color="#f58518", linewidth=1.2, label="linear adv acc")
        ax_acc.axhline(0.5, color="#9d9d9d", linewidth=0.9, linestyle="--", label="chance")
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.legend(frameon=False, fontsize=8)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Linear adversary train accuracy")
    ax_acc.grid(True, alpha=0.25, linewidth=0.6)

    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def _torch_mlp_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int = 400,
) -> float:
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(int(x_train.shape[1]), 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    ).to(device)
    x_tr = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device=device)
    y_tr = torch.from_numpy(np.asarray(y_train, dtype=np.int64)).to(device=device)
    x_te = torch.from_numpy(np.asarray(x_test, dtype=np.float32)).to(device=device)
    y_te = torch.from_numpy(np.asarray(y_test, dtype=np.int64)).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for _ in range(int(epochs)):
        logits = model(x_tr)
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_te).argmax(dim=1)
        acc = (pred == y_te).to(torch.float32).mean()
    return float(acc.detach().cpu())


@pytest.mark.skipif(
    os.environ.get("RUN_PR_ADV_DECODING_DIAGNOSTIC", "") != "1",
    reason="set RUN_PR_ADV_DECODING_DIAGNOSTIC=1 to run adversarial PR decoding diagnostic",
)
def test_adversarial_projection_suppresses_linear_decode_but_keeps_mlp_decode(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the adversarial PR decoding diagnostic.")
    device = torch.device("cuda")
    z_train, y_train, z_test, y_test = _balanced_binary_gaussian_data(
        n_per_class=256,
        seed=123,
        test_frac=0.25,
    )

    raw_linear_acc = _logistic_accuracy(z_train, y_train, z_test, y_test)
    # Set PR_ADV_DECODING_DIAGNOSTIC_LAMBDA_ADV=0 to omit lambda_adv_eff * adv_ce from the
    # encoder loss (recon - lambda_pr * PR only); adversary metrics are still logged.
    lambda_adv = float(os.environ.get("PR_ADV_DECODING_DIAGNOSTIC_LAMBDA_ADV", "1.0"))
    cfg = PRAutoencoderConfig(
        z_dim=2,
        h_dim=8,
        hidden1=64,
        hidden2=96,
        train_epochs=600,
        train_batch_size=128,
        train_lr=1e-3,
        lambda_pr=2e-2,
        adversarial_categorical=True,
        lambda_adv=lambda_adv,
        adv_warmup_epochs=50,
        adv_ramp_epochs=150,
        adv_steps=4,
        adv_num_classes=2,
        adv_source_sha256="diagnostic-linear-vs-mlp",
    )
    built = train_or_load_pr_autoencoder(
        config=cfg,
        seed=11,
        device=device,
        cache_dir=tmp_path,
        force_retrain=True,
        train_z=z_train,
        train_y=y_train,
        logger=None,
    )
    plot_dir = Path(os.environ.get("PR_ADV_DECODING_DIAGNOSTIC_PLOT_DIR", tmp_path))
    curve_stem = (
        "pr_adv_training_curves_no_adv_ce"
        if float(lambda_adv) == 0.0
        else "pr_adv_training_curves"
    )
    curve_png = _plot_adversarial_pr_training_metrics(built.metrics, plot_dir, stem=curve_stem)
    print(f"[pr_adv_diagnostic] training curves: {curve_png}")
    print(f"[pr_adv_diagnostic] training curves: {curve_png.with_suffix('.svg')}")

    h_train = _encode_with_pr_model(built.model, z_train, device=device)
    h_test = _encode_with_pr_model(built.model, z_test, device=device)
    linear_acc = _logistic_accuracy(h_train, y_train, h_test, y_test)
    mlp_acc = _torch_mlp_accuracy(h_train, y_train, h_test, y_test, device=device, seed=17)

    summary = (
        f"raw_linear_acc={raw_linear_acc:.4f}, projected_linear_acc={linear_acc:.4f}, "
        f"projected_mlp_acc={mlp_acc:.4f}, "
        f"final_adv_acc={float(built.metrics['adv_acc'][-1]):.4f}, "
        f"final_recon={float(built.metrics['recon'][-1]):.6f}, final_pr={float(built.metrics['pr'][-1]):.4f}"
    )
    assert raw_linear_acc >= 0.90, summary
    assert linear_acc <= 0.75, summary
    assert mlp_acc >= 0.85, summary
    assert mlp_acc - linear_acc >= 0.15, summary
