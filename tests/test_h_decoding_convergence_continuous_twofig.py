import json
from pathlib import Path

import numpy as np
import pytest

from fisher import h_decoding_convergence_twofig as conv2
from fisher import h_decoding_twofig as twofig


def _write_dataset_npz(path: Path, *, theta_type: str = "continuous", family: str = "randamp_gaussian_sqrtd") -> None:
    meta = {
        "version": 2,
        "dataset_family": family,
        "theta_type": theta_type,
        "theta_encoding": "one_hot" if theta_type == "categorical" else "native",
        "seed": 7,
        "num_categories": 2,
    }
    theta_all = np.linspace(0.0, 1.0, 8, dtype=np.float64).reshape(-1, 1)
    if theta_type == "categorical":
        theta_all = np.eye(2, dtype=np.float64)[np.arange(8) % 2]
    x_all = np.column_stack([np.linspace(-1.0, 1.0, 8), np.linspace(1.0, -1.0, 8)])
    train_idx = np.arange(6, dtype=np.int64)
    validation_idx = np.arange(6, 8, dtype=np.int64)
    meta_bytes = json.dumps(meta).encode("utf-8")
    np.savez_compressed(
        path,
        meta_json_utf8=np.frombuffer(meta_bytes, dtype=np.uint8),
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )


def _write_loss_npz(path: Path, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray([1.0, 0.7, 0.5], dtype=np.float64)
    np.savez_compressed(
        path,
        theta_field_method=np.asarray([method], dtype=object),
        prior_enable=np.bool_(False),
        score_train_losses=vals,
        score_val_losses=vals + 0.1,
        score_val_monitor_losses=vals + 0.2,
    )


def _write_results_npz(path: Path, dataset_npz: Path) -> None:
    h_gt = np.asarray([[np.nan, 0.3], [0.3, np.nan]], dtype=np.float64)
    dec_ref = np.asarray([[np.nan, 0.8], [0.8, np.nan]], dtype=np.float64)
    h_sw = h_gt.reshape(1, 1, 2, 2)
    dec_sw = dec_ref.reshape(1, 2, 2)
    np.savez_compressed(
        path,
        n=np.asarray([4], dtype=np.int64),
        n_ref=np.int64(6),
        method_names=np.asarray(["bin_gaussian"], dtype=object),
        theta_bin_centers=np.asarray([0.25, 0.75], dtype=np.float64),
        h_gt_sqrt=h_gt,
        decode_ref=dec_ref,
        decode_sweep=dec_sw,
        h_sqrt_sweep=h_sw,
        corr_h_vs_gt=np.asarray([[1.0]], dtype=np.float64),
        nmse_h_vs_gt=np.asarray([[0.0]], dtype=np.float64),
        corr_decode_vs_ref=np.asarray([1.0], dtype=np.float64),
        nmse_decode_vs_ref=np.asarray([0.0], dtype=np.float64),
        wall_seconds=np.asarray([[0.0]], dtype=np.float64),
        dataset_family=np.asarray("randamp_gaussian_sqrtd", dtype=np.str_),
        dataset_npz=np.asarray(str(dataset_npz), dtype=np.str_),
        dataset_meta_seed=np.int64(7),
        dataset_pool_size=np.int64(8),
        training_losses_root=np.asarray(str(path.parent / "training_losses"), dtype=np.str_),
        theta_binning_mode=np.asarray(["theta1"], dtype=object),
    )


def test_parser_methods_default_and_old_selectors_rejected():
    parser = conv2.build_parser()
    args = parser.parse_args(["--dataset-npz", "dummy.npz"])
    assert args.methods == conv2.DEFAULT_METHODS
    assert args.lxf_low_rank_dim is None
    assert parser.parse_args(["--dataset-npz", "dummy.npz", "--lxf-low-rank-dim", "4"]).lxf_low_rank_dim == 4
    assert conv2._parse_methods(args.methods) == [
        "theta_flow",
        "x_flow",
        "linear_x_flow_t",
        "bin_gaussian",
    ]
    assert conv2._parse_methods("vae-x-flow,vae_xflow_sir_lrank,vae-bin-gaussian,vae-ctsm-v") == [
        "vae_x_flow",
        "vae_xflow_sir_lrank",
        "vae_bin_gaussian",
        "vae_ctsm_v",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-npz", "dummy.npz", "--theta-field-method", "x_flow"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-npz", "dummy.npz", "--theta-field-methods", "x_flow"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-npz", "dummy.npz", "--theta-field-rows", "x_flow"])


def test_unknown_methods_fail_clearly():
    with pytest.raises(ValueError, match="Unknown --methods token"):
        conv2._parse_methods("not_a_method")


def test_twofig_parser_accepts_vae_rows():
    parser = twofig.build_parser()
    args = parser.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-methods",
            "vae-x-flow,vae-xflow-sir-lrank,vae-bin-gaussian,vae-ctsm-v",
        ]
    )
    rows = twofig._parse_theta_field_rows(args)
    assert [r.method for r in rows] == ["vae_x_flow", "vae_xflow_sir_lrank", "vae_bin_gaussian", "vae_ctsm_v"]


def test_twofig_initial_validation_uses_vae_base_method(monkeypatch):
    parser = twofig.build_parser()
    args = parser.parse_args(
        [
            "--dataset-npz",
            "dummy.npz",
            "--theta-field-methods",
            "vae-ctsm-v",
        ]
    )
    rows = twofig._parse_theta_field_rows(args)
    validation_row = next((r for r in rows if r.method not in twofig._NO_TRAIN_METHODS), rows[0])
    validation_method = twofig._VAE_WRAPPED_METHODS.get(validation_row.method, validation_row.method)
    args.theta_field_method = "theta_flow" if validation_method in twofig._NO_TRAIN_METHODS else validation_method

    seen = {}

    def fake_validate_cli(args_in):
        seen["theta_field_method"] = args_in.theta_field_method

    monkeypatch.setattr(twofig.conv, "_validate_cli", fake_validate_cli)
    twofig.conv._validate_cli(args)
    twofig._validate_cli_for_rows(args, rows)
    assert seen["theta_field_method"] == "ctsm_v"


def test_visualization_only_convergence_outputs(tmp_path, monkeypatch):
    dataset_npz = tmp_path / "dataset.npz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_dataset_npz(dataset_npz)
    _write_results_npz(out_dir / "h_decoding_convergence_results.npz", dataset_npz)
    _write_loss_npz(out_dir / "training_losses" / "bin_gaussian" / "n_000004.npz", "bin_gaussian")

    def fake_png(output_dir, *, sweep_svg, corr_nmse_svg, loss_panel_svg):
        p = Path(output_dir) / "h_decoding_categorical_twofig_all_columns.png"
        p.write_bytes(b"png")
        return str(p)

    monkeypatch.setattr(conv2, "_write_all_columns_png", fake_png)
    conv2.main(
        [
            "--dataset-npz",
            str(dataset_npz),
            "--dataset-family",
            "randamp_gaussian_sqrtd",
            "--output-dir",
            str(out_dir),
            "--methods",
            "bin_gaussian",
            "--n-list",
            "4",
            "--n-ref",
            "6",
            "--num-theta-bins",
            "2",
            "--visualization-only",
            "--device",
            "cuda",
        ]
    )
    for name in (
        "h_decoding_convergence_results.npz",
        "h_decoding_convergence_sweep.svg",
        "h_decoding_convergence_corr_nmse.svg",
        "h_decoding_convergence_training_losses_panel.svg",
        "h_decoding_convergence_all_columns.png",
        "h_decoding_convergence_summary.txt",
    ):
        assert (out_dir / name).is_file(), name
    with np.load(out_dir / "h_decoding_convergence_results.npz", allow_pickle=True) as z:
        for key in (
            "method_names",
            "h_sqrt_sweep",
            "corr_h_vs_gt",
            "nmse_h_vs_gt",
            "corr_decode_vs_ref",
            "nmse_decode_vs_ref",
        ):
            assert key in z.files
    summary = (out_dir / "h_decoding_convergence_summary.txt").read_text()
    assert "workflow: continuous_twofig" in summary
    assert "methods: bin_gaussian" in summary


def test_categorical_npz_rejected(tmp_path):
    dataset_npz = tmp_path / "cat.npz"
    _write_dataset_npz(dataset_npz, theta_type="categorical", family="random_mog_categorical")
    parser = conv2.build_parser()
    args = parser.parse_args(
        [
            "--dataset-npz",
            str(dataset_npz),
            "--dataset-family",
            "random_mog_categorical",
        ]
    )
    with pytest.raises(ValueError, match="study_h_decoding_categorical_twofig.py"):
        conv2._validate_continuous_dataset(args)
