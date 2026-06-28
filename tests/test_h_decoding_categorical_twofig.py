"""Unit tests for categorical H-decoding twofig helpers."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from fisher.data import ToyCategoricalMultiRingsDataset, ToyCategoricalRandomMoGDataset
from fisher.h_decoding_categorical_twofig import (
    build_parser,
    h_sq_category_from_sample_directed,
    h_sq_directed_from_delta_l,
    hellinger_gt_sq_category_matrix,
    main,
    parse_methods,
    theta_flow_categorical_hellinger_sqrt,
    _validate_cached_cli,
    _save_method_training_loss_npz,
    _validation_only_work_sweep_subset,
    _binned_gaussian_delta_l,
    _fit_lda_weighted_projection,
    _fit_pca_weighted_projection,
    _fit_pls_weighted_projection,
    _binary_llr_1_minus_0_to_delta_l,
    _binary_delta_l_to_raw_llr_1_minus_0,
    _pairwise_delta_l_to_raw_llr,
    _iqr_zoom_ylim_from_pooled_methods,
    _pairwise_raw_llr_metrics,
    _raw_llr_metrics,
    _save_pairwise_raw_llr_est_vs_true_figure,
    _save_raw_binary_llr_est_vs_true_figure,
)
from fisher.h_matrix import HMatrixEstimator
from fisher.llr_divergence import (
    directed_kl_from_delta_l,
    sym_kl_category_from_sample_directed,
    symmetric_kl_gaussian_diag_matrix,
)
from fisher import h_decoding_convergence as conv
from fisher.h_decoding_convergence_methods import prepare_categorical_binning_for_convergence
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.svg_utils import concatenate_pngs_horizontally, concatenate_svgs_horizontally, concatenate_svgs_horizontally_to_png


def _load_llr_scatter_study_module():
    path = Path(__file__).resolve().parent / "study_h_decoding_categorical_llr_scatter.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_categorical_llr_scatter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bench_llr_module():
    path = Path(__file__).resolve().parents[1] / "bin" / "bench_llr.py"
    spec = importlib.util.spec_from_file_location("bench_llr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tiny_svg(path: Path, *, width: int = 10, height: int = 8) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"></svg>',
        encoding="utf-8",
    )
    return str(path)


def _write_tiny_png(path: Path, *, width: int = 10, height: int = 8, color=(255, 0, 0, 255)) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (width, height), color).save(path, format="PNG")
    return str(path)


def test_concatenate_svgs_horizontally_viewbox_order(tmp_path: Path) -> None:
    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"
    out = tmp_path / "out.svg"
    a.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 20"><rect id="a" width="10" height="20"/></svg>',
        encoding="utf-8",
    )
    b.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="30pt" height="15pt"><circle id="b" cx="5" cy="5" r="5"/></svg>',
        encoding="utf-8",
    )

    got = concatenate_svgs_horizontally([a, b], out, spacing=5.0)

    assert got == str(out)
    root = ET.parse(out).getroot()
    assert root.attrib["viewBox"] == "0 0 45 20"
    cols = [child for child in list(root) if child.tag.endswith("svg")]
    assert [col.attrib["x"] for col in cols] == ["0", "15"]
    assert [col.attrib["viewBox"] for col in cols] == ["0 0 10 20", "0 0 30 15"]


def test_concatenate_svgs_horizontally_normalized_height(tmp_path: Path) -> None:
    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"
    out = tmp_path / "out.svg"
    a.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>',
        encoding="utf-8",
    )
    b.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 10"><circle cx="5" cy="5" r="5"/></svg>',
        encoding="utf-8",
    )

    concatenate_svgs_horizontally([a, b], out, spacing=5.0, target_height=20.0, valign="center")

    root = ET.parse(out).getroot()
    assert root.attrib["viewBox"] == "0 0 85 20"
    cols = [child for child in list(root) if child.tag.endswith("svg")]
    assert [col.attrib["x"] for col in cols] == ["0", "25"]
    assert [col.attrib["y"] for col in cols] == ["0", "0"]
    assert [col.attrib["width"] for col in cols] == ["20", "60"]
    assert [col.attrib["height"] for col in cols] == ["20", "20"]


def test_concatenate_svgs_horizontally_prefixes_duplicate_ids(tmp_path: Path) -> None:
    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"
    out = tmp_path / "out.svg"
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<defs><clipPath id="clip"><rect id="shape" width="10" height="10"/></clipPath></defs>'
        '<g clip-path="url(#clip)"><use href="#shape" xlink:href="#shape" '
        'xmlns:xlink="http://www.w3.org/1999/xlink"/></g>'
        "</svg>"
    )
    a.write_text(source, encoding="utf-8")
    b.write_text(source, encoding="utf-8")

    concatenate_svgs_horizontally([a, b], out, spacing=0.0)

    text = out.read_text(encoding="utf-8")
    assert 'id="svg0_clip"' in text
    assert 'id="svg1_clip"' in text
    assert "url(#svg0_clip)" in text
    assert "url(#svg1_clip)" in text
    assert '#svg0_shape' in text
    assert '#svg1_shape' in text


def test_concatenate_svgs_horizontally_to_png_uses_dpi(tmp_path: Path) -> None:
    from PIL import Image

    a = tmp_path / "a.svg"
    b = tmp_path / "b.svg"
    out = tmp_path / "out.png"
    a.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 36"><rect width="72" height="36"/></svg>',
        encoding="utf-8",
    )
    b.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72"><circle cx="36" cy="36" r="20"/></svg>',
        encoding="utf-8",
    )

    got = concatenate_svgs_horizontally_to_png([a, b], out, spacing=0.0, dpi=300)

    assert got == str(out)
    with Image.open(out) as im:
        assert im.format == "PNG"
        assert im.size == (600, 300)


def test_concatenate_pngs_horizontally_normalizes_height(tmp_path: Path) -> None:
    from PIL import Image

    a = Path(_write_tiny_png(tmp_path / "a.png", width=20, height=10, color=(255, 0, 0, 128)))
    b = Path(_write_tiny_png(tmp_path / "b.png", width=10, height=20, color=(0, 0, 255, 255)))
    out = tmp_path / "combined.png"

    got = concatenate_pngs_horizontally([a, b], out, spacing=5, target_height=20, valign="center")

    assert got == str(out)
    with Image.open(out) as im:
        assert im.format == "PNG"
        assert im.mode == "RGB"
        assert im.size == (55, 20)
        assert im.getpixel((0, 0)) != (255, 0, 0, 128)


def test_concatenate_svgs_horizontally_to_png_removes_empty_failed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a.svg"
    out = tmp_path / "out.png"
    a.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 36"><rect width="72" height="36"/></svg>',
        encoding="utf-8",
    )
    out.write_bytes(b"")

    def fake_run(cmd, check):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"")
        raise RuntimeError("rsvg failed")

    monkeypatch.setattr("fisher.svg_utils.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="rsvg failed"):
        concatenate_svgs_horizontally_to_png([a], out, dpi=300)
    assert not out.exists()


def test_llr_scatter_combined_loss_outputs_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = _load_llr_scatter_study_module()
    method = "linear_x_flow_t"
    loss_npz = tmp_path / "training_losses" / study._sanitize_row_label(method) / "n_000123.npz"
    study._save_method_training_loss_npz(
        loss_npz,
        method_name=method,
        result={"train_out": {"train_losses": [2.0, 1.0], "val_losses": [2.5, 1.5]}},
    )
    assert loss_npz.is_file()
    with np.load(loss_npz, allow_pickle=True) as z:
        assert z["theta_field_method"].tolist() == [method]
        assert z["score_train_losses"].tolist() == [2.0, 1.0]

    llr_svg = Path(_write_tiny_svg(tmp_path / "llr.svg", width=20, height=5)).resolve()
    llr_png = Path(_write_tiny_png(tmp_path / "llr.png", width=40, height=10)).resolve()
    loss_svg = Path(_write_tiny_svg(tmp_path / "loss.svg", width=12, height=10)).resolve()
    combined_svg = tmp_path / "llr_est_vs_true_all_with_losses.svg"
    combined_png = tmp_path / "llr_est_vs_true_all_with_losses.png"

    def fake_svg_to_png(source_paths, out_path, *, spacing=24.0, dpi=300, target_height=None, valign="center"):
        assert source_paths == [loss_svg]
        assert dpi == 300
        _write_tiny_png(Path(out_path), width=12, height=10)
        return str(out_path)

    def fake_concat_png(source_paths, out_path, *, spacing=100, target_height=None, valign="center"):
        assert source_paths[0] == llr_png
        assert Path(source_paths[1]).name == "loss_panel.png"
        assert spacing == 100
        assert target_height == 10
        assert valign == "center"
        Path(out_path).write_bytes(b"png")
        return str(out_path)

    monkeypatch.setattr(study, "concatenate_svgs_horizontally_to_png", fake_svg_to_png)
    monkeypatch.setattr(study, "concatenate_pngs_horizontally", fake_concat_png)
    got_svg, got_png = study._write_combined_llr_loss_outputs(
        llr_svg=llr_svg,
        llr_png=llr_png,
        loss_panel_svg=loss_svg,
        combined_svg=combined_svg,
        combined_png=combined_png,
    )

    assert got_svg == combined_svg.resolve()
    assert got_png == combined_png.resolve()
    assert combined_svg.is_file()
    assert combined_png.read_bytes() == b"png"
    root = ET.parse(combined_svg).getroot()
    assert root.attrib["viewBox"] == "0 0 76 10"

    pca_svg = Path(_write_tiny_svg(tmp_path / "dataset_pca.svg", width=10, height=5)).resolve()
    pca_png = Path(_write_tiny_png(tmp_path / "dataset_pca.png", width=20, height=10)).resolve()
    mse_svg = Path(_write_tiny_svg(tmp_path / "dataset_pca_llr_mse.svg", width=15, height=5)).resolve()
    mse_png = Path(_write_tiny_png(tmp_path / "dataset_pca_llr_mse.png", width=30, height=10)).resolve()
    combined_three_svg = tmp_path / "llr_est_vs_true_all_with_losses_and_dataset_pca.svg"
    combined_three_png = tmp_path / "llr_est_vs_true_all_with_losses_and_dataset_pca.png"

    def fake_three_concat_png(source_paths, out_path, *, spacing=100, target_height=None, valign="center"):
        assert source_paths[0] == pca_png
        assert source_paths[1] == mse_png
        assert Path(source_paths[2]).name == "loss_panel.png"
        assert source_paths[3] == llr_png
        assert spacing == 100
        assert target_height == 10
        assert valign == "center"
        Path(out_path).write_bytes(b"three-png")
        return str(out_path)

    monkeypatch.setattr(study, "concatenate_pngs_horizontally", fake_three_concat_png)
    got_three_svg, got_three_png = study._write_combined_llr_loss_dataset_pca_outputs(
        llr_svg=llr_svg,
        llr_png=llr_png,
        loss_panel_svg=loss_svg,
        dataset_pca_svg=pca_svg,
        dataset_pca_png=pca_png,
        dataset_pca_llr_mse_svg=mse_svg,
        dataset_pca_llr_mse_png=mse_png,
        combined_svg=combined_three_svg,
        combined_png=combined_three_png,
    )

    assert got_three_svg == combined_three_svg.resolve()
    assert got_three_png == combined_three_png.resolve()
    assert combined_three_png.read_bytes() == b"three-png"
    root_three = ET.parse(combined_three_svg).getroot()
    assert root_three.attrib["viewBox"] == "0 0 174 10"
    cols = [child for child in list(root_three) if child.tag.endswith("svg")]
    assert [col.attrib["viewBox"] for col in cols] == [
        "0 0 10 5",
        "0 0 15 5",
        "0 0 12 10",
        "0 0 20 5",
    ]


def test_llr_scatter_per_sample_llr_mse_uses_row_and_column_offdiag() -> None:
    study = _load_llr_scatter_study_module()
    true = np.zeros((3, 3), dtype=np.float64)
    est = np.array(
        [
            [99.0, 1.0, 2.0],
            [3.0, 99.0, np.nan],
            [4.0, 5.0, 99.0],
        ],
        dtype=np.float64,
    )

    got = study._per_sample_llr_mse(est, true)

    assert got[0] == pytest.approx((1.0 + 4.0 + 9.0 + 16.0) / 4.0)
    assert got[1] == pytest.approx((9.0 + 1.0 + 25.0) / 3.0)
    assert got[2] == pytest.approx((16.0 + 4.0 + 25.0) / 3.0)


def test_llr_scatter_dataset_pca_projection_outputs(tmp_path: Path) -> None:
    study = _load_llr_scatter_study_module()
    x_eval = np.array(
        [
            [-2.0, 0.0, 0.5],
            [-1.0, 0.2, 0.4],
            [1.0, -0.1, -0.3],
            [2.0, 0.0, -0.5],
        ],
        dtype=np.float64,
    )
    bins_eval = np.array([0, 0, 1, 1], dtype=np.int64)

    svg_path, png_path = study._save_dataset_pca_projection_figure(
        x_eval,
        bins_eval,
        k_cat=2,
        out_base=tmp_path / "dataset_pca_projection",
    )

    assert svg_path == (tmp_path / "dataset_pca_projection.svg").resolve()
    assert png_path == (tmp_path / "dataset_pca_projection.png").resolve()
    assert svg_path.is_file()
    assert png_path.is_file()
    root = ET.parse(svg_path).getroot()
    assert root.tag.endswith("svg")

    mse_svg_path, mse_png_path = study._save_dataset_pca_llr_mse_projection_figure(
        x_eval,
        np.array([0.1, 0.2, 1.5, 2.0], dtype=np.float64),
        method_label="ctsm_v_binary",
        out_base=tmp_path / "dataset_pca_llr_mse_ctsm_v_binary",
    )

    assert mse_svg_path == (tmp_path / "dataset_pca_llr_mse_ctsm_v_binary.svg").resolve()
    assert mse_png_path == (tmp_path / "dataset_pca_llr_mse_ctsm_v_binary.png").resolve()
    assert mse_svg_path.is_file()
    assert mse_png_path.is_file()
    root = ET.parse(mse_svg_path).getroot()
    assert root.tag.endswith("svg")


def test_llr_scatter_summary_records_loss_and_combined_paths(tmp_path: Path) -> None:
    study = _load_llr_scatter_study_module()
    summary = tmp_path / "llr_scatter_summary.txt"
    args = SimpleNamespace(
        output_dir=tmp_path,
        num_categories=2,
        n_eval=123,
        pr_project=True,
        device="cuda",
    )
    paths = {
        "dataset_npz": (tmp_path / "random_mog_categorical.npz").resolve(),
        "work_dataset_npz": (tmp_path / "random_mog_categorical_pr5.npz").resolve(),
        "results_npz": (tmp_path / "llr_scatter_results.npz").resolve(),
        "llr_svg": (tmp_path / "llr_est_vs_true_all.svg").resolve(),
        "llr_png": (tmp_path / "llr_est_vs_true_all.png").resolve(),
        "training_losses_root": (tmp_path / "training_losses").resolve(),
        "loss_panel_svg": (tmp_path / "h_decoding_categorical_llr_scatter_training_losses_panel.svg").resolve(),
        "combined_svg": (tmp_path / "llr_est_vs_true_all_with_losses.svg").resolve(),
        "combined_png": (tmp_path / "llr_est_vs_true_all_with_losses.png").resolve(),
        "dataset_pca_svg": (tmp_path / "dataset_pca_projection.svg").resolve(),
        "dataset_pca_png": (tmp_path / "dataset_pca_projection.png").resolve(),
        "dataset_pca_llr_mse_svg": (tmp_path / "dataset_pca_llr_mse_ctsm_v_binary.svg").resolve(),
        "dataset_pca_llr_mse_png": (tmp_path / "dataset_pca_llr_mse_ctsm_v_binary.png").resolve(),
        "combined_with_dataset_pca_svg": (
            tmp_path / "llr_est_vs_true_all_with_losses_and_dataset_pca.svg"
        ).resolve(),
        "combined_with_dataset_pca_png": (
            tmp_path / "llr_est_vs_true_all_with_losses_and_dataset_pca.png"
        ).resolve(),
    }

    study._write_summary(
        summary,
        args=args,
        methods=["x_flow"],
        eval_split="all",
        n_eval_matrix=123,
        wall_seconds=np.asarray([1.25], dtype=np.float64),
        **paths,
    )

    text = summary.read_text(encoding="utf-8")
    assert f"training_losses_root: {paths['training_losses_root']}" in text
    assert f"h_decoding_categorical_llr_scatter_training_losses_panel.svg: {paths['loss_panel_svg']}" in text
    assert f"llr_est_vs_true_all_with_losses.svg: {paths['combined_svg']}" in text
    assert f"llr_est_vs_true_all_with_losses.png: {paths['combined_png']}" in text
    assert f"dataset_pca_projection.svg: {paths['dataset_pca_svg']}" in text
    assert f"dataset_pca_projection.png: {paths['dataset_pca_png']}" in text
    assert f"dataset_pca_llr_mse_ctsm_v_binary.svg: {paths['dataset_pca_llr_mse_svg']}" in text
    assert f"dataset_pca_llr_mse_ctsm_v_binary.png: {paths['dataset_pca_llr_mse_png']}" in text
    assert (
        "llr_est_vs_true_all_with_losses_and_dataset_pca.svg: "
        f"{paths['combined_with_dataset_pca_svg']}"
    ) in text
    assert (
        "llr_est_vs_true_all_with_losses_and_dataset_pca.png: "
        f"{paths['combined_with_dataset_pca_png']}"
    ) in text


def test_bench_llr_parser_n_eval_default_and_override() -> None:
    bench = _load_bench_llr_module()

    args = bench.build_parser().parse_args([])
    assert args.n_eval == 600
    assert args.device == "cuda:1"

    args2 = bench.build_parser().parse_args(["--n-eval", "123", "--methods", "bin_gaussian"])
    assert args2.n_eval == 123
    assert args2.methods == "bin_gaussian"


def test_bench_llr_combined_writer_outputs_svg_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _load_bench_llr_module()
    llr_svg = Path(_write_tiny_svg(tmp_path / "llr_est_vs_true_all.svg", width=20, height=5)).resolve()
    llr_png = Path(_write_tiny_png(tmp_path / "llr_est_vs_true_all.png", width=40, height=10)).resolve()
    loss_svg = Path(_write_tiny_svg(tmp_path / "bench_llr_training_losses_panel.svg", width=12, height=10)).resolve()
    combined_svg = tmp_path / "bench_llr.svg"
    combined_png = tmp_path / "bench_llr.png"

    def fake_svg_to_png(source_paths, out_path, *, spacing=24.0, dpi=300, target_height=None, valign="center"):
        assert source_paths == [loss_svg]
        assert dpi == 300
        _write_tiny_png(Path(out_path), width=12, height=10)
        return str(out_path)

    def fake_concat_png(source_paths, out_path, *, spacing=100, target_height=None, valign="center"):
        assert source_paths[0] == llr_png
        assert Path(source_paths[1]).name == "loss_panel.png"
        assert spacing == 100
        assert target_height == 10
        assert valign == "center"
        Path(out_path).write_bytes(b"png")
        return str(out_path)

    monkeypatch.setattr(bench, "concatenate_svgs_horizontally_to_png", fake_svg_to_png)
    monkeypatch.setattr(bench, "concatenate_pngs_horizontally", fake_concat_png)

    got_svg, got_png = bench._write_combined_llr_loss_outputs(
        llr_svg=llr_svg,
        llr_png=llr_png,
        loss_panel_svg=loss_svg,
        combined_svg=combined_svg,
        combined_png=combined_png,
    )

    assert got_svg == combined_svg.resolve()
    assert got_png == combined_png.resolve()
    assert combined_svg.is_file()
    assert combined_png.read_bytes() == b"png"
    root = ET.parse(combined_svg).getroot()
    assert root.attrib["viewBox"] == "0 0 76 10"


def test_bench_llr_diagnostic_metric_bars_include_rmse_and_pearson(tmp_path: Path) -> None:
    bench = _load_bench_llr_module()
    pair_labels = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    bins = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    true = np.array(
        [
            [1.0, 2.0, np.nan, 3.0, 4.0, np.nan],
            [5.0, np.nan, 6.0, 7.0, np.nan, 8.0],
            [np.nan, 9.0, 10.0, np.nan, 11.0, 12.0],
        ],
        dtype=np.float64,
    )
    est = true + 0.1
    metrics = {"method_a": bench._pairwise_raw_llr_metrics(est, true)}

    bench._save_pairwise_raw_llr_est_vs_true_figure(
        {"method_a": est},
        true,
        pair_labels,
        bins,
        out_base=tmp_path / "llr_est_vs_true_all",
        metrics_by_method=metrics,
    )

    svg = tmp_path / "llr_est_vs_true_all.svg"
    png = tmp_path / "llr_est_vs_true_all.png"
    assert svg.is_file()
    assert png.is_file()
    text = svg.read_text(encoding="utf-8")
    assert "Full range RMSE" in text
    assert "Full range Pearson r" in text
    assert "Full range mean error" in text
    assert text.count("true raw LLR in [-8, 8]") >= 3


def test_bench_llr_summary_records_loss_llr_combined_and_results_paths(tmp_path: Path) -> None:
    bench = _load_bench_llr_module()
    summary = tmp_path / "bench_llr_summary.txt"
    args = SimpleNamespace(
        output_dir=tmp_path,
        dataset_npz=tmp_path / "random_mog_categorical.npz",
        num_categories=3,
        n_eval=123,
        pr_project=False,
        device="cuda",
    )
    paths = {
        "results_npz": (tmp_path / "bench_llr_results.npz").resolve(),
        "bench_llr_training_losses_panel.svg": (tmp_path / "bench_llr_training_losses_panel.svg").resolve(),
        "llr_est_vs_true_all.svg": (tmp_path / "llr_est_vs_true_all.svg").resolve(),
        "llr_est_vs_true_all.png": (tmp_path / "llr_est_vs_true_all.png").resolve(),
        "bench_llr.svg": (tmp_path / "bench_llr.svg").resolve(),
        "bench_llr.png": (tmp_path / "bench_llr.png").resolve(),
        "training_losses_root": (tmp_path / "training_losses").resolve(),
    }

    bench._write_summary(summary, args=args, methods=["bin_gaussian"], eval_split="all", paths=paths)

    text = summary.read_text(encoding="utf-8")
    assert f"results_npz: {paths['results_npz']}" in text
    assert f"bench_llr_training_losses_panel.svg: {paths['bench_llr_training_losses_panel.svg']}" in text
    assert f"llr_est_vs_true_all.svg: {paths['llr_est_vs_true_all.svg']}" in text
    assert f"llr_est_vs_true_all.png: {paths['llr_est_vs_true_all.png']}" in text
    assert f"bench_llr.svg: {paths['bench_llr.svg']}" in text
    assert f"bench_llr.png: {paths['bench_llr.png']}" in text
    assert f"training_losses_root: {paths['training_losses_root']}" in text


def test_parse_methods_aliases_and_dedup() -> None:
    assert parse_methods("x-flow, X_FLOW, binary-classifier") == ["x_flow", "binary_classifier"]
    assert parse_methods("vae-x-flow, vae_x_flow") == ["vae_x_flow"]
    assert parse_methods("vae-xflow-sir-lrank, vae_xflow_sir_lrank") == ["vae_xflow_sir_lrank"]
    assert parse_methods("bin-gaussian, bin_gaussian, x_flow") == ["bin_gaussian", "x_flow"]
    assert parse_methods("vae-bin-gaussian, vae_bin_gaussian") == ["vae_bin_gaussian"]
    assert parse_methods("bin_gaussian_cate") == ["bin_gaussian"]
    assert parse_methods("contrastive-soft-categorical, contrastive_soft_categorical") == [
        "contrastive_soft_categorical"
    ]
    assert parse_methods("theta-flow-cate, theta_flow_cate, thetaflow-cate") == ["theta_flow_cate"]
    assert parse_methods("ctsm-v, ctsm_v, ctsm") == ["ctsm_v"]
    assert parse_methods("ctsm-v-binary, ctsm_v_binary, ctsm_binary") == ["ctsm_v_binary"]
    assert parse_methods("vae-ctsm-v, vae_ctsm_v") == ["vae_ctsm_v"]
    assert parse_methods("latent-belief-ctsm-v-binary, latent_ctsm_v_binary") == [
        "latent_belief_ctsm_v_binary"
    ]
    assert parse_methods(
        "latent_belief_ctsm_v_binary_inner_post, "
        "latent-belief-ctsm-v-binary-inner-post, "
        "latent_belief_ctsm_v_binary_innner_post"
    ) == ["latent_belief_ctsm_v_binary_inner_post"]
    assert parse_methods("lda-ctsm-v, lda_ctsm_v, ldactsm-v") == ["lda_ctsm_v"]
    assert parse_methods("pls-ctsm-v, pls_ctsm_v, plsctsm-v") == ["pls_ctsm_v"]
    assert parse_methods("pca-ctsm-v, pca_ctsm_v, pcactsm-v") == ["pca_ctsm_v"]
    with pytest.raises(ValueError, match="Unknown method"):
        parse_methods("theta_flow")


def test_fit_lda_weighted_projection_shapes_weights_and_order() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0, 0.1],
            [-1.0, 0.2, -0.1],
            [1.0, 0.0, 0.0],
            [2.0, -0.1, 0.2],
        ],
        dtype=np.float64,
    )
    bins_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_val = x_train[:2] + 0.5
    x_all = x_train.copy()

    z_train, z_val, z_all, meta = _fit_lda_weighted_projection(
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        x_all=x_all,
        shrinkage=1e-2,
        eps=1e-6,
        max_dim=2,
    )

    assert z_train.shape == (4, 2)
    assert z_val.shape == (2, 2)
    assert z_all.shape == (4, 2)
    weights = np.asarray(meta["lda_ctsm_weights"], dtype=np.float64)
    eigvals = np.asarray(meta["lda_ctsm_eigenvalues"], dtype=np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(2.0)
    assert np.all(eigvals[:-1] >= eigvals[1:] - 1e-12)
    assert np.asarray(meta["lda_ctsm_transform"]).shape == (2, 3)


def test_fit_pls_weighted_projection_shapes_weights_and_metadata() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0, 0.1],
            [-1.0, 0.2, -0.1],
            [1.0, 0.0, 0.0],
            [2.0, -0.1, 0.2],
            [2.5, 0.1, -0.2],
        ],
        dtype=np.float64,
    )
    bins_train = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    x_val = x_train[:2] + 0.25
    x_all = x_train.copy()

    z_train, z_val, z_all, meta = _fit_pls_weighted_projection(
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        x_all=x_all,
        eps=1e-6,
        max_dim=2,
        scale_x=True,
        max_iter=100,
        tol=1e-7,
    )

    assert z_train.shape == (5, 2)
    assert z_val.shape == (2, 2)
    assert z_all.shape == (5, 2)
    weights = np.asarray(meta["pls_ctsm_weights"], dtype=np.float64)
    scores = np.asarray(meta["pls_ctsm_covariance_scores"], dtype=np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(2.0)
    assert np.all(scores >= -1e-12)
    assert np.asarray(meta["pls_ctsm_transform"]).shape == (2, 3)
    assert bool(meta["pls_ctsm_scale_x"])


def test_fit_pca_weighted_projection_shapes_weights_and_metadata() -> None:
    x_train = np.array(
        [
            [-2.0, 0.0, 0.1],
            [-1.0, 0.2, -0.1],
            [1.0, 0.0, 0.0],
            [2.0, -0.1, 0.2],
        ],
        dtype=np.float64,
    )
    x_val = x_train[:2] + 0.25
    x_all = x_train.copy()

    z_train, z_val, z_all, meta = _fit_pca_weighted_projection(
        x_train=x_train,
        x_val=x_val,
        x_all=x_all,
        eps=1e-6,
        max_dim=2,
        scale_x=True,
    )

    assert z_train.shape == (4, 2)
    assert z_val.shape == (2, 2)
    assert z_all.shape == (4, 2)
    weights = np.asarray(meta["pca_ctsm_weights"], dtype=np.float64)
    explained = np.asarray(meta["pca_ctsm_explained_variance"], dtype=np.float64)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(2.0)
    assert np.all(explained[:-1] >= explained[1:] - 1e-12)
    assert np.asarray(meta["pca_ctsm_components"]).shape == (3, 2)
    assert np.asarray(meta["pca_ctsm_transform"]).shape == (2, 3)
    assert bool(meta["pca_ctsm_scale_x"])


def test_decode_accuracy_color_limits() -> None:
    from fisher.h_decoding_twofig import _decode_accuracy_color_limits

    a = np.array([[np.nan, 0.55], [0.55, np.nan]])
    vmin, vmax = _decode_accuracy_color_limits(a)
    assert vmin == pytest.approx(0.5) and vmax == 1.0
    b = np.array([[np.nan, 0.35], [0.35, np.nan]])
    vmin2, vmax2 = _decode_accuracy_color_limits(b)
    assert vmin2 == pytest.approx(0.35) and vmax2 == 1.0


def test_hellinger_gt_sq_category_matrix_shape_diag_bounds() -> None:
    ds = ToyCategoricalRandomMoGDataset(x_dim=2, num_categories=4, seed=0)
    h2 = hellinger_gt_sq_category_matrix(ds)
    assert h2.shape == (4, 4)
    assert np.allclose(np.diag(h2), 0.0)
    assert float(np.min(h2)) >= 0.0
    assert float(np.max(h2)) <= 1.0 + 1e-9


def test_category_aggregation_small_delta_l() -> None:
    n = 6
    k = 2
    rng = np.random.default_rng(0)
    c = rng.normal(size=(n, n)).astype(np.float64)
    delta_l = HMatrixEstimator.compute_delta_l(c)
    h_dir = h_sq_directed_from_delta_l(delta_l)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    h_cat = h_sq_category_from_sample_directed(h_dir, labels, k_cat=k)
    assert h_cat.shape == (k, k)
    assert np.allclose(np.diag(h_cat), 0.0)
    assert np.all(h_cat >= 0.0)


def test_directed_kl_from_delta_l_negates_and_zeroes_diagonal() -> None:
    delta_l = np.array(
        [
            [3.0, -2.0, 1.5],
            [4.0, -7.0, -0.5],
            [0.25, 2.5, 9.0],
        ],
        dtype=np.float64,
    )
    got = directed_kl_from_delta_l(delta_l)
    expected = -delta_l
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(got, expected)


def test_symmetric_kl_category_aggregation_small_delta_l() -> None:
    delta_l = np.array(
        [
            [0.0, -1.0, -3.0, -5.0],
            [-2.0, 0.0, -7.0, -11.0],
            [-13.0, -17.0, 0.0, -19.0],
            [-23.0, -29.0, -31.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    directed = directed_kl_from_delta_l(delta_l)
    got = sym_kl_category_from_sample_directed(directed, labels, k_cat=2)

    d01 = float(np.mean(directed[np.ix_([0, 1], [2, 3])]))
    d10 = float(np.mean(directed[np.ix_([2, 3], [0, 1])]))
    expected = np.array([[0.0, 0.5 * (d01 + d10)], [0.5 * (d01 + d10), 0.0]], dtype=np.float64)
    assert np.allclose(got, expected)


def test_symmetric_kl_gaussian_diag_matrix_matches_1d_manual_case() -> None:
    means = np.array([[0.0], [2.0]], dtype=np.float64)
    variances = np.array([[1.0], [4.0]], dtype=np.float64)
    got = symmetric_kl_gaussian_diag_matrix(means, variances)
    manual = 0.25 * ((1.0 / 4.0) + (4.0 / 1.0) + (2.0**2) * (1.0 / 1.0 + 1.0 / 4.0) - 2.0)
    assert got.shape == (2, 2)
    assert np.allclose(got, got.T)
    assert np.allclose(np.diag(got), 0.0)
    assert np.all(got >= 0.0)
    assert got[0, 1] == pytest.approx(manual)


def test_ctsm_pair_models_accept_one_hot_theta() -> None:
    from fisher.ctsm_models import ToyPairConditionedTimeScoreNet, ToyPairConditionedTimeScoreNetFiLM

    x = np.zeros((4, 2), dtype=np.float32)
    t = np.full((4, 1), 0.5, dtype=np.float32)
    m = np.eye(3, dtype=np.float32)[[0, 1, 2, 0]]
    d = np.eye(3, dtype=np.float32)[[1, 2, 0, 1]] - m
    for cls in (ToyPairConditionedTimeScoreNet, ToyPairConditionedTimeScoreNetFiLM):
        model = cls(dim=2, hidden_dim=8, theta_dim=3)
        y = model.forward_full(
            torch.from_numpy(x),
            torch.from_numpy(t),
            torch.from_numpy(m),
            torch.from_numpy(d),
        )
        assert tuple(y.shape) == (4, 2)


def test_ctsm_binary_model_accepts_no_theta_conditioning() -> None:
    from fisher.ctsm_models import ToyBinaryTimeScoreNet

    x = torch.zeros((4, 2), dtype=torch.float32)
    t = torch.full((4, 1), 0.5, dtype=torch.float32)
    model = ToyBinaryTimeScoreNet(dim=2, hidden_dim=8)

    y = model.forward_full(x, t)

    assert tuple(y.shape) == (4, 2)
    assert tuple(model(x, t).shape) == (4, 1)


def test_latent_belief_ctsm_binary_model_shapes() -> None:
    from fisher.ctsm_models import ToyLatentBeliefBinaryTimeScoreNet

    x = torch.zeros((4, 3), dtype=torch.float32)
    t = torch.full((4, 1), 0.5, dtype=torch.float32)
    model = ToyLatentBeliefBinaryTimeScoreNet(dim=3, h_dim=5, hidden_dim=8)

    mean, std = model.posterior(x, t)
    b1, b2 = model.sample_two_readouts(x, t)
    s_mean = model.posterior_mean_vector(x, t, n_mc=2)

    assert tuple(mean.shape) == (4, 5)
    assert tuple(std.shape) == (4, 5)
    assert torch.all(std > 0)
    assert tuple(b1.shape) == (4, 3)
    assert tuple(b2.shape) == (4, 3)
    assert tuple(s_mean.shape) == (4, 3)
    assert tuple(model(x, t, n_mc=2).shape) == (4, 1)


def test_latent_belief_ctsm_loss_backward_and_inference() -> None:
    from fisher.ctsm_models import ToyLatentBeliefBinaryTimeScoreNet
    from fisher.ctsm_objectives import latent_belief_ctsm_v_inner_posterior_loss, latent_belief_ctsm_v_two_sample_loss
    from fisher.ctsm_paths import TwoSB
    from fisher.shared_fisher_est import estimate_latent_belief_ctsm_v_binary_log_ratio

    model = ToyLatentBeliefBinaryTimeScoreNet(dim=2, h_dim=3, hidden_dim=8)
    path = TwoSB(dim=2, var=2.0)
    x0 = torch.randn((6, 2), dtype=torch.float32)
    x1 = torch.randn((6, 2), dtype=torch.float32)

    loss = latent_belief_ctsm_v_two_sample_loss(model, path, x0, x1, t_eps=0.01)
    loss.backward()
    model.zero_grad(set_to_none=True)
    inner_loss = latent_belief_ctsm_v_inner_posterior_loss(model, path, x0, x1, t_eps=0.01, nh=2)
    inner_loss.backward()
    llr = estimate_latent_belief_ctsm_v_binary_log_ratio(
        model,
        np.zeros((5, 2), dtype=np.float32),
        device=torch.device("cpu"),
        batch_size=3,
        eps1=0.01,
        eps2=0.01,
        n_time=3,
        n_mc_eval=2,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(inner_loss)
    assert float(inner_loss.detach()) >= 0.0
    assert any(p.grad is not None for p in model.parameters())
    assert llr.shape == (5,)
    assert np.all(np.isfinite(llr))


def test_ctsm_binary_llr_transform_matches_delta_convention() -> None:
    llr = np.array([2.0, 3.0, -5.0, 7.0], dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)

    got = _binary_llr_1_minus_0_to_delta_l(llr, bins)

    expected = np.array(
        [
            [0.0, 2.0, 0.0, 2.0],
            [-3.0, 0.0, -3.0, 0.0],
            [0.0, -5.0, 0.0, -5.0],
            [-7.0, 0.0, -7.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(got, expected)


def test_llr_scatter_raw_binary_reconstruction_from_delta() -> None:
    study = _load_llr_scatter_study_module()
    llr = np.array([2.0, 3.0, -5.0, 7.0], dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)
    delta = _binary_llr_1_minus_0_to_delta_l(llr, bins)

    got = study._binary_delta_l_to_raw_llr_1_minus_0(delta, bins)

    assert np.allclose(got, llr)


def test_llr_scatter_raw_binary_metrics_are_finite() -> None:
    study = _load_llr_scatter_study_module()
    true = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    est = np.array([-0.8, 0.1, 1.1, 1.7], dtype=np.float64)

    got = study._raw_llr_metrics(est, true)

    assert np.isfinite(got["llr_raw_rmse"])
    assert np.isfinite(got["llr_raw_pearson_r"])
    assert got["llr_raw_rmse"] > 0.0
    assert got["llr_raw_pearson_r"] > 0.9


def test_llr_scatter_raw_binary_figure_writes_svg_png(tmp_path: Path) -> None:
    study = _load_llr_scatter_study_module()
    true = np.array([-2.0, 1.5, 0.5, -1.0], dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)
    est = np.array([-1.8, 1.4, 0.7, -0.9], dtype=np.float64)
    metrics = {"method_a": study._raw_llr_metrics(est, true)}

    study._save_raw_binary_llr_est_vs_true_figure(
        {"method_a": est},
        true,
        bins,
        out_base=tmp_path / "llr_est_vs_true_all",
        metrics_by_method=metrics,
    )

    svg = tmp_path / "llr_est_vs_true_all.svg"
    png = tmp_path / "llr_est_vs_true_all.png"
    assert svg.is_file()
    assert png.is_file()
    text = svg.read_text(encoding="utf-8")
    assert "class 0" in text
    assert "class 1" in text


def test_twofig_raw_binary_reconstruction_from_delta() -> None:
    llr = np.array([1.5, -2.0, 0.25, 4.0], dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)
    delta = _binary_llr_1_minus_0_to_delta_l(llr, bins)

    got = _binary_delta_l_to_raw_llr_1_minus_0(delta, bins)

    assert np.allclose(got, llr)


def test_twofig_raw_binary_metrics_are_finite() -> None:
    true = np.array([-2.0, -0.5, 0.5, 3.0], dtype=np.float64)
    est = np.array([-1.7, -0.6, 0.75, 2.8], dtype=np.float64)

    got = _raw_llr_metrics(est, true)

    assert np.isfinite(got["llr_raw_rmse"])
    assert np.isfinite(got["llr_raw_pearson_r"])
    assert got["llr_raw_rmse"] > 0.0
    assert got["llr_raw_pearson_r"] > 0.9


def test_twofig_raw_binary_figure_writes_svg_png(tmp_path: Path) -> None:
    true = np.array([-2.0, 1.5, 0.5, -1.0], dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)
    est_a = np.array([-1.8, 1.4, 0.7, -0.9], dtype=np.float64)
    est_b = np.array([-2.2, 1.7, 0.3, -1.2], dtype=np.float64)
    metrics = {
        "method_a": _raw_llr_metrics(est_a, true),
        "method_b": _raw_llr_metrics(est_b, true),
    }

    _save_raw_binary_llr_est_vs_true_figure(
        {"method_a": est_a, "method_b": est_b},
        true,
        bins,
        out_base=tmp_path / "llr_est_vs_true_all",
        metrics_by_method=metrics,
    )

    svg = tmp_path / "llr_est_vs_true_all.svg"
    png = tmp_path / "llr_est_vs_true_all.png"
    assert svg.is_file()
    assert png.is_file()
    text = svg.read_text(encoding="utf-8")
    assert "method_a" in text
    assert "method_b" in text
    assert "class 0" in text
    assert "class 1" in text


def test_twofig_pairwise_raw_llr_reconstruction_k3() -> None:
    bins = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    delta = np.zeros((6, 6), dtype=np.float64)
    expected = np.full((3, 6), np.nan, dtype=np.float64)
    pair_values = {
        (0, 1): np.array([10.0, 11.0, np.nan, 13.0, 14.0, np.nan], dtype=np.float64),
        (0, 2): np.array([20.0, np.nan, 22.0, 23.0, np.nan, 25.0], dtype=np.float64),
        (1, 2): np.array([np.nan, 31.0, 32.0, np.nan, 34.0, 35.0], dtype=np.float64),
    }
    for pidx, ((a, b), values) in enumerate(pair_values.items()):
        expected[pidx] = values
        cols_a = bins == a
        cols_b = bins == b
        for i, yi in enumerate(bins):
            if int(yi) == a:
                delta[i, cols_b] = values[i]
            elif int(yi) == b:
                delta[i, cols_a] = -values[i]

    pair_labels, got = _pairwise_delta_l_to_raw_llr(delta, bins, k_cat=3)

    assert pair_labels.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert np.allclose(got, expected, equal_nan=True)


def test_twofig_pairwise_raw_llr_metrics_are_finite() -> None:
    true = np.array([[1.0, 2.0, np.nan], [np.nan, -1.0, 4.0]], dtype=np.float64)
    est = np.array([[1.1, 1.8, np.nan], [np.nan, -0.8, 3.7]], dtype=np.float64)

    got = _pairwise_raw_llr_metrics(est, true)

    assert np.isfinite(got["llr_pairwise_raw_rmse"])
    assert np.isfinite(got["llr_pairwise_raw_pearson_r"])
    assert got["llr_pairwise_raw_rmse"] > 0.0


def test_twofig_pairwise_raw_llr_metrics_macro_average_and_band() -> None:
    true = np.array(
        [
            [-10.0, -4.0, 0.0, 4.0, 10.0],
            [-12.0, -2.0, 2.0, 6.0, 12.0],
        ],
        dtype=np.float64,
    )
    est_a = true + np.array(
        [
            [10.0, 1.0, -1.0, 1.0, -10.0],
            [20.0, 2.0, -2.0, 2.0, -20.0],
        ],
        dtype=np.float64,
    )
    est_b = true + np.array(
        [
            [1.0, -2.0, 2.0, -2.0, 1.0],
            [-3.0, 3.0, -3.0, 3.0, -3.0],
        ],
        dtype=np.float64,
    )

    got_a = _pairwise_raw_llr_metrics(est_a, true)
    got_b = _pairwise_raw_llr_metrics(est_b, true)

    expected_pair_rmse_a = np.array(
        [
            np.sqrt(np.mean(np.array([10.0, 1.0, -1.0, 1.0, -10.0]) ** 2)),
            np.sqrt(np.mean(np.array([20.0, 2.0, -2.0, 2.0, -20.0]) ** 2)),
        ],
        dtype=np.float64,
    )
    expected_pair_r_a = np.array([np.corrcoef(est_a[pidx], true[pidx])[0, 1] for pidx in range(2)])
    expected_pair_bias_a = np.array([0.2, 0.4], dtype=np.float64)
    expected_band_rmse_a = np.array(
        [
            np.sqrt(np.mean(np.array([1.0, -1.0, 1.0]) ** 2)),
            np.sqrt(np.mean(np.array([2.0, -2.0, 2.0]) ** 2)),
        ],
        dtype=np.float64,
    )
    expected_band_bias_a = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)

    assert got_a["llr_pairwise_raw_rmse_mean_pair"] == pytest.approx(float(np.mean(expected_pair_rmse_a)))
    assert got_a["llr_pairwise_raw_pearson_r_mean_pair"] == pytest.approx(float(np.mean(expected_pair_r_a)))
    assert got_a["llr_pairwise_raw_bias_mean_pair"] == pytest.approx(float(np.mean(expected_pair_bias_a)))
    assert got_a["llr_pairwise_raw_bias_mean_pair_true_in_m8_p8"] == pytest.approx(
        float(np.mean(expected_band_bias_a))
    )
    assert np.isfinite(got_a["llr_pairwise_raw_pearson_r_mean_pair"])
    assert got_a["llr_pairwise_raw_rmse_mean_pair_true_in_m8_p8"] == pytest.approx(
        float(np.mean(expected_band_rmse_a))
    )
    assert got_a["llr_pairwise_raw_rmse_mean_pair_true_in_m8_p8"] < got_a["llr_pairwise_raw_rmse_mean_pair"]
    assert np.isfinite(got_a["llr_pairwise_raw_pearson_r_mean_pair_true_in_m8_p8"])
    assert got_b["llr_pairwise_raw_rmse_mean_pair"] == pytest.approx(
        (np.sqrt(14.0 / 5.0) + 3.0) / 2.0
    )
    assert got_b["llr_pairwise_raw_rmse_mean_pair_true_in_m8_p8"] == pytest.approx(
        (2.0 + 3.0) / 2.0
    )
    assert got_b["llr_pairwise_raw_bias_mean_pair_true_in_m8_p8"] == pytest.approx(1.0 / 6.0)


def test_twofig_iqr_zoom_ylim_uses_pooled_method_values() -> None:
    got = _iqr_zoom_ylim_from_pooled_methods(
        [
            np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            np.array([0.15, 0.25, 0.35, 1000.0], dtype=np.float64),
        ]
    )

    assert got is not None
    lo, hi = got
    assert lo < 0.0
    assert 0.35 < hi < 1000.0


def test_twofig_pairwise_raw_llr_figure_writes_svg_png_k3(tmp_path: Path) -> None:
    bins = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    pair_labels = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    true = np.array(
        [
            [10.0, 11.0, np.nan, 13.0, 14.0, np.nan],
            [20.0, np.nan, 22.0, 23.0, np.nan, 25.0],
            [np.nan, 31.0, 32.0, np.nan, 34.0, 35.0],
        ],
        dtype=np.float64,
    )
    est = true + np.array(
        [
            [0.1, -0.1, np.nan, 0.2, -0.2, np.nan],
            [-0.2, np.nan, 0.1, -0.1, np.nan, 0.3],
            [np.nan, 0.3, -0.2, np.nan, 0.1, -0.1],
        ],
        dtype=np.float64,
    )
    metrics = {"method_a": _pairwise_raw_llr_metrics(est, true)}

    _save_pairwise_raw_llr_est_vs_true_figure(
        {"method_a": est},
        true,
        pair_labels,
        bins,
        out_base=tmp_path / "llr_est_vs_true_all",
        metrics_by_method=metrics,
    )

    svg = tmp_path / "llr_est_vs_true_all.svg"
    png = tmp_path / "llr_est_vs_true_all.png"
    assert svg.is_file()
    assert png.is_file()
    text = svg.read_text(encoding="utf-8")
    assert "classes 0 vs 1" in text
    assert "classes 0 vs 2" in text
    assert "classes 1 vs 2" in text
    assert "method_a" in text
    assert "Full range RMSE" in text
    assert "Full range Pearson r" in text
    assert "Full range mean error" in text
    assert "Mean error" in text
    assert "mean(est - true)" in text
    assert "true raw LLR in [-8, 8]" in text
    assert text.count("true raw LLR in [-8, 8]") >= 3
    assert "y zoom (IQR pooled methods)" in text
    assert "fill-opacity: 0.8" not in text


def test_h_matrix_ctsm_accepts_one_hot_theta() -> None:
    from fisher.ctsm_models import ToyPairConditionedTimeScoreNet

    theta = np.eye(3, dtype=np.float64)
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    model = ToyPairConditionedTimeScoreNet(dim=2, hidden_dim=8, theta_dim=3)
    est = HMatrixEstimator(
        model_post=model,
        model_prior=None,
        sigma_eval=1e-5,
        field_method="ctsm_v",
        device=torch.device("cpu"),
        pair_batch_size=64,
        ctsm_int_n_time=3,
    )
    got = est.compute_ctsm_delta_l_matrix(theta, x)
    assert got.shape == (3, 3)
    assert np.allclose(np.diag(got), 0.0)


def test_ctsm_v_binary_dispatch_uses_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray | int] = {}

    def fake_ctsm_binary(*args, **kwargs):
        seen["bins_train"] = np.asarray(kwargs["bins_train"], dtype=np.int64)
        seen["bins_val"] = np.asarray(kwargs["bins_val"], dtype=np.int64)
        seen["bins_all"] = np.asarray(kwargs["bins_all"], dtype=np.int64)
        seen["k_cat"] = int(kwargs["k_cat"])
        n = int(np.asarray(kwargs["x_all"]).shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_binary_delta", fake_ctsm_binary)
    args = SimpleNamespace()
    theta = np.eye(2, dtype=np.float64)[[0, 1, 0, 1]]
    x = np.zeros((4, 2), dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)

    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="ctsm_v_binary",
        theta_train=theta[:2],
        x_train=x[:2],
        theta_val=theta[2:],
        x_val=x[2:],
        theta_all=theta,
        x_all=x,
        bins_train=bins[:2],
        bins_val=bins[2:],
        bins_all=bins,
        k_cat=2,
    )

    assert result["delta_l"].shape == (4, 4)
    assert np.array_equal(seen["bins_train"], bins[:2])
    assert np.array_equal(seen["bins_val"], bins[2:])
    assert np.array_equal(seen["bins_all"], bins)
    assert seen["k_cat"] == 2


def test_latent_belief_ctsm_v_binary_dispatch_uses_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray | int] = {}

    def fake_latent_ctsm_binary(*args, **kwargs):
        seen["bins_train"] = np.asarray(kwargs["bins_train"], dtype=np.int64)
        seen["bins_val"] = np.asarray(kwargs["bins_val"], dtype=np.int64)
        seen["bins_all"] = np.asarray(kwargs["bins_all"], dtype=np.int64)
        seen["k_cat"] = int(kwargs["k_cat"])
        n = int(np.asarray(kwargs["x_all"]).shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_latent_belief_ctsm_v_binary_delta", fake_latent_ctsm_binary)
    args = SimpleNamespace()
    theta = np.eye(2, dtype=np.float64)[[0, 1, 0, 1]]
    x = np.zeros((4, 2), dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)

    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="latent_belief_ctsm_v_binary",
        theta_train=theta[:2],
        x_train=x[:2],
        theta_val=theta[2:],
        x_val=x[2:],
        theta_all=theta,
        x_all=x,
        bins_train=bins[:2],
        bins_val=bins[2:],
        bins_all=bins,
        k_cat=2,
    )

    assert result["delta_l"].shape == (4, 4)
    assert np.array_equal(seen["bins_train"], bins[:2])
    assert np.array_equal(seen["bins_val"], bins[2:])
    assert np.array_equal(seen["bins_all"], bins)
    assert seen["k_cat"] == 2


def test_latent_belief_ctsm_v_binary_inner_post_dispatch_uses_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray | int | str] = {}

    def fake_latent_ctsm_binary(*args, **kwargs):
        seen["bins_train"] = np.asarray(kwargs["bins_train"], dtype=np.int64)
        seen["bins_val"] = np.asarray(kwargs["bins_val"], dtype=np.int64)
        seen["bins_all"] = np.asarray(kwargs["bins_all"], dtype=np.int64)
        seen["k_cat"] = int(kwargs["k_cat"])
        seen["method_name"] = str(kwargs["method_name"])
        n = int(np.asarray(kwargs["x_all"]).shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_latent_belief_ctsm_v_binary_delta", fake_latent_ctsm_binary)
    args = SimpleNamespace()
    theta = np.eye(2, dtype=np.float64)[[0, 1, 0, 1]]
    x = np.zeros((4, 2), dtype=np.float64)
    bins = np.array([0, 1, 0, 1], dtype=np.int64)

    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="latent_belief_ctsm_v_binary_inner_post",
        theta_train=theta[:2],
        x_train=x[:2],
        theta_val=theta[2:],
        x_val=x[2:],
        theta_all=theta,
        x_all=x,
        bins_train=bins[:2],
        bins_val=bins[2:],
        bins_all=bins,
        k_cat=2,
    )

    assert result["delta_l"].shape == (4, 4)
    assert np.array_equal(seen["bins_train"], bins[:2])
    assert np.array_equal(seen["bins_val"], bins[2:])
    assert np.array_equal(seen["bins_all"], bins)
    assert seen["k_cat"] == 2
    assert seen["method_name"] == "latent_belief_ctsm_v_binary_inner_post"


def test_lda_ctsm_v_dispatch_transforms_x_before_ctsm(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray] = {}

    def fake_ctsm(*args, **kwargs):
        seen["x_train"] = np.asarray(kwargs["x_train"], dtype=np.float64)
        seen["x_val"] = np.asarray(kwargs["x_val"], dtype=np.float64)
        seen["x_all"] = np.asarray(kwargs["x_all"], dtype=np.float64)
        n = int(seen["x_all"].shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_ctsm)
    args = SimpleNamespace(lda_ctsm_shrinkage=1e-2, lda_ctsm_eps=1e-6, lda_ctsm_max_dim=1)
    theta = np.eye(2, dtype=np.float64)[[0, 0, 1, 1]]
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.1],
        ],
        dtype=np.float64,
    )
    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="lda_ctsm_v",
        theta_train=theta,
        x_train=x,
        theta_val=theta[:2],
        x_val=x[:2],
        theta_all=theta,
        x_all=x,
        bins_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bins_val=np.array([0, 0], dtype=np.int64),
        bins_all=np.array([0, 0, 1, 1], dtype=np.int64),
        k_cat=2,
    )

    assert seen["x_train"].shape == (4, 1)
    assert seen["x_val"].shape == (2, 1)
    assert seen["x_all"].shape == (4, 1)
    assert "lda_ctsm_weights" in result
    assert str(result["ctsm_theta_encoding"][0]) == "one_hot_lda_weighted_x"


def test_pls_ctsm_v_dispatch_transforms_x_before_ctsm(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray] = {}

    def fake_ctsm(*args, **kwargs):
        seen["x_train"] = np.asarray(kwargs["x_train"], dtype=np.float64)
        seen["x_val"] = np.asarray(kwargs["x_val"], dtype=np.float64)
        seen["x_all"] = np.asarray(kwargs["x_all"], dtype=np.float64)
        n = int(seen["x_all"].shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_ctsm)
    args = SimpleNamespace(
        pls_ctsm_eps=1e-6,
        pls_ctsm_max_dim=1,
        pls_ctsm_scale_x=False,
        pls_ctsm_max_iter=100,
        pls_ctsm_tol=1e-6,
    )
    theta = np.eye(2, dtype=np.float64)[[0, 0, 1, 1, 1]]
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.1],
            [2.4, -0.2],
        ],
        dtype=np.float64,
    )
    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="pls_ctsm_v",
        theta_train=theta,
        x_train=x,
        theta_val=theta[:2],
        x_val=x[:2],
        theta_all=theta,
        x_all=x,
        bins_train=np.array([0, 0, 1, 1, 1], dtype=np.int64),
        bins_val=np.array([0, 0], dtype=np.int64),
        bins_all=np.array([0, 0, 1, 1, 1], dtype=np.int64),
        k_cat=2,
    )

    assert seen["x_train"].shape == (5, 1)
    assert seen["x_val"].shape == (2, 1)
    assert seen["x_all"].shape == (5, 1)
    assert "pls_ctsm_weights" in result
    assert str(result["ctsm_theta_encoding"][0]) == "one_hot_pls_weighted_x"


def test_pca_ctsm_v_dispatch_transforms_x_before_ctsm(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    seen: dict[str, np.ndarray] = {}

    def fake_ctsm(*args, **kwargs):
        seen["x_train"] = np.asarray(kwargs["x_train"], dtype=np.float64)
        seen["x_val"] = np.asarray(kwargs["x_val"], dtype=np.float64)
        seen["x_all"] = np.asarray(kwargs["x_all"], dtype=np.float64)
        n = int(seen["x_all"].shape[0])
        return {"delta_l": np.zeros((n, n), dtype=np.float64), "train_out": None}

    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_ctsm)
    args = SimpleNamespace(
        pca_ctsm_eps=1e-6,
        pca_ctsm_max_dim=1,
        pca_ctsm_scale_x=False,
    )
    theta = np.eye(2, dtype=np.float64)[[0, 0, 1, 1]]
    x = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.1],
            [2.0, 0.1],
        ],
        dtype=np.float64,
    )
    result = cat._train_one_method(
        args,
        dev=torch.device("cpu"),
        method_name="pca_ctsm_v",
        theta_train=theta,
        x_train=x,
        theta_val=theta[:2],
        x_val=x[:2],
        theta_all=theta,
        x_all=x,
        bins_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bins_val=np.array([0, 0], dtype=np.int64),
        bins_all=np.array([0, 0, 1, 1], dtype=np.int64),
        k_cat=2,
    )

    assert seen["x_train"].shape == (4, 1)
    assert seen["x_val"].shape == (2, 1)
    assert seen["x_all"].shape == (4, 1)
    assert "pca_ctsm_weights" in result
    assert str(result["ctsm_theta_encoding"][0]) == "one_hot_pca_weighted_x"


def test_theta_flow_categorical_helper_shape_symmetry_zero_diag(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    def fake_train(*args, **kwargs):
        return {"train_losses": [1.0, 0.5], "val_losses": [1.2, 0.7], "val_monitor_losses": [1.2, 0.8]}

    class FakeEstimator:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, theta, x, *, restore_original_order=False):
            labels = np.asarray(theta, dtype=np.int64).reshape(-1)
            h = np.zeros((labels.size, labels.size), dtype=np.float64)
            h[labels[:, None] != labels[None, :]] = 0.25
            return SimpleNamespace(h_directed=h)

    monkeypatch.setattr(cat, "_build_theta_flow_post_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cat, "_build_theta_flow_prior_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cat, "train_conditional_theta_flow_model", fake_train)
    monkeypatch.setattr(cat, "train_prior_theta_flow_model", fake_train)
    monkeypatch.setattr(cat, "HMatrixEstimator", FakeEstimator)
    args = SimpleNamespace(
        clf_min_class_count=2,
        flow_epochs=2,
        flow_batch_size=4,
        flow_lr=1e-3,
        flow_early_patience=10,
        flow_early_min_delta=0.0,
        flow_early_ema_alpha=0.5,
        flow_restore_best=True,
        flow_scheduler="cosine",
        flow_fm_t_eps=0.05,
        flow_likelihood_finetune_epochs=0,
        theta_flow_posterior_only_likelihood=False,
        prior_epochs=2,
        prior_batch_size=4,
        prior_lr=1e-3,
        prior_early_patience=10,
        prior_early_min_delta=0.0,
        prior_early_ema_alpha=0.5,
        prior_restore_best=True,
        h_batch_size=64,
        flow_ode_steps=8,
        flow_likelihood_exact_divergence=False,
        log_every=100,
    )
    x_train = np.arange(18, dtype=np.float64).reshape(9, 2)
    bins_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    x_val = np.arange(12, dtype=np.float64).reshape(6, 2)
    bins_val = np.array([0, 0, 1, 1, 2, 2])
    result = theta_flow_categorical_hellinger_sqrt(
        args,
        dev=SimpleNamespace(),
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        bins_val=bins_val,
        x_all=x_train,
        bins_all=bins_train,
        k_cat=3,
    )
    h = result["h_sqrt"]
    assert h.shape == (3, 3)
    assert np.allclose(h, h.T, equal_nan=True)
    assert np.allclose(np.diag(h), 0.0)
    assert np.allclose(h[np.triu_indices(3, 1)], 0.5)
    assert int(result["theta_flow_cate_num_valid_pairs"]) == 3


def test_theta_flow_categorical_degeneracy_skips_low_train_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    monkeypatch.setattr(cat, "_build_theta_flow_post_model", lambda *args, **kwargs: pytest.fail("should skip"))
    args = SimpleNamespace(clf_min_class_count=3)
    result = theta_flow_categorical_hellinger_sqrt(
        args,
        dev=SimpleNamespace(),
        x_train=np.zeros((3, 2)),
        bins_train=np.array([0, 0, 1]),
        x_val=np.zeros((2, 2)),
        bins_val=np.array([0, 1]),
        x_all=np.zeros((3, 2)),
        bins_all=np.array([0, 0, 1]),
        k_cat=2,
    )
    h = result["h_sqrt"]
    assert np.isnan(h[0, 1]) and np.isnan(h[1, 0])
    assert int(result["theta_flow_cate_num_valid_pairs"]) == 0
    assert int(result["theta_flow_cate_num_skipped_pairs"]) == 1


def test_cli_defaults_from_parser() -> None:
    p = build_parser()
    ns = p.parse_args([])
    assert ns.dataset_family == "random_mog_categorical"
    assert int(ns.num_categories) == 2
    assert ns.n_list == "80,200,400,600"
    assert int(ns.n_ref) == 10000
    assert str(ns.device) == "cuda:1"


def test_cli_accepts_multi_rings_dataset_family() -> None:
    p = build_parser()
    ns = p.parse_args(["--dataset-family", "multi_rings_radial", "--num-categories", "4"])
    assert ns.dataset_family == "multi_rings_radial"
    assert int(ns.num_categories) == 4


def test_hellinger_gt_category_matrix_multi_rings() -> None:
    ds = ToyCategoricalMultiRingsDataset(num_categories=3, seed=0)
    h2 = hellinger_gt_sq_category_matrix(ds)
    assert h2.shape == (3, 3)
    assert np.allclose(h2, h2.T)
    assert np.allclose(np.diag(h2), 0.0)
    assert 0.0 < h2[0, 1] < h2[0, 2] < 1.0


def test_main_rejects_num_categories_lt2() -> None:
    with pytest.raises(ValueError, match="num-categories"):
        main(["--num-categories", "1", "--device", "cpu"])


def test_main_rejects_mismatched_dataset_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    theta = np.eye(2, dtype=np.float64)[np.array([0, 1, 0, 1], dtype=np.int64)]
    x = np.zeros((4, 2), dtype=np.float64)
    bundle = SharedDatasetBundle(
        meta={"dataset_family": "multi_rings_radial", "theta_type": "categorical", "num_categories": 2},
        theta_all=theta,
        x_all=x,
        train_idx=np.array([0, 1], dtype=np.int64),
        validation_idx=np.array([2, 3], dtype=np.int64),
        theta_train=theta[:2],
        x_train=x[:2],
        theta_validation=theta[2:],
        x_validation=x[2:],
    )
    fake_npz = tmp_path / "multi_rings_radial.npz"
    fake_npz.write_bytes(b"placeholder")
    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)

    with pytest.raises(ValueError, match="Expected random_mog_categorical NPZ"):
        cat.main(
            [
                "--dataset-npz",
                str(fake_npz),
                "--num-categories",
                "2",
                "--n-ref",
                "4",
                "--n-list",
                "2",
                "--methods",
                "bin_gaussian",
                "--device",
                "cpu",
            ]
        )


def test_visualization_only_cached_shapes(tmp_path: Path) -> None:
    k = 3
    methods = ["binary_classifier", "theta_flow_cate"]
    ns = [20, 40]
    h_gt = np.full((k, k), 0.1, dtype=np.float64)
    np.fill_diagonal(h_gt, 0.0)
    dec_ref = np.full((k, k), 0.5, dtype=np.float64)
    np.fill_diagonal(dec_ref, np.nan)
    h_sw = np.random.RandomState(1).rand(len(methods), len(ns), k, k).astype(np.float64)
    dec_sw = np.random.RandomState(2).rand(len(ns), k, k).astype(np.float64)
    corr_h = np.random.RandomState(3).rand(len(methods), len(ns))
    nmse_h = np.random.RandomState(4).rand(len(methods), len(ns))
    corr_d = np.random.RandomState(5).rand(len(ns))
    nmse_d = np.random.RandomState(6).rand(len(ns))
    fake_ds = tmp_path / "fake.npz"
    fake_ds.write_bytes(b"not a real npz")
    npz_path = tmp_path / "h_decoding_categorical_twofig_results.npz"
    np.savez_compressed(
        str(npz_path),
        n=np.asarray(ns, dtype=np.int64),
        n_ref=np.int64(40),
        num_categories=np.int64(k),
        method_names=np.asarray(methods, dtype=object),
        theta_bin_centers=np.arange(k, dtype=np.float64).reshape(-1, 1),
        h_gt_sqrt=h_gt,
        decode_ref=dec_ref,
        decode_sweep=dec_sw,
        h_sqrt_sweep=h_sw,
        corr_h_vs_gt=corr_h,
        nmse_h_vs_gt=nmse_h,
        corr_decode_vs_ref=corr_d,
        nmse_decode_vs_ref=nmse_d,
        native_dataset_npz=np.asarray([str(fake_ds.resolve())], dtype=object),
        eval_split=np.asarray(["all"], dtype=object),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    npz_path.rename(out_dir / "h_decoding_categorical_twofig_results.npz")
    for method in methods:
        loss_root = out_dir / "training_losses" / method
        loss_root.mkdir(parents=True)
        for n in ns:
            _save_method_training_loss_npz(
                loss_root / f"n_{int(n):06d}.npz",
                method_name=method,
                result={},
            )
    argv = [
        "--visualization-only",
        "--output-dir",
        str(out_dir),
        "--dataset-npz",
        str(fake_ds),
        "--num-categories",
        str(k),
        "--n-ref",
        "40",
        "--n-list",
        "20,40",
        "--methods",
        "binary_classifier,theta-flow-cate",
        "--device",
        "cpu",
    ]
    main(argv)
    assert (out_dir / "h_decoding_categorical_twofig_sweep.svg").is_file()
    assert (out_dir / "h_decoding_categorical_twofig_corr_nmse.svg").is_file()
    assert (out_dir / "h_decoding_categorical_twofig_training_losses_panel.svg").is_file()
    all_columns = out_dir / "h_decoding_categorical_twofig_all_columns.png"
    assert all_columns.is_file()
    summary = (out_dir / "h_decoding_categorical_twofig_summary.txt").read_text(encoding="utf-8")
    assert f"h_decoding_categorical_twofig_all_columns.png: {all_columns.resolve()}" in summary


def test_visualization_only_rerenders_cached_scatter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    k = 2
    ns = [20]
    methods = ["bin_gaussian"]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fake_ds = tmp_path / "fake.npz"
    fake_ds.write_bytes(b"not a real npz")
    h_gt = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    dec_ref = np.array([[np.nan, 0.7], [0.7, np.nan]], dtype=np.float64)
    scatter_true = np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float64)
    scatter_est = np.array([[[-1.8, -1.2, 1.1, 100.0]]], dtype=np.float64)
    metric_names = np.array(["llr_pairwise_raw_rmse", "llr_pairwise_raw_pearson_r"], dtype=object)
    metric_values = np.array([[1.0, 0.5]], dtype=np.float64)
    np.savez_compressed(
        out_dir / "h_decoding_categorical_twofig_results.npz",
        n=np.asarray(ns, dtype=np.int64),
        n_ref=np.int64(40),
        num_categories=np.int64(k),
        method_names=np.asarray(methods, dtype=object),
        theta_bin_centers=np.arange(k, dtype=np.float64).reshape(-1, 1),
        h_gt_sqrt=h_gt,
        decode_ref=dec_ref,
        decode_sweep=np.asarray([dec_ref], dtype=np.float64),
        h_sqrt_sweep=np.asarray([[h_gt]], dtype=np.float64),
        corr_h_vs_gt=np.asarray([[1.0]], dtype=np.float64),
        nmse_h_vs_gt=np.asarray([[0.0]], dtype=np.float64),
        corr_decode_vs_ref=np.asarray([1.0], dtype=np.float64),
        nmse_decode_vs_ref=np.asarray([0.0], dtype=np.float64),
        native_dataset_npz=np.asarray([str(fake_ds.resolve())], dtype=object),
        eval_split=np.asarray(["all"], dtype=object),
        scatter_llr_pair_labels=np.asarray([[0, 1]], dtype=np.int64),
        scatter_true_llr_pairwise=scatter_true,
        scatter_llr_pairwise_est=scatter_est,
        scatter_llr_pairwise_metric_method_names=np.asarray(methods, dtype=object),
        scatter_llr_pairwise_metric_names=metric_names,
        scatter_llr_pairwise_metric_values=metric_values,
    )
    loss_root = out_dir / "training_losses" / methods[0]
    loss_root.mkdir(parents=True)
    _save_method_training_loss_npz(loss_root / "n_000020.npz", method_name=methods[0], result={})
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(out_dir / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(out_dir / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(out_dir / "loss.svg"))
    monkeypatch.setattr(cat, "_write_all_columns_png", lambda *args, **kwargs: str(_write_tiny_png(out_dir / "all.png")))

    cat.main(
        [
            "--visualization-only",
            "--output-dir",
            str(out_dir),
            "--dataset-npz",
            str(fake_ds),
            "--num-categories",
            str(k),
            "--n-ref",
            "40",
            "--n-list",
            "20",
            "--methods",
            ",".join(methods),
            "--device",
            "cpu",
        ]
    )

    svg = out_dir / "llr_est_vs_true_all.svg"
    png = out_dir / "llr_est_vs_true_all.png"
    assert svg.is_file()
    assert png.is_file()
    text = svg.read_text(encoding="utf-8")
    assert "y zoom (IQR pooled methods)" in text
    assert "Full range mean error" in text
    assert "Mean error" in text
    assert "true raw LLR in [-8, 8]" in text
    assert text.count("true raw LLR in [-8, 8]") >= 3
    summary = (out_dir / "h_decoding_categorical_twofig_summary.txt").read_text(encoding="utf-8")
    assert f"llr_est_vs_true_all.svg: {svg.resolve()}" in summary
    assert f"llr_est_vs_true_all.png: {png.resolve()}" in summary


def test_build_parser_eval_split() -> None:
    p = build_parser()
    assert p.parse_args([]).eval_split == "all"
    assert p.parse_args(["--eval-split", "validation", "--device", "cpu"]).eval_split == "validation"
    with pytest.raises(SystemExit):
        p.parse_args(["--hellinger-eval-split", "validation", "--device", "cpu"])
    with pytest.raises(SystemExit):
        p.parse_args(["--binary-classifier-h-on-full-pool"])


def test_visualization_only_cache_eval_split_validation() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="validation",
    )
    cached = {
        "n": np.asarray([20, 40], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
        "eval_split": np.asarray(["validation"], dtype=object),
    }
    _validate_cached_cli(args, cached, ["binary_classifier"], [20, 40])


def test_visualization_only_cache_legacy_all_only() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="all",
    )
    cached = {
        "n": np.asarray([20], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
    }
    _validate_cached_cli(args, cached, ["binary_classifier"], [20])
    cached["hellinger_eval_split"] = np.asarray(["all"], dtype=object)
    _validate_cached_cli(args, cached, ["binary_classifier"], [20])


def test_visualization_only_cache_rejects_legacy_validation() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="all",
    )
    cached = {
        "n": np.asarray([20], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
        "hellinger_eval_split": np.asarray(["validation"], dtype=object),
    }
    with pytest.raises(ValueError, match="legacy hellinger_eval_split"):
        _validate_cached_cli(args, cached, ["binary_classifier"], [20])


def test_visualization_only_cache_rejects_requested_validation_against_legacy() -> None:
    args = SimpleNamespace(
        n_ref=40,
        num_categories=3,
        dataset_npz=Path("/tmp/fake_dataset.npz"),
        eval_split="validation",
    )
    cached = {
        "n": np.asarray([20], dtype=np.int64),
        "n_ref": np.int64(40),
        "num_categories": np.int64(3),
        "method_names": np.asarray(["binary_classifier"], dtype=object),
    }
    with pytest.raises(ValueError, match="no eval_split metadata"):
        _validate_cached_cli(args, cached, ["binary_classifier"], [20])


def test_validation_only_work_sweep_subset_aligns_with_bundle_validation() -> None:
    repo = Path(__file__).resolve().parent.parent
    npz = repo / "data" / "random_mog_categorical_xdim2_default" / "random_mog_categorical.npz"
    if not npz.is_file():
        pytest.skip("benchmark-cate native NPZ not in checkout")
    bundle = load_shared_dataset_npz(npz)
    meta = dict(bundle.meta)
    k_cat = int(meta.get("num_categories", 5))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(bundle.theta_all, k_cat)
    perm = np.arange(int(bundle.theta_all.shape[0]), dtype=np.int64)
    subset_w = conv._subset_bundle(bundle, perm, 200, meta, bin_idx_all=bin_idx_all)
    vs = _validation_only_work_sweep_subset(subset_w)
    assert int(vs.bundle.x_all.shape[0]) == int(subset_w.bundle.x_validation.shape[0])
    assert np.array_equal(vs.bin_all, subset_w.bin_validation)


def test_binned_gaussian_delta_l_matches_shared_diag_log_ratio() -> None:
    x = np.array([[-1.0], [1.0], [4.0], [6.0]], dtype=np.float64)
    bins = np.array([0, 0, 1, 1], dtype=np.int64)
    subset = SimpleNamespace(bundle=SimpleNamespace(x_all=x), bin_all=bins)

    got = _binned_gaussian_delta_l(subset, 2, variance_floor=1e-6)

    means = np.array([[0.0], [5.0]], dtype=np.float64)
    var = np.array([1.0], dtype=np.float64)
    log_norm = np.log(2.0 * np.pi * var[0])
    logp_by_bin = -0.5 * (((x - means.reshape(1, 2)) ** 2) / var[0] + log_norm)
    c_matrix = logp_by_bin[:, bins]
    expected = HMatrixEstimator.compute_delta_l(c_matrix)
    assert got.shape == (4, 4)
    assert np.allclose(got, expected)
    assert np.allclose(np.diag(got), 0.0)


def test_eval_split_validation_controls_classifier_and_decoding_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    theta = np.eye(2, dtype=np.float64)[np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)]
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    meta = {"dataset_family": "random_mog_categorical", "theta_type": "categorical", "num_categories": 2, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=x[:3],
        theta_validation=theta[3:],
        x_validation=x[3:],
    )
    seen: dict[str, list] = {"method": [], "llr": [], "decode": []}

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(
        cat,
        "hellinger_gt_sq_category_matrix",
        lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]),
    )
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    def fake_pairwise(**kwargs):
        x_eval = kwargs.get("decode_x_all")
        bins_eval = kwargs.get("decode_bin_all")
        if x_eval is None:
            x_eval = kwargs["subset"].bundle.x_all
        if bins_eval is None:
            bins_eval = kwargs["subset"].bin_all
        seen["decode"].append(
            (
                len(kwargs["subset"].bundle.x_train),
                len(np.asarray(x_eval)),
                tuple(np.asarray(bins_eval, dtype=np.int64).tolist()),
            )
        )
        return np.array([[np.nan, 0.5], [0.5, np.nan]])

    monkeypatch.setattr(cat.conv, "_pairwise_clf_from_bundle", fake_pairwise)
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    def fake_train_one_method(*args, **kwargs):
        seen["method"].append((len(kwargs["x_all"]), tuple(kwargs["bins_all"].tolist())))
        return {"delta_l": np.zeros((len(kwargs["x_all"]), len(kwargs["x_all"])), dtype=np.float64)}

    monkeypatch.setattr(cat, "_train_one_method", fake_train_one_method)

    dbg = cat._debug_categorical_module()
    orig_llr = dbg._llr_comparison_metrics

    def fake_llr(delta, true_delta):
        seen["llr"].append((int(np.asarray(delta).shape[0]), tuple(np.asarray(true_delta).shape)))
        return orig_llr(delta, true_delta)

    monkeypatch.setattr(dbg, "_llr_comparison_metrics", fake_llr)

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "binary_classifier",
            "--eval-split",
            "validation",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    assert len(seen["method"]) == 1
    assert len(seen["decode"]) == 2
    assert seen["decode"][-1][0] == 3
    assert seen["decode"][-1][1] == 3
    assert seen["method"][0][0] == 3
    assert seen["decode"][-1][2] == seen["method"][0][1]
    assert any(shape == (3, 3) for _, shape in seen["llr"])

    with np.load(tmp_path / "out" / "h_decoding_categorical_twofig_results.npz", allow_pickle=True) as z:
        assert str(z["eval_split"][0]) == "validation"
        assert "hellinger_eval_split" not in z.files
        assert "classifier_llr_h_eval_pool" not in z.files
        assert "binary_classifier_h_on_full_pool" not in z.files
        assert "decode_hellinger_ref_sqrt" not in z.files
        assert "decode_hellinger_sweep_sqrt" not in z.files
        assert "corr_decode_hellinger_vs_gt" not in z.files
        assert "nmse_decode_hellinger_vs_gt" not in z.files


def test_eval_split_all_controls_classifier_and_decoding_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    theta = np.eye(2, dtype=np.float64)[np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)]
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    meta = {"dataset_family": "random_mog_categorical", "theta_type": "categorical", "num_categories": 2, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=x[:3],
        theta_validation=theta[3:],
        x_validation=x[3:],
    )
    seen: dict[str, list] = {"method": [], "decode": []}

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))
    def fake_pairwise(**kwargs):
        x_eval = kwargs.get("decode_x_all")
        bins_eval = kwargs.get("decode_bin_all")
        if x_eval is None:
            x_eval = kwargs["subset"].bundle.x_all
        if bins_eval is None:
            bins_eval = kwargs["subset"].bin_all
        seen["decode"].append(
            (
                len(kwargs["subset"].bundle.x_train),
                len(np.asarray(x_eval)),
                tuple(np.asarray(bins_eval, dtype=np.int64).tolist()),
            )
        )
        return np.array([[np.nan, 0.5], [0.5, np.nan]])

    monkeypatch.setattr(cat.conv, "_pairwise_clf_from_bundle", fake_pairwise)
    monkeypatch.setattr(
        cat,
        "_train_one_method",
        lambda *args, **kwargs: seen["method"].append((len(kwargs["x_all"]), tuple(kwargs["bins_all"].tolist())))
        or {"delta_l": np.zeros((len(kwargs["x_all"]), len(kwargs["x_all"])), dtype=np.float64)},
    )

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "binary_classifier",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    assert seen["decode"][-1][0] == 3
    assert seen["decode"][-1][1] == 6
    assert seen["method"][0][0] == 6


def test_bin_gaussian_writes_finite_llr_scatter_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    theta = np.eye(2, dtype=np.float64)[labels]
    x = np.array(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        dtype=np.float64,
    )
    meta = {
        "dataset_family": "random_mog_categorical",
        "theta_type": "categorical",
        "num_categories": 2,
        "train_frac": 0.5,
    }
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(4, dtype=np.int64),
        validation_idx=np.arange(4, 6, dtype=np.int64),
        theta_train=theta[:4],
        x_train=x[:4],
        theta_validation=theta[4:],
        x_validation=x[4:],
    )

    def true_loglik(x_all: np.ndarray, theta_all: np.ndarray, meta: dict) -> np.ndarray:
        x_eval = np.asarray(x_all, dtype=np.float64)[:, 0]
        bins = np.argmax(np.asarray(theta_all, dtype=np.float64), axis=1)
        return x_eval.reshape(-1, 1) * (2.0 * bins.reshape(1, -1) - 1.0)

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", true_loglik)
    monkeypatch.setattr(
        cat.conv,
        "_pairwise_clf_from_bundle",
        lambda **kwargs: np.array([[np.nan, 0.5], [0.5, np.nan]]),
    )
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "bin_gaussian",
            "--device",
            "cpu",
        ]
    )

    with np.load(tmp_path / "out" / "h_decoding_categorical_twofig_results.npz", allow_pickle=True) as z:
        methods = [str(x) for x in z["method_names"]]
        assert methods == ["bin_gaussian"]
        assert np.isfinite(np.asarray(z["llr_pearson_offdiag"], dtype=np.float64)[0, 0])
        raw_methods = [str(x) for x in z["scatter_llr_pairwise_metric_method_names"]]
        assert raw_methods == ["bin_gaussian"]
        raw_est = np.asarray(z["scatter_llr_pairwise_est"], dtype=np.float64)
        assert raw_est.shape[:2] == (1, 1)
        assert np.isfinite(raw_est).any()
        raw_metric_names = [str(x) for x in z["scatter_llr_pairwise_metric_names"]]
        raw_metric_values = np.asarray(z["scatter_llr_pairwise_metric_values"], dtype=np.float64)
        r_idx = raw_metric_names.index("llr_pairwise_raw_pearson_r")
        assert np.isfinite(raw_metric_values[0, r_idx])


def test_pr_project_classifier_rows_use_work_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    theta = np.eye(2, dtype=np.float64)[labels]
    native_x = 1000.0 + np.arange(12, dtype=np.float64).reshape(6, 2)
    work_x = np.arange(24, dtype=np.float64).reshape(6, 4)
    native_meta = {
        "dataset_family": "random_mog_categorical",
        "theta_type": "categorical",
        "num_categories": 2,
        "train_frac": 0.5,
    }
    work_meta = dict(native_meta)
    work_meta.update({"pr_autoencoder_embedded": True, "pr_autoencoder_z_dim": 2})
    native_bundle = SharedDatasetBundle(
        meta=native_meta,
        theta_all=theta,
        x_all=native_x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=native_x[:3],
        theta_validation=theta[3:],
        x_validation=native_x[3:],
    )
    work_bundle = SharedDatasetBundle(
        meta=work_meta,
        theta_all=theta,
        x_all=work_x,
        train_idx=np.arange(3, dtype=np.int64),
        validation_idx=np.arange(3, 6, dtype=np.int64),
        theta_train=theta[:3],
        x_train=work_x[:3],
        theta_validation=theta[3:],
        x_validation=work_x[3:],
    )
    calls = {"load": 0, "decode": [], "method": []}

    def fake_load(path):
        calls["load"] += 1
        return native_bundle if calls["load"] == 1 else work_bundle

    def record_shape(arr: np.ndarray) -> tuple[int, int, float]:
        a = np.asarray(arr, dtype=np.float64)
        return (int(a.shape[0]), int(a.shape[1]), float(np.max(a)))

    def fake_pairwise(**kwargs):
        calls["decode"].append((record_shape(kwargs["decode_x_train"]), record_shape(kwargs["decode_x_all"])))
        return np.array([[np.nan, 0.5], [0.5, np.nan]])

    def fake_train_one_method(*args, **kwargs):
        calls["method"].append((record_shape(kwargs["x_train"]), record_shape(kwargs["x_all"])))
        return {"delta_l": np.zeros((len(kwargs["x_all"]), len(kwargs["x_all"])), dtype=np.float64)}

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "_ensure_pr_projected_npz", lambda *args, **kwargs: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", fake_load)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    monkeypatch.setattr(cat.conv, "_pairwise_clf_from_bundle", fake_pairwise)
    monkeypatch.setattr(cat, "_train_one_method", fake_train_one_method)
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "native.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--pr-project",
            "--pr-output-npz",
            str(tmp_path / "work.npz"),
            "--decode-source-npz",
            str(tmp_path / "missing-source.npz"),
            "--num-categories",
            "2",
            "--n-ref",
            "6",
            "--n-list",
            "6",
            "--methods",
            "binary_classifier",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    assert len(calls["decode"]) == 2
    assert len(calls["method"]) == 1
    for group in ("decode", "method"):
        for train_shape, eval_shape in calls[group]:
            assert train_shape[1] == 4
            assert eval_shape[1] == 4
            assert train_shape[2] < 1000.0
            assert eval_shape[2] < 1000.0


def test_binary_classifier_method_still_writes_method_row_on_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    rng = np.random.default_rng(0)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 20)
    theta = np.eye(2, dtype=np.float64)[labels]
    x = np.empty((labels.size, 2), dtype=np.float64)
    x[labels == 0] = rng.normal(loc=-1.0, scale=0.2, size=(int(np.sum(labels == 0)), 2))
    x[labels == 1] = rng.normal(loc=1.0, scale=0.2, size=(int(np.sum(labels == 1)), 2))
    meta = {"dataset_family": "random_mog_categorical", "theta_type": "categorical", "num_categories": 2, "train_frac": 0.5}
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta,
        x_all=x,
        train_idx=np.arange(20, dtype=np.int64),
        validation_idx=np.arange(20, 40, dtype=np.int64),
        theta_train=theta[:20],
        x_train=x[:20],
        theta_validation=theta[20:],
        x_validation=x[20:],
    )

    monkeypatch.setattr(cat, "_ensure_dataset", lambda args: None)
    monkeypatch.setattr(cat, "load_shared_dataset_npz", lambda path: bundle)
    monkeypatch.setattr(cat, "hellinger_gt_sq_category_matrix", lambda gen_ds: np.array([[0.0, 0.25], [0.25, 0.0]]))
    monkeypatch.setattr(cat, "build_dataset_from_meta", lambda meta: object())
    monkeypatch.setattr(cat, "compute_true_conditional_loglik_matrix", lambda x_all, theta_all, meta: np.zeros((len(x_all), len(x_all))))
    monkeypatch.setattr(cat, "_render_method_sweep_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "sweep.svg"))
    monkeypatch.setattr(cat, "_render_corr_nmse_two_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "corr.svg"))
    monkeypatch.setattr(cat, "_render_row_n_training_losses_panel", lambda **kwargs: _write_tiny_svg(tmp_path / "loss.svg"))

    cat.main(
        [
            "--dataset-npz",
            str(tmp_path / "fake.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-categories",
            "2",
            "--n-ref",
            "40",
            "--n-list",
            "40",
            "--methods",
            "binary_classifier",
            "--clf-min-class-count",
            "2",
            "--no-scatter-diagnostics",
            "--device",
            "cpu",
        ]
    )

    with np.load(tmp_path / "out" / "h_decoding_categorical_twofig_results.npz", allow_pickle=True) as z:
        methods = [str(x) for x in z["method_names"]]
        i = methods.index("binary_classifier")
        h = np.asarray(z["h_sqrt_sweep"][i, 0], dtype=np.float64)
        assert h.shape == (2, 2)
        assert np.allclose(np.diag(h), 0.0)
        assert np.isfinite(h[np.triu_indices(2, 1)]).all()
        assert "h_sq_lr_affinity_sweep" not in z.files
        assert "corr_h_lr_affinity_vs_gt" not in z.files
        assert "nmse_h_lr_affinity_vs_gt" not in z.files
        assert "decode_hellinger_ref_sqrt" not in z.files
        assert "decode_hellinger_sweep_sqrt" not in z.files
        assert "corr_decode_hellinger_vs_gt" not in z.files
        assert "nmse_decode_hellinger_vs_gt" not in z.files


def test_pairwise_clf_from_bundle_decode_bin_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    subset = SimpleNamespace(
        bundle=SimpleNamespace(
            x_train=np.zeros((4, 2), dtype=np.float64),
            x_all=np.zeros((6, 2), dtype=np.float64),
        ),
        bin_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bin_all=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    seen = {}

    def fake_accuracy(x_tr, bin_tr, x_ev, bin_ev, n_bins, **kwargs):
        seen["x_ev_rows"] = int(np.asarray(x_ev).shape[0])
        seen["bin_ev"] = tuple(np.asarray(bin_ev, dtype=np.int64).tolist())
        return np.array([[np.nan, 0.75], [0.75, np.nan]]), None, None, None

    monkeypatch.setattr(conv.vhb, "pairwise_bin_logistic_accuracy_train_val", fake_accuracy)
    got = conv._pairwise_clf_from_bundle(
        args=SimpleNamespace(),
        meta={},
        subset=subset,
        output_dir=str(tmp_path),
        n_bins=2,
        clf_min_class_count=2,
        clf_random_state=0,
        decode_x_all=np.ones((2, 2), dtype=np.float64),
        decode_bin_all=np.array([0, 1], dtype=np.int64),
    )
    assert got[0, 1] == pytest.approx(0.75)
    assert seen == {"x_ev_rows": 2, "bin_ev": (0, 1)}


def test_pairwise_clf_from_bundle_decode_bin_override_row_mismatch(tmp_path: Path) -> None:
    subset = SimpleNamespace(
        bundle=SimpleNamespace(
            x_train=np.zeros((4, 2), dtype=np.float64),
            x_all=np.zeros((6, 2), dtype=np.float64),
        ),
        bin_train=np.array([0, 0, 1, 1], dtype=np.int64),
        bin_all=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="decode_x_all rows 3 != decode_bin_all 2"):
        conv._pairwise_clf_from_bundle(
            args=SimpleNamespace(),
            meta={},
            subset=subset,
            output_dir=str(tmp_path),
            n_bins=2,
            clf_min_class_count=2,
            clf_random_state=0,
            decode_x_all=np.ones((3, 2), dtype=np.float64),
            decode_bin_all=np.array([0, 1], dtype=np.int64),
        )


def test_resolve_lxf_low_rank_dim_clamps_to_x_dim(capsys: pytest.CaptureFixture[str]) -> None:
    from fisher.linear_x_flow import resolve_lxf_low_rank_dim

    assert resolve_lxf_low_rank_dim(3, 2, log_prefix="[test] ") == 2
    captured = capsys.readouterr().out
    assert "warning" in captured.lower()
    assert "using r=2" in captured
    assert resolve_lxf_low_rank_dim(2, 5) == 2


def test_fit_sir_projection_clamps_to_available_rank(capsys: pytest.CaptureFixture[str]) -> None:
    from fisher.h_decoding_convergence_methods import _fit_sir_projection

    theta = np.eye(2, dtype=np.float64)[np.array([0, 1, 0, 1, 0, 1])]
    x = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.1, 0.0, 0.0],
            [1.1, 0.1, 0.0, 0.1, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.1],
            [1.2, 0.2, 0.1, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    z_train, z_val, z_all, meta = _fit_sir_projection(
        x_train=x,
        theta_train=theta,
        x_val=x,
        x_all=x,
        sir_dim=3,
        num_bins=10,
        ridge=1e-6,
    )

    captured = capsys.readouterr().out
    assert "available SIR rank 1" in captured
    assert "using r=1" in captured
    assert int(meta["sir_dim"]) == 1
    assert z_train.shape == (6, 1)
    assert z_val.shape == (6, 1)
    assert z_all.shape == (6, 1)


def test_fit_sir_projection_caps_requested_rank_with_sir_max_rank(capsys: pytest.CaptureFixture[str]) -> None:
    from fisher.h_decoding_convergence_methods import _fit_sir_projection

    theta = np.linspace(0.0, 1.0, 8, dtype=np.float64).reshape(-1, 1)
    x = np.column_stack(
        [
            theta[:, 0],
            theta[:, 0] ** 2,
        ]
    ).astype(np.float64)

    z_train, z_val, z_all, meta = _fit_sir_projection(
        x_train=x,
        theta_train=theta,
        x_val=x,
        x_all=x,
        sir_dim=3,
        num_bins=4,
        ridge=1e-6,
    )

    captured = capsys.readouterr().out
    assert "available SIR rank 2" in captured
    assert "using r=2" in captured
    assert "--lxf-low-rank-dim=3 exceeds x_dim=2" not in captured
    assert int(meta["sir_dim"]) == 2
    assert z_train.shape == (8, 2)
    assert z_val.shape == (8, 2)
    assert z_all.shape == (8, 2)


def test_fit_sir_projection_auto_rank_uses_eigenvalue_ratio(capsys: pytest.CaptureFixture[str]) -> None:
    from fisher.h_decoding_convergence_methods import _fit_sir_projection

    theta = np.linspace(-1.0, 1.0, 12, dtype=np.float64).reshape(-1, 1)
    x = np.column_stack(
        [
            theta[:, 0],
            theta[:, 0] ** 2,
            0.25 * theta[:, 0] ** 3,
        ]
    ).astype(np.float64)

    z_train, z_val, z_all, meta = _fit_sir_projection(
        x_train=x,
        theta_train=theta,
        x_val=x,
        x_all=x,
        sir_dim=None,
        num_bins=4,
        ridge=1e-6,
    )

    captured = capsys.readouterr().out
    eig = np.asarray(meta["sir_eigenvalues_all"], dtype=np.float64)
    clipped = np.clip(eig, 0.0, None)
    total = float(np.sum(clipped))
    k90 = int(np.searchsorted(np.cumsum(clipped) / total, 0.90, side="left")) + 1
    expected_rank = min(k90 + 1, int(eig.shape[0]))
    assert "auto rank selected" in captured
    assert str(np.asarray(meta["sir_rank_mode"], dtype=object).reshape(-1)[0]) == "auto_90_plus1"
    assert int(meta["sir_rank_requested"]) == 0
    assert float(meta["sir_rank_auto_threshold"]) == pytest.approx(0.90)
    assert int(meta["sir_dim"]) == expected_rank
    assert z_train.shape == (12, expected_rank)
    assert z_val.shape == (12, expected_rank)
    assert z_all.shape == (12, expected_rank)


def test_fit_sir_projection_auto_rank_zero_eigenvalues_falls_back_to_one() -> None:
    from fisher.h_decoding_convergence_methods import _fit_sir_projection

    theta = np.linspace(0.0, 1.0, 8, dtype=np.float64).reshape(-1, 1)
    x = np.ones((8, 3), dtype=np.float64)

    z_train, z_val, z_all, meta = _fit_sir_projection(
        x_train=x,
        theta_train=theta,
        x_val=x,
        x_all=x,
        sir_dim=None,
        num_bins=4,
        ridge=1e-6,
    )

    assert int(meta["sir_dim"]) == 1
    assert str(np.asarray(meta["sir_rank_mode"], dtype=object).reshape(-1)[0]) == "auto_90_plus1"
    assert z_train.shape == (8, 1)
    assert z_val.shape == (8, 1)
    assert z_all.shape == (8, 1)


def test_categorical_lxf_low_rank_dim_default_auto_and_manual_override() -> None:
    from fisher.h_decoding_categorical_twofig import build_parser

    parser = build_parser()
    args_auto = parser.parse_args([])
    args_manual = parser.parse_args(["--lxf-low-rank-dim", "3"])

    assert args_auto.lxf_low_rank_dim is None
    assert args_manual.lxf_low_rank_dim == 3
