"""Continuous twofig-style entrypoint for ``study_h_decoding_convergence.py``."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from global_setting import DATA_DIR

from fisher import h_decoding_convergence as conv
from fisher import h_decoding_twofig as twofig
from fisher.h_decoding_categorical_twofig import _write_all_columns_png
from fisher.shared_dataset_io import load_shared_dataset_npz


DEFAULT_METHODS = "theta_flow,x_flow,linear_x_flow_t,bin_gaussian"

_TWOFIG_TO_CONVERGENCE_FILES = {
    "h_decoding_twofig_results.npz": "h_decoding_convergence_results.npz",
    "h_decoding_twofig_sweep.svg": "h_decoding_convergence_sweep.svg",
    "h_decoding_twofig_corr_nmse.svg": "h_decoding_convergence_corr_nmse.svg",
    "h_decoding_twofig_training_losses_panel.svg": "h_decoding_convergence_training_losses_panel.svg",
    "h_decoding_twofig_summary.txt": "h_decoding_convergence_summary.txt",
}


def _remove_arg(parser: argparse.ArgumentParser, *option_strings: str) -> None:
    actions = [
        parser._option_string_actions[o]
        for o in option_strings
        if o in parser._option_string_actions
    ]
    for action in set(actions):
        for opt in list(action.option_strings):
            parser._option_string_actions.pop(opt, None)
        if action in parser._actions:
            parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)


def build_parser() -> argparse.ArgumentParser:
    p = twofig.build_parser()
    p.description = (
        "Continuous-dataset H/decoding convergence study using the shared twofig compute/render pipeline. "
        "Categorical NPZs are rejected; use bin/study_h_decoding_categorical_twofig.py for those."
    )
    p.set_defaults(output_dir=str(Path(DATA_DIR) / "h_decoding_convergence"))
    _remove_arg(p, "--theta-field-method")
    _remove_arg(p, "--theta-field-methods")
    _remove_arg(p, "--theta-field-rows")
    p.add_argument(
        "--methods",
        type=str,
        default=DEFAULT_METHODS,
        help=(
            "Comma-separated methods to sweep. Supported values are the continuous twofig row "
            "method aliases, without per-row ':arch' suffixes."
        ),
    )
    return p


def _parse_methods(methods_raw: str) -> list[str]:
    toks = [t.strip() for t in str(methods_raw or "").split(",") if t.strip()]
    if not toks:
        raise ValueError("--methods must contain at least one method.")
    out: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if ":" in tok:
            raise ValueError(
                f"Invalid --methods token {tok!r}; per-method ':arch' overrides are not supported "
                "by study_h_decoding_convergence.py. Use global --flow-arch."
            )
        try:
            method = twofig._normalize_theta_field_method_local(tok)
        except Exception as exc:
            raise ValueError(f"Unknown --methods token {tok!r}.") from exc
        if method not in seen:
            seen.add(method)
            out.append(method)
    return out


def _validate_continuous_dataset(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
    meta_family = str(meta.get("dataset_family", ""))
    if meta_family != str(args.dataset_family):
        raise ValueError(
            f"NPZ meta dataset_family={meta_family!r} does not match --dataset-family={str(args.dataset_family)!r}."
        )
    if str(meta.get("theta_type", "")).strip().lower() == "categorical":
        raise ValueError(
            "study_h_decoding_convergence.py now runs the continuous twofig workflow and rejects "
            "categorical NPZs. Use bin/study_h_decoding_categorical_twofig.py for categorical datasets."
        )
    return meta


def _scalar_str(z: Any, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.size == 0:
        return default
    x = arr.reshape(-1)[0]
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _write_convergence_npz_from_twofig(output_dir: str) -> str:
    src = os.path.join(output_dir, "h_decoding_twofig_results.npz")
    dst = os.path.join(output_dir, "h_decoding_convergence_results.npz")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Expected twofig results NPZ is missing: {src}")
    with np.load(src, allow_pickle=True) as z:
        payload = {k: np.asarray(z[k]) for k in z.files}
        if "theta_field_rows" in z.files:
            methods = np.asarray(z["theta_field_rows"], dtype=np.str_)
        elif "theta_field_methods" in z.files:
            methods = np.asarray(z["theta_field_methods"], dtype=np.str_)
        else:
            raise KeyError(f"{src} missing theta_field_rows/theta_field_methods.")
        if "h_binned_sweep" not in z.files:
            raise KeyError(f"{src} missing h_binned_sweep.")
        if "corr_h_binned_vs_gt_mc" not in z.files:
            raise KeyError(f"{src} missing corr_h_binned_vs_gt_mc.")
        if "corr_decode_vs_ref_shared" not in z.files:
            raise KeyError(f"{src} missing corr_decode_vs_ref_shared.")

        payload.update(
            method_names=methods,
            h_sqrt_sweep=np.asarray(z["h_binned_sweep"], dtype=np.float64),
            corr_h_vs_gt=np.asarray(z["corr_h_binned_vs_gt_mc"], dtype=np.float64),
            corr_decode_vs_ref=np.asarray(z["corr_decode_vs_ref_shared"], dtype=np.float64),
        )
        if "nmse_h_binned_vs_gt_mc" in z.files:
            payload["nmse_h_vs_gt"] = np.asarray(z["nmse_h_binned_vs_gt_mc"], dtype=np.float64)
        if "nmse_decode_vs_ref_shared" in z.files:
            payload["nmse_decode_vs_ref"] = np.asarray(z["nmse_decode_vs_ref_shared"], dtype=np.float64)
        payload["results_schema"] = np.asarray(["continuous_convergence_twofig"], dtype=object)
    np.savez_compressed(dst, **payload)
    return dst


def _write_twofig_npz_from_convergence(output_dir: str, *, meta: dict[str, Any]) -> str:
    src = os.path.join(output_dir, "h_decoding_convergence_results.npz")
    dst = os.path.join(output_dir, "h_decoding_twofig_results.npz")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            "Visualization-only mode requires the new continuous twofig-style results file:\n"
            f"  {src}\n"
            "Legacy single-method h_decoding_convergence_results.npz files are not migrated."
        )
    with np.load(src, allow_pickle=True) as z:
        required = (
            "n",
            "n_ref",
            "method_names",
            "theta_bin_centers",
            "h_gt_sqrt",
            "decode_ref",
            "decode_sweep",
            "h_sqrt_sweep",
            "corr_h_vs_gt",
            "corr_decode_vs_ref",
            "wall_seconds",
        )
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(
                f"{src} is not a new continuous twofig-style convergence NPZ; missing keys: {missing}"
            )
        payload = {k: np.asarray(z[k]) for k in z.files}
        methods = np.asarray(z["method_names"], dtype=np.str_)
        payload.update(
            theta_field_rows=methods,
            theta_field_methods=methods,
            theta_field_row_methods=methods,
            theta_field_row_arches=np.asarray([""] * int(methods.size), dtype=np.str_),
            h_binned_sweep=np.asarray(z["h_sqrt_sweep"], dtype=np.float64),
            corr_h_binned_vs_gt_mc=np.asarray(z["corr_h_vs_gt"], dtype=np.float64),
            corr_decode_vs_ref_shared=np.asarray(z["corr_decode_vs_ref"], dtype=np.float64),
        )
        if "nmse_h_vs_gt" in z.files:
            payload["nmse_h_binned_vs_gt_mc"] = np.asarray(z["nmse_h_vs_gt"], dtype=np.float64)
        if "nmse_decode_vs_ref" in z.files:
            payload["nmse_decode_vs_ref_shared"] = np.asarray(z["nmse_decode_vs_ref"], dtype=np.float64)
        if "dataset_meta_seed" not in payload:
            payload["dataset_meta_seed"] = np.int64(int(meta.get("seed", 0)))
        if "dataset_pool_size" not in payload:
            payload["dataset_pool_size"] = np.int64(0)
    np.savez_compressed(dst, **payload)
    return dst


def _copy_prefixed_outputs(output_dir: str) -> dict[str, str]:
    copied: dict[str, str] = {}
    for src_name, dst_name in _TWOFIG_TO_CONVERGENCE_FILES.items():
        src = os.path.join(output_dir, src_name)
        dst = os.path.join(output_dir, dst_name)
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            copied[dst_name] = dst
    sweep = os.path.join(output_dir, "h_decoding_convergence_sweep.svg")
    corr = os.path.join(output_dir, "h_decoding_convergence_corr_nmse.svg")
    loss = os.path.join(output_dir, "h_decoding_convergence_training_losses_panel.svg")
    if all(os.path.isfile(p) for p in (sweep, corr, loss)):
        try:
            copied["h_decoding_convergence_all_columns.png"] = _write_all_columns_png(
                output_dir,
                sweep_svg=sweep,
                corr_nmse_svg=corr,
                loss_panel_svg=loss,
                out_name="h_decoding_convergence_all_columns.png",
            )
        except TypeError:
            copied["h_decoding_convergence_all_columns.png"] = _write_all_columns_png(
                output_dir,
                sweep_svg=sweep,
                corr_nmse_svg=corr,
                loss_panel_svg=loss,
            )
            default_png = os.path.join(output_dir, "h_decoding_categorical_twofig_all_columns.png")
            want_png = os.path.join(output_dir, "h_decoding_convergence_all_columns.png")
            if os.path.isfile(default_png) and default_png != want_png:
                shutil.copy2(default_png, want_png)
                copied["h_decoding_convergence_all_columns.png"] = want_png
    return copied


def _rewrite_summary(output_dir: str, *, args: argparse.Namespace, visualization_only: bool) -> str:
    path = os.path.join(output_dir, "h_decoding_convergence_summary.txt")
    out_npz = os.path.abspath(os.path.join(output_dir, "h_decoding_convergence_results.npz"))
    lines = [
        "study_h_decoding_convergence",
        f"workflow: continuous_twofig",
        f"visualization_only: {bool(visualization_only)}",
        f"output_dir: {os.path.abspath(output_dir)}",
        f"dataset_npz: {os.path.abspath(str(args.dataset_npz))}",
        f"dataset_family: {str(args.dataset_family)}",
        f"n_list: {args.n_list}",
        f"n_ref: {int(args.n_ref)}",
        f"methods: {args.methods}",
        f"device: {args.device}",
        f"results_npz: {out_npz}",
        "h_decoding_convergence_sweep.svg: "
        f"{os.path.abspath(os.path.join(output_dir, 'h_decoding_convergence_sweep.svg'))}",
        "h_decoding_convergence_corr_nmse.svg: "
        f"{os.path.abspath(os.path.join(output_dir, 'h_decoding_convergence_corr_nmse.svg'))}",
        "h_decoding_convergence_training_losses_panel.svg: "
        f"{os.path.abspath(os.path.join(output_dir, 'h_decoding_convergence_training_losses_panel.svg'))}",
    ]
    png = os.path.join(output_dir, "h_decoding_convergence_all_columns.png")
    if os.path.isfile(png):
        lines.append(f"h_decoding_convergence_all_columns.png: {os.path.abspath(png)}")
    loss_root = os.path.join(output_dir, "training_losses")
    if os.path.isdir(loss_root):
        lines.append(f"training_losses_root: {os.path.abspath(loss_root)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _twofig_argv_from_args(args: argparse.Namespace, methods: list[str]) -> list[str]:
    argv: list[str] = []
    skip = {"methods"}
    for action in build_parser()._actions:
        if not action.option_strings:
            continue
        dest = action.dest
        if dest in skip or dest == "help":
            continue
        if not hasattr(args, dest):
            continue
        opt = action.option_strings[-1]
        val = getattr(args, dest)
        if isinstance(action, argparse._StoreTrueAction):
            if bool(val):
                argv.append(opt)
        elif isinstance(action, argparse._StoreFalseAction):
            if not bool(val):
                argv.append(opt)
        else:
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                argv.append(opt)
                argv.extend(str(x) for x in val)
            else:
                argv.extend([opt, str(val)])
    argv.extend(["--theta-field-methods", ",".join(methods)])
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir = os.path.abspath(str(args.output_dir))
    args.dataset_npz = os.path.abspath(str(args.dataset_npz))
    methods = _parse_methods(args.methods)
    meta = _validate_continuous_dataset(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if bool(getattr(args, "visualization_only", False)):
        _write_twofig_npz_from_convergence(args.output_dir, meta=meta)
    twofig.main(_twofig_argv_from_args(args, methods))
    _write_convergence_npz_from_twofig(args.output_dir)
    _copy_prefixed_outputs(args.output_dir)
    summary = _rewrite_summary(
        args.output_dir,
        args=args,
        visualization_only=bool(getattr(args, "visualization_only", False)),
    )
    print("[convergence] Saved:", flush=True)
    for name in (
        "h_decoding_convergence_results.npz",
        "h_decoding_convergence_sweep.svg",
        "h_decoding_convergence_corr_nmse.svg",
        "h_decoding_convergence_training_losses_panel.svg",
        "h_decoding_convergence_all_columns.png",
    ):
        p = os.path.join(args.output_dir, name)
        if os.path.isfile(p):
            print(f"  - {os.path.abspath(p)}", flush=True)
    print(f"  - {os.path.abspath(summary)}", flush=True)
