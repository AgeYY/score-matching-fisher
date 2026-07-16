#!/usr/bin/env python3
"""Cache minimally processed instantaneous voltage features for BCI IV-2a."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import (  # noqa: E402
    extract_voltage_features,
    list_training_recordings,
    save_features_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf-dir", type=Path, default=ROOT / "data/bci_iv_2a/raw/gdf")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_20uv",
    )
    parser.add_argument("--tmin", type=float, default=-1.5)
    parser.add_argument("--tmax", type=float, default=3.5)
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument(
        "--all-native-time-points",
        action="store_true",
        help="Retain every native sample from tmin through tmax instead of the requested sparse grid.",
    )
    parser.add_argument("--voltage-unit-microvolts", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0", help="Recorded for project command consistency; unused.")
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("Project runs must specify --device cuda:0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in list_training_recordings(args.gdf_dir):
        print(f"[voltage] {source.name}", flush=True)
        features = extract_voltage_features(
            source,
            tmin=args.tmin,
            tmax=args.tmax,
            step_seconds=args.step_seconds,
            voltage_unit_microvolts=args.voltage_unit_microvolts,
            all_native_time_points=args.all_native_time_points,
        )
        destination = save_features_npz(args.output_dir / f"{source.stem}.npz", features)
        record = dict(features.metadata)
        record["feature_file"] = str(destination.resolve())
        records.append(record)
        print(
            f"[voltage] {source.stem}: clean={features.features.shape[0]} "
            f"shape={features.features.shape} range="
            f"[{features.features.min():.4g}, {features.features.max():.4g}] "
            f"saved={destination}",
            flush=True,
        )
    manifest = args.output_dir / "qc_manifest.json"
    manifest.write_text(
        json.dumps({"recordings": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[voltage] Saved QC manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
