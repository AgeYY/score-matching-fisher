#!/usr/bin/env python3
"""Validate BCI IV-2a training recordings and cache log-bandpower features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import (  # noqa: E402
    extract_log_bandpower_features,
    list_training_recordings,
    save_features_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf-dir", type=Path, default=ROOT / "data/bci_iv_2a/raw/gdf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/bci_iv_2a/processed/log_bandpower")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in list_training_recordings(args.gdf_dir):
        print(f"[preprocess] {source.name}", flush=True)
        features = extract_log_bandpower_features(source)
        destination = save_features_npz(args.output_dir / f"{source.stem}.npz", features)
        record = dict(features.metadata)
        record["feature_file"] = str(destination.resolve())
        records.append(record)
        print(
            f"[preprocess] {source.stem}: clean={features.features.shape[0]} "
            f"shape={features.features.shape} saved={destination}",
            flush=True,
        )
    manifest = args.output_dir / "qc_manifest.json"
    manifest.write_text(json.dumps({"recordings": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[preprocess] Saved QC manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
