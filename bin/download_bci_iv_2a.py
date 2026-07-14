#!/usr/bin/env python3
"""Download the official BCI Competition IV 2a archive and extract A01T--A09T."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
EXPECTED_SHA256 = "65fe93cb766e4b00ece69a200312d81f54bba17c642406bd922d913a8aedc024"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data/bci_iv_2a/raw")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "BCICIV_2a_gdf.zip"
    if not archive.exists():
        print(f"[download] {URL} -> {archive}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    got = digest(archive)
    if got != EXPECTED_SHA256:
        raise RuntimeError(f"SHA256 mismatch: expected {EXPECTED_SHA256}, got {got}")
    gdf_dir = args.output_dir / "gdf"
    gdf_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        for subject in range(1, 10):
            name = f"A{subject:02d}T.gdf"
            if name not in names:
                raise RuntimeError(f"Archive lacks {name}")
            destination = gdf_dir / name
            if not destination.exists():
                with source.open(name) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"[download] Verified {archive} ({got})", flush=True)
    print(f"[download] Training recordings: {gdf_dir}", flush=True)


if __name__ == "__main__":
    main()
