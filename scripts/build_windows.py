"""Build the windowed dataset from raw PhysioNet records.

Run via: ``python -m scripts.build_windows --config configs/data.yaml``

Output: a versioned ``dataset_wW_sS_strategy_v1.npz`` under the configured
processed directory. Includes metadata and annotation hashes; never replaces
the legacy ``afdb.npz`` / ``ltafdb.npz`` files or an existing versioned archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.artifacts import save_windows, window_filename
from src.data.build import build_windows
from src.utils.artifacts import file_digest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    window_size = cfg["windowing"]["window_size"]
    stride = cfg["windowing"]["stride"]
    label_strategy = cfg["windowing"]["label_strategy"]
    min_rr = cfg["preprocessing"]["clip_rr_seconds"]["min"]
    max_rr = cfg["preprocessing"]["clip_rr_seconds"]["max"]

    for name, spec in cfg["datasets"].items():
        if not spec.get("enabled", False):
            continue
        dataset_dir = raw_dir / spec["physionet_id"].replace("/", "_")
        out_path = out_dir / window_filename(name, window_size, stride, label_strategy)
        if out_path.exists():
            raise FileExistsError(f"{out_path} existe déjà : choisir un autre répertoire de sortie.")
        X, y, groups = build_windows(dataset_dir, window=window_size, stride=stride,
                                     label_strategy=label_strategy, min_rr=min_rr,
                                     max_rr=max_rr, verbose=True)
        source_files = sorted(p for p in dataset_dir.iterdir() if p.suffix in {".hea", ".atr", ".qrs"})
        save_windows(out_path, X, y, groups, {
            "dataset": name, "window": window_size, "stride": stride,
            "label_strategy": label_strategy, "min_rr": min_rr, "max_rr": max_rr,
            "raw_annotation_sha256": {p.name: file_digest(p) for p in source_files},
        })
        print(f"[ok] {name}: {X.shape} -> {out_path}")


if __name__ == "__main__":
    main()
