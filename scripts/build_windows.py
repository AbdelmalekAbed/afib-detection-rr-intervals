"""Build the windowed dataset from raw PhysioNet records.

Run via: ``python -m scripts.build_windows --config configs/data.yaml``

Output: a single ``.npz`` per dataset under ``data/processed/`` with arrays
``X`` (n, window_size), ``y`` (n,), ``patient_id`` (n,).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.loader import list_records, load_record_metadata
from src.data.rr_extract import clean_rr_series, extract_rr_series
from src.data.windowing import build_windowed_dataset


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
        series_by_patient = {}
        for rec_name in list_records(dataset_dir):
            try:
                meta = load_record_metadata(dataset_dir, rec_name)
                series = extract_rr_series(meta)
                series = clean_rr_series(series, min_rr=min_rr, max_rr=max_rr)
                if len(series.rr_seconds) >= window_size:
                    series_by_patient[rec_name] = series
            except Exception as e:
                print(f"  skip {rec_name}: {e}")

        if not series_by_patient:
            print(f"[skip] {name}: no usable records")
            continue

        windowed = build_windowed_dataset(series_by_patient, window_size, stride, label_strategy)
        out_path = out_dir / f"{name}.npz"
        np.savez_compressed(out_path, X=windowed.X, y=windowed.y, patient_id=windowed.patient_id)
        print(f"[ok] {name}: {windowed.X.shape} -> {out_path}")


if __name__ == "__main__":
    main()
