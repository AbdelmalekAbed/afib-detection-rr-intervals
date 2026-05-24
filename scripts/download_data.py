"""Download all enabled datasets from configs/data.yaml.

Run via: ``python -m scripts.download_data --config configs/data.yaml``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.loader import download_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg["datasets"].items():
        if not spec.get("enabled", False):
            continue
        print(f"[download] {name} -> {spec['physionet_id']}")
        target = download_dataset(spec["physionet_id"], raw_dir)
        print(f"  ok: {target}")


if __name__ == "__main__":
    main()
