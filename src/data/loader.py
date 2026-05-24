"""Download and access PhysioNet datasets via the wfdb library."""
from __future__ import annotations

from pathlib import Path

import wfdb


def download_dataset(physionet_id: str, raw_dir: Path) -> Path:
    """Download a PhysioNet dataset into ``raw_dir / physionet_id``.

    Returns the local directory containing the downloaded records.
    Skips download if the target directory already contains files.
    """
    target = raw_dir / physionet_id.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        return target
    wfdb.dl_database(physionet_id, dl_dir=str(target))
    return target


def list_records(dataset_dir: Path) -> list[str]:
    """Return record names (stems) found in a downloaded PhysioNet directory."""
    return sorted({p.stem for p in dataset_dir.glob("*.hea")})


def load_record(dataset_dir: Path, record_name: str) -> tuple[wfdb.Record, wfdb.Annotation]:
    """Load a record with its reference annotation file (``.atr``)."""
    record = wfdb.rdrecord(str(dataset_dir / record_name))
    annotation = wfdb.rdann(str(dataset_dir / record_name), extension="atr")
    return record, annotation
