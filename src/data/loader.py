"""Download and access PhysioNet datasets via the wfdb library.

Some PhysioNet datasets (notably AFDB) split beat positions and rhythm annotations
across two annotation files:

- ``.qrs`` — machine-detected beat positions (typically all symbols ``N``)
- ``.atr`` — reference rhythm annotations (``(AFIB``, ``(N``...) using ``+`` symbols

Others (NSRDB, LTAFDB) store everything in a single ``.atr`` file.

This module exposes a uniform ``load_record_metadata`` that returns *both* a beat
annotation and a rhythm annotation, picking the right source automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import wfdb

BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?",
}


@dataclass
class RecordMetadata:
    """Lightweight bundle of what we need to extract RR + labels.

    We deliberately avoid loading the ECG signal itself: a few AFDB records ship
    without their ``.dat`` files, and we don't use the raw signal anyway.
    """
    record_name: str
    fs: float
    beat_ann: wfdb.Annotation
    rhythm_ann: wfdb.Annotation


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


def _count_beat_symbols(ann: wfdb.Annotation | None) -> int:
    if ann is None:
        return 0
    return sum(1 for s in ann.symbol if s in BEAT_SYMBOLS)


def load_record_metadata(dataset_dir: Path, record_name: str) -> RecordMetadata:
    """Load fs and the right beat/rhythm annotations for a record.

    Strategy:
      1. Read ``.hea`` for sampling frequency (no signal loaded).
      2. Load ``.atr`` (always present).
      3. If a ``.qrs`` file exists *and* contains more beats than ``.atr``,
         use ``.qrs`` as the beat source (AFDB pattern). Rhythms always come
         from ``.atr``.
    """
    header = wfdb.rdheader(str(dataset_dir / record_name))
    fs = float(header.fs)

    atr_ann = wfdb.rdann(str(dataset_dir / record_name), extension="atr")
    qrs_path = dataset_dir / f"{record_name}.qrs"
    qrs_ann = None
    if qrs_path.exists():
        qrs_ann = wfdb.rdann(str(dataset_dir / record_name), extension="qrs")

    beat_ann = atr_ann
    if _count_beat_symbols(qrs_ann) > _count_beat_symbols(atr_ann):
        beat_ann = qrs_ann

    return RecordMetadata(
        record_name=record_name,
        fs=fs,
        beat_ann=beat_ann,
        rhythm_ann=atr_ann,
    )
