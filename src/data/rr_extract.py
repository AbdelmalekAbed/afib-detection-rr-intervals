"""Convert PhysioNet beat + rhythm annotations into per-beat RR-interval series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wfdb

from src.data.loader import BEAT_SYMBOLS, RecordMetadata

AFIB_RHYTHM_CODES = {"(AFIB", "(AFL"}


@dataclass
class RRSeries:
    """Per-record RR series with beat-aligned AFib labels."""
    record_name: str
    rr_seconds: np.ndarray
    is_afib: np.ndarray
    fs: float


def _rhythm_per_position(
    target_samples: np.ndarray,
    rhythm_ann: wfdb.Annotation,
) -> np.ndarray:
    """For each target sample, return the rhythm code active at that time.

    Rhythm codes are inferred from ``aux_note`` entries that start with ``(``.
    The rhythm at the *last* such marker preceding a target sample is taken.
    """
    if rhythm_ann.aux_note is None:
        return np.full(len(target_samples), "", dtype=object)

    rhythm_samples_all = np.asarray(rhythm_ann.sample)
    notes = np.asarray(
        [(note or "").strip().rstrip("\x00") for note in rhythm_ann.aux_note]
    )
    mask = np.array([n.startswith("(") for n in notes])
    rhythm_samples = rhythm_samples_all[mask]
    rhythm_codes = notes[mask]

    if len(rhythm_samples) == 0:
        return np.full(len(target_samples), "", dtype=object)

    order = np.argsort(rhythm_samples)
    rhythm_samples = rhythm_samples[order]
    rhythm_codes = rhythm_codes[order]

    idx = np.searchsorted(rhythm_samples, target_samples, side="right") - 1
    out = np.full(len(target_samples), "", dtype=object)
    valid = idx >= 0
    out[valid] = rhythm_codes[idx[valid]]
    return out


def extract_rr_series(meta: RecordMetadata) -> RRSeries:
    """Build an :class:`RRSeries` from a :class:`RecordMetadata` bundle.

    Beats come from ``meta.beat_ann`` (may be ``.qrs`` for AFDB or ``.atr`` for
    NSRDB/LTAFDB). Rhythms always come from ``meta.rhythm_ann`` (``.atr``).
    """
    beat_samples_all = np.asarray(meta.beat_ann.sample)
    beat_symbols = np.asarray(meta.beat_ann.symbol)
    is_beat = np.array([s in BEAT_SYMBOLS for s in beat_symbols])
    beat_samples = beat_samples_all[is_beat]

    if len(beat_samples) < 2:
        return RRSeries(meta.record_name, np.empty(0, np.float32), np.empty(0, np.int8), meta.fs)

    rhythm_per_beat = _rhythm_per_position(beat_samples, meta.rhythm_ann)
    is_afib_beat = np.array([r in AFIB_RHYTHM_CODES for r in rhythm_per_beat])

    beat_times = beat_samples.astype(np.float64) / meta.fs
    rr = np.diff(beat_times).astype(np.float32)
    is_afib_rr = is_afib_beat[1:].astype(np.int8)

    return RRSeries(meta.record_name, rr, is_afib_rr, meta.fs)


def clean_rr_series(series: RRSeries, min_rr: float = 0.3, max_rr: float = 2.0) -> RRSeries:
    """Drop physiologically implausible RR values and the corresponding labels."""
    keep = (series.rr_seconds >= min_rr) & (series.rr_seconds <= max_rr)
    return RRSeries(
        record_name=series.record_name,
        rr_seconds=series.rr_seconds[keep],
        is_afib=series.is_afib[keep],
        fs=series.fs,
    )
