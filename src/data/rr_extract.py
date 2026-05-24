"""Convert PhysioNet beat annotations into per-beat RR-interval series with AFib labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wfdb

AFIB_RHYTHM_CODES = {"(AFIB", "(AFL"}
BEAT_SYMBOLS = {"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"}


@dataclass
class RRSeries:
    """Per-record RR series with beat-aligned AFib labels."""
    record_name: str
    rr_seconds: np.ndarray
    is_afib: np.ndarray
    fs: float


def extract_rr_series(record: wfdb.Record, annotation: wfdb.Annotation) -> RRSeries:
    """Build an :class:`RRSeries` from a wfdb record + annotation pair.

    Rhythm annotations (``aux_note`` entries like ``(AFIB``) are forward-filled to
    every beat; the RR series is the diff of successive beat sample indices divided
    by the sampling frequency.
    """
    fs = float(record.fs)
    sample_idx = np.asarray(annotation.sample)
    symbols = np.asarray(annotation.symbol)
    aux = annotation.aux_note if annotation.aux_note is not None else [""] * len(symbols)

    beat_mask = np.array([s in BEAT_SYMBOLS for s in symbols])
    beat_times = sample_idx[beat_mask].astype(np.float64) / fs

    current_rhythm = ""
    rhythm_per_event: list[str] = []
    for note in aux:
        cleaned = (note or "").strip().rstrip("\x00")
        if cleaned.startswith("("):
            current_rhythm = cleaned
        rhythm_per_event.append(current_rhythm)
    rhythm_per_event_arr = np.asarray(rhythm_per_event)[beat_mask]
    is_afib_beat = np.array([r in AFIB_RHYTHM_CODES for r in rhythm_per_event_arr])

    rr = np.diff(beat_times)
    is_afib_rr = is_afib_beat[1:]

    return RRSeries(
        record_name=record.record_name,
        rr_seconds=rr.astype(np.float32),
        is_afib=is_afib_rr.astype(np.int8),
        fs=fs,
    )


def clean_rr_series(series: RRSeries, min_rr: float = 0.3, max_rr: float = 2.0) -> RRSeries:
    """Drop physiologically implausible RR values and the corresponding labels."""
    keep = (series.rr_seconds >= min_rr) & (series.rr_seconds <= max_rr)
    return RRSeries(
        record_name=series.record_name,
        rr_seconds=series.rr_seconds[keep],
        is_afib=series.is_afib[keep],
        fs=series.fs,
    )
