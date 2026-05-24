"""Segment RR series into fixed-size windows with patient-level metadata."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.rr_extract import RRSeries


@dataclass
class WindowedDataset:
    """Stacked windows with per-window patient id and label.

    The ``patient_id`` column is the key to enforce patient-level splits — never
    shuffle without grouping on it.
    """
    X: np.ndarray
    y: np.ndarray
    patient_id: np.ndarray


def segment_record(
    series: RRSeries,
    window_size: int,
    stride: int,
    label_strategy: str = "majority",
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a single record and produce (X, y).

    ``label_strategy`` is ``"majority"`` (window labeled AFib if ≥50% of beats are AFib)
    or ``"all"`` (only AFib if every beat in the window is AFib).
    """
    rr = series.rr_seconds
    lab = series.is_afib
    n = len(rr)
    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int8)

    starts = np.arange(0, n - window_size + 1, stride)
    X = np.stack([rr[s : s + window_size] for s in starts])

    if label_strategy == "majority":
        y = np.array([(lab[s : s + window_size].mean() >= 0.5) for s in starts], dtype=np.int8)
    elif label_strategy == "all":
        y = np.array([lab[s : s + window_size].all() for s in starts], dtype=np.int8)
    else:
        raise ValueError(f"Unknown label_strategy: {label_strategy}")

    return X.astype(np.float32), y


def build_windowed_dataset(
    series_by_patient: dict[str, RRSeries],
    window_size: int,
    stride: int,
    label_strategy: str = "majority",
) -> WindowedDataset:
    """Concatenate windowed records across patients, preserving patient ids."""
    Xs, ys, pids = [], [], []
    for pid, series in series_by_patient.items():
        X, y = segment_record(series, window_size, stride, label_strategy)
        if len(X) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        pids.append(np.full(len(X), pid))
    return WindowedDataset(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        patient_id=np.concatenate(pids, axis=0),
    )
