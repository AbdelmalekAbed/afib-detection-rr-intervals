"""Sanity tests for windowing and HRV features (don't require a downloaded dataset)."""
from __future__ import annotations

import numpy as np

from src.data.rr_extract import RRSeries
from src.data.windowing import segment_record
from src.features.hrv import feature_vector

# Smoke test of the loader/extract chain on a real PhysioNet record is skipped
# if the data has not been downloaded yet — keeps unit tests pure.


def make_series(n: int, afib_fraction: float, seed: int = 0) -> RRSeries:
    rng = np.random.default_rng(seed)
    rr = rng.uniform(0.6, 0.9, size=n).astype(np.float32)
    n_afib = int(n * afib_fraction)
    is_afib = np.zeros(n, dtype=np.int8)
    is_afib[:n_afib] = 1
    return RRSeries("synthetic", rr, is_afib, fs=250.0)


def test_segment_record_basic():
    series = make_series(120, afib_fraction=0.5)
    X, y = segment_record(series, window_size=30, stride=15)
    assert X.shape[1] == 30
    assert len(X) == len(y)
    assert X.dtype == np.float32


def test_segment_too_short_returns_empty():
    series = make_series(10, afib_fraction=0.0)
    X, y = segment_record(series, window_size=30, stride=15)
    assert len(X) == 0 and len(y) == 0


def test_hrv_feature_vector_shape():
    rr = np.array([0.7, 0.72, 0.69, 0.75, 0.71], dtype=np.float32)
    feats = feature_vector(rr)
    assert feats.shape == (8,)
    assert np.all(np.isfinite(feats))
