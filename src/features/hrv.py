"""Classical HRV features used by the tree-based baselines."""
from __future__ import annotations

import numpy as np


def rmssd(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    diffs = np.diff(rr)
    return float(np.sqrt(np.mean(diffs**2)))


def sdnn(rr: np.ndarray) -> float:
    return float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0


def pnn50(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    diffs = np.abs(np.diff(rr))
    return float((diffs > 0.05).mean())


def shannon_entropy(rr: np.ndarray, bins: int = 16) -> float:
    if len(rr) == 0:
        return 0.0
    hist, _ = np.histogram(rr, bins=bins, density=True)
    p = hist[hist > 0]
    return float(-(p * np.log(p)).sum())


def coefficient_of_variation(rr: np.ndarray) -> float:
    if len(rr) == 0 or rr.mean() == 0:
        return 0.0
    return float(rr.std(ddof=1) / rr.mean())


def feature_vector(rr: np.ndarray) -> np.ndarray:
    """Return a fixed-order HRV feature vector for a single window."""
    return np.array(
        [
            rmssd(rr),
            sdnn(rr),
            pnn50(rr),
            shannon_entropy(rr),
            coefficient_of_variation(rr),
            float(rr.mean()),
            float(rr.min()) if len(rr) else 0.0,
            float(rr.max()) if len(rr) else 0.0,
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "rmssd",
    "sdnn",
    "pnn50",
    "shannon_entropy",
    "cv",
    "mean_rr",
    "min_rr",
    "max_rr",
]


def featurize_windows(X: np.ndarray) -> np.ndarray:
    """Apply :func:`feature_vector` to every row of a (n, window_size) array."""
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {X.shape}")
    return np.stack([feature_vector(row) for row in X])
