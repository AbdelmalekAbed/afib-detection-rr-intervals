"""Patient-level cross-validation splits.

CRITICAL: every split function here groups by patient id. Never write a function
that shuffles windows directly — that causes data leakage and is the #1 mistake
in the AFib literature.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def patient_kfold(patient_ids: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Yield ``(train_idx, val_idx)`` such that no patient appears in both folds."""
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(np.zeros_like(y), y, groups=patient_ids):
        assert not set(patient_ids[train_idx]) & set(patient_ids[val_idx]), "patient leak"
        yield train_idx, val_idx
