"""Non-negotiable: every CV fold must group by patient id.

Data leakage via window-level splits is the most common cause of inflated AFib
scores in the literature; this test fails loudly if the split logic ever drifts.
"""
from __future__ import annotations

import numpy as np

from src.utils.splits import patient_kfold


def test_no_patient_appears_in_both_train_and_val():
    patient_ids = np.array(["p" + str(i // 4) for i in range(40)])
    y = np.random.default_rng(0).integers(0, 2, size=40)
    for train_idx, val_idx in patient_kfold(patient_ids, y, n_splits=4):
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        assert not (train_patients & val_patients), "patient appears in both splits"
        assert len(val_patients) > 0
        assert len(train_patients) > 0


def test_all_patients_are_used_across_folds():
    patient_ids = np.array(["p" + str(i // 5) for i in range(50)])
    y = np.random.default_rng(0).integers(0, 2, size=50)
    seen = set()
    for _, val_idx in patient_kfold(patient_ids, y, n_splits=5):
        seen.update(patient_ids[val_idx])
    assert seen == set(patient_ids)
