"""Evaluation metrics with patient-level aggregation support."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float((y_pred == y_true).mean()),
    }


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def per_patient_f1(y_true: np.ndarray, y_score: np.ndarray, patient_ids: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Mean F1 across patients — penalizes models that only work on a few subjects."""
    scores: list[float] = []
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        if mask.sum() == 0 or y_true[mask].sum() == 0 or y_true[mask].sum() == mask.sum():
            continue
        y_pred = (y_score[mask] >= threshold).astype(int)
        scores.append(f1_score(y_true[mask], y_pred, zero_division=0))
    return {"mean_patient_f1": float(np.mean(scores)) if scores else 0.0, "n_patients_scored": len(scores)}
