"""Tests des métriques et des descripteurs HRV sur des cas à réponse connue.

L'enjeu principal est la règle d'exclusion du F1 par patient : les patients dont
l'enregistrement ne contient qu'une seule classe sont écartés. Cette règle
conditionne l'interprétation de tous les tableaux du mémoire, elle doit donc
être verrouillée par un test.
"""
from __future__ import annotations

import numpy as np

from src.features.hrv import (
    FEATURE_NAMES,
    coefficient_of_variation,
    feature_vector,
    featurize_windows,
    pnn50,
    rmssd,
    sdnn,
)
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def test_threshold_metrics_on_a_hand_checked_case():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.6, 0.4, 0.9])   # à 0,5 : 1 FP, 1 FN, 1 VP, 1 VN
    m = threshold_metrics(y, score, threshold=0.5)

    assert m["sensitivity"] == 0.5      # 1 VP sur 2 positifs
    assert m["specificity"] == 0.5      # 1 VN sur 2 négatifs
    assert m["accuracy"] == 0.5
    assert abs(m["f1"] - 0.5) < 1e-9    # précision 0,5 et rappel 0,5


def test_perfect_and_inverted_rankings():
    y = np.array([0, 0, 1, 1])
    assert ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert ranking_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 0.0


def test_per_patient_f1_excludes_single_class_patients():
    """Un patient entièrement en FA (ou entièrement sain) n'est pas notable."""
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.9, 0.1, 0.2])
    groups = np.array(["a", "a", "a", "a", "b", "b", "c", "c"])
    #   patient a : les deux classes           -> noté
    #   patient b : 100 % FA                   -> exclu
    #   patient c : 0 % FA                     -> exclu
    out = per_patient_f1(y, score, groups, threshold=0.5)

    assert out["n_patients_scored"] == 1
    assert out["mean_patient_f1"] == 1.0     # le patient a est parfaitement classé


def test_per_patient_f1_averages_patients_not_windows():
    """Chaque patient pèse identiquement, quel que soit son nombre de fenêtres."""
    # Patient « petit » parfaitement classé (2 fenêtres),
    # patient « grand » totalement raté (100 fenêtres).
    y = np.concatenate([[0, 1], np.tile([0, 1], 50)])
    score = np.concatenate([[0.1, 0.9], np.tile([0.9, 0.1], 50)])
    groups = np.array(["petit"] * 2 + ["grand"] * 100)
    out = per_patient_f1(y, score, groups, threshold=0.5)

    assert out["n_patients_scored"] == 2
    # Une moyenne par fenêtre serait écrasée par le patient « grand » ;
    # la moyenne par patient reste à mi-chemin.
    assert abs(out["mean_patient_f1"] - 0.5) < 1e-9


def test_per_patient_f1_returns_zero_when_nothing_is_scorable():
    y = np.array([1, 1, 1, 1])
    score = np.array([0.9, 0.9, 0.9, 0.9])
    groups = np.array(["a", "a", "b", "b"])
    out = per_patient_f1(y, score, groups)
    assert out == {"mean_patient_f1": 0.0, "n_patients_scored": 0}


# --------------------------------------------------------------------------- #
# Descripteurs HRV
# --------------------------------------------------------------------------- #
def test_constant_series_has_zero_variability():
    """Un rythme parfaitement régulier : tous les descripteurs de dispersion nuls."""
    rr = np.full(20, 0.8, dtype=np.float32)
    assert rmssd(rr) == 0.0
    assert sdnn(rr) == 0.0
    assert pnn50(rr) == 0.0
    assert coefficient_of_variation(rr) == 0.0


def test_rmssd_matches_its_definition():
    rr = np.array([0.8, 1.0, 0.6], dtype=np.float32)   # différences : +0,2 puis −0,4
    expected = np.sqrt((0.2**2 + 0.4**2) / 2)
    assert abs(rmssd(rr) - expected) < 1e-6


def test_pnn50_counts_differences_above_50_ms():
    rr = np.array([0.80, 0.82, 0.95, 0.96], dtype=np.float32)
    # différences : 0,02 (non) / 0,13 (oui) / 0,01 (non) -> 1 sur 3
    assert abs(pnn50(rr) - 1 / 3) < 1e-6


def test_irregular_series_has_higher_variability_than_regular_one():
    """Propriété qui fonde toute l'approche : la FA se voit dans la dispersion."""
    rng = np.random.default_rng(0)
    regular = rng.normal(0.80, 0.01, 200).astype(np.float32)      # sinusal
    irregular = rng.uniform(0.40, 1.20, 200).astype(np.float32)   # fibrillation

    assert rmssd(irregular) > 5 * rmssd(regular)
    assert coefficient_of_variation(irregular) > coefficient_of_variation(regular)


def test_feature_vector_shape_and_order():
    rr = np.array([0.7, 0.72, 0.69, 0.75, 0.71], dtype=np.float32)
    feats = feature_vector(rr)

    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))
    assert abs(feats[FEATURE_NAMES.index("mean_rr")] - rr.mean()) < 1e-6
    assert abs(feats[FEATURE_NAMES.index("min_rr")] - rr.min()) < 1e-6
    assert abs(feats[FEATURE_NAMES.index("max_rr")] - rr.max()) < 1e-6


def test_featurize_windows_is_row_wise():
    X = np.random.default_rng(1).uniform(0.6, 1.0, size=(7, 60)).astype(np.float32)
    F = featurize_windows(X)

    assert F.shape == (7, len(FEATURE_NAMES))
    assert np.allclose(F[3], feature_vector(X[3]))
