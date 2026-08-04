"""Tests de l'extraction RR + étiquetage — la partie la plus risquée du pipeline.

L'appariement entre les positions de battements et les annotations de rythme est
l'endroit où une erreur silencieuse fausserait toutes les étiquettes sans faire
échouer quoi que ce soit. Ces tests construisent des annotations synthétiques
dont on connaît la réponse exacte.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import RecordMetadata
from src.data.rr_extract import clean_rr_series, extract_rr_series


class FakeAnnotation:
    """Imite le strict nécessaire d'un ``wfdb.Annotation``."""

    def __init__(self, sample, symbol, aux_note=None):
        self.sample = np.asarray(sample)
        self.symbol = list(symbol)
        self.aux_note = aux_note


def make_meta(beat_samples, rhythm_samples, rhythm_notes, fs=250.0) -> RecordMetadata:
    beats = FakeAnnotation(beat_samples, ["N"] * len(beat_samples))
    rhythms = FakeAnnotation(rhythm_samples, ["+"] * len(rhythm_samples), rhythm_notes)
    return RecordMetadata(record_name="synth", fs=fs, beat_ann=beats, rhythm_ann=rhythms)


def test_rr_intervals_are_differences_of_beat_times():
    """Un battement par seconde à 250 Hz doit donner des RR de 1 s."""
    fs = 250.0
    beats = np.arange(0, 6) * int(fs)          # 0, 1, 2, 3, 4, 5 secondes
    meta = make_meta(beats, [0], ["(N"], fs=fs)
    series = extract_rr_series(meta)

    assert len(series.rr_seconds) == len(beats) - 1
    assert np.allclose(series.rr_seconds, 1.0)


def test_labels_follow_the_last_rhythm_marker():
    """Le rythme d'un battement est celui du dernier marqueur qui le précède."""
    fs = 250.0
    beats = np.arange(0, 10) * int(fs)
    # Sinusal de 0 s à 4 s, FA à partir de 4 s.
    meta = make_meta(beats, [0, 4 * int(fs)], ["(N", "(AFIB"], fs=fs)
    series = extract_rr_series(meta)

    # is_afib est aligné sur les RR, donc sur les battements 1..9.
    # Les battements aux instants 4 s et au-delà (indices 4 à 9) sont en FA.
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    assert np.array_equal(series.is_afib, expected)


def test_atrial_flutter_is_labelled_positive():
    """Le flutter (AFL) partage le risque clinique de la FA : il est positif."""
    fs = 250.0
    beats = np.arange(0, 5) * int(fs)
    meta = make_meta(beats, [0], ["(AFL"], fs=fs)
    assert extract_rr_series(meta).is_afib.all()


def test_other_rhythms_are_labelled_negative():
    """Un rythme non listé (ici bloc auriculo-ventriculaire) reste négatif."""
    fs = 250.0
    beats = np.arange(0, 5) * int(fs)
    meta = make_meta(beats, [0], ["(B"], fs=fs)
    assert not extract_rr_series(meta).is_afib.any()


def test_beats_before_any_rhythm_marker_are_negative():
    """Sans rythme déclaré en amont, on ne suppose pas la FA."""
    fs = 250.0
    beats = np.arange(0, 6) * int(fs)
    meta = make_meta(beats, [3 * int(fs)], ["(AFIB"], fs=fs)
    series = extract_rr_series(meta)
    assert series.is_afib[0] == 0
    assert series.is_afib[-1] == 1


def test_non_beat_symbols_are_ignored():
    """Les annotations qui ne sont pas des battements ne créent pas de RR."""
    fs = 250.0
    samples = [0, 125, 250, 375, 500]
    symbols = ["N", "+", "N", "~", "N"]   # seuls 0, 250 et 500 sont des battements
    beats = FakeAnnotation(samples, symbols)
    rhythms = FakeAnnotation([0], ["+"], ["(N"])
    meta = RecordMetadata("synth", fs, beats, rhythms)

    series = extract_rr_series(meta)
    assert len(series.rr_seconds) == 2
    assert np.allclose(series.rr_seconds, 1.0)


def test_too_few_beats_returns_empty_series():
    meta = make_meta([0], [0], ["(N"])
    series = extract_rr_series(meta)
    assert len(series.rr_seconds) == 0
    assert len(series.is_afib) == 0


@pytest.mark.parametrize("rr,keep", [
    (0.2, False),   # trop court : impossible physiologiquement
    (0.3, True),    # borne basse incluse
    (1.0, True),
    (2.0, True),    # borne haute incluse
    (2.5, False),   # trop long
])
def test_clean_rr_series_bounds(rr, keep):
    from src.data.rr_extract import RRSeries

    series = RRSeries("synth", np.array([rr], np.float32), np.array([1], np.int8), 250.0)
    cleaned = clean_rr_series(series)
    assert (len(cleaned.rr_seconds) == 1) is keep


def test_clean_rr_series_drops_labels_together_with_values():
    """Un RR écarté doit emporter son étiquette, sans décalage."""
    from src.data.rr_extract import RRSeries

    rr = np.array([0.8, 5.0, 0.9, 0.1, 0.7], np.float32)
    lab = np.array([0, 1, 1, 0, 0], np.int8)
    cleaned = clean_rr_series(RRSeries("synth", rr, lab, 250.0))

    assert np.allclose(cleaned.rr_seconds, [0.8, 0.9, 0.7])
    assert np.array_equal(cleaned.is_afib, [0, 1, 0])
