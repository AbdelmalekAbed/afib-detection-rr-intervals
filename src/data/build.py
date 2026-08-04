"""Construction du jeu de fenêtres à partir des enregistrements PhysioNet bruts.

Cette fonction était auparavant recopiée à l'identique dans chaque script de
phase. La factoriser ici garantit que toutes les expériences du dépôt partent
exactement des mêmes fenêtres — condition nécessaire pour que les tableaux
comparatifs du mémoire soient légitimes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.loader import list_records, load_record_metadata
from src.data.rr_extract import clean_rr_series, extract_rr_series
from src.data.windowing import build_windowed_dataset


def build_windows(
    dataset_dir: Path,
    window: int,
    stride: int,
    label_strategy: str = "majority",
    min_rr: float = 0.3,
    max_rr: float = 2.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne ``(X, y, patient_ids)`` pour un répertoire PhysioNet téléchargé.

    Les enregistrements illisibles ou trop courts sont ignorés ; leur nombre est
    signalé si ``verbose`` est vrai, afin qu'une base partiellement téléchargée
    ne passe pas inaperçue.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_dir} est introuvable — lancer `make data` pour télécharger les bases."
        )

    series_by_patient: dict[str, object] = {}
    skipped: list[str] = []
    for rec in list_records(dataset_dir):
        try:
            meta = load_record_metadata(dataset_dir, rec)
            series = clean_rr_series(extract_rr_series(meta), min_rr=min_rr, max_rr=max_rr)
        except Exception as exc:  # enregistrement corrompu ou incomplet
            skipped.append(f"{rec} ({exc})")
            continue
        if len(series.rr_seconds) >= window:
            series_by_patient[rec] = series
        else:
            skipped.append(f"{rec} (trop court)")

    if not series_by_patient:
        raise RuntimeError(f"aucun enregistrement exploitable dans {dataset_dir}")

    if verbose and skipped:
        print(f"[build_windows] {len(skipped)} enregistrement(s) ignoré(s) : {', '.join(skipped[:5])}"
              + (" ..." if len(skipped) > 5 else ""))

    windowed = build_windowed_dataset(
        series_by_patient, window_size=window, stride=stride, label_strategy=label_strategy
    )
    return (
        windowed.X.astype(np.float32),
        windowed.y.astype(np.int8),
        np.asarray(windowed.patient_id),
    )
