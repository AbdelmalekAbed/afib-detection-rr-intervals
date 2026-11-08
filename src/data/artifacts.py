"""Versioned window archives. Legacy caches are readable only by explicit opt-in."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def window_filename(dataset: str, window: int, stride: int, strategy: str) -> str:
    return f"{dataset}_w{window}_s{stride}_{strategy}_v{SCHEMA_VERSION}.npz"


def validate_arrays(X, y, groups, expected_window: int) -> None:
    if X.ndim != 2 or X.shape[1] != expected_window:
        raise ValueError(f"Fenêtres attendues : {expected_window} RR ; forme reçue : {X.shape}.")
    if y.ndim != 1 or groups.ndim != 1 or len(X) != len(y) or len(y) != len(groups) or not len(y):
        raise ValueError("Fenêtres, étiquettes et patients doivent être non vides et alignés.")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("RR non finis ou étiquettes non binaires.")


def save_windows(path: Path, X, y, groups, metadata: dict) -> None:
    validate_arrays(X, y, groups, metadata["window"])
    info = {**metadata, "schema_version": SCHEMA_VERSION}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, X=X, y=y, patient_id=np.asarray(groups).astype(str),
                            metadata=np.asarray(json.dumps(info, sort_keys=True)))


def load_windows(path: Path, *, expected_window: int, expected_metadata: dict | None = None,
                 allow_legacy: bool = False) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        X, y, groups = archive["X"], archive["y"], archive["patient_id"].astype(str)
        validate_arrays(X, y, groups, expected_window)
        if "metadata" not in archive:
            if not allow_legacy:
                raise ValueError(f"{path} : cache historique sans métadonnées ; reconstruire une archive versionnée.")
            if expected_metadata:
                raise ValueError("Les métadonnées d'un cache historique ne peuvent pas être vérifiées.")
            warnings.warn("Cache historique : seule la forme des données a été vérifiée.", stacklevel=2)
            metadata = {"legacy": True, "window": expected_window}
        else:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("window") != expected_window:
                raise ValueError("Version ou taille de fenêtre incohérente dans les métadonnées.")
            for key, value in (expected_metadata or {}).items():
                if metadata.get(key) != value:
                    raise ValueError(f"Métadonnée incompatible : {key}.")
    return {"X": X, "y": y, "patient_id": groups, "metadata": metadata}
