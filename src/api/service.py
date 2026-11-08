"""Verified local artifact access and frozen ONNX inference for the dashboard."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score

from src.data.artifacts import load_windows
from src.utils.artifacts import file_digest
from src.utils.metrics import threshold_metrics
from src.validation import preprocess_windows


class ArtifactError(RuntimeError):
    """The local candidate is absent or inconsistent."""


class ResearchService:
    def __init__(self, catalog_path: Path):
        self.catalog_path = catalog_path
        self._lock = Lock()
        try:
            self.catalog = json.loads(catalog_path.read_text())
            if self.catalog["schema_version"] != 1:
                raise ValueError("Schéma incompatible")
            self.root = Path(self.catalog["artifact_root"]).resolve()
            self._verified: set[str] = set()
        except (OSError, ValueError, KeyError) as exc:
            raise ArtifactError("Catalogue absent ou invalide. Exécuter make web-prepare.") from exc

    def path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or name not in self.catalog["references"]:
            raise ArtifactError("Artefact non référencé.")
        # Verify once before use; the application never writes these files.
        with self._lock:
            if name not in self._verified:
                try:
                    if file_digest(path) != self.catalog["references"][name]:
                        raise ArtifactError("Intégrité des artefacts : empreinte incompatible.")
                except OSError as exc:
                    raise ArtifactError("Artefact requis introuvable.") from exc
                self._verified.add(name)
        return path

    @lru_cache(maxsize=4)
    def scores(self, dataset: str, model: str) -> dict:
        entry = self.catalog["datasets"][dataset]["models"][model]
        with np.load(self.path(entry["scores"]), allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        data["groups"] = data["groups"].astype(str)
        n = len(data["y"])
        if data["y"].ndim != 1 or any(data[k].shape != (n,) for k in ("groups", "scores")):
            raise ArtifactError("Scores désalignés.")
        if not np.isfinite(data["scores"]).all() or not np.isin(data["y"], [0, 1]).all():
            raise ArtifactError("Scores invalides.")
        return data

    @lru_cache(maxsize=2)
    def windows(self, dataset: str) -> dict:
        data = load_windows(
            self.path(self.catalog["datasets"][dataset]["windows"]),
            expected_window=60,
            expected_metadata={
                "dataset": dataset,
                "stride": 30,
                "label_strategy": "majority",
                "min_rr": 0.3,
                "max_rr": 2.0,
            },
        )
        scores = self.scores(dataset, "cnn")
        if not np.array_equal(data["y"], scores["y"]) or not np.array_equal(
            data["patient_id"], scores["groups"]
        ):
            raise ArtifactError("Fenêtres et scores désalignés.")
        return data

    def summary(self) -> dict:
        for dataset in ("afdb", "ltafdb"):
            for model in ("cnn", "hrv"):
                self.scores(dataset, model)
        return {
            "version": "optimization_v2",
            "datasets": self.catalog["datasets"],
            "compression": self.catalog["compression"],
            "latency_configuration": self.catalog["latency_configuration"],
            "protocol": self.catalog["protocol"],
            "limitations": self.catalog["limitations"],
            "folds": [
                {
                    "fold": f["fold"],
                    "threshold": f["threshold"],
                    "normalization": f["normalization"],
                }
                for f in self.catalog["folds"]
            ],
        }

    @staticmethod
    def decisions(data: dict, policy: str) -> np.ndarray:
        if policy == "fixed":
            return data["scores"] >= 0.5
        return (
            data["scores"] >= data["thresholds"]
            if "thresholds" in data
            else data["calibrated_decisions"].astype(bool)
        )

    @lru_cache(maxsize=8)
    def patients(self, dataset: str, model: str, policy: str) -> list[dict]:
        data = self.scores(dataset, model)
        decisions = self.decisions(data, policy)
        rows = []
        for patient in np.unique(data["groups"]):
            mask = data["groups"] == patient
            y = data["y"][mask]
            mixed = len(np.unique(y)) == 2
            rows.append(
                {
                    "id": str(patient),
                    "n_windows": len(y),
                    "burden": float(y.mean()),
                    "f1": float(f1_score(y, decisions[mask], zero_division=0)) if mixed else None,
                    "scorable": mixed,
                }
            )
        return sorted(
            rows, key=lambda p: (p["f1"] is None, p["f1"] if p["f1"] is not None else 1, p["id"])
        )

    def trace(
        self, dataset: str, model: str, policy: str, patient: str, start: int, limit: int
    ) -> dict:
        data = self.scores(dataset, model)
        indices = np.flatnonzero(data["groups"] == patient)
        if not len(indices):
            raise KeyError(patient)
        y = data["y"][indices]
        decisions = self.decisions(data, policy)[indices]
        summary = {
            **threshold_metrics(y, decisions.astype(float)),
            "burden": float(y.mean()),
            "scorable": len(np.unique(y)) == 2,
        }
        summary["patient_f1"] = summary["f1"] if summary["scorable"] else None
        selected = indices[start : start + limit]
        rows = [
            {
                "index": start + i,
                "score": float(data["scores"][idx]),
                "label": int(data["y"][idx]),
                "decision": int(decisions[start + i]),
                "threshold": float(data["thresholds"][idx])
                if policy == "calibrated" and "thresholds" in data
                else 0.5,
            }
            for i, idx in enumerate(selected)
        ]
        return {
            "patient": patient,
            "total": len(indices),
            "start": start,
            "points": rows,
            "metrics": summary,
            "protocol": "patient_oof" if dataset == "afdb" else "external_ensemble",
            "threshold_line_is_decision_rule": dataset == "afdb" or policy == "fixed",
        }

    def window(self, dataset: str, patient: str, index: int) -> dict:
        data = self.windows(dataset)
        indices = np.flatnonzero(data["patient_id"] == patient)
        if not len(indices) or index >= len(indices):
            raise KeyError(patient)
        idx = indices[index]
        return {
            "patient": patient,
            "index": index,
            "rr": data["X"][idx].tolist(),
            "label": int(data["y"][idx]),
            "unit": "seconds",
        }

    @lru_cache(maxsize=1)
    def sessions(self) -> list:
        import onnxruntime as ort

        sessions = []
        for fold in self.catalog["folds"]:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            path = self.path(fold["checkpoints"]["onnx_fp32"]["path"])
            sessions.append(
                ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                )
            )
        return sessions

    def predict(self, rr: list[float]) -> dict:
        X = np.asarray(rr, dtype=np.float32)[None, :]
        if X.shape != (1, 60) or not np.isfinite(X).all() or np.any((X < 0.3) | (X > 2.0)):
            raise ValueError("Fournir exactement 60 RR finis, entre 0,3 et 2 secondes.")
        predictions = []
        for fold, session in zip(self.catalog["folds"], self.sessions(), strict=True):
            values = preprocess_windows(X, fold["normalization"])
            score = float(expit(session.run(None, {"rr_window": values})[0]).item())
            predictions.append(
                {
                    "fold": fold["fold"],
                    "score": score,
                    "threshold": fold["threshold"],
                    "decision": int(score >= fold["threshold"]),
                }
            )
        score = float(np.mean([p["score"] for p in predictions]))
        return {
            "score": score,
            "fixed_decision": int(score >= 0.5),
            "calibrated_vote": int(sum(p["decision"] for p in predictions) >= 3),
            "positive_votes": sum(p["decision"] for p in predictions),
            "members": predictions,
            "runtime": "ONNX FP32 · CPU · lot de 1",
            "scope": "Ensemble de cinq modèles AFDB ; démonstration pédagogique.",
        }
