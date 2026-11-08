"""Nested HRV gradient-boosting tuning, with thresholds learned inside outer train."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils.metrics import per_patient_f1
from src.utils.splits import patient_kfold
from src.validation import choose_threshold


def suggest_hrv_params(trial: optuna.Trial) -> dict:
    return {
        "max_iter": trial.suggest_categorical("max_iter", [120, 240, 400]),
        "learning_rate": trial.suggest_float("learning_rate", .025, .15, log=True),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [7, 15, 31]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 160),
        "l2_regularization": trial.suggest_float("l2_regularization", .1, 30., log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def make_hrv_model(params: dict, seed: int = 42):
    # No shuffled window-level validation for early stopping.
    return HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)


def nested_hrv_oof(features: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                   n_splits: int = 5, inner_splits: int = 3, n_trials: int = 16,
                   seed: int = 42, suggest: Callable = suggest_hrv_params,
                   model_factory: Callable = make_hrv_model,
                   progress_callback: Callable | None = None,
                   fold_callback: Callable | None = None) -> dict:
    if n_trials < 1:
        raise ValueError("Au moins un essai est nécessaire.")
    if len(features) != len(y) or len(groups) != len(y):
        raise ValueError("Descripteurs, étiquettes et groupes non alignés.")
    scores = np.full(len(y), np.nan, dtype=np.float32)
    thresholds = np.full(len(y), np.nan, dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    details = []
    for k, (tr, te) in enumerate(patient_kfold(groups, y, n_splits=n_splits)):
        # Everything closed over by objective is restricted to the outer train.
        F, labels, patients = features[tr], y[tr], groups[tr]
        inner = list(patient_kfold(patients, labels, n_splits=inner_splits))
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=seed + 1000 * k, n_startup_trials=5))

        def objective(trial, train_features=F, train_labels=labels,
                      train_patients=patients, train_splits=inner):
            params = suggest(trial)
            oof = np.full(len(train_labels), np.nan, dtype=np.float32)
            for j, (fit_idx, val_idx) in enumerate(train_splits):
                model = model_factory(params, seed + j)
                model.fit(train_features[fit_idx], train_labels[fit_idx])
                oof[val_idx] = model.predict_proba(train_features[val_idx])[:, 1]
            threshold = choose_threshold(train_labels, oof, train_patients)
            trial.set_user_attr("inner_threshold", threshold)
            return per_patient_f1(train_labels, oof, train_patients, threshold)["mean_patient_f1"]

        def on_trial(study, trial, fold_id=k):
            if progress_callback:
                progress_callback({"fold": fold_id, "trial": trial.number, "value": trial.value,
                                   "params": trial.params, "attributes": trial.user_attrs})

        study.optimize(objective, n_trials=n_trials, callbacks=[on_trial])
        model = model_factory(study.best_params, seed + 1000 * k + 999)
        model.fit(features[tr], y[tr])
        scores[te] = model.predict_proba(features[te])[:, 1]
        thresholds[te] = study.best_trial.user_attrs["inner_threshold"]
        fold[te] = k
        detail = {
            "fold": k, "params": study.best_params,
            "threshold": study.best_trial.user_attrs["inner_threshold"],
            "selection_patients": np.unique(patients).tolist(),
            "test_patients": np.unique(groups[te]).tolist(),
            "inner_folds": [{"train_patients": np.unique(patients[a]).tolist(),
                             "validation_patients": np.unique(patients[b]).tolist()} for a, b in inner],
            "trials": [{"number": t.number, "value": t.value, "params": t.params,
                        "attributes": t.user_attrs} for t in study.trials],
        }
        details.append(detail)
        if fold_callback:
            fold_callback(k, model, detail, te, scores[te], thresholds[te])
    if not np.isfinite(scores).all() or not np.isfinite(thresholds).all() or (fold < 0).any():
        raise RuntimeError("Évaluation externe HRV incomplète.")
    return {"scores": scores, "thresholds": thresholds, "fold": fold, "details": details}
