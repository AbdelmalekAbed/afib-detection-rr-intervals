"""Patient-disjoint model/epoch/threshold selection inside each outer training fold.

Outer labels are used only after selection and refitting. Historical report
archives are deliberately not read here. A fixed architecture remains a
development configuration; only ``suggest_params`` enables nested tuning.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.artifacts import validate_arrays
from src.utils.metrics import per_patient_f1
from src.utils.seed import set_seed
from src.utils.splits import patient_kfold


@dataclass
class FitResult:
    state: dict
    best_epoch: int
    scores: np.ndarray | None


@dataclass
class NestedResult:
    y_score: np.ndarray
    fold: np.ndarray
    thresholds: np.ndarray
    states: list[dict]
    details: list[dict]


def normalize_windows(X: np.ndarray) -> np.ndarray:
    return ((X - X.mean(axis=1, keepdims=True)) /
            (X.std(axis=1, keepdims=True) + 1e-6)).astype(np.float32)


def preprocess_windows(X: np.ndarray, mode: str = "zscore_per_window") -> np.ndarray:
    """RR-only transforms; relative centering preserves the coefficient of variation."""
    if mode == "zscore_per_window":
        return normalize_windows(X)
    if mode == "relative_per_window":
        mean = X.mean(axis=1, keepdims=True)
        if not np.isfinite(X).all() or (mean <= 0).any():
            raise ValueError("La normalisation relative exige des RR de moyenne positive et finie.")
        return ((X - mean) / mean).astype(np.float32)
    raise ValueError(f"Normalisation inconnue : {mode}")


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size doit être positif.")
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).float()
            out[start:start + batch_size] = torch.sigmoid(model(xb)).numpy()
    return out


def fit_torch(factory: Callable[[], nn.Module], X: np.ndarray, y: np.ndarray, *,
              epochs: int, batch_size: int, lr: float, weight_decay: float, seed: int,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              patience: int = 3, monitor: str = "loss", use_pos_weight: bool = True) -> FitResult:
    """Fit from a seed set BEFORE construction; validation is an inner split only."""
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Paramètres d'entraînement invalides.")
    if monitor not in {"loss", "auroc"}:
        raise ValueError("monitor doit être loss ou auroc.")
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val et y_val doivent être fournis ensemble.")
    n_pos = int(y.sum())
    if not 0 < n_pos < len(y):
        raise ValueError("L'entraînement nécessite les deux classes.")
    set_seed(seed)
    model = factory()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                     torch.from_numpy(y.astype(np.float32))),
                        batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor([(len(y) - n_pos) / n_pos]) if use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    best = float("inf") if monitor == "loss" else -float("inf")
    best_state, best_scores, best_epoch, bad = None, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Perte d'entraînement non finie.")
            loss.backward()
            optimizer.step()
        if X_val is None:
            best_epoch = epoch
            continue
        model.eval()
        logits = []
        with torch.inference_mode():
            for start in range(0, len(X_val), batch_size):
                logits.append(model(torch.from_numpy(X_val[start:start + batch_size]).float()))
            val_logits = torch.cat(logits)
            value = float(criterion(val_logits, torch.from_numpy(y_val.astype(np.float32))))
            scores = torch.sigmoid(val_logits).numpy()
        if monitor == "auroc":
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_val)) < 2:
                raise ValueError("AUROC non définie pour une validation mono-classe.")
            value = float(roc_auc_score(y_val, scores))
        if not np.isfinite(value):
            raise FloatingPointError("Critère de validation non fini.")
        improved = value < best - 1e-4 if monitor == "loss" else value > best + 1e-4
        if improved:
            best, best_epoch, bad = value, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_scores = scores.copy()
        else:
            bad += 1
            if bad >= patience:
                break
    if X_val is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return FitResult(best_state, best_epoch, best_scores)


def choose_threshold(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                     grid: np.ndarray | None = None) -> float:
    """Optimize mean patient F1 using INNER predictions; prefer 0.5 in a tie."""
    if not (len(y) == len(scores) == len(groups)) or not np.isfinite(scores).all():
        raise ValueError("Scores de calibration non finis ou non alignés.")
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid, dtype=float)
    if not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("Grille de seuils invalide.")
    if grid.ndim != 1:
        raise ValueError("La grille doit être un vecteur.")
    by_patient = []
    for pid in np.unique(groups):
        mask = groups == pid
        truth = y[mask].astype(bool)
        n_pos = int(truth.sum())
        if not 0 < n_pos < len(truth):
            continue
        decisions = scores[mask][None, :] >= grid[:, None]
        tp = (decisions & truth[None, :]).sum(axis=1)
        fp = (decisions & ~truth[None, :]).sum(axis=1)
        fn = n_pos - tp
        denom = 2 * tp + fp + fn
        by_patient.append(np.divide(2 * tp, denom, out=np.zeros(len(grid)), where=denom > 0))
    values = np.mean(by_patient, axis=0) if by_patient else np.zeros(len(grid))
    tied = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))
    return float(grid[tied[np.argmin(np.abs(grid[tied] - 0.5))]])


def select_on_training(factory: Callable[[], nn.Module], X: np.ndarray, y: np.ndarray,
                       groups: np.ndarray, *, inner_splits: int, epochs: int,
                       batch_size: int, lr: float, weight_decay: float, patience: int,
                       seed: int, monitor: str = "loss", use_pos_weight: bool = True,
                       suggest_params: Callable | None = None,
                       parameterized_factory: Callable | None = None,
                       n_trials: int = 0, preprocess_candidates: bool = False,
                       default_normalization: str = "zscore_per_window",
                       objective_metric: str = "fixed_patient_f1",
                       startup_trials: int = 10, initial_params: list[dict] | None = None,
                       parameter_budget: int | None = None,
                       progress_callback: Callable[[dict], None] | None = None) -> dict:
    """All arguments are restricted to the outer TRAINING patients."""
    splits = list(patient_kfold(groups, y, n_splits=inner_splits))
    if objective_metric not in {"fixed_patient_f1", "calibrated_patient_f1"}:
        raise ValueError("Objectif de recherche inconnu.")
    inner_records = [{"train_patients": np.unique(groups[tr]).tolist(),
                      "validation_patients": np.unique(groups[va]).tolist()} for tr, va in splits]

    def candidate(params):
        candidate_factory = (lambda: parameterized_factory(params)) if params else factory
        transformed = (preprocess_windows(X, params.get("normalization", default_normalization))
                       if preprocess_candidates else X)
        scores = np.full(len(y), np.nan, dtype=np.float32)
        best_epochs = []
        for j, (tr, va) in enumerate(splits):
            fitted = fit_torch(candidate_factory, transformed[tr], y[tr], X_val=transformed[va], y_val=y[va],
                               epochs=epochs, batch_size=batch_size, lr=params.get("lr", lr),
                               weight_decay=params.get("weight_decay", weight_decay),
                               patience=patience, seed=seed + j, monitor=monitor,
                               use_pos_weight=params.get("use_pos_weight", use_pos_weight))
            scores[va] = fitted.scores
            best_epochs.append(fitted.best_epoch)
        return scores, best_epochs

    params, trials = {}, []
    if suggest_params is not None:
        import optuna
        if n_trials < 1 or parameterized_factory is None:
            raise ValueError("La recherche imbriquée exige des essais et une fabrique paramétrée.")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=startup_trials))
        for initial in initial_params or []:
            study.enqueue_trial(initial)
        cache = {}

        def objective(trial):
            p = suggest_params(trial)
            if parameter_budget is not None:
                set_seed(seed)
                n_parameters = sum(v.numel() for v in parameterized_factory(p).parameters())
                trial.set_user_attr("n_parameters", n_parameters)
                if n_parameters > parameter_budget:
                    raise optuna.TrialPruned("Parameter budget exceeded")
            scores, best_epochs = candidate(p)
            cache[trial.number] = (p, scores, best_epochs)
            threshold = choose_threshold(y, scores, groups) if objective_metric == "calibrated_patient_f1" else .5
            trial.set_user_attr("inner_threshold", threshold)
            return per_patient_f1(y, scores, groups, threshold)["mean_patient_f1"]

        def on_trial(study, trial):
            if progress_callback is not None:
                progress_callback({"trial": trial.number, "state": trial.state.name,
                                   "value": trial.value, "params": trial.params,
                                   "attributes": trial.user_attrs})

        study.optimize(objective, n_trials=n_trials, callbacks=[on_trial])
        params, inner_scores, best_epochs = cache[study.best_trial.number]
        trials = [{"number": t.number, "value": t.value, "params": t.params,
                   "state": t.state.name, "attributes": t.user_attrs} for t in study.trials]
    else:
        inner_scores, best_epochs = candidate({})
    return {
        "params": params, "epochs": max(1, int(np.median(best_epochs))),
        "threshold": choose_threshold(y, inner_scores, groups),
        "inner_best_epochs": best_epochs, "inner_folds": inner_records,
        "selection_patients": np.unique(groups).tolist(), "trials": trials,
        "objective_metric": objective_metric,
        "normalization": params.get("normalization", default_normalization),
        "use_pos_weight": params.get("use_pos_weight", use_pos_weight),
    }


def fit_selected(factory: Callable[[], nn.Module], X: np.ndarray, y: np.ndarray,
                 groups: np.ndarray, *, epochs: int, lr: float, weight_decay: float,
                 seed: int, batch_size: int = 512, patience: int = 3,
                 inner_splits: int = 3) -> tuple[FitResult, dict]:
    """Select inside training patients, then refit them all. X is normalized."""
    selection = select_on_training(factory, X, y, groups, inner_splits=inner_splits,
                                   epochs=epochs, batch_size=batch_size, lr=lr,
                                   weight_decay=weight_decay, patience=patience, seed=seed)
    fitted = fit_torch(factory, X, y, epochs=selection["epochs"], lr=lr,
                       weight_decay=weight_decay, seed=seed + 999, batch_size=batch_size)
    return fitted, selection


def nested_torch_oof(factory: Callable[[], nn.Module], X: np.ndarray, y: np.ndarray,
                     groups: np.ndarray, *, n_splits: int = 5, inner_splits: int = 3,
                     epochs: int = 12, batch_size: int = 512, lr: float = 1e-3,
                     weight_decay: float = 0.0, patience: int = 3, seed: int = 42,
                     monitor: str = "loss", use_pos_weight: bool = True,
                     suggest_params: Callable | None = None,
                     parameterized_factory: Callable | None = None,
                     n_trials: int = 0, default_normalization: str = "zscore_per_window",
                     objective_metric: str = "fixed_patient_f1", startup_trials: int = 10,
                     initial_params: list[dict] | None = None, parameter_budget: int | None = None,
                     progress_callback: Callable[[dict], None] | None = None,
                     fold_callback: Callable | None = None) -> NestedResult:
    validate_arrays(X, y, groups, X.shape[1])
    if inner_splits < 2:
        raise ValueError("Au moins deux plis internes sont nécessaires.")
    splits = list(patient_kfold(groups, y, n_splits=n_splits))
    if any(len(np.unique(groups[tr])) < inner_splits for tr, _ in splits):
        raise ValueError("Nombre de patients insuffisant pour les plis internes.")
    scores = np.full(len(y), np.nan, dtype=np.float32)
    thresholds = np.full(len(y), np.nan, dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    states, details = [], []
    for k, (tr, te) in enumerate(splits):
        fold_seed = seed + k * 1000
        def report_progress(event, fold_id=k):
            if progress_callback is not None:
                progress_callback({"fold": fold_id, **event})

        selection = select_on_training(
            factory, X[tr], y[tr], groups[tr], inner_splits=inner_splits,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            patience=patience, seed=fold_seed, monitor=monitor, use_pos_weight=use_pos_weight,
            suggest_params=suggest_params, parameterized_factory=parameterized_factory,
            n_trials=n_trials, preprocess_candidates=True,
            default_normalization=default_normalization, objective_metric=objective_metric,
            startup_trials=startup_trials, initial_params=initial_params,
            parameter_budget=parameter_budget, progress_callback=report_progress)
        params = selection["params"]
        Xn = preprocess_windows(X, selection["normalization"])
        selected_factory = (lambda p=params: parameterized_factory(p)) if params else factory
        # Refit ALL outer training patients, for the epoch count selected internally.
        fitted = fit_torch(selected_factory, Xn[tr], y[tr], epochs=selection["epochs"],
                           batch_size=batch_size, lr=params.get("lr", lr),
                           weight_decay=params.get("weight_decay", weight_decay),
                           seed=fold_seed + 999, use_pos_weight=selection["use_pos_weight"])
        set_seed(fold_seed + 999)
        model = selected_factory()
        model.load_state_dict(fitted.state)
        scores[te] = predict(model, Xn[te], batch_size)
        thresholds[te] = selection["threshold"]
        fold[te] = k
        states.append(fitted.state)
        details.append({"fold": k, "seed": fold_seed, **selection,
                        "test_patients": np.unique(groups[te]).tolist(),
                        "effective_lr": params.get("lr", lr),
                        "effective_weight_decay": params.get("weight_decay", weight_decay)})
        if fold_callback is not None:
            fold_callback(k, fitted.state, details[-1], te, scores[te], thresholds[te])
    if not np.isfinite(scores).all() or (fold < 0).any():
        raise RuntimeError("Prédictions externes incomplètes.")
    return NestedResult(scores, fold, thresholds, states, details)
