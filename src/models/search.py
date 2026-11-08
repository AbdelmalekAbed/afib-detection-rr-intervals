"""Shared, predeclared CNN-LSTM search spaces; no report artifacts are read."""
from __future__ import annotations

import optuna

from src.models.cnn_lstm import CNNLSTM


def suggest_optimized_params(trial: optuna.Trial) -> dict:
    """Declared optimization v2: preserve RR variability and calibrate internally."""
    return {
        "normalization": trial.suggest_categorical("normalization", ["zscore_per_window", "relative_per_window"]),
        "use_pos_weight": trial.suggest_categorical("use_pos_weight", [False, True]),
        "n_cnn_blocks": trial.suggest_int("n_cnn_blocks", 2, 3),
        "cnn_first": trial.suggest_categorical("cnn_first", [8, 16]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "cnn_dropout": trial.suggest_float("cnn_dropout", .05, .35),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [16, 32, 48]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
        "head_hidden": trial.suggest_categorical("head_hidden", [16, 32]),
        "head_dropout": trial.suggest_float("head_dropout", .15, .5),
        "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True),
    }


def initial_optimization_candidates() -> list[dict]:
    """Generic paired controls, declared without reading historical best models."""
    common = {"use_pos_weight": False, "n_cnn_blocks": 2, "cnn_first": 16,
              "kernel_size": 5, "cnn_dropout": .2, "lstm_hidden": 32, "lstm_layers": 1,
              "head_hidden": 16, "head_dropout": .3, "lr": .001, "weight_decay": .0001}
    return [{**common, "normalization": mode} for mode in
            ("zscore_per_window", "relative_per_window")]


def suggest_params(trial: optuna.Trial, phase: str) -> dict:
    """Espace de recherche, identique à celui du notebook de la phase concernée.

    Phase 3 : exploration large, sans pénalisation L2.
    Phase 3.5 : plages resserrées vers les petites architectures, régularisation
    renforcée (dropout plus élevé, pénalisation L2 ajoutée) — l'objectif étant
    cette fois de viser une empreinte mémoire réduite.
    """
    if phase == "3":
        return {
            "n_cnn_blocks": trial.suggest_int("n_cnn_blocks", 2, 3),
            "cnn_first": trial.suggest_categorical("cnn_first", [8, 16, 24]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "cnn_dropout": trial.suggest_float("cnn_dropout", 0.1, 0.4),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [16, 32, 48]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "head_hidden": trial.suggest_categorical("head_hidden", [16, 32]),
            "head_dropout": trial.suggest_float("head_dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "weight_decay": 0.0,
        }
    return {
        "n_cnn_blocks": trial.suggest_int("n_cnn_blocks", 2, 3),
        "cnn_first": trial.suggest_categorical("cnn_first", [8, 16]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "cnn_dropout": trial.suggest_float("cnn_dropout", 0.2, 0.5),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [16, 32, 48]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
        "head_hidden": trial.suggest_categorical("head_hidden", [16, 32]),
        "head_dropout": trial.suggest_float("head_dropout", 0.3, 0.6),
        "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 3e-3, log=True),
    }


def model_from_params(p: dict, input_length: int) -> CNNLSTM:
    """Le nombre de filtres double à chaque bloc : (c, 2c) ou (c, 2c, 4c)."""
    c = p["cnn_first"]
    channels = (c, c * 2) if p["n_cnn_blocks"] == 2 else (c, c * 2, c * 4)
    return CNNLSTM(
        input_length=input_length,
        cnn_channels=channels,
        kernel_size=p["kernel_size"],
        pool_size=2,
        cnn_dropout=p["cnn_dropout"],
        batch_norm=True,
        lstm_hidden=p["lstm_hidden"],
        lstm_layers=p["lstm_layers"],
        bidirectional=True,
        lstm_dropout=0.2 if p["lstm_layers"] > 1 else 0.0,
        head_hidden=p["head_hidden"],
        head_dropout=p["head_dropout"],
    )
