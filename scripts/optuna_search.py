"""Recherche d'hyperparamètres par Optuna — version exécutable en ligne de commande.

Ce script reprend, à l'identique, l'espace de recherche et la fonction objectif
des notebooks ``03_ablation.ipynb`` (phase 3) et ``04_phase35_w60.ipynb``
(phase 3.5). Il existe pour que la recherche décrite à l'annexe du mémoire soit
rejouable sans ouvrir un notebook, et pour que l'espace de recherche vive dans
un fichier versionné plutôt que dans une cellule.

La fonction objectif maximise le **F1 moyen par patient** calculé sur les scores
hors-pli d'une validation croisée groupée — et non le F1 par fenêtre — afin
d'optimiser directement le critère qui sert ensuite à juger le modèle.

Exemples :
    python -m scripts.optuna_search --phase 35 --n-trials 25
    python -m scripts.optuna_search --phase 3 --n-trials 30 --window 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch
from optuna.samplers import TPESampler

from src.baselines.cv import crossval_torch_oof
from src.data.build import build_windows
from src.models.cnn_lstm import CNNLSTM
from src.utils.metrics import per_patient_f1
from src.utils.seed import set_seed

SEED = 42


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["3", "35"], default="35")
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--window", type=int, default=None,
                    help="Défaut : 30 pour la phase 3, 60 pour la phase 3.5.")
    ap.add_argument("--stride", type=int, default=None, help="Défaut : window / 2.")
    ap.add_argument("--n-splits-search", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Défaut : 8 pour la phase 3, 15 pour la phase 3.5.")
    ap.add_argument("--dataset", type=str, default="afdb")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    window = args.window or (30 if args.phase == "3" else 60)
    stride = args.stride or window // 2
    epochs = args.epochs or (8 if args.phase == "3" else 15)
    patience = 2 if args.phase == "3" else 4
    early_stop = "loss" if args.phase == "3" else "auroc"
    out_path = args.out or PROJECT_ROOT / "reports" / f"optuna_phase{args.phase}_rerun.json"

    set_seed(SEED)
    torch.set_num_threads(4)

    t0 = time.time()
    X, y, groups = build_windows(
        PROJECT_ROOT / "data" / "raw" / args.dataset,
        window=window, stride=stride, verbose=True,
    )
    print(f"[données] {args.dataset} w={window} : X={X.shape}, "
          f"charge de FA {y.mean():.3f} ({time.time() - t0:.1f}s)")
    print(f"[recherche] phase {args.phase}, {args.n_trials} essais, "
          f"{args.n_splits_search} plis, {epochs} époques par essai")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.phase)
        try:
            oof = crossval_torch_oof(
                model_factory=lambda: model_from_params(params, window),
                X=X, y=y, groups=groups,
                n_splits=args.n_splits_search, epochs=epochs, batch_size=512,
                lr=params["lr"], patience=patience, seed=SEED,
                weight_decay=params["weight_decay"], early_stop_metric=early_stop,
            )
        except Exception as exc:
            print(f"  essai {trial.number} abandonné : {exc}")
            return 0.0
        return per_patient_f1(y, oof.y_score, groups, threshold=0.5)["mean_patient_f1"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best.setdefault("weight_decay", 0.0)
    n_params = model_from_params(best, window).num_parameters()

    print(f"\n[résultat] meilleur F1/patient = {study.best_value:.4f}")
    print(f"[résultat] {n_params:,} paramètres")
    for k, v in best.items():
        print(f"    {k:14s} = {v}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "phase": args.phase,
        "window": window, "stride": stride,
        "n_trials": args.n_trials, "n_splits_search": args.n_splits_search,
        "epochs": epochs, "seed": SEED,
        "best_value_mean_patient_f1": float(study.best_value),
        "best_params": best,
        "n_params": int(n_params),
        "trials": [{"number": t.number, "value": t.value, "params": t.params}
                   for t in study.trials],
    }, indent=2))
    print(f"[sauvegarde] {out_path}")

    df_rows = [{"number": t.number, "value": t.value, **t.params} for t in study.trials]
    import pandas as pd
    pd.DataFrame(df_rows).to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[sauvegarde] {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
