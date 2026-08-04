"""Entraînement du CNN-LSTM — `python -m src.train --config configs/train.yaml`.

Reproduit en ligne de commande la procédure des phases 3.5 / 4b : validation
croisée à 5 plis groupée par patient, une entropie croisée binaire pondérée par
le déséquilibre de chaque pli, un arrêt anticipé sur la perte de validation.

Sorties (dans ``artifacts/<experiment_name>/``) :
  - ``fold{k}.pt``      : poids de la meilleure époque de chaque pli
  - ``oof_scores.npz``  : scores hors-pli, étiquettes, patients, indice de pli
  - ``metrics.json``    : métriques agrégées + configuration utilisée

Le fichier ``oof_scores.npz`` est le point d'entrée de ``src.evaluate``.
La recherche d'hyperparamètres vit dans ``scripts/optuna_search.py``, qui
réutilise les mêmes briques.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.cv import zscore_per_window
from src.data.build import build_windows
from src.models.cnn_lstm import CNNLSTM
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.utils.seed import set_seed
from src.utils.splits import patient_kfold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement du CNN-LSTM sur intervalles RR.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="afdb",
                   help="Nom du répertoire sous data/raw/ (afdb, ltafdb...).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Écrase logging.out_dir du fichier de configuration.")
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    """Charge configs/train.yaml et y fusionne les configurations référencées."""
    with path.open() as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parent.parent
    for key, rel in (cfg.get("defaults") or {}).items():
        sub_path = root / rel
        if sub_path.is_file():
            with sub_path.open() as f:
                cfg[key] = yaml.safe_load(f)
    return cfg


def build_model(model_cfg: dict, input_length: int) -> CNNLSTM:
    """Instancie le CNN-LSTM décrit par configs/model_cnn_lstm.yaml."""
    m = model_cfg["model"]
    cnn, lstm, head = m["cnn"], m["lstm"], m["head"]
    channels = tuple(cnn["channels"][: cnn["n_blocks"]])
    return CNNLSTM(
        input_length=input_length,
        cnn_channels=channels,
        kernel_size=cnn["kernel_size"],
        pool_size=cnn["pool_size"],
        cnn_dropout=cnn["dropout"],
        batch_norm=cnn["batch_norm"],
        lstm_hidden=lstm["hidden_size"],
        lstm_layers=lstm["num_layers"],
        bidirectional=lstm["bidirectional"],
        lstm_dropout=lstm["dropout"] if lstm["num_layers"] > 1 else 0.0,
        head_hidden=head["hidden"],
        head_dropout=head["dropout"],
        out_classes=m["out_classes"],
    )


def train_fold(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int, batch_size: int, lr: float, weight_decay: float,
    patience: int, pos_weight: float, seed: int,
) -> tuple[dict, np.ndarray]:
    """Entraîne un pli et retourne (poids de la meilleure époque, scores de validation)."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr.astype(np.float32))
    Xv = torch.from_numpy(X_va).float()
    yv = torch.from_numpy(y_va.astype(np.float32))

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_scores: np.ndarray | None = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            crit(model(xb), yb).backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = float(crit(logits, yv).item())
            scores = torch.sigmoid(logits).cpu().numpy()

        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_scores = scores
        else:
            bad += 1
            if bad >= patience:
                print(f"    arrêt anticipé à l'époque {epoch + 1}")
                break

    return best_state, (best_scores if best_scores is not None else scores)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    tr_cfg = cfg["training"]
    win_cfg = cfg["data"]["windowing"]
    pre_cfg = cfg["data"]["preprocessing"]
    n_splits = args.n_splits or cfg["data"]["split"]["n_folds"]
    epochs = args.epochs or tr_cfg["epochs"]
    threshold = cfg["evaluation"]["decision_threshold"]

    set_seed(tr_cfg["seed"])
    torch.set_num_threads(4)

    out_dir = args.out_dir or Path(cfg["logging"]["out_dir"]) / cfg["logging"]["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    root = args.config.resolve().parent.parent
    dataset_dir = root / cfg["data"]["paths"]["raw_dir"] / args.dataset
    t0 = time.time()
    X, y, groups = build_windows(
        dataset_dir,
        window=win_cfg["window_size"],
        stride=win_cfg["stride"],
        label_strategy=win_cfg["label_strategy"],
        min_rr=pre_cfg["clip_rr_seconds"]["min"],
        max_rr=pre_cfg["clip_rr_seconds"]["max"],
        verbose=True,
    )
    print(f"[données] {args.dataset} : X={X.shape}, {len(np.unique(groups))} patients, "
          f"charge de FA {y.mean():.3f} ({time.time() - t0:.1f}s)")

    Xn = zscore_per_window(X)
    model_cfg = cfg["model"]
    n_params = build_model(model_cfg, win_cfg["window_size"]).num_parameters()
    print(f"[modèle] {n_params:,} paramètres entraînables")

    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_arr = np.full(len(y), -1, dtype=np.int8)
    for k, (tr, va) in enumerate(patient_kfold(groups, y, n_splits=n_splits)):
        t0 = time.time()
        n_pos = max(int(y[tr].sum()), 1)
        pos_weight = max(len(tr) - n_pos, 1) / n_pos
        model = build_model(model_cfg, win_cfg["window_size"])
        state, scores = train_fold(
            model, Xn[tr], y[tr], Xn[va], y[va],
            epochs=epochs, batch_size=tr_cfg["batch_size"],
            lr=tr_cfg["optimizer"]["lr"], weight_decay=tr_cfg["optimizer"]["weight_decay"],
            patience=tr_cfg["early_stopping"]["patience"],
            pos_weight=pos_weight, seed=tr_cfg["seed"] + k,
        )
        oof[va] = scores
        fold_arr[va] = k
        torch.save(state, out_dir / f"fold{k}.pt")
        print(f"  pli {k} : {len(va)} fenêtres de validation, "
              f"{len(np.unique(groups[va]))} patients ({time.time() - t0:.1f}s)")

    metrics = {
        **ranking_metrics(y, oof),
        **threshold_metrics(y, oof, threshold),
        **per_patient_f1(y, oof, groups, threshold),
    }
    print(f"\n[hors-pli] AUROC={metrics['auroc']:.4f}  F1/fenêtre={metrics['f1']:.4f}  "
          f"F1/patient={metrics['mean_patient_f1']:.4f} "
          f"({metrics['n_patients_scored']} patients notables)")

    np.savez(out_dir / "oof_scores.npz", y=y.astype(np.int8),
             groups=groups.astype(str), fold=fold_arr, scores=oof)
    (out_dir / "metrics.json").write_text(json.dumps({
        "dataset": args.dataset,
        "n_params": int(n_params),
        "n_splits": n_splits,
        "epochs": epochs,
        "threshold": threshold,
        "data": {"n_samples": int(len(y)), "n_patients": int(len(np.unique(groups))),
                 "afib_rate": float(y.mean())},
        "metrics": metrics,
        "config": {"training": tr_cfg, "windowing": win_cfg},
    }, indent=2))
    print(f"[sauvegarde] {out_dir}/ (poids, oof_scores.npz, metrics.json)")


if __name__ == "__main__":
    main()
