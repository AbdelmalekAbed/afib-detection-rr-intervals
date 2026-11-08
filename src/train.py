"""Nested patient-level training. New runs never overwrite historical reports.

Optuna, epoch selection and threshold calibration use outer TRAINING patients
only. ``--trials 0`` disables hyperparameter tuning and is explicitly recorded
as a fixed development configuration, not a nested-search estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.artifacts import load_windows
from src.data.build import build_windows
from src.models.cnn_lstm import CNNLSTM
from src.models.search import (
    initial_optimization_candidates,
    model_from_params,
    suggest_optimized_params,
    suggest_params,
)
from src.utils.artifacts import (
    file_digest,
    new_run_dir,
    provenance,
    require_new_directory,
    snapshot_sources,
    write_json,
)
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.validation import nested_torch_oof


def load_config(path: Path) -> dict:
    """Charge configs/train.yaml et y fusionne les configurations référencées."""
    with path.open() as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parent.parent
    for key, rel in (cfg.get("defaults") or {}).items():
        sub_path = root / rel
        if not sub_path.is_file():
            raise FileNotFoundError(f"Configuration manquante : {sub_path}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", default="afdb")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--inner-splits", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--windows", type=Path, help="Archive RR versionnée, vérifiée avant entraînement.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    training, data = cfg["training"], cfg["data"]
    win = data["windowing"]
    protocol = cfg.get("protocol", {})
    n_trials = args.trials if args.trials is not None else protocol.get("trials", 3)
    if n_trials < 0:
        parser.error("--trials doit être positif ou nul.")
    for name in ("epochs", "n_splits", "inner_splits"):
        value = getattr(args, name)
        if value is not None and value < (1 if name == "epochs" else 2):
            parser.error(f"--{name.replace('_', '-')} est trop petit.")
    training["epochs"] = args.epochs if args.epochs is not None else training["epochs"]
    training["seed"] = args.seed if args.seed is not None else training["seed"]
    data["split"]["n_folds"] = args.n_splits if args.n_splits is not None else data["split"]["n_folds"]
    protocol["trials"] = n_trials
    protocol["inner_splits"] = args.inner_splits or protocol.get("inner_splits", 3)
    cfg["protocol"] = protocol
    monitor = {"val_loss": "loss", "val_auroc": "auroc"}.get(training["early_stopping"]["monitor"])
    if monitor is None:
        parser.error("Critère d'arrêt anticipé inconnu.")
    out_dir = (require_new_directory(args.out_dir) if args.out_dir else
               new_run_dir(Path(cfg["logging"]["out_dir"]), "cnn_lstm_nested_v1"))
    root = args.config.resolve().parent.parent
    run_provenance = provenance([args.config, root / "src" / "validation.py"])
    snapshot_sources(out_dir / "source_snapshot")
    clip = data["preprocessing"]["clip_rr_seconds"]
    if args.windows:
        cached = load_windows(args.windows, expected_window=win["window_size"], expected_metadata={
            "dataset": args.dataset, "stride": win["stride"], "label_strategy": win["label_strategy"],
            "min_rr": clip["min"], "max_rr": clip["max"]})
        X, y, groups = cached["X"], cached["y"], cached["patient_id"]
        for name, digest in cached["metadata"].get("raw_annotation_sha256", {}).items():
            if file_digest(root / data["paths"]["raw_dir"] / args.dataset / name) != digest:
                raise ValueError(f"Annotation source modifiée : {name}")
        run_provenance["windows_sha256"] = file_digest(args.windows)
    else:
        X, y, groups = build_windows(root / data["paths"]["raw_dir"] / args.dataset,
                                 window=win["window_size"], stride=win["stride"],
                                 label_strategy=win["label_strategy"], min_rr=clip["min"],
                                 max_rr=clip["max"], verbose=True)
    torch.set_num_threads(4)
    print(f"[données] {args.dataset} {X.shape}, {len(np.unique(groups))} patients", flush=True)

    def progress(event):
        with (out_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def checkpoint(k, state, details, indices, scores, thresholds):
        with (out_dir / f"fold{k}.pt").open("xb") as stream:
            torch.save(state, stream)
        write_json(out_dir / f"fold{k}_selection.json", details)
        with (out_dir / f"fold{k}_oof.npz").open("xb") as stream:
            np.savez(stream, indices=indices, scores=scores, thresholds=thresholds)
        print(f"[pli {k}] terminé : {details['normalization']}, "
              f"époques={details['epochs']}, seuil={details['threshold']:.2f}", flush=True)

    optimized = protocol.get("search_space") == "optimized_v2"
    result = nested_torch_oof(
        lambda: build_model(cfg["model"], X.shape[1]), X, y, groups,
        n_splits=data["split"]["n_folds"], inner_splits=protocol["inner_splits"],
        epochs=training["epochs"], batch_size=training["batch_size"],
        lr=training["optimizer"]["lr"], weight_decay=training["optimizer"]["weight_decay"],
        patience=training["early_stopping"]["patience"], seed=training["seed"],
        suggest_params=(suggest_optimized_params if optimized else
                        lambda trial: suggest_params(trial, "35")) if n_trials else None,
        parameterized_factory=lambda params: model_from_params(params, X.shape[1]),
        n_trials=n_trials,
        monitor=monitor, objective_metric=protocol.get("objective_metric", "fixed_patient_f1"),
        startup_trials=protocol.get("startup_trials", 10),
        initial_params=initial_optimization_candidates() if optimized and n_trials else None,
        parameter_budget=protocol.get("parameter_budget"),
        progress_callback=progress, fold_callback=checkpoint,
    )
    fixed_threshold = cfg["evaluation"]["decision_threshold"]
    fixed = {**ranking_metrics(y, result.y_score),
             **threshold_metrics(y, result.y_score, fixed_threshold),
             **per_patient_f1(y, result.y_score, groups, fixed_threshold)}
    # Binary decisions support the same patient weighting despite per-fold thresholds.
    calibrated_predictions = (result.y_score >= result.thresholds).astype(np.float32)
    calibrated = {**threshold_metrics(y, calibrated_predictions),
                  **per_patient_f1(y, calibrated_predictions, groups)}
    calibrated.pop("threshold")
    with (out_dir / "oof_scores.npz").open("xb") as stream:
        np.savez(stream, y=y, groups=groups.astype(str), fold=result.fold,
                 scores=result.y_score, thresholds=result.thresholds)
    write_json(out_dir / "metrics.json", {
        "protocol": "nested_patient_cv_v1", "dataset": args.dataset,
        "hyperparameter_selection": "inner_optuna" if n_trials else "fixed_development_config",
        "config": cfg, "window": X.shape[1], "folds": result.details,
        "data": {"n_samples": len(y), "n_patients": len(np.unique(groups)), "afib_rate": float(y.mean())},
        "metrics": fixed, "calibrated_metrics": calibrated,
        "provenance": run_provenance,
        "limitations": ["Retrospective development on previously inspected public datasets.",
                         "Internal CV is not a new prospective clinical validation."],
    })
    print(f"[résultat] F1 patient seuil fixe={fixed['mean_patient_f1']:.4f}, "
          f"seuil interne={calibrated['mean_patient_f1']:.4f}", flush=True)
    print(f"[sorties] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
