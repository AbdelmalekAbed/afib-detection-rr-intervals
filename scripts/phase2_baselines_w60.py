"""Phase 2 — baselines recalculées à WINDOW=60 (configuration finale).

Toutes les baselines sont évaluées sous le MÊME découpage patient 5-fold que le
CNN-LSTM final (Phase 4, w=60), afin que la comparaison soit honnête et
cohérente. Ajoute un vrai gradient boosting (HistGradientBoosting de
scikit-learn) sur les mêmes descripteurs HRV que la forêt aléatoire.

Sortie : reports/phase2_baselines_w60.json + tableau imprimé.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score

from src.baselines.cv import crossval_sklearn_oof, crossval_torch_oof
from src.data.build import build_windows
from src.features.hrv import coefficient_of_variation, featurize_windows
from src.models.baselines import CNNOnly, LSTMOnly, build_hrv_baseline
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.utils.seed import set_seed
from src.utils.splits import patient_kfold

WINDOW, STRIDE, N_SPLITS, SEED = 60, 30, 5, 42


def build_afdb_windows(window: int, stride: int):
    return build_windows(PROJECT_ROOT / "data" / "raw" / "afdb", window=window, stride=stride)


def metric_block(y, score, groups, hard=None):
    out = dict(ranking_metrics(y, score))
    if hard is None:
        out.update(threshold_metrics(y, score, 0.5))
        out.update(per_patient_f1(y, score, groups, 0.5))
    else:
        tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
        out.update({
            "threshold": float("nan"),
            "f1": float(f1_score(y, hard, zero_division=0)),
            "sensitivity": float(tp / (tp + fn) if (tp + fn) else 0.0),
            "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
            "accuracy": float((hard == y).mean()),
        })
        s = []
        for pid in np.unique(groups):
            m = groups == pid
            if y[m].sum() == 0 or y[m].sum() == m.sum():
                continue
            s.append(f1_score(y[m], hard[m], zero_division=0))
        out.update({"mean_patient_f1": float(np.mean(s)) if s else 0.0, "n_patients_scored": len(s)})
    return out


def rule_oof(cv, y, groups, n_splits):
    """Règle métier : seuil sur le coefficient de variation, choisi par F1 sur le train de chaque fold."""
    pred = np.full(len(y), -1, dtype=np.int8)
    for tr, va in patient_kfold(groups, y, n_splits=n_splits):
        cand = np.linspace(np.quantile(cv[tr], 0.01), np.quantile(cv[tr], 0.99), 80)
        best_t, best_f1 = 0.0, -1.0
        for t_ in cand:
            p = (cv[tr] >= t_).astype(int)
            tp = ((p == 1) & (y[tr] == 1)).sum()
            fp = ((p == 1) & (y[tr] == 0)).sum()
            fn = ((p == 0) & (y[tr] == 1)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t_
        pred[va] = (cv[va] >= best_t).astype(int)
    return pred


def gb_factory():
    try:
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            class_weight="balanced", random_state=42,
        )
    except TypeError:  # older sklearn without class_weight
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=42,
        )


def main():
    set_seed(SEED)
    torch.set_num_threads(4)
    t0 = time.time()
    X, y, groups = build_afdb_windows(WINDOW, STRIDE)
    print(f"[data] w={WINDOW}: X={X.shape} afib={y.mean():.3f} patients={len(np.unique(groups))} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    F = featurize_windows(X)
    cv = np.array([coefficient_of_variation(r) for r in X], dtype=np.float32)
    print(f"[feat] HRV features {F.shape} ({time.time()-t0:.1f}s)", flush=True)

    results = {}
    # Scores hors-pli conservés pour toutes les analyses aval (intervalles de
    # confiance, étude du seuil, F1 par patient) — cf. scripts/phase6_analysis.py.
    oof_table: dict[str, np.ndarray] = {}
    fold_arr = np.full(len(y), -1, dtype=np.int8)
    for k, (_, va) in enumerate(patient_kfold(groups, y, n_splits=N_SPLITS)):
        fold_arr[va] = k

    pred = rule_oof(cv, y, groups, N_SPLITS)
    results["rule_cv"] = metric_block(y, cv, groups, hard=pred)
    oof_table["rule_cv"] = pred.astype(np.float32)  # décisions dures 0/1
    print(f"[rule] F1/patient={results['rule_cv']['mean_patient_f1']:.3f}", flush=True)

    oof = crossval_sklearn_oof(lambda: build_hrv_baseline(n_estimators=300), F, y, groups, N_SPLITS)
    results["hrv_rf"] = metric_block(y, oof.y_score, groups)
    oof_table["hrv_rf"] = oof.y_score
    print(f"[rf]   F1/patient={results['hrv_rf']['mean_patient_f1']:.3f}", flush=True)

    oof = crossval_sklearn_oof(gb_factory, F, y, groups, N_SPLITS)
    results["gb_hist"] = metric_block(y, oof.y_score, groups)
    oof_table["gb_hist"] = oof.y_score
    print(f"[gb]   F1/patient={results['gb_hist']['mean_patient_f1']:.3f}", flush=True)

    oof = crossval_torch_oof(
        lambda: CNNOnly(input_length=WINDOW, channels=(16, 32), kernel=5),
        X, y, groups, n_splits=N_SPLITS, epochs=12, batch_size=512, lr=1e-3, patience=3, seed=SEED,
    )
    results["cnn_only"] = metric_block(y, oof.y_score, groups)
    oof_table["cnn_only"] = oof.y_score
    print(f"[cnn]  F1/patient={results['cnn_only']['mean_patient_f1']:.3f}", flush=True)

    oof = crossval_torch_oof(
        lambda: LSTMOnly(hidden=32, layers=1, bidirectional=True),
        X, y, groups, n_splits=N_SPLITS, epochs=12, batch_size=512, lr=1e-3, patience=3, seed=SEED,
    )
    results["lstm_only"] = metric_block(y, oof.y_score, groups)
    oof_table["lstm_only"] = oof.y_score
    print(f"[lstm] F1/patient={results['lstm_only']['mean_patient_f1']:.3f}", flush=True)

    meta = {"window": WINDOW, "stride": STRIDE, "n_splits": N_SPLITS, "seed": SEED,
            "n_samples": int(len(y)), "n_patients": int(len(np.unique(groups))), "afib_rate": float(y.mean())}
    (PROJECT_ROOT / "reports" / "phase2_baselines_w60.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2))

    np.savez(
        PROJECT_ROOT / "reports" / "phase2_oof_scores.npz",
        y=y.astype(np.int8), groups=groups.astype(str), fold=fold_arr,
        **{k: v.astype(np.float32) for k, v in oof_table.items()},
    )
    print("[save] reports/phase2_oof_scores.npz")

    labels = {"rule_cv": "Règle métier (CV)", "hrv_rf": "Forêt aléatoire (HRV)",
              "gb_hist": "Gradient boosting", "cnn_only": "CNN 1D seul", "lstm_only": "LSTM seul"}
    print(f"\n{'Méthode':<24}{'F1/pat':>8}{'AUROC':>8}{'Sens':>8}{'Spéc':>8}{'F1win':>8}")
    for k, lab in labels.items():
        r = results[k]
        print(f"{lab:<24}{r['mean_patient_f1']:>8.3f}{r['auroc']:>8.3f}{r['sensitivity']:>8.3f}{r['specificity']:>8.3f}{r['f1']:>8.3f}")
    print("\n[done] saved reports/phase2_baselines_w60.json")


if __name__ == "__main__":
    main()
