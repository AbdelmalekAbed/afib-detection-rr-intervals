"""Phase 5 — Robustesse cross-dataset (AFDB → LTAFDB).

Question centrale : le plafond F1/patient ≈ 0.72 mesuré sur AFDB en Phases 3-4
est-il un plafond d'information (intrinsèque aux features RR) ou un artefact
d'apprentissage spécifique à AFDB ?

Pipeline :
  1. Construire les fenêtres AFDB et LTAFDB w=60 / stride=30 (mêmes paramètres
     que Phase 3.5 / Phase 4).
  2. Entraîner l'architecture Phase 3.5 (~30k params) sur l'ensemble AFDB
     → modèle "source" (gel de référence).
  3. Évaluer ce modèle source en **zero-shot** sur LTAFDB → quantifie le transfert.
  4. 5-fold patient-level CV sur LTAFDB avec la même archi, entraîné de zéro
     → mesure le plafond LTAFDB intrinsèque.
  5. Pour chaque fold LTAFDB, partir du modèle source AFDB et faire un
     fine-tuning court sur les autres folds → mesure le bénéfice du pré-entraînement.
  6. Sauvegarder ``reports/phase5_results.json``, ``phase5_cross_dataset.csv``,
     ``phase5_oof_scores.npz`` et les figures ``18_phase5_*.png``.

Sortie figures :
  - 18_phase5_f1_bar.png  : F1/patient pour les 4 configurations.
  - 18_phase5_perpatient_box.png : distribution par patient des F1, par config.
  - 18_phase5_finetune_gain.png : zero-shot vs fine-tuned, par fold LTAFDB.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.cv import zscore_per_window
from src.data.loader import list_records, load_record_metadata
from src.data.rr_extract import clean_rr_series, extract_rr_series
from src.data.windowing import build_windowed_dataset
from src.models.cnn_lstm import CNNLSTM
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.utils.seed import set_seed
from src.utils.splits import patient_kfold

WINDOW = 60
STRIDE = 30
SEED = 42
N_SPLITS = 5
EPOCHS_SOURCE = 12
EPOCHS_SCRATCH = 12
EPOCHS_FINETUNE = 6
BATCH_SIZE = 512
PATIENCE = 3


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_windows(dataset: str, window: int, stride: int):
    """Build w=60 windows from raw PhysioNet data for AFDB or LTAFDB."""
    dataset_dir = PROJECT_ROOT / "data" / "raw" / dataset
    series_by_patient = {}
    for rec in list_records(dataset_dir):
        try:
            meta = load_record_metadata(dataset_dir, rec)
            s = clean_rr_series(extract_rr_series(meta))
            if len(s.rr_seconds) >= window:
                series_by_patient[rec] = s
        except Exception:
            continue
    w = build_windowed_dataset(series_by_patient, window_size=window, stride=stride, label_strategy="majority")
    return w.X.astype(np.float32), w.y.astype(np.int8), np.asarray(w.patient_id)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(best: dict, input_length: int) -> CNNLSTM:
    if best["n_cnn_blocks"] == 2:
        channels = (best["cnn_first"], best["cnn_first"] * 2)
    else:
        channels = (best["cnn_first"], best["cnn_first"] * 2, best["cnn_first"] * 4)
    return CNNLSTM(
        input_length=input_length,
        cnn_channels=channels,
        kernel_size=best["kernel_size"],
        pool_size=2,
        cnn_dropout=best["cnn_dropout"],
        batch_norm=True,
        lstm_hidden=best["lstm_hidden"],
        lstm_layers=best["lstm_layers"],
        bidirectional=True,
        lstm_dropout=0.2 if best["lstm_layers"] > 1 else 0.0,
        head_hidden=best["head_hidden"],
        head_dropout=best["head_dropout"],
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray | None, y_va: np.ndarray | None,
    epochs: int, lr: float, weight_decay: float,
    batch_size: int, patience: int, pos_weight: float, seed: int,
) -> tuple[dict, np.ndarray | None]:
    """Train; if a val split is given, do early stopping on val loss and return the best state."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr.astype(np.float32))
    has_val = X_va is not None and y_va is not None
    if has_val:
        Xv = torch.from_numpy(X_va).float()
        yv = torch.from_numpy(y_va.astype(np.float32))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor([pos_weight], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_scores = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            crit(model(xb), yb).backward()
            optim.step()
        if not has_val:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            continue
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(crit(val_logits, yv).item())
            scores = torch.sigmoid(val_logits).cpu().numpy()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_scores = scores
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best_scores


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            out[i : i + batch_size] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def metric_block(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict:
    return {
        **ranking_metrics(y, scores),
        **threshold_metrics(y, scores, threshold=0.5),
        **per_patient_f1(y, scores, groups, threshold=0.5),
    }


def patient_f1_list(y: np.ndarray, scores: np.ndarray, groups: np.ndarray, thr: float = 0.5):
    out = {}
    for pid in np.unique(groups):
        m = groups == pid
        if m.sum() == 0 or y[m].sum() == 0 or y[m].sum() == m.sum():
            continue
        out[str(pid)] = float(f1_score(y[m], (scores[m] >= thr).astype(int), zero_division=0))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--epochs-source", type=int, default=EPOCHS_SOURCE)
    ap.add_argument("--epochs-scratch", type=int, default=EPOCHS_SCRATCH)
    ap.add_argument("--epochs-finetune", type=int, default=EPOCHS_FINETUNE)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports")
    ap.add_argument("--ckpt-dir", type=Path, default=PROJECT_ROOT / "reports" / "checkpoints")
    ap.add_argument("--params-json", type=Path,
                    default=PROJECT_ROOT / "reports" / "phase35_best_params.json")
    ap.add_argument("--out-prefix", type=str, default="phase5")
    ap.add_argument("--fig-prefix", type=str, default="18_phase5")
    args = ap.parse_args()

    set_seed(SEED)
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Best params (Phase 3.5 archi)
    raw = json.loads(args.params_json.read_text())
    best = raw["best_params"] if "best_params" in raw else raw
    weight_decay = float(best.get("weight_decay", 0.0))
    print(f"[params] {args.params_json}: {best}")

    # 2. Datasets
    t0 = time.time()
    Xa, ya, ga = build_windows("afdb", WINDOW, STRIDE)
    print(f"[data] AFDB w={WINDOW}: X={Xa.shape}, AFib={ya.mean():.3f}, "
          f"patients={len(np.unique(ga))} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    Xl, yl, gl = build_windows("ltafdb", WINDOW, STRIDE)
    print(f"[data] LTAFDB w={WINDOW}: X={Xl.shape}, AFib={yl.mean():.3f}, "
          f"patients={len(np.unique(gl))} ({time.time()-t0:.1f}s)")
    Xan = zscore_per_window(Xa)
    Xln = zscore_per_window(Xl)

    n_params = sum(p.numel() for p in make_model(best, WINDOW).parameters())
    print(f"[params] model parameter count: {n_params:,}")

    # 3. Source training : AFDB full → 1 modèle. On garde 1 fold pour early-stop interne.
    print("[source] training on AFDB (val = fold 0 for early stopping)")
    folds = list(patient_kfold(ga, ya, n_splits=args.n_splits))
    tr0, va0 = folds[0]
    n_pos = max(int(ya[tr0].sum()), 1)
    pw_src = (len(ya[tr0]) - n_pos) / n_pos
    src_model = make_model(best, WINDOW)
    t0 = time.time()
    src_state, _ = train(
        src_model, Xan[tr0], ya[tr0], Xan[va0], ya[va0],
        epochs=args.epochs_source, lr=best["lr"], weight_decay=weight_decay,
        batch_size=BATCH_SIZE, patience=PATIENCE, pos_weight=pw_src, seed=SEED,
    )
    src_model.load_state_dict(src_state)
    torch.save(src_state, args.ckpt_dir / "phase5_source_afdb.pt")
    print(f"  source trained in {time.time()-t0:.1f}s")

    # AFDB internal benchmark from this single model on fold 0 val
    scores_a_internal = predict(src_model, Xan[va0])
    afdb_internal = metric_block(ya[va0], scores_a_internal, ga[va0])
    print(f"[afdb_internal_fold0] F1/patient={afdb_internal['mean_patient_f1']:.4f} "
          f"AUROC={afdb_internal['auroc']:.4f}")

    # 4. Zero-shot LTAFDB
    t0 = time.time()
    scores_zs = predict(src_model, Xln)
    zs_metrics = metric_block(yl, scores_zs, gl)
    zs_perpat = patient_f1_list(yl, scores_zs, gl)
    print(f"[ltafdb_zero_shot] F1/patient={zs_metrics['mean_patient_f1']:.4f} "
          f"AUROC={zs_metrics['auroc']:.4f} F1={zs_metrics['f1']:.4f} "
          f"({time.time()-t0:.1f}s)")

    # 5. LTAFDB-only 5-fold from scratch — measure intrinsic LTAFDB ceiling
    print(f"[ltafdb_scratch] {args.n_splits}-fold patient-level OOF — from-scratch CNN-LSTM on LTAFDB")
    oof_scratch = np.full(len(yl), np.nan, dtype=np.float32)
    fold_arr = np.full(len(yl), -1, dtype=np.int8)
    for k, (tr, va) in enumerate(patient_kfold(gl, yl, n_splits=args.n_splits)):
        n_pos = max(int(yl[tr].sum()), 1)
        pw = (len(yl[tr]) - n_pos) / n_pos
        m = make_model(best, WINDOW)
        t0 = time.time()
        _, scores_va = train(
            m, Xln[tr], yl[tr], Xln[va], yl[va],
            epochs=args.epochs_scratch, lr=best["lr"], weight_decay=weight_decay,
            batch_size=BATCH_SIZE, patience=PATIENCE, pos_weight=pw, seed=SEED + k,
        )
        oof_scratch[va] = scores_va
        fold_arr[va] = k
        print(f"  fold {k}: trained in {time.time()-t0:.1f}s, val_size={len(va)}")
    scratch_metrics = metric_block(yl, oof_scratch, gl)
    scratch_perpat = patient_f1_list(yl, oof_scratch, gl)
    print(f"[ltafdb_scratch] F1/patient={scratch_metrics['mean_patient_f1']:.4f} "
          f"AUROC={scratch_metrics['auroc']:.4f}")

    # 6. Fine-tuning AFDB→LTAFDB : start from source, fine-tune on 4 folds, eval on 1
    print(f"[ltafdb_finetuned] {args.n_splits}-fold patient-level OOF — AFDB-pretrained + LTAFDB fine-tune")
    oof_ft = np.full(len(yl), np.nan, dtype=np.float32)
    finetune_lr = best["lr"] * 0.3
    for k, (tr, va) in enumerate(patient_kfold(gl, yl, n_splits=args.n_splits)):
        n_pos = max(int(yl[tr].sum()), 1)
        pw = (len(yl[tr]) - n_pos) / n_pos
        m = make_model(best, WINDOW)
        m.load_state_dict(copy.deepcopy(src_state))
        t0 = time.time()
        _, scores_va = train(
            m, Xln[tr], yl[tr], Xln[va], yl[va],
            epochs=args.epochs_finetune, lr=finetune_lr, weight_decay=weight_decay,
            batch_size=BATCH_SIZE, patience=PATIENCE, pos_weight=pw, seed=SEED + 100 + k,
        )
        oof_ft[va] = scores_va
        print(f"  fold {k}: fine-tuned in {time.time()-t0:.1f}s")
    ft_metrics = metric_block(yl, oof_ft, gl)
    ft_perpat = patient_f1_list(yl, oof_ft, gl)
    print(f"[ltafdb_finetuned] F1/patient={ft_metrics['mean_patient_f1']:.4f} "
          f"AUROC={ft_metrics['auroc']:.4f}")

    # 7. Persist results
    out_json = {
        "config": {
            "window": WINDOW, "stride": STRIDE, "n_splits": args.n_splits,
            "epochs_source": args.epochs_source, "epochs_scratch": args.epochs_scratch,
            "epochs_finetune": args.epochs_finetune, "batch_size": BATCH_SIZE,
            "patience": PATIENCE, "seed": SEED, "n_params": int(n_params),
        },
        "best_params": best,
        "datasets": {
            "afdb": {"n_samples": int(len(ya)), "n_patients": int(len(np.unique(ga))),
                      "afib_rate": float(ya.mean())},
            "ltafdb": {"n_samples": int(len(yl)), "n_patients": int(len(np.unique(gl))),
                        "afib_rate": float(yl.mean())},
        },
        "configurations": {
            "afdb_internal_fold0": afdb_internal,
            "ltafdb_zero_shot": zs_metrics,
            "ltafdb_scratch_5fold": scratch_metrics,
            "ltafdb_finetuned_5fold": ft_metrics,
        },
        "per_patient_f1": {
            "ltafdb_zero_shot": zs_perpat,
            "ltafdb_scratch_5fold": scratch_perpat,
            "ltafdb_finetuned_5fold": ft_perpat,
        },
    }
    (args.out_dir / f"{args.out_prefix}_results.json").write_text(json.dumps(out_json, indent=2))
    print(f"[save] reports/{args.out_prefix}_results.json")

    import pandas as pd
    rows = [
        {"config": "afdb_internal_fold0", **afdb_internal},
        {"config": "ltafdb_zero_shot", **zs_metrics},
        {"config": "ltafdb_scratch_5fold", **scratch_metrics},
        {"config": "ltafdb_finetuned_5fold", **ft_metrics},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / f"{args.out_prefix}_cross_dataset.csv", index=False)
    print(f"[save] reports/{args.out_prefix}_cross_dataset.csv\n{df.to_string(index=False)}")

    np.savez(
        args.out_dir / f"{args.out_prefix}_oof_scores.npz",
        y_ltafdb=yl.astype(np.int8), groups_ltafdb=gl.astype(str), fold=fold_arr,
        zero_shot=scores_zs.astype(np.float32),
        scratch=oof_scratch.astype(np.float32),
        finetuned=oof_ft.astype(np.float32),
    )
    print(f"[save] reports/{args.out_prefix}_oof_scores.npz")

    # 8. Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["AFDB\ninternal", "LTAFDB\nzero-shot", "LTAFDB\nscratch", "LTAFDB\nfine-tuned"]
    f1p = [afdb_internal["mean_patient_f1"], zs_metrics["mean_patient_f1"],
           scratch_metrics["mean_patient_f1"], ft_metrics["mean_patient_f1"]]
    auroc = [afdb_internal["auroc"], zs_metrics["auroc"],
             scratch_metrics["auroc"], ft_metrics["auroc"]]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, f1p, color=colors)
    axes[0].set_ylabel("F1 moyen par patient")
    axes[0].set_title("Phase 5 — F1/patient par configuration")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(f1p):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].bar(labels, auroc, color=colors)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Phase 5 — AUROC par configuration")
    axes[1].set_ylim(0.5, 1)
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(auroc):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.fig_prefix}_f1_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-patient distribution (zero-shot vs scratch vs finetuned)
    keys = sorted(set(zs_perpat) & set(scratch_perpat) & set(ft_perpat))
    arr_zs = np.array([zs_perpat[k] for k in keys])
    arr_sc = np.array([scratch_perpat[k] for k in keys])
    arr_ft = np.array([ft_perpat[k] for k in keys])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([arr_zs, arr_sc, arr_ft], labels=["zero-shot", "scratch", "fine-tuned"],
               showmeans=True, meanline=True,
               medianprops=dict(color="black"),
               meanprops=dict(color="red", linestyle="--"))
    ax.set_ylabel("F1 par patient (LTAFDB)")
    ax.set_title(f"Phase 5 — Distribution F1 par patient sur LTAFDB (n={len(keys)})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.fig_prefix}_perpatient_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fine-tune gain (paired delta)
    delta = arr_ft - arr_zs
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = np.argsort(arr_zs)
    ax.bar(range(len(keys)), arr_zs[order], color="#d62728", label="zero-shot", alpha=0.7)
    ax.bar(range(len(keys)), delta[order], bottom=arr_zs[order],
           color="#2ca02c", label="gain par fine-tune", alpha=0.85)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([keys[i] for i in order], rotation=70, fontsize=7)
    ax.set_ylabel("F1 par patient (LTAFDB)")
    ax.set_title("Phase 5 — gain du fine-tuning AFDB→LTAFDB par patient")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.fig_prefix}_finetune_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] figures in {fig_dir}/{args.fig_prefix}_*.png")
    print("\n[summary]")
    print(f"  AFDB internal       F1/patient = {afdb_internal['mean_patient_f1']:.4f}")
    print(f"  LTAFDB zero-shot    F1/patient = {zs_metrics['mean_patient_f1']:.4f}")
    print(f"  LTAFDB scratch      F1/patient = {scratch_metrics['mean_patient_f1']:.4f}")
    print(f"  LTAFDB fine-tuned   F1/patient = {ft_metrics['mean_patient_f1']:.4f}")


if __name__ == "__main__":
    main()
