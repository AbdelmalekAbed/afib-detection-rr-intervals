"""Évaluation — `python -m src.evaluate --config configs/train.yaml`.

Deux modes complémentaires.

**Sans ``--external``** : relit les scores hors-pli produits par ``src.train``
et en tire le rapport complet — métriques globales, métriques par patient,
matrice de confusion, courbes ROC et précision-rappel.

**Avec ``--external <base>``** : recharge les poids sauvegardés, applique le
modèle à une base jamais vue (typiquement LTAFDB) et rapporte les mêmes
métriques. Les scores des plis sont moyennés, ce qui revient à évaluer un
ensemble des cinq modèles.

Sorties : ``<in-dir>/eval_<étiquette>.json`` et les figures correspondantes
dans ``reports/figures/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.baselines.cv import zscore_per_window
from src.data.build import build_windows
from src.train import build_model, load_config
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation du CNN-LSTM entraîné.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--external", type=str, default=None,
                   help="Base externe à évaluer (ex. ltafdb). Sans cette option, "
                        "les scores hors-pli existants sont analysés.")
    p.add_argument("--in-dir", type=Path, default=None,
                   help="Répertoire produit par src.train (défaut : déduit de la configuration).")
    p.add_argument("--fig-dir", type=Path, default=Path("reports/figures"))
    return p.parse_args()


def confusion_counts(y: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "VP": int(((y_pred == 1) & (y == 1)).sum()),
        "VN": int(((y_pred == 0) & (y == 0)).sum()),
        "FP": int(((y_pred == 1) & (y == 0)).sum()),
        "FN": int(((y_pred == 0) & (y == 1)).sum()),
    }


def patient_table(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                  threshold: float) -> list[dict]:
    """Détail par patient : charge de FA, effectif, F1 (ou None si non notable)."""
    from sklearn.metrics import f1_score

    rows = []
    for pid in np.unique(groups):
        m = groups == pid
        n_pos, n_tot = int(y[m].sum()), int(m.sum())
        scorable = 0 < n_pos < n_tot
        rows.append({
            "patient": str(pid),
            "n_windows": n_tot,
            "afib_burden": n_pos / n_tot,
            "f1": (float(f1_score(y[m], (scores[m] >= threshold).astype(int), zero_division=0))
                   if scorable else None),
        })
    return sorted(rows, key=lambda r: r["afib_burden"])


def plot_curves(y: np.ndarray, scores: np.ndarray, label: str, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y, scores)
    ax1.plot(fpr, tpr, lw=1.8, color="#1f77b4")
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
    ax1.set_xlabel("1 − spécificité")
    ax1.set_ylabel("sensibilité")
    ax1.set_title(f"Courbe ROC — {label}")
    ax1.grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(y, scores)
    ax2.plot(rec, prec, lw=1.8, color="#d62728")
    ax2.axhline(y.mean(), ls="--", lw=1, color="#888888", label="classifieur aléatoire")
    ax2.set_xlabel("rappel")
    ax2.set_ylabel("précision")
    ax2.set_title(f"Courbe précision-rappel — {label}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"eval_{label}_roc_pr.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {fig_dir}/eval_{label}_roc_pr.{{png,pdf}}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    threshold = cfg["evaluation"]["decision_threshold"]
    win_cfg = cfg["data"]["windowing"]
    pre_cfg = cfg["data"]["preprocessing"]

    in_dir = args.in_dir or (Path(cfg["logging"]["out_dir"]) / cfg["logging"]["experiment_name"])
    if not in_dir.is_dir():
        raise FileNotFoundError(
            f"{in_dir} est introuvable — lancer d'abord "
            f"`python -m src.train --config {args.config}`."
        )

    if args.external is None:
        label = "interne"
        npz = np.load(in_dir / "oof_scores.npz", allow_pickle=True)
        y, groups, scores = npz["y"].astype(int), npz["groups"].astype(str), npz["scores"]
        print(f"[interne] scores hors-pli relus depuis {in_dir}/oof_scores.npz")
    else:
        label = args.external
        root = args.config.resolve().parent.parent
        X, y, groups = build_windows(
            root / cfg["data"]["paths"]["raw_dir"] / args.external,
            window=win_cfg["window_size"], stride=win_cfg["stride"],
            label_strategy=win_cfg["label_strategy"],
            min_rr=pre_cfg["clip_rr_seconds"]["min"], max_rr=pre_cfg["clip_rr_seconds"]["max"],
            verbose=True,
        )
        y = y.astype(int)
        Xn = zscore_per_window(X)
        ckpts = sorted(in_dir.glob("fold*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"aucun poids fold*.pt dans {in_dir}")
        print(f"[externe] {args.external} : X={X.shape}, moyenne de {len(ckpts)} modèles")

        acc = np.zeros(len(y), dtype=np.float64)
        for ckpt in ckpts:
            model = build_model(cfg["model"], win_cfg["window_size"])
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                for i in range(0, len(Xn), 512):
                    xb = torch.from_numpy(Xn[i : i + 512]).float()
                    acc[i : i + 512] += torch.sigmoid(model(xb)).numpy()
        scores = (acc / len(ckpts)).astype(np.float32)

    metrics = {
        **ranking_metrics(y, scores),
        **threshold_metrics(y, scores, threshold),
        **per_patient_f1(y, scores, groups, threshold),
    }
    counts = confusion_counts(y, (scores >= threshold).astype(int))
    rows = patient_table(y, scores, groups, threshold)

    print(f"\n=== Rapport d'évaluation — {label} ===")
    print(f"  fenêtres             : {len(y)}  (charge de FA {y.mean():.3f})")
    print(f"  patients             : {len(np.unique(groups))} "
          f"dont {metrics['n_patients_scored']} notables")
    print(f"  VP={counts['VP']}  VN={counts['VN']}  FP={counts['FP']}  FN={counts['FN']}")
    print(f"  AUROC                : {metrics['auroc']:.4f}")
    print(f"  AUPRC                : {metrics['auprc']:.4f}")
    print(f"  F1 par fenêtre       : {metrics['f1']:.4f}")
    print(f"  sensibilité / spéc.  : {metrics['sensitivity']:.4f} / {metrics['specificity']:.4f}")
    print(f"  F1 moyen par patient : {metrics['mean_patient_f1']:.4f}")

    worst = [r for r in rows if r["f1"] is not None][:5]
    print("\n  5 patients les plus difficiles (charge de FA croissante) :")
    for r in worst:
        print(f"    {r['patient']:>8}  charge={r['afib_burden']:6.3f}  "
              f"fenêtres={r['n_windows']:5d}  F1={r['f1']:.3f}")

    out = {"label": label, "threshold": threshold, "confusion": counts,
           "metrics": metrics, "per_patient": rows}
    out_path = in_dir / f"eval_{label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[sauvegarde] {out_path}")

    plot_curves(y, scores, label, args.fig_dir)


if __name__ == "__main__":
    main()
