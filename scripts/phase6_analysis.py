"""Phase 6 — Analyses statistiques transverses sur les scores hors-pli (OOF).

Ce script ne réentraîne aucun modèle : il relit les scores OOF déjà produits par
les phases 2, 4, 4b et 5, et répond à trois questions que les tableaux de
résultats bruts laissent ouvertes.

1. **Incertitude.** Un F1 par patient de 0,88 contre 0,72 est-il un écart réel ?
   On produit un intervalle de confiance à 95 % par bootstrap sur les patients
   (ré-échantillonnage des sujets, pas des fenêtres), plus l'écart-type entre
   plis.

2. **Sensibilité au seuil.** Toutes les métriques du mémoire sont calculées au
   seuil 0,5, alors que les modèles profonds sont entraînés avec une pondération
   de la classe positive qui décale leurs probabilités. On mesure ce que coûte
   ce choix : seuil optimal a posteriori, seuil réglé honnêtement sur les autres
   plis (imbriqué), et borne haute avec un seuil par patient.

3. **Origine du plafond.** On relie le F1 de chaque patient à sa charge de FA
   (proportion de fenêtres positives) pour montrer d'où vient l'écart entre le
   F1 par fenêtre et le F1 par patient.

Sorties :
  - reports/phase6_analysis.json
  - reports/phase6_per_patient.csv
  - reports/figures/19_confidence_intervals.{png,pdf}
  - reports/figures/20_threshold_sensitivity.{png,pdf}
  - reports/figures/21_f1_vs_afib_burden.{png,pdf}
  - reports/figures/22_window_vs_patient.{png,pdf}

Exécution depuis la racine du dépôt :
    python -m scripts.phase6_analysis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
N_BOOTSTRAP = 2000
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)

# Libellés français utilisés dans les figures et le mémoire.
LABELS = {
    "rule_cv": "Règle métier (CV)",
    "hrv_rf": "Forêt aléatoire (HRV)",
    "gb_hist": "Gradient boosting (HRV)",
    "cnn_only": "CNN 1D seul",
    "lstm_only": "LSTM seul",
    "cnn_lstm_100k": "CNN-LSTM 100 k",
    "cnn_lstm_30k": "CNN-LSTM 30 k",
    "cnn_lstm_30k_int8": "CNN-LSTM 30 k + INT8",
}


# --------------------------------------------------------------------------- #
# Briques de calcul
# --------------------------------------------------------------------------- #
def scorable_patients(y: np.ndarray, groups: np.ndarray) -> list:
    """Patients notables : ceux dont l'enregistrement contient les deux classes.

    Un patient intégralement en FA (ou intégralement en rythme sinusal) n'a pas
    de F1 défini de façon informative — il est exclu, comme dans
    ``src.utils.metrics.per_patient_f1``.
    """
    out = []
    for pid in np.unique(groups):
        m = groups == pid
        n_pos = int(y[m].sum())
        if 0 < n_pos < int(m.sum()):
            out.append(pid)
    return out


def per_patient_f1_vector(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, threshold: float, pids: list
) -> np.ndarray:
    """F1 de chaque patient notable, dans l'ordre de ``pids``."""
    out = np.empty(len(pids), dtype=np.float64)
    for i, pid in enumerate(pids):
        m = groups == pid
        out[i] = f1_score(y[m], (score[m] >= threshold).astype(int), zero_division=0)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    """IC 95 % par bootstrap percentile sur les *patients*.

    On ré-échantillonne les patients avec remise : c'est l'unité statistique
    pertinente ici, les fenêtres d'un même patient n'étant pas indépendantes.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "mean": float(values.mean()),
        "std_between_patients": float(values.std(ddof=1)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_patients": int(n),
    }


def per_fold_std(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, fold: np.ndarray, threshold: float
) -> dict:
    """Moyenne et écart-type du F1 par patient calculés pli par pli."""
    vals = []
    for k in np.unique(fold[fold >= 0]):
        m = fold == k
        pids = scorable_patients(y[m], groups[m])
        if not pids:
            continue
        vals.append(per_patient_f1_vector(y[m], score[m], groups[m], threshold, pids).mean())
    if not vals:
        return {"fold_mean": float("nan"), "fold_std": float("nan"), "n_folds": 0}
    return {
        "fold_mean": float(np.mean(vals)),
        "fold_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n_folds": len(vals),
    }


def threshold_study(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, fold: np.ndarray
) -> dict:
    """Quatre régimes de choix du seuil, du plus naïf au plus optimiste.

    - ``fixed_0.5``   : ce que rapporte le mémoire.
    - ``best_global`` : meilleur seuil unique choisi *sur les données de test*
      (optimiste, sert de référence haute pour un seuil global).
    - ``nested``      : seuil choisi sur les autres plis puis appliqué au pli
      courant — c'est la seule estimation honnête d'un gain réalisable.
    - ``per_patient_oracle`` : meilleur seuil pour chaque patient pris
      séparément. Non réalisable en pratique (il faut connaître la vérité
      terrain), mais borne supérieure de ce qu'un calibrage par sujet
      apporterait.
    """
    pids = scorable_patients(y, groups)
    curve = [per_patient_f1_vector(y, score, groups, t, pids).mean() for t in THRESHOLD_GRID]
    curve = np.asarray(curve)
    i_best = int(np.argmax(curve))

    nested = []
    for k in np.unique(fold[fold >= 0]):
        tr, te = fold != k, fold == k
        pids_tr = scorable_patients(y[tr], groups[tr])
        pids_te = scorable_patients(y[te], groups[te])
        if not pids_tr or not pids_te:
            continue
        c = [per_patient_f1_vector(y[tr], score[tr], groups[tr], t, pids_tr).mean()
             for t in THRESHOLD_GRID]
        t_star = float(THRESHOLD_GRID[int(np.argmax(c))])
        nested.append(
            per_patient_f1_vector(y[te], score[te], groups[te], t_star, pids_te).mean()
        )

    oracle = []
    for pid in pids:
        m = groups == pid
        oracle.append(
            max(f1_score(y[m], (score[m] >= t).astype(int), zero_division=0)
                for t in THRESHOLD_GRID)
        )

    return {
        "fixed_0.5": float(per_patient_f1_vector(y, score, groups, 0.5, pids).mean()),
        "best_global_threshold": float(THRESHOLD_GRID[i_best]),
        "best_global_f1": float(curve[i_best]),
        "nested_f1": float(np.mean(nested)) if nested else float("nan"),
        "per_patient_oracle_f1": float(np.mean(oracle)),
        "curve_thresholds": THRESHOLD_GRID.tolist(),
        "curve_f1": curve.tolist(),
    }


def burden_breakdown(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, threshold: float = 0.5
) -> dict:
    """F1 par patient, croisé avec la charge de FA (part de fenêtres positives)."""
    rows = []
    for pid in np.unique(groups):
        m = groups == pid
        n_pos, n_tot = int(y[m].sum()), int(m.sum())
        burden = n_pos / n_tot
        if 0 < n_pos < n_tot:
            f1 = float(f1_score(y[m], (score[m] >= threshold).astype(int), zero_division=0))
            scorable = True
        else:
            f1, scorable = float("nan"), False
        rows.append({"patient": str(pid), "afib_burden": burden,
                     "n_windows": n_tot, "f1": f1, "scorable": scorable})
    ok = [r for r in rows if r["scorable"]]
    low = [r["f1"] for r in ok if r["afib_burden"] < 0.10]
    high = [r["f1"] for r in ok if r["afib_burden"] >= 0.10]
    return {
        "rows": rows,
        "low_burden_lt10pct": {"n": len(low), "mean_f1": float(np.mean(low)) if low else float("nan")},
        "high_burden_ge10pct": {"n": len(high), "mean_f1": float(np.mean(high)) if high else float("nan")},
    }


# --------------------------------------------------------------------------- #
# Collecte des scores OOF de toutes les phases
# --------------------------------------------------------------------------- #
def load_all_afdb_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Rassemble les scores OOF AFDB des phases 2, 4 et 4b sur la même grille.

    Les trois phases partagent exactement les mêmes fenêtres (AFDB, W=60,
    pas=30) et le même découpage patient, donc leurs vecteurs OOF sont alignés
    index par index — c'est ce qui rend la comparaison légitime.
    """
    p2 = np.load(REPORTS / "phase2_oof_scores.npz", allow_pickle=True)
    y = p2["y"].astype(int)
    groups = p2["groups"].astype(str)
    fold = p2["fold"]

    scores = {k: p2[k].astype(np.float64) for k in
              ("rule_cv", "hrv_rf", "gb_hist", "cnn_only", "lstm_only")}

    p4 = np.load(REPORTS / "phase4_oof_scores.npz", allow_pickle=True)
    p4b = np.load(REPORTS / "phase4b_oof_scores.npz", allow_pickle=True)
    if not (np.array_equal(p4["y"].astype(int), y) and np.array_equal(p4b["y"].astype(int), y)):
        raise RuntimeError("les fenêtres des phases 2 / 4 / 4b ne coïncident pas")

    scores["cnn_lstm_100k"] = p4["fp32"].astype(np.float64)
    scores["cnn_lstm_30k"] = p4b["fp32"].astype(np.float64)
    scores["cnn_lstm_30k_int8"] = p4b["int8_dynamic"].astype(np.float64)
    return y, groups, fold, scores


def window_level_f1(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> float:
    return float(f1_score(y, (score >= threshold).astype(int), zero_division=0))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save_fig(fig, stem: str) -> None:
    """Enregistre chaque figure en PDF vectoriel (LaTeX) et en PNG 300 dpi."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    pdf_dir = PROJECT_ROOT / "report" / "figures"
    if pdf_dir.is_dir():
        fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight")


def fig_confidence_intervals(ci: dict, order: list) -> None:
    import matplotlib.pyplot as plt

    names = [LABELS[k] for k in order]
    means = [ci[k]["mean"] for k in order]
    lo = [ci[k]["mean"] - ci[k]["ci95_low"] for k in order]
    hi = [ci[k]["ci95_high"] - ci[k]["mean"] for k in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ypos = np.arange(len(order))
    colors = ["#d62728" if "HRV" in n else "#1f77b4" for n in names]
    ax.errorbar(means, ypos, xerr=[lo, hi], fmt="o", capsize=4, lw=1.4,
                color="#333333", ecolor="#888888", markersize=7, zorder=3)
    for i, (m, c) in enumerate(zip(means, colors, strict=True)):
        ax.plot(m, i, "o", color=c, markersize=7, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("$F_1$ moyen par patient (barres : IC 95 % par bootstrap sur les patients)")
    ax.set_title("Incertitude des comparaisons — AFDB, 23 patients notés")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 1)
    save_fig(fig, "19_confidence_intervals")
    plt.close(fig)


def fig_threshold(thr: dict, order: list) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in order:
        t = thr[k]
        ax.plot(t["curve_thresholds"], t["curve_f1"], lw=1.6, label=LABELS[k])
    ax.axvline(0.5, color="#d62728", ls="--", lw=1.2, label="seuil utilisé (0,5)")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("$F_1$ moyen par patient")
    ax.set_title("Sensibilité du $F_1$ par patient au seuil de décision")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    save_fig(fig, "20_threshold_sensitivity")
    plt.close(fig)


def fig_burden(burden: dict, key_a: str, key_b: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, color, marker in ((key_a, "#d62728", "o"), (key_b, "#1f77b4", "s")):
        rows = [r for r in burden[key]["rows"] if r["scorable"]]
        ax.scatter([r["afib_burden"] * 100 for r in rows], [r["f1"] for r in rows],
                   s=55, color=color, marker=marker, alpha=0.8, label=LABELS[key])
    ax.axvline(10, color="#555555", ls=":", lw=1.2)
    ax.text(10.4, 1.055, "charge de FA = 10 %", fontsize=8, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel("Charge de FA du patient (% de fenêtres positives, échelle log)")
    ax.set_ylabel("$F_1$ du patient (seuil 0,5)")
    ax.set_title("D'où vient le plafond : les patients à faible charge de FA")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.03, 1.12)
    save_fig(fig, "21_f1_vs_afib_burden")
    plt.close(fig)


def fig_window_vs_patient(win: dict, pat: dict, order: list) -> None:
    import matplotlib.pyplot as plt

    names = [LABELS[k] for k in order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w / 2, [win[k] for k in order], w, label="$F_1$ par fenêtre", color="#9ecae1")
    ax.bar(x + w / 2, [pat[k]["mean"] for k in order], w, label="$F_1$ par patient", color="#08519c")
    for i, k in enumerate(order):
        ax.annotate(f"−{100*(win[k]-pat[k]['mean']):.0f} pts",
                    (i, max(win[k], pat[k]["mean"]) + 0.02),
                    ha="center", fontsize=8, color="#b30000")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("$F_1$")
    ax.set_ylim(0, 1.12)
    ax.set_title("Effet du protocole d'évaluation : par fenêtre contre par patient")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "22_window_vs_patient")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    y, groups, fold, scores = load_all_afdb_scores()
    pids = scorable_patients(y, groups)
    print(f"[data] {len(y)} fenêtres, {len(np.unique(groups))} patients "
          f"({len(pids)} notables), charge de FA globale {y.mean():.3f}")

    order = ["rule_cv", "hrv_rf", "gb_hist", "cnn_only", "lstm_only",
             "cnn_lstm_100k", "cnn_lstm_30k", "cnn_lstm_30k_int8"]

    ci, folds, thr, burden, winf1 = {}, {}, {}, {}, {}
    for k in order:
        s = scores[k]
        vec = per_patient_f1_vector(y, s, groups, 0.5, pids)
        ci[k] = bootstrap_ci(vec)
        folds[k] = per_fold_std(y, s, groups, fold, 0.5)
        winf1[k] = window_level_f1(y, s)
        burden[k] = burden_breakdown(y, s, groups)
        # La règle métier ne produit que des décisions dures : l'étude du seuil
        # n'a pas de sens pour elle.
        if k != "rule_cv":
            thr[k] = threshold_study(y, s, groups, fold)
        print(f"[{k:20s}] F1/pat={ci[k]['mean']:.4f} "
              f"IC95=[{ci[k]['ci95_low']:.4f}, {ci[k]['ci95_high']:.4f}] "
              f"σ_plis={folds[k]['fold_std']:.4f} F1/fen={winf1[k]:.4f}")

    print("\n[seuil] gain réalisable (seuil réglé sur les autres plis)")
    for k, t in thr.items():
        print(f"  {LABELS[k]:24s} 0,5 → {t['fixed_0.5']:.4f} | "
              f"imbriqué {t['nested_f1']:.4f} | "
              f"global optimal {t['best_global_f1']:.4f} (t*={t['best_global_threshold']:.2f}) | "
              f"oracle/patient {t['per_patient_oracle_f1']:.4f}")

    print("\n[charge] F1 moyen selon la charge de FA du patient")
    for k in ("gb_hist", "cnn_lstm_100k"):
        b = burden[k]
        print(f"  {LABELS[k]:24s} <10 % : {b['low_burden_lt10pct']['mean_f1']:.3f} "
              f"(n={b['low_burden_lt10pct']['n']})  |  "
              f"≥10 % : {b['high_burden_ge10pct']['mean_f1']:.3f} "
              f"(n={b['high_burden_ge10pct']['n']})")

    # --- Écart entre les deux familles, avec son incertitude ------------------
    rng = np.random.default_rng(SEED)
    v_gb = per_patient_f1_vector(y, scores["gb_hist"], groups, 0.5, pids)
    v_dl = per_patient_f1_vector(y, scores["cnn_lstm_100k"], groups, 0.5, pids)
    diff = v_gb - v_dl
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOTSTRAP)])
    gap = {
        "mean_difference": float(diff.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "n_patients_gb_better": int((diff > 0).sum()),
        "n_patients": int(len(diff)),
    }
    print(f"\n[écart] gradient boosting − CNN-LSTM 100 k : "
          f"{gap['mean_difference']:+.4f} "
          f"IC95=[{gap['ci95_low']:+.4f}, {gap['ci95_high']:+.4f}] "
          f"({gap['n_patients_gb_better']}/{gap['n_patients']} patients en faveur du GB)")

    # --- Figures --------------------------------------------------------------
    fig_confidence_intervals(ci, order)
    fig_threshold(thr, ["gb_hist", "hrv_rf", "cnn_lstm_100k", "cnn_lstm_30k"])
    fig_burden(burden, "gb_hist", "cnn_lstm_100k")
    fig_window_vs_patient(winf1, ci, order)
    print(f"\n[save] figures 19 à 22 dans {FIGURES}/ (PNG + PDF)")

    # --- Persistance ----------------------------------------------------------
    out = {
        "config": {"threshold": 0.5, "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
                   "n_windows": int(len(y)), "n_patients": int(len(np.unique(groups))),
                   "n_patients_scored": len(pids), "afib_rate": float(y.mean())},
        "confidence_intervals": ci,
        "per_fold": folds,
        "window_level_f1": winf1,
        "threshold_study": {k: {kk: vv for kk, vv in v.items()
                                if not kk.startswith("curve_")} for k, v in thr.items()},
        "threshold_curves": {k: {"thresholds": v["curve_thresholds"], "f1": v["curve_f1"]}
                             for k, v in thr.items()},
        "burden_breakdown": {k: {"low_burden_lt10pct": v["low_burden_lt10pct"],
                                 "high_burden_ge10pct": v["high_burden_ge10pct"]}
                             for k, v in burden.items()},
        "gb_minus_cnnlstm": gap,
    }
    (REPORTS / "phase6_analysis.json").write_text(json.dumps(out, indent=2))
    print("[save] reports/phase6_analysis.json")

    import pandas as pd
    rows = []
    for k in order:
        for r in burden[k]["rows"]:
            rows.append({"method": k, "label": LABELS[k], **r})
    pd.DataFrame(rows).to_csv(REPORTS / "phase6_per_patient.csv", index=False)
    print("[save] reports/phase6_per_patient.csv")


if __name__ == "__main__":
    main()
