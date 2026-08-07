"""Régénère les figures exploratoires (phase 1) sur les deux bases du projet.

Les figures d'origine venaient de ``notebooks/01_eda.ipynb``, exécuté à l'époque
où NSRDB était encore envisagée et où les fenêtres faisaient 30 battements. Deux
d'entre elles — le diagramme de Poincaré et les séries brutes — utilisaient un
enregistrement NSRDB comme exemple de rythme sinusal, alors qu'aucune expérience
rapportée n'emploie cette base.

Ce script les reconstruit à partir d'AFDB et LTAFDB uniquement, avec les
paramètres de fenêtrage réellement utilisés (60 battements, pas de 30). Le
panneau « rythme sinusal » s'appuie désormais sur un enregistrement AFDB à très
faible charge de FA — ce qui est plus représentatif du problème traité : le
rythme sinusal que le modèle doit reconnaître est celui d'un patient à risque,
pas celui d'un sujet sain.

Lancer depuis la racine du dépôt :
    python -m scripts.eda_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.build import build_windows
from src.data.loader import load_record_metadata
from src.data.rr_extract import RRSeries, clean_rr_series, extract_rr_series

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
FIG_DIR = REPO_ROOT / "reports" / "figures"

WINDOW, STRIDE = 60, 30
MIN_RR, MAX_RR = 0.3, 2.0

# Enregistrements servant d'exemples visuels (tous AFDB).
# 0,8 % de FA : parmi les enregistrements à très faible charge, c'est celui dont
# le diagramme de Poincaré montre le plus nettement le nuage allongé du rythme
# sinusal, sans être dominé par des amas d'extrasystoles.
REC_SINUS = "00735"
REC_AFIB = "07162"       # 100 % de FA
REC_TRANSITION = "05121"  # alternance nette sinusal / FA

CLASS_COLORS = {"NSR": "#1f77b4", "AFib": "#d62728"}
DS_COLORS = {"afdb": "#1b9e77", "ltafdb": "#d95f02"}
DS_LABELS = {"afdb": "AFDB", "ltafdb": "LTAFDB"}


def load_raw_rr(dataset: str, record: str) -> RRSeries:
    """Série RR battement par battement d'un enregistrement, sans fenêtrage."""
    meta = load_record_metadata(RAW_DIR / dataset, record)
    return clean_rr_series(extract_rr_series(meta), min_rr=MIN_RR, max_rr=MAX_RR)


def slice_seconds(series: RRSeries, max_seconds: float = 300.0):
    """Retourne (temps, rr, label) sur les ``max_seconds`` premières secondes."""
    t = np.cumsum(series.rr_seconds)
    keep = t <= max_seconds
    return t[keep], series.rr_seconds[keep], series.is_afib[keep]


def figure_dataset_summary(stats: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(stats)

    ax = axes[0]
    nsr = [stats[n]["n_windows"] - stats[n]["n_afib"] for n in names]
    afib = [stats[n]["n_afib"] for n in names]
    ax.bar(names, nsr, color=CLASS_COLORS["NSR"], label="NSR")
    ax.bar(names, afib, bottom=nsr, color=CLASS_COLORS["AFib"], label="AFib")
    for i, n in enumerate(names):
        ax.text(i, stats[n]["n_windows"] * 1.02, f"{stats[n]['n_windows']:,}".replace(",", " "),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(stats[n]["n_windows"] for n in names) * 1.15)
    ax.set_ylabel("Nombre de fenêtres")
    ax.set_xlabel("Base")
    ax.set_title(f"Volume de fenêtres de {WINDOW} battements (NSR vs FA)")
    ax.legend(title="Classe")

    ax = axes[1]
    rates = [stats[n]["afib_rate"] for n in names]
    ax.bar(names, rates, color="#444")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.set_ylabel("Taux de FA (proportion de fenêtres)")
    ax.set_xlabel("Base")
    ax.set_title("Prévalence de la FA par base")

    fig.suptitle("Résumé quantitatif des deux bases utilisées", y=1.03, fontsize=13)
    fig.savefig(FIG_DIR / "01_dataset_summary.png")
    plt.close(fig)


def figure_patient_rate_hist(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in stats:
        rates = stats[name]["patient_rates"]
        ax.hist(rates, bins=20, range=(0, 1), alpha=0.55,
                label=f"{DS_LABELS[name]} (n={len(rates)})", color=DS_COLORS[name])
    ax.set_xlabel("Proportion de fenêtres en FA, par patient")
    ax.set_ylabel("Nombre de patients")
    ax.set_title("Distribution de la charge de FA à l'échelle du patient")
    ax.legend()
    fig.savefig(FIG_DIR / "03b_patient_afib_rate_hist.png")
    plt.close(fig)


def figure_poincare() -> None:
    panels = [
        ("Rythme sinusal", REC_SINUS),
        ("FA permanente", REC_AFIB),
        ("Rythme mixte", REC_TRANSITION),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, rec) in zip(axes, panels, strict=True):
        series = load_raw_rr("afdb", rec)
        rr, lab = series.rr_seconds, series.is_afib[1:]
        x, y = rr[:-1], rr[1:]
        for cls, mask in (("NSR", lab == 0), ("AFib", lab == 1)):
            ax.scatter(x[mask], y[mask], s=4, alpha=0.35, color=CLASS_COLORS[cls], label=cls)
        lim = (0.25, 1.8)
        ax.plot(lim, lim, "k--", lw=0.6, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"RR$_n$ (s)")
        ax.set_ylabel(r"RR$_{n+1}$ (s)")
        ax.set_title(f"{title}\nAFDB / {rec} — {series.is_afib.mean():.1%} de FA")
        ax.legend(loc="upper left", markerscale=2)
    fig.suptitle("Diagrammes de Poincaré — signature visuelle de la FA", y=1.02, fontsize=13)
    fig.savefig(FIG_DIR / "04_poincare_afib_vs_nsr.png")
    plt.close(fig)


def figure_raw_series() -> None:
    panels = [
        ("Rythme sinusal", REC_SINUS),
        ("FA permanente", REC_AFIB),
        ("Transition sinusal / FA", REC_TRANSITION),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8))
    for ax, (title, rec) in zip(axes, panels, strict=True):
        series = load_raw_rr("afdb", rec)
        t, rr, lab = slice_seconds(series, max_seconds=300.0)
        ax.plot(t, rr, color="#444", lw=0.7, alpha=0.85)
        for cls, mask in (("NSR", lab == 0), ("AFib", lab == 1)):
            ax.scatter(t[mask], rr[mask], s=8, color=CLASS_COLORS[cls], label=cls)
        ax.set_ylim(0.25, 1.8)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("RR (s)")
        ax.set_title(f"{title} — AFDB / record {rec}")
        ax.legend(loc="upper right", markerscale=1.5)
    fig.suptitle("Séries RR brutes — trois régimes typiques (5 min)", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_raw_rr_examples.png")
    plt.close(fig)


def main() -> None:
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}
    for name in ("afdb", "ltafdb"):
        X, y, pid = build_windows(RAW_DIR / name, window=WINDOW, stride=STRIDE)
        stats[name] = {
            "n_windows": len(y),
            "n_afib": int(y.sum()),
            "afib_rate": float(y.mean()),
            "patient_rates": np.array([y[pid == p].mean() for p in np.unique(pid)]),
        }
        print(f"[{name}] {len(y):,} fenêtres · {y.mean():.1%} de FA · "
              f"{len(np.unique(pid))} patients", flush=True)

    figure_dataset_summary(stats)
    figure_patient_rate_hist(stats)
    figure_poincare()
    figure_raw_series()
    print(f"[ok] 4 figures régénérées dans {FIG_DIR}")


if __name__ == "__main__":
    main()
