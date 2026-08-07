# AFib CNN-LSTM

> Optimisation d'un modèle hybride **CNN-LSTM** pour la détection de la **fibrillation auriculaire** à partir des séries d'**intervalles RR**, avec un focus déploiement sur ressources contraintes (CPU / wearable).

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Phase%207%20%E2%80%94%20Report-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-AFib%20Sandbox-FF4B4B?logo=streamlit)](app/README.md)
[![Report](https://img.shields.io/badge/Report-63%20pages-008080?logo=overleaf)](report/README.md)

## Live Demo

An interactive Streamlit playground (**AFib Sandbox**, 5 pages) lets you scrub through
real patient timelines, swap compression variants live, and sketch RR series by hand.

- Hosted demo: **[TODO: Streamlit Cloud URL]**
- Local launch: `make demo` (or `streamlit run app/streamlit_app.py`)
- Walkthrough: [`app/README.md`](app/README.md)

## Rapport (mémoire)

Le mémoire est rédigé en **LaTeX** (plan IMRAD strict, directives Sayadi : 12 pt
Times-like, interligne 1,5, marges 25 mm) : **63 pages**, 6 chapitres, 20 figures,
2 algorithmes en pseudo-code, 22 références au format IEEE, résumé FR + abstract EN.
Il compile sans aucun avertissement.

- Source LaTeX : [`report/`](report/) — `make report-pdf`
- Guide de compilation et état du document : [`report/README.md`](report/README.md)
- 32 figures disponibles dans [`report/figures/`](report/figures/)

---

## Problématique

La fibrillation auriculaire (AFib) est l'arythmie cardiaque la plus fréquente et un facteur de risque majeur d'AVC. Sa détection à partir des seuls **intervalles RR** (et non de l'ECG complet) permet un déploiement sur dispositifs portables (smartwatches, patchs). Ce projet conçoit, optimise et évalue un modèle hybride **CNN-LSTM** sur cet axe, en équilibrant :

1. **Performance** — F1 / AUROC / AUPRC patient-level
2. **Efficacité** — taille mémoire et latence CPU (quantization, pruning, ONNX)
3. **Robustesse** — généralisation cross-dataset (AFDB → LTAFDB)

## Datasets

| Dataset | Rôle | Patients | Fenêtres (60 battements) | FA |
|---|---|---|---|---|
| [MIT-BIH AFib DB](https://physionet.org/content/afdb/) | Entraînement + validation interne (`GroupKFold` par patient) | 25 | 40 613 | 43,6 % |
| [Long-Term AFib DB](https://physionet.org/content/ltafdb/) | Évaluation externe — aucun réglage sur cette base | 84 | 299 727 | 58,1 % |

Ces deux bases sont libres et téléchargées automatiquement via `make data`. Pour la justification du couple, le pipeline d'extraction RR, les bases écartées (NSRDB, CinC 2017) et les limites, voir [`docs/datasets.md`](docs/datasets.md).

## Quick start

```bash
# Cloner
git clone https://github.com/AbdelmalekAbed/afib-detection-rr-intervals.git
cd afib-detection-rr-intervals

# Option A — Makefile (uv + venv + extras)
make setup
make data
make train
make eval
make demo

# Option B — pip seul
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly                # pour la démo
streamlit run app/streamlit_app.py
```

Cibles `make` disponibles : `setup`, `data`, `train`, `eval`, `demo`, `test`, `lint`,
`format`, `clean` — voir `make help`.

### Reproductibilité

`requirements.txt` liste les contraintes hautes (`>=`) du `pyproject.toml`.
Pour reproduire **à l'identique** l'environnement utilisé pour générer les figures
et les checkpoints du dépôt, utiliser le lockfile :

```bash
pip install -r requirements-lock.txt
```

Le lockfile fige les versions exactes (Python 3.12, PyTorch CPU 2.x, ONNX
Runtime, scikit-learn, Optuna, etc.) et est régénéré à chaque verrouillage de
phase. Tous les résultats numériques du README et du rapport ont été obtenus
sous cet environnement.

## Phases & notebooks

| Phase | Sujet | Notebook |
|---|---|---|
| 1 | EDA — distributions RR, Poincaré, déséquilibre par patient | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| 2 | Baselines (règle métier, HRV+RF, HRV+gradient boosting, CNN seul, LSTM seul) sous CV patient-level partagée | [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb) |
| 3 | CNN-LSTM + Optuna (archi ~100k params), ablation, courbes d'apprentissage | [`notebooks/03_ablation.ipynb`](notebooks/03_ablation.ipynb) |
| 3.5 | Régularisation focalisée fenêtre w=60 (archi ~30k params) | [`notebooks/04_phase35_w60.ipynb`](notebooks/04_phase35_w60.ipynb) |
| 4 / 4b | Compression — quantization, pruning, ONNX (archi 100k puis 30k → 87 KB INT8) | [`notebooks/05_phase4_compression.ipynb`](notebooks/05_phase4_compression.ipynb) |
| 5 | Robustesse cross-dataset AFDB → LTAFDB (zero-shot, scratch, fine-tune) | [`notebooks/06_phase5_cross_dataset.ipynb`](notebooks/06_phase5_cross_dataset.ipynb) |
| 6 | Démo interactive Streamlit (AFib Sandbox, 5 pages) | [`app/README.md`](app/README.md) |
| 6 bis | Analyses statistiques — IC bootstrap, sensibilité au seuil, F1 vs charge de FA | [`scripts/phase6_analysis.py`](scripts/phase6_analysis.py) |
| 7 | Rapport LaTeX (IMRAD strict, conforme directives Sayadi) | [`report/README.md`](report/README.md) |

## Structure du repo

```
afib-cnn-lstm/
├── src/                 # code source
│   ├── data/            # loader, extraction RR, segmentation
│   ├── features/        # features HRV pour baselines
│   ├── models/          # CNN-LSTM, baselines, compression
│   ├── utils/           # reproductibilité, logging
│   ├── train.py
│   └── evaluate.py
├── configs/             # YAML (data, modèle, training)
├── notebooks/           # EDA, ablation, compression
├── app/streamlit_app.py # démo
├── tests/               # 29 tests unitaires (découpage patient, étiquetage RR, métriques)
├── reports/             # résultats JSON/CSV, scores hors-pli, figures
├── report/              # source LaTeX du mémoire
└── scripts/             # téléchargement, phases 2 à 6, recherche Optuna
```

## État d'avancement

Projet structuré en 7 phases sur 16 semaines.

- [x] Phase 0 — Cadrage & setup
- [x] Phase 1 — Pipeline de données & EDA
- [x] Phase 2 — Baselines (5 méthodes sous CV patient-level partagée)
- [x] Phase 3 — CNN-LSTM & optimisation Optuna (+ Phase 3.5 : régularisation focalisée w=60)
- [x] Phase 4 — Compression (quantization, pruning, ONNX) sur archi Phase 3
- [x] Phase 4b — Compression sur archi Phase 3.5 (~30k params) — atteint la cible <200 KB
- [x] Phase 5 — Robustesse cross-dataset (AFDB → LTAFDB)
- [x] Phase 6 — Démo Streamlit (AFib Sandbox, 5 pages interactives Plotly)
- [x] Phase 6 bis — Analyses statistiques (IC bootstrap, seuil, charge de FA)
- [~] Phase 7 — Rapport + polish (en cours)
  - [x] Mémoire LaTeX complet (63 pages, 6 chapitres IMRAD, 22 références IEEE)
  - [x] Compilation propre : 0 avertissement, 0 référence non résolue
  - [x] 20 figures insérées + 2 algorithmes en pseudo-code
  - [x] Tous les chiffres du mémoire vérifiés contre les JSON de `reports/`
  - [x] `src/train.py` et `src/evaluate.py` réellement implémentés
  - [x] `scripts/optuna_search.py` — recherche rejouable hors notebook
  - [x] 29 tests unitaires (dont l'étiquetage RR, non couvert auparavant)
  - [ ] Membres du jury à renseigner sur la page de garde
  - [ ] Déploiement Streamlit Cloud (URL publique)
  - [ ] Ré-export vectoriel des 28 figures issues des notebooks

## Résultats clés

Toutes les valeurs ci-dessous sont extraites des fichiers JSON versionnés dans
[`reports/`](reports/) — jamais recopiées à la main. Évaluation en validation
croisée 5 plis groupée par patient sur AFDB, fenêtre de 60 battements,
seuil de décision 0,5.

| Modèle | F1 **patient** | F1 fenêtre | AUROC | Taille | Source |
|---|---|---|---|---|---|
| Règle métier (coefficient de variation) | 0.635 | 0.814 | 0.906 | — | `phase2_baselines_w60.json` |
| HRV + Random Forest                     | 0.850 | 0.933 | 0.988 | — | `phase2_baselines_w60.json` |
| **HRV + Gradient boosting**             | **0.883** | 0.953 | 0.992 | — | `phase2_baselines_w60.json` |
| CNN 1D seul                             | 0.653 | 0.853 | 0.930 | — | `phase2_baselines_w60.json` |
| LSTM seul                               | 0.631 | 0.811 | 0.922 | — | `phase2_baselines_w60.json` |
| CNN-LSTM (Phase 3, 100 k params)        | 0.717 | 0.894 | 0.971 | 403 KB | `phase4_results.json` |
| CNN-LSTM (Phase 3.5, 30 k params)       | 0.680 | 0.881 | 0.959 | 133 KB | `phase4b_results.json` |
| CNN-LSTM 30 k + INT8 (Phase 4b)         | 0.680 | 0.881 | 0.959 | **87 KB** | `phase4b_results.json` |
| CNN-LSTM 30 k → LTAFDB zero-shot (Ph. 5) | 0.713 | 0.945 | 0.970 | 87 KB | `phase5_results.json` |

**Trois lectures à retenir.**

1. **Les descripteurs HRV battent le deep learning.** Sous évaluation par
   patient, le gradient boosting sur 8 descripteurs HRV classiques atteint 0.883
   contre 0.717 pour le meilleur CNN-LSTM. Sur 23 patients notés, des
   descripteurs conçus depuis vingt ans pour la FA encapsulent une connaissance
   a priori que le réseau doit réapprendre à partir de très peu de sujets.

2. **L'écart avec la littérature est un artefact de protocole.** Les mêmes
   modèles mesurés *par fenêtre* donnent 0.953 (GB) et 0.894 (CNN-LSTM), soit
   exactement l'ordre de grandeur des articles publiés. C'est l'agrégation par
   patient qui fait chuter le score, pas l'architecture.

3. **Le « plafond » est porté par une poignée de patients.** Les 8 patients dont
   moins de 10 % de l'enregistrement est en FA ont un F1 moyen de 0.36 ; les 15
   autres atteignent 0.85. Quelques faux positifs suffisent à effondrer la
   précision quand la classe positive est rare. Analyse détaillée :
   [`scripts/phase6_analysis.py`](scripts/phase6_analysis.py).

L'intérêt du modèle profond se situe donc ailleurs que dans le F1 brut :
traitement de bout en bout sans extraction manuelle de descripteurs, 87 KB après
quantification INT8 sans aucune perte, et F1 patient préservé en transfert
zero-shot vers LTAFDB.

## Disclaimer médical

Ce projet est à but de **recherche et d'éducation uniquement**. Il ne constitue **pas un dispositif médical** et ne doit pas être utilisé pour un diagnostic ou une décision clinique.

## Licence

[MIT](LICENSE)
