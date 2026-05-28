# AFib CNN-LSTM

> Optimisation d'un modèle hybride **CNN-LSTM** pour la détection de la **fibrillation auriculaire** à partir des séries d'**intervalles RR**, avec un focus déploiement sur ressources contraintes (CPU / wearable).

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-red.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-AFib%20Sandbox-FF4B4B?logo=streamlit)](app/README.md)

## Live Demo

An interactive Streamlit playground (**AFib Sandbox**, 5 pages) lets you scrub through
real patient timelines, swap compression variants live, and sketch RR series by hand.

- Hosted demo: **[TODO: Streamlit Cloud URL]**
- Local launch: `make demo` (or `streamlit run app/streamlit_app.py`)
- Walkthrough: [`app/README.md`](app/README.md)

---

## Problématique

La fibrillation auriculaire (AFib) est l'arythmie cardiaque la plus fréquente et un facteur de risque majeur d'AVC. Sa détection à partir des seuls **intervalles RR** (et non de l'ECG complet) permet un déploiement sur dispositifs portables (smartwatches, patchs). Ce projet conçoit, optimise et évalue un modèle hybride **CNN-LSTM** sur cet axe, en équilibrant :

1. **Performance** — F1 / AUROC / AUPRC patient-level
2. **Efficacité** — taille mémoire et latence CPU (quantization, pruning, ONNX)
3. **Robustesse** — généralisation cross-dataset (AFDB → LTAFDB)

## Datasets

| Dataset | Rôle | Fiche |
|---|---|---|
| [MIT-BIH AFib DB](https://physionet.org/content/afdb/) | Entraînement + validation principale | [`docs/datasets/afdb.md`](docs/datasets/afdb.md) |
| [MIT-BIH NSR DB](https://physionet.org/content/nsrdb/) | Exemples négatifs supplémentaires | [`docs/datasets/nsrdb.md`](docs/datasets/nsrdb.md) |
| [Long-Term AFib DB](https://physionet.org/content/ltafdb/) | Évaluation externe (généralisation) | [`docs/datasets/ltafdb.md`](docs/datasets/ltafdb.md) |

Tous les datasets sont libres et téléchargés automatiquement via la commande `make data`. Pour la motivation de chaque dataset et les pièges méthodologiques associés, voir [`docs/datasets/README.md`](docs/datasets/README.md).

## Quick start

```bash
# Cloner
git clone https://github.com/<user>/afib-cnn-lstm.git
cd afib-cnn-lstm

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
| 2 | Baselines HRV + RF, LR, MLP, XGBoost sous CV patient-level partagée | [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb) |
| 3 | CNN-LSTM + Optuna (archi ~100k params), ablation, courbes d'apprentissage | [`notebooks/03_ablation.ipynb`](notebooks/03_ablation.ipynb) |
| 3.5 | Régularisation focalisée fenêtre w=60 (archi ~30k params) | [`notebooks/04_phase35_w60.ipynb`](notebooks/04_phase35_w60.ipynb) |
| 4 / 4b | Compression — quantization, pruning, ONNX (archi 100k puis 30k → 87 KB INT8) | [`notebooks/05_phase4_compression.ipynb`](notebooks/05_phase4_compression.ipynb) |
| 5 | Robustesse cross-dataset AFDB → LTAFDB (zero-shot, scratch, fine-tune) | [`notebooks/06_phase5_cross_dataset.ipynb`](notebooks/06_phase5_cross_dataset.ipynb) |
| 6 | Démo interactive Streamlit (AFib Sandbox, 5 pages) | [`app/README.md`](app/README.md) |

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
├── tests/               # tests unitaires (pipeline de données critique)
├── reports/             # figures et rapport final
└── scripts/             # scripts utilitaires (téléchargement, prétraitement)
```

## État d'avancement

Projet structuré en 7 phases sur 16 semaines.

- [x] Phase 0 — Cadrage & setup
- [x] Phase 1 — Pipeline de données & EDA
- [x] Phase 2 — Baselines (4 modèles sous CV patient-level partagée)
- [x] Phase 3 — CNN-LSTM & optimisation Optuna (+ Phase 3.5 : régularisation focalisée w=60)
- [x] Phase 4 — Compression (quantization, pruning, ONNX) sur archi Phase 3
- [x] Phase 4b — Compression sur archi Phase 3.5 (~30k params) — atteint la cible <200 KB
- [x] Phase 5 — Robustesse cross-dataset (AFDB → LTAFDB)
- [x] Phase 6 — Démo Streamlit (AFib Sandbox, 5 pages interactives Plotly)
- [ ] Phase 7 — Rapport + polish

## Résultats clés

> *À compléter à la fin du projet.*

| Modèle | F1 (patient) | AUROC | Taille | Latence CPU |
|---|---|---|---|---|
| HRV + RF (baseline) | — | — | — | — |
| CNN seul | — | — | — | — |
| LSTM seul | — | — | — | — |
| **CNN-LSTM (notre)** | — | — | — | — |
| CNN-LSTM quantisé | — | — | — | — |

## Disclaimer médical

Ce projet est à but de **recherche et d'éducation uniquement**. Il ne constitue **pas un dispositif médical** et ne doit pas être utilisé pour un diagnostic ou une décision clinique.

## Licence

[MIT](LICENSE)
