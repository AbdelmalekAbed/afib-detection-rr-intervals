# AFib CNN-LSTM

> Optimisation d'un modèle hybride **CNN-LSTM** pour la détection de la **fibrillation auriculaire** à partir des séries d'**intervalles RR**, avec un focus déploiement sur ressources contraintes (CPU / wearable).

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-red.svg)]()

---

## Problématique

La fibrillation auriculaire (AFib) est l'arythmie cardiaque la plus fréquente et un facteur de risque majeur d'AVC. Sa détection à partir des seuls **intervalles RR** (et non de l'ECG complet) permet un déploiement sur dispositifs portables (smartwatches, patchs). Ce projet conçoit, optimise et évalue un modèle hybride **CNN-LSTM** sur cet axe, en équilibrant :

1. **Performance** — F1 / AUROC / AUPRC patient-level
2. **Efficacité** — taille mémoire et latence CPU (quantization, pruning, ONNX)
3. **Robustesse** — généralisation cross-dataset (AFDB → LTAFDB)

## Datasets

| Dataset | Rôle |
|---|---|
| [MIT-BIH AFib DB](https://physionet.org/content/afdb/) | Entraînement + validation principale |
| [MIT-BIH NSR DB](https://physionet.org/content/nsrdb/) | Exemples négatifs supplémentaires |
| [Long-Term AFib DB](https://physionet.org/content/ltafdb/) | Évaluation externe (généralisation) |

Tous les datasets sont libres et téléchargés automatiquement via la commande `make data`.

## Quick start

```bash
# Cloner
git clone https://github.com/<user>/afib-cnn-lstm.git
cd afib-cnn-lstm

# Installer
make setup

# Télécharger et prétraiter les données
make data

# Entraîner le modèle de référence
make train

# Évaluer (rapport sur test interne + LTAFDB externe)
make eval

# Lancer la démo Streamlit
make demo
```

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

Voir [`/home/abdou/.claude/plans/je-veux-que-tu-synchronous-meadow.md`](../.claude/plans/je-veux-que-tu-synchronous-meadow.md) pour le planning détaillé en 7 phases (16 semaines).

- [x] Phase 0 — Cadrage & setup
- [ ] Phase 1 — Pipeline de données & EDA
- [ ] Phase 2 — Baselines
- [ ] Phase 3 — CNN-LSTM & optimisation Optuna
- [ ] Phase 4 — Compression (quantization, pruning, ONNX)
- [ ] Phase 5 — Robustesse cross-dataset
- [ ] Phase 6 — Démo Streamlit
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
