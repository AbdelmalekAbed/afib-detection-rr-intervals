# Fiabilisation technique — protocole corrigé v1

Les résultats historiques de `reports/`, les checkpoints existants, les caches
`data/processed/afdb.npz` et `ltafdb.npz`, et le mémoire `report/` sont conservés.
Les nouvelles expériences écrivent dans `artifacts/`, ignoré par Git. Une baisse
de score sous un protocole indépendant ne signifie pas une destruction du travail
initial : elle distingue l'expérience historique de sa réévaluation.

## Corrections

- L'entropie HRV utilise les probabilités des cases de l'histogramme, en nats.
  Son identifiant est `hrv_v2_probability_entropy`. Les arbres historiques n'ont
  pas été réécrits ni requalifiés comme entraînés avec ce descripteur corrigé.
- Les graines sont fixées avant l'instanciation des modèles dans le moteur
  corrigé ; le mélange des lots utilise un générateur explicite.
- Pour chaque pli externe, la recherche Optuna, le choix de l'époque et celui du
  seuil utilisent uniquement les patients du train externe. Le modèle est ensuite
  réentraîné sur tous ces patients, pendant le nombre d'époques sélectionné.
  Les patients du test externe ne servent qu'à la prédiction et à l'évaluation.
- Le seuil est choisi sur les prédictions des plis internes. Les décisions des
  cinq tests externes sont agrégées avec un poids identique pour chaque patient
  notable, y compris lorsque les plis comptent des effectifs différents.
- Le `weight_decay` effectif est explicite dans les entraînements corrigés et la
  compression. Les archives historiques ont pu utiliser d'autres valeurs.
- Les caches nouvellement construits ont un nom incluant fenêtre, pas et
  stratégie, ainsi que des métadonnées et empreintes des annotations sources.
  Un cache ancien exige une lecture explicite et une vérification de dimension.
- Les répertoires de nouvelles expériences refusent de remplacer des sorties
  existantes. Les notebooks 02, 03 et 04 conservent leurs sorties affichées ; une
  réexécution écrit dans un dossier unique sous `artifacts/notebooks/`.
- L'étude de seuil des anciens scores est renommée *exploratoire*, car régler
  le seuil sur d'autres prédictions OOF ne constitue pas une validation imbriquée.
- L'export ONNX choisit explicitement l'exporteur TorchScript, l'opset 17 et des
  poids incorporés. Son test vérifie l'inférence avec un batch différent du batch
  d'export. L'exporteur et la quantification actuelle émettent des avertissements
  de dépréciation PyTorch ; leur migration reste un chantier distinct.

## Commandes corrigées

```bash
# Cinq baselines, entropie corrigée, sélection d'époque interne pour les réseaux.
.venv/bin/python -m scripts.phase2_baselines_w60

# Recherche imbriquée : 5 plis externes, 3 internes, 3 essais par pli par défaut.
# Trois essais constituent un budget de validation initial, pas une optimisation exhaustive.
.venv/bin/python -m src.train --config configs/train.yaml

# Examiner une expérience précise ; le dossier est indiqué par la commande train.
.venv/bin/python -m src.evaluate --config configs/train.yaml --in-dir <run>
.venv/bin/python -m src.evaluate --config configs/train.yaml --in-dir <run> --external ltafdb

# Expériences de compression et de transfert à architecture historique fixée.
# Elles isolent désormais l'arrêt anticipé du test, mais ne refont pas une
# recherche d'architecture imbriquée : cette limite est inscrite dans leurs JSON.
.venv/bin/python -m scripts.phase4_compression
.venv/bin/python -m scripts.phase5_cross_dataset

# Reconstruire des caches versionnés sans remplacer les anciens noms.
.venv/bin/python -m scripts.build_windows --config configs/data.yaml

# Tests et contrôle statique.
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check --no-cache src scripts tests
```

`--trials 0` désactive la recherche et conserve une configuration de développement
fixe, explicitement indiquée comme telle. Les résultats ne doivent pas être
présentés comme ceux d'une recherche imbriquée. La recherche autonome
`scripts.optuna_search` reste un outil de développement, pas un score de test.

`src.evaluate` recharge les paramètres propres à chaque pli : une recherche
imbriquée peut sélectionner des architectures différentes. L'évaluation externe
utilise la moyenne de leurs probabilités au seuil fixe configuré ; le seuil ne
doit pas être optimisé sur LTAFDB.

Le modèle source du script de transfert est maintenant réentraîné sur tout AFDB
après sélection du nombre d'époques sur un holdout AFDB. La référence interne
est celle du modèle de sélection, pas une performance sur des patients inconnus
du modèle source final. Le modèle source reste FP32 : un résultat de transfert
INT8 doit être mesuré séparément.

## Traces de la première validation

Le manifeste `artifacts/reliability_v1/historical_manifest.json` conserve les
empreintes de 162 fichiers existants. Les expériences réelles sont placées dans
`baselines/` et `cnn_lstm_nested/` sous ce même répertoire.

Les dossiers `smoke_compression/` et `smoke_transfer/` contiennent exclusivement
des tests d'intégration sur données synthétiques. Leurs métriques et leurs
latences ne doivent jamais être utilisées comme résultats scientifiques.

### Mesures du premier passage (8 novembre 2026)

Sur AFDB, 40 613 fenêtres et 23 patients notables :

| Modèle | F1 patient historique | F1 patient corrigé, seuil 0,5 | Seuil choisi en interne |
|---|---:|---:|---:|
| Règle CV | 0,6354 | 0,6354 | — |
| HRV + Random Forest | 0,8504 | 0,8741 | — |
| HRV + Gradient boosting | 0,8833 | 0,8788 | — |
| CNN seul | 0,6532 | 0,6623 | 0,6914 |
| LSTM seul | 0,6314 | 0,6006 | 0,5995 |
| CNN-LSTM, recherche imbriquée initiale | non directement comparable | 0,6495 | 0,6833 |

La comparaison des arbres isole la correction du descripteur HRV sous les mêmes
plis. Les réseaux changent aussi de procédure d'entraînement. Le CNN-LSTM explore
trois candidats par pli et retient des architectures différentes (5 377 à 91 905
paramètres) ; il ne s'agit pas d'une réévaluation contrôlée de l'ancien réseau
100k à architecture identique. Ce budget initial ne permet pas de conclure au
maximum de performance du protocole corrigé.

Les cinq checkpoints ont été rechargés et comparés à leurs prédictions OOF sur
128 fenêtres de chaque pli : écart maximal inférieur à `1,2e-7`.

Leur ensemble a ensuite été évalué sur LTAFDB : 299 727 fenêtres, 84 patients,
dont 62 notables, au seuil 0,5 fixé sans réglage externe. Résultats : F1 patient
**0,7034**, F1 fenêtre **0,9365**, AUROC **0,9665**. Cet ensemble de cinq modèles
diffère du modèle source unique de la phase 5 historique ; ses résultats ne
doivent pas être présentés comme la même expérience.

Les métriques précises et contrôles de rechargement figurent dans
`artifacts/reliability_v1/comparison.json`. La suite complète compte **45 tests
réussis**. Ruff et `git diff --check` passent. Les empreintes des 162 fichiers
historiques ont été revérifiées sans changement.

## Limites et suite convenue

Ces validations restent rétrospectives, sur des bases publiques déjà explorées.
Une séparation correcte des plis ne remplace pas une nouvelle cohorte clinique.
Le F1 principal conserve la convention historique d'exclusion des patients
mono-classe ; les métriques globales les incluent.

Après validation technique : optimiser avec un budget interne déclaré, refaire
les expériences nécessaires de compression et de transfert, puis stabiliser les
résultats. Étudier ensuite l'interface, notamment l'option React + FastAPI. Le
mémoire sera ajusté à partir des résultats stabilisés, puis la présentation de
soutenance sera préparée. Aucun chiffre du mémoire ne doit être remplacé sur la
seule base d'un test d'intégration ou d'un essai de recherche isolé.
