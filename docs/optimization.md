# Optimisation v2 — plan du passage

Ce passage suit la fiabilisation décrite dans `docs/reliability.md`. Les sorties
historiques et celles du premier passage restent intactes ; le mémoire et
l'interface ne sont pas modifiés pendant cette étape.

## Protocole fixé avant ce passage

- AFDB : mêmes fenêtres de 60 RR, pas de 30, mêmes cinq plis externes par patient.
- Trois plis internes, entièrement contenus dans le train de chaque pli externe.
- Critère principal de recherche : F1 moyen par patient après choix du seuil
  sur les prédictions internes. Le F1 au seuil 0,5 reste également rapporté.
- CNN-LSTM : huit essais par pli, seize époques au maximum, arrêt anticipé
  sur AUROC interne avec patience de trois époques, budget de 100 000 paramètres.
  Deux contrôles génériques sont proposés en premier avec une architecture
  identique, l'un en z-score et l'autre en RR centrés divisés par leur moyenne.
  Aucun de ces contrôles ne recharge les anciens meilleurs hyperparamètres.
- La recherche teste la normalisation et la pondération de la perte au même
  titre que les hyperparamètres du réseau. Le modèle reste un CNN-LSTM à entrée
  RR seule. La normalisation relative conserve le coefficient de variation ; le
  z-score supprime l'échelle de variabilité de chaque fenêtre.
- Gradient boosting HRV : seize essais par pli, sans validation aléatoire de
  fenêtres pour l'arrêt anticipé. Les nombres d'itérations, taux d'apprentissage,
  feuilles, effectifs minimaux, régularisation et pondération sont réglés en interne.
- LTAFDB ne sert ni au choix des hyperparamètres ni à celui des seuils.
- Pour LTAFDB, la moyenne des probabilités au seuil 0,5 reste le résultat externe
  principal. Un vote majoritaire utilisant les seuils internes AFDB de chaque
  modèle sera aussi rapporté séparément. Cette règle est fixée avant de calculer
  les prédictions externes de ce passage ; elle ne sera pas choisie selon le
  meilleur résultat LTAFDB observé.
- Les budgets restent limités : un résultat inférieur au passage précédent sera
  conservé et communiqué, sans sélection a posteriori des seuls plis favorables.

Les scripts sauvegardent les paramètres effectifs, le journal des essais,
les checkpoints par pli et une copie du code au début de l'expérience. Les
arguments CLI qui remplacent le YAML sont inscrits dans la configuration sauvegardée.

## Commandes

```bash
.venv/bin/python -m src.train --config configs/train_optimized.yaml \
  --windows artifacts/reliability_v1/windows/afdb_w60_s30_majority_v1.npz \
  --out-dir artifacts/optimization_v2/cnn_lstm

.venv/bin/python -m scripts.tune_hrv \
  --windows artifacts/reliability_v1/windows/afdb_w60_s30_majority_v1.npz \
  --out-dir artifacts/optimization_v2/hrv
```

Les expériences sont rétrospectives : AFDB et LTAFDB ont déjà été inspectées dans
le projet. Ce passage améliore la méthode de développement et l'indépendance
opérationnelle des plis ; il ne constitue pas une nouvelle validation clinique.

## Résultats du passage

Les 40 essais neuronaux (dont cinq écartés pour dépassement du budget de
paramètres) et les 80 essais HRV ont été exécutés. La normalisation relative a
été retenue par les cinq sélections internes du CNN-LSTM.

| Mesure | Passage de fiabilisation v1 | Optimisation v2 |
|---|---:|---:|
| CNN-LSTM, AFDB F1 patient au seuil 0,5 | 0,6495 | 0,8555 |
| CNN-LSTM, AFDB F1 patient calibré en interne | 0,6833 | 0,8653 |
| CNN-LSTM, AFDB F1 fenêtre au seuil 0,5 | 0,8553 | 0,9577 |
| CNN-LSTM, LTAFDB F1 patient, moyenne des scores au seuil 0,5 | 0,7034 | 0,8319 |
| CNN-LSTM, LTAFDB F1 fenêtre au seuil 0,5 | 0,9365 | 0,9662 |
| Gradient boosting HRV, AFDB F1 patient au seuil 0,5 | 0,8788 | 0,8926 |

Pour le vote utilisant les seuils gelés sur AFDB, le CNN-LSTM atteint 0,8582 sur
LTAFDB. Le gradient boosting optimisé atteint 0,8327 au seuil fixe et 0,8467 avec
ce vote. Les résultats des deux règles sont conservés ; la règle n'a pas été
choisie en fonction de son résultat sur LTAFDB.

La différence appariée sur AFDB entre CNN-LSTM optimisé calibré et pilote
calibré vaut +0,1820, IC 95 % bootstrap conditionnel [+0,0941 ; +0,2814], avec
21 patients améliorés sur 23. En revanche, la différence face au gradient
boosting de référence est de −0,0135, IC [−0,0666 ; +0,0220] : cette expérience
ne démontre pas une supériorité nette d'une famille sur l'autre.

Sur les huit patients à moins de 10 % de fenêtres positives, le F1 calibré du
CNN-LSTM passe de 0,4272 à 0,7148. C'est une observation descriptive de la même
cohorte, pas une garantie sur de nouveaux patients.

La recherche élargie change plusieurs facteurs : normalisation, réglages,
pondération de la perte et critère de sélection. Le gain global ne peut donc
pas être attribué quantitativement à la seule normalisation. Les deux contrôles
internes génériques utilisent toutefois la même architecture et les mêmes
graines, ce qui documente l'effet de la représentation au sein de l'apprentissage.

## Compression et contrat d'inférence

Le F1 patient AFDB calibré est de 0,86527 en FP32, 0,86513 en INT8 par lots et
0,86498 en INT8 fenêtre par fenêtre. ONNX FP32 reproduit le F1 de référence.
Le contrôle de rechargement INT8 compare des lots identiques : la quantification
dynamique des activations dépend de la composition du lot. L'écart maximal de
score entre lots de 512 et fenêtres isolées atteint 0,123 sur AFDB, même si le
F1 agrégé change très peu. Ces scores ne doivent pas être assimilés à des
probabilités cliniquement calibrées.

Les poids INT8 occupent en moyenne 58,66 Kio par membre, au maximum 122,40 Kio.
L'ensemble de cinq modèles occupe 293,29 Kio, contre 665,54 Kio pour les poids
FP32. Il faut distinguer cette somme de la taille d'un modèle unique. Ces
tailles excluent le runtime et la mémoire vive d'inférence.

La moyenne des latences médianes, pour une fenêtre sur CPU à quatre threads,
est de 0,271 ms en FP32, 0,409 ms en INT8 et 0,040 ms avec ONNX Runtime. Ce sont
des mesures locales sur le processeur de développement, pas sur microcontrôleur.

La vérification externe complète a également été exécutée pour chaque format :

| Inférence sur les 299 727 fenêtres LTAFDB | F1 patient au seuil fixe | F1 patient avec vote calibré |
|---|---:|---:|
| FP32, lots de 512 | 0,83194 | 0,85822 |
| INT8, lots de 512 | 0,83209 | 0,85830 |
| INT8, fenêtres isolées, un thread | 0,83214 | 0,85826 |
| ONNX FP32, lots de 512 | 0,83194 | 0,85822 |

Les 53 tests passent, ainsi que Ruff et le contrôle d'espaces Git. Les warnings
de dépréciation PyTorch de l'exporteur et de la quantification restent connus ;
ils ne sont pas présentés comme une exécution sans avertissement.

L'intégration dans la future interface devra charger **avec chaque modèle** sa
normalisation, ses paramètres et son seuil depuis le manifeste. Les nouveaux
checkpoints attendent la représentation sélectionnée ; l'ancienne démo applique
encore le prétraitement historique et utilise ses propres anciens artefacts.

Les fichiers détaillés sont générés sous `artifacts/optimization_v2/`. Le script
`scripts.finalize_optimization` vérifie les évaluations FP32, INT8 et ONNX ainsi
que l'intégrité des fichiers historiques avant de produire un manifeste local
et une archive de candidat. Il ne déploie rien et ne modifie pas le mémoire.
