# MIT-BIH Normal Sinus Rhythm Database (NSRDB)

> **Rôle dans le projet :** source supplémentaire d'**exemples négatifs** (rythme sinusal normal) pour enrichir la diversité des sujets sains et limiter l'overfitting.

## En une phrase

NSRDB contient des enregistrements **longs et propres** de personnes **sans aucune arythmie détectée** — c'est notre "référence du normal" pour aider le modèle à mieux apprendre la frontière entre AFib et rythme sain.

## Origine et contexte

- **Publié par** : MIT-BIH Arrhythmia Laboratory (Beth Israel Deaconess).
- **Année** : années 1980, sur PhysioNet depuis 1999.
- **Lien officiel** : <https://physionet.org/content/nsrdb/1.0.0/>
- **Identifiant PhysioNet** : `nsrdb`.

## Composition

| Caractéristique | Valeur |
|---|---|
| Nombre d'enregistrements | **18** |
| Sujets | 5 hommes (26-45 ans), 13 femmes (20-50 ans) |
| Durée par enregistrement | ~24 heures (Holter) |
| Fréquence d'échantillonnage | 128 Hz |
| Particularité | Aucune arythmie significative détectée chez ces sujets |

> ⚠️ La **fréquence d'échantillonnage diffère** d'AFDB (250 Hz). C'est sans impact direct sur les intervalles RR (qui sont des temps en secondes, indépendants de fs), mais à garder en tête si on voulait un jour mélanger des features dérivées du signal brut.

## Pourquoi on a besoin de ce dataset

Le problème avec [AFDB](afdb.md) seul : ses patients sont **tous connus pour faire de l'AFib**. Même quand ils sont en rythme sinusal, ils ne sont pas représentatifs d'une "vraie" population saine — ils ont d'autres particularités cardiaques sous-jacentes.

Si on entraîne uniquement sur AFDB, le modèle apprend à séparer :
- "AFib chez patient à risque" vs "non-AFib chez patient à risque"

… ce qui généralise mal à :
- "AFib chez patient grand public" vs "rythme normal chez personne saine"

**NSRDB comble ce trou** en fournissant des heures de **rythme sinusal vraiment normal**.

## Annotations

Les annotations NSRDB indiquent quasi-exclusivement :
- `N` (battement normal) à chaque position,
- aucun marqueur de rythme `(AFIB`.

Notre fonction `extract_rr_series` produit donc des séries où **toutes les étiquettes sont `0`** (non-AFib). C'est exactement ce qu'on veut : un grand pool d'exemples négatifs propres.

## Comment on l'utilise dans le projet

1. **Téléchargement** : `make data` → `data/raw/nsrdb/`.
2. **Extraction RR** : même pipeline qu'AFDB.
3. **Fusion à l'entraînement** : on **concatène** les fenêtres NSRDB aux fenêtres « non-AFib » d'AFDB pour le split d'entraînement.
4. **Patient grouping respecté** : chaque sujet NSRDB a son propre `patient_id` et n'apparaît jamais dans deux folds.
5. **Pas inclus dans le test externe** : NSRDB sert pendant la cross-validation, pas comme test de généralisation.

## Effet attendu dans les expériences

| Configuration | Effet attendu sur la métrique |
|---|---|
| AFDB seul | Surapprentissage des spécificités patients, AUC élevé mais fragile |
| AFDB + NSRDB | Spécificité (vrais négatifs) plus solide, légère baisse possible de sensibilité, **meilleure généralisation à LTAFDB** |

Ce sera un **point d'ablation à reporter** dans le rapport (« avec vs sans NSRDB »).

## Pièges à connaître

- **Sujets jeunes** (20-50 ans). La variabilité du rythme cardiaque (HRV) est naturellement plus élevée chez les jeunes que chez les âgés. Conséquence : nos exemples négatifs sont **plus variables** que ceux d'AFDB, ce qui peut rendre la frontière de décision plus difficile à apprendre — mais aussi plus robuste.
- **Pas d'ectopiques** chez la majorité des sujets : un modèle entraîné uniquement sur NSRDB ne saura pas distinguer une extrasystole isolée d'un début d'AFib. C'est pour ça qu'on **mixe** AFDB (qui contient les ectopiques) + NSRDB.
- **18 patients seulement** : utile en complément, mais on ne peut pas s'en contenter comme source unique de négatifs.

## En résumé

- **AFDB seul** : "AFib vs non-AFib chez patients arythmiques" → biais.
- **AFDB + NSRDB** : "AFib vs (non-AFib + cœur sain)" → frontière plus représentative.

NSRDB est l'**augmentation des négatifs** qu'on s'offre gratuitement pour rendre le modèle plus crédible cliniquement.
