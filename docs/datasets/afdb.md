# MIT-BIH Atrial Fibrillation Database (AFDB)

> **Rôle dans le projet :** dataset principal d'entraînement et de validation interne (5-fold cross-validation au niveau patient).

## En une phrase

L'AFDB est la **base de référence historique** pour la détection de la fibrillation auriculaire à partir de l'ECG, composée d'**enregistrements Holter** longs (~10 heures) de patients connus pour faire de l'AFib.

## Origine et contexte

- **Publié par** : MIT Beth Israel Deaconess Medical Center.
- **Année** : début des années 1990, distribué sur PhysioNet depuis 1999.
- **Lien officiel** : <https://physionet.org/content/afdb/1.0.0/>
- **Identifiant PhysioNet** : `afdb` (utilisé dans notre `configs/data.yaml`).
- **Citation académique** : Moody, G. B. & Mark, R. G. *A new method for detecting atrial fibrillation using R-R intervals.* Computers in Cardiology (1983).

## Composition

| Caractéristique | Valeur |
|---|---|
| Nombre d'enregistrements | **25** (initialement 23, deux ajoutés ensuite) |
| Durée par enregistrement | ~10 heures |
| Fréquence d'échantillonnage | 250 Hz |
| Nombre de leads ECG | 2 |
| Total approximatif de battements | ~1.2 million |
| Population | Adultes ambulatoires (Holter), connus pour épisodes d'AFib |

Chaque enregistrement contient des **alternances** entre rythme normal et épisodes d'AFib — c'est précisément ce qui en fait un dataset idéal pour apprendre à les **distinguer**.

## Format des annotations

Les fichiers `.atr` contiennent deux types d'informations utiles :

1. **Positions des battements** (sample index) avec leur symbole :
   - `N` = battement normal,
   - `V` = battement ventriculaire,
   - `A` = auriculaire prématuré,
   - etc.
2. **Annotations de rythme** dans `aux_note`, encodées par des chaînes débutant par `(` :
   - `(N` = rythme sinusal normal,
   - `(AFIB` = fibrillation auriculaire,
   - `(AFL` = flutter auriculaire,
   - `(J` = rythme jonctionnel.

Notre fonction `src/data/rr_extract.py:extract_rr_series` propage la dernière annotation de rythme à chaque battement suivant (jusqu'à un nouveau marqueur), puis attribue `is_afib = 1` si le rythme est `(AFIB` ou `(AFL`.

## Spécificités à garder en tête

- **25 patients seulement.** C'est très peu en deep learning. Sans précaution :
  - Risque énorme de **mémoriser** les caractéristiques d'un patient au lieu d'apprendre l'AFib.
  - Risque de **data leakage** si des fenêtres d'un même patient se retrouvent dans train *et* test.
  → Notre split est **GroupKFold par patient** (`src/utils/splits.py`), avec un test unitaire dédié.

- **Déséquilibre par patient.** Certains patients ont 90 % d'AFib, d'autres 10 %. La métrique « accuracy globale » est trompeuse — on rapporte donc F1, sensibilité, spécificité, AUROC et AUPRC.

- **Données Holter réelles, donc bruitées.** Les annotations restent une référence très fiable, mais les RR contiennent des artefacts (battements manqués, mouvement). Notre `clean_rr_series` filtre les RR < 0.3 s et > 2.0 s pour limiter l'impact.

## Comment on l'utilise dans le projet

1. **Téléchargement** : `make data` → `scripts/download_data.py` → `data/raw/afdb/`.
2. **Extraction** : on transforme chaque `.atr` en séquence RR + labels par battement.
3. **Fenêtrage** : 30 RR par fenêtre, recouvrement 50 %.
4. **Split** : 5-fold cross-validation **au niveau patient** (5 × 5 patients ≈ pas exactement, mais répartition équilibrée).
5. **Entraînement** : le CNN-LSTM voit uniquement ces fenêtres.
6. **Évaluation interne** : F1 par patient (moyenne ± écart-type sur les 5 folds).

## Pourquoi ce dataset et pas un autre comme dataset principal ?

| Critère | AFDB | LTAFDB | CinC 2017 |
|---|---|---|---|
| Annotations beat-level précises | ✅ très précises | ⚠️ moins précises | ❌ niveau enregistrement |
| Alternance NSR/AFib chez le même patient | ✅ oui | ✅ oui | ❌ une seule classe par enregistrement |
| Taille raisonnable pour itérer vite sur CPU | ✅ ~25 patients | ❌ 84 patients × 24h | ✅ 8528 mais segments courts |
| Standard de référence dans la littérature | ✅ **le** standard | référence secondaire | challenge |

AFDB est donc le **meilleur compromis** pour entraîner et **comparer honnêtement** à la littérature.

## Limites à mentionner dans le rapport

- Petite taille → généralisation incertaine. C'est précisément pourquoi on utilise [LTAFDB](ltafdb.md) en évaluation externe.
- Population de patients connus pour faire de l'AFib → biais de sélection. Les sujets « vraiment sains » sont sous-représentés → c'est pourquoi on ajoute [NSRDB](nsrdb.md).
- Annotations parfois imparfaites (10 % des records ont des erreurs documentées dans la doc PhysioNet).
