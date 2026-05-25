# Long-Term Atrial Fibrillation Database (LTAFDB)

> **Rôle dans le projet :** **test de généralisation externe**. Le modèle est entraîné uniquement sur AFDB+NSRDB, puis évalué « à l'aveugle » sur LTAFDB — sans aucun réentraînement ni fine-tuning.

## En une phrase

LTAFDB contient des enregistrements **très longs (24-25 heures)** de patients faisant de l'AFib **paroxystique ou persistante** — un environnement réaliste pour tester si notre modèle généralise au-delà du dataset d'entraînement.

## Origine et contexte

- **Publié par** : Boston Children's Hospital et collaborateurs (groupe de Lewis Glass / Ary Goldberger).
- **Année** : 2007, distribué sur PhysioNet.
- **Lien officiel** : <https://physionet.org/content/ltafdb/1.0.0/>
- **Identifiant PhysioNet** : `ltafdb`.
- **Citation académique** : Petrutiu, S., Sahakian, A. V., Swiryn, S. *Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans.* Europace (2007).

## Composition

| Caractéristique | Valeur |
|---|---|
| Nombre d'enregistrements | **84** |
| Durée par enregistrement | **24-25 heures** |
| Fréquence d'échantillonnage | 128 Hz |
| Population | Patients adultes avec AFib paroxystique ou persistante diagnostiquée |
| Taille totale (brute) | ~3-4 Go |

C'est de **loin** le plus gros des trois datasets du projet, à la fois en patients (3.3× AFDB) et en durée (2.4× AFDB). En contrepartie, les annotations sont moins précises (voir plus bas).

## Pourquoi un test externe ?

C'est **le point méthodologique** qui sépare un projet de niveau étudiant d'un projet de niveau ingénieur :

> **Évaluer un modèle uniquement par cross-validation sur le dataset d'entraînement surestime systématiquement ses performances réelles.**

Référence : Mousavi & Afghah (ICASSP 2019) montrent que le delta intra-patient vs inter-patient est typiquement de 5 à 15 points de F1.

Notre protocole :
1. Le modèle est entraîné et validé en GroupKFold sur AFDB + NSRDB.
2. **Une fois le modèle final figé**, on le déploie sur LTAFDB **sans aucun ajustement**.
3. La métrique reportée sur LTAFDB est notre **estimateur honnête** de la performance en conditions réelles.

Si le F1 chute de plus de 10-15 points entre interne et externe, c'est un signal d'overfitting au dataset d'entraînement — à documenter et discuter dans le rapport.

## Spécificités à garder en tête

### Annotations moins fines qu'AFDB

- Les positions de battements (`.atr` symbols) restent fiables.
- Les **annotations de rythme** (`(AFIB`, `(N`, …) sont parfois moins précises temporellement : transitions parfois marquées avec quelques secondes de décalage.
- Conséquence : les exemples « ambigus » près des transitions seront un peu plus bruités. C'est une **caractéristique du monde réel**, pas un bug.

### Diversité des présentations

LTAFDB contient à la fois :
- AFib **paroxystique** : épisodes courts entrecoupés de rythme normal,
- AFib **persistante** : long maintien en AFib.

C'est intéressant car le ratio AFib/non-AFib varie énormément entre patients → un bon stress test pour la métrique « per-patient F1 ».

### Volume = inférence longue

84 enregistrements × 24h × 128 Hz ≈ ~1 milliard de samples. Heureusement, on travaille sur les RR (~85 000 battements par enregistrement, ~7 millions au total) → reste très gérable sur CPU.

## Comment on l'utilise dans le projet

1. **Téléchargement** : `make data` → `data/raw/ltafdb/`. C'est le plus long (~3-4 Go).
2. **Extraction RR** : pipeline identique aux autres datasets.
3. **Fenêtrage** : mêmes paramètres (30 RR, recouvrement 50 %) **pour rester comparable**.
4. **Aucun split CV** : LTAFDB est **un seul bloc** utilisé comme test.
5. **Évaluation** :
   - Métriques au niveau fenêtre (F1, AUROC, AUPRC).
   - Métriques **par patient** + intervalle de confiance bootstrap (1000 tirages).
   - Comparaison directe interne (AFDB CV) vs externe (LTAFDB) → tableau du rapport.

## Ce qu'on espère observer

| Scénario | Interprétation |
|---|---|
| F1 LTAFDB ≈ F1 AFDB (delta < 5 pts) | Excellent — le modèle généralise vraiment |
| F1 LTAFDB ~ 5-10 pts en dessous | Acceptable — domain shift modéré, à discuter dans le rapport |
| F1 LTAFDB > 15 pts en dessous | Overfitting dataset-spécifique → revoir la régularisation, augmenter NSRDB, repenser le windowing |

Quel que soit le résultat, **on le rapporte honnêtement**. Une baisse documentée est plus crédible qu'un score interne « parfait » sans test externe.

## Pourquoi ce dataset et pas autre chose comme test externe ?

| Dataset envisagé | Avantage | Inconvénient |
|---|---|---|
| **LTAFDB** ✅ | 84 patients, 24h, populations différentes d'AFDB | Lourd à télécharger |
| CinC 2017 | 8 528 enregistrements | Segments **très courts** (~30s), tâche différente |
| Apnea-ECG (apnea) | Long-terme | Pas annoté pour AFib |
| Aucun test externe | Plus rapide | **Score interne non crédible** → projet faible pour portfolio |

LTAFDB coche toutes les cases : annotations AFib, populations différentes, format compatible.

## Limites à mentionner dans le rapport

- Annotations rythmes parfois imprécises (peut pénaliser légèrement notre F1 reporté).
- Pas de variabilité géographique : tous les patients viennent du même hôpital → si on voulait aller plus loin, un test sur des données européennes ou asiatiques serait un plus.
- 128 Hz uniquement → si à l'avenir on voulait remonter aux ondes P, l'échantillonnage est limite. Pour l'approche RR-only, aucun impact.
