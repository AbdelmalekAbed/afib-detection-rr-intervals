# Les données du projet

Deux bases PhysioNet, et deux seulement : **AFDB** pour l'entraînement et la
validation interne, **LTAFDB** pour le test externe. Cette page documente ce
qu'elles sont, pourquoi ce couple précis, comment le pipeline les transforme en
exemples, et ce qui a été écarté.

> Source : <https://physionet.org>. Les deux bases sont libres pour la recherche
> et téléchargées automatiquement par `make data`
> (voir [`configs/data.yaml`](../configs/data.yaml)).

## 1. Pourquoi ce couple

Le choix ne porte pas sur une base isolée mais sur le **couple**. Trois critères
sont imposés par la problématique (F1 **par patient**, découpage `GroupKFold`
par patient, entrée RR seule) :

| Critère | Pourquoi il est éliminatoire |
|---|---|
| Annotations **au battement** + marqueurs de **rythme** | Sans les deux, pas d'intervalles RR ni de labels temporels |
| **Alternance** rythme sinusal / FA chez le même patient | Un patient mono-classe n'a pas de F1 individuel interprétable |
| Plusieurs heures par patient, patients identifiables | Condition du découpage et de la métrique par patient |

**AFDB** est la base de référence historique : c'est *dans* l'article qui
l'accompagne (Moody & Mark, 1983) qu'a été proposée la première méthode de
détection de FA à partir des seuls intervalles RR. Toute la littérature RR
ultérieure y rapporte ses résultats — c'est donc la seule base qui rende ce
travail comparable à l'état de l'art.

**LTAFDB** est le test externe. Sa valeur tient au fait qu'elle diffère d'AFDB
sur **quatre axes simultanément** : nombre de patients, durée des
enregistrements, fréquence d'échantillonnage et prévalence de la FA. Le décalage
de domaine est réel, pas cosmétique.

## 2. Les deux bases

| | **AFDB** | **LTAFDB** |
|---|---|---|
| Nom complet | MIT-BIH Atrial Fibrillation Database | Long-Term Atrial Fibrillation Database |
| Lien | <https://physionet.org/content/afdb/> | <https://physionet.org/content/ltafdb/> |
| Référence | Moody & Mark, 1983 | Petrutiu, Sahakian & Swiryn, 2007 |
| Patients | **25** | **84** |
| Durée par enregistrement | ~10 h | 24–25 h |
| Fréquence d'échantillonnage | **250 Hz** (précision 4 ms) | **128 Hz** (précision ~7,8 ms) |
| Rôle dans le projet | entraînement + validation interne (`GroupKFold`, 5 plis) | **test externe**, aucun réglage sur cette base |
| Fenêtres de 60 battements | **40 613** | **299 727** |
| Fenêtres en FA | **43,6 %** | **58,1 %** |
| Patients notables (F1 calculable) | **23** / 25 | **62** / 84 |

**Patients « notables ».** Le F1 par patient n'a pas de sens pour un sujet
mono-classe. Sur AFDB, deux patients (`7162`, `7859`) sont en FA 100 % du temps ;
sur LTAFDB, 22 patients sont soit intégralement en FA soit intégralement en
rythme sinusal. Ils sont exclus du **calcul de la moyenne** — jamais de
l'entraînement — et le nombre de patients notés est reporté avec chaque
résultat (champ `n_patients_scored` dans les JSON de `reports/`).

**LTAFDB est intrinsèquement plus facile** (58,1 % de FA contre 43,6 %). C'est
une réserve à garder en tête : le F1 obtenu en transfert ne se compare pas
naïvement au F1 interne.

## 3. Comment sont faites les annotations

Deux informations distinctes coexistent dans les fichiers PhysioNet :

1. **Les positions de battements** — index temporel de chaque onde R, avec un
   symbole de type (`N` normal, `V` ventriculaire, `A` auriculaire prématuré…).
2. **Les marqueurs de rythme** — chaînes commençant par `(`, posées **aux
   transitions** uniquement : `(N` sinusal, `(AFIB` fibrillation, `(AFL`
   flutter, `(J` jonctionnel.

Les marqueurs de rythme ne sont pas posés à chaque battement : sur le record
AFDB `04015`, il y en a **15 pour 44 005 battements**. D'où le mécanisme de
propagation de [`src/data/rr_extract.py`](../src/data/rr_extract.py) : le
dernier rythme annoté vaut jusqu'au marqueur suivant. Un battement est étiqueté
positif si son rythme courant est `(AFIB` **ou** `(AFL` (`AFIB_RHYTHM_CODES`).

### Le piège `.qrs` contre `.atr`

Les deux bases ne rangent pas ces informations au même endroit :

| Enregistrement | `.atr` | `.qrs` | Source retenue |
|---|---|---|---|
| **AFDB `04015`** | 15 annotations, **toutes des marqueurs de rythme** → **0 battement** | 44 005 battements | `.qrs` pour les battements |
| **LTAFDB `00`** | 106 247 battements (`N`, `V`, `A`) + 89 marqueurs de rythme | 106 139 battements | `.atr` pour les battements |

Lire naïvement le `.atr` partout — le réflexe normal, c'est le fichier standard
de PhysioNet — produirait **zéro battement sur AFDB, donc un dataset vide**.
Lire le `.qrs` partout perdrait des battements et surtout les rythmes sur
LTAFDB.

La règle appliquée par
[`src/data/loader.py`](../src/data/loader.py) est unique et automatique :
lire le `.atr` (toujours présent), lire le `.qrs` s'il existe, **compter les
symboles de battement dans chacun et retenir celui qui en a le plus** ; les
**rythmes viennent toujours du `.atr`**.

**Nuance à connaître.** Sur AFDB, les positions du `.qrs` sont *détectées
automatiquement* (toutes étiquetées `N`) ; ce sont les **étiquettes de rythme**
du `.atr` qui constituent la référence experte. Sur LTAFDB, les battements sont
typés.

## 4. Du fichier brut à l'exemple d'entraînement

Le signal ECG brut (`.dat`) n'est **jamais chargé** : seuls le `.hea` (pour la
fréquence d'échantillonnage) et les annotations le sont. C'est ce qui rend
l'approche RR-only légère.

```
Bases PhysioNet ──► Extraction RR ──► Fenêtrage ──► Découpage patient ──► Modèle ──► Évaluation
 (AFDB, LTAFDB)                    (60 battements)     (GroupKFold)              (F1, AUROC)
```

1. **Extraction** — `RR[i] = (sample[i+1] - sample[i]) / fs`, en **secondes**.
   La division par `fs` est ce qui rend les RR comparables entre 250 Hz et
   128 Hz, et donc le transfert cross-dataset possible.
2. **Nettoyage** — élimination des RR physiologiquement impossibles
   (< 0,3 s ou > 2,0 s). Aucune correction de battement ectopique n'est
   appliquée : c'est un choix assumé, discuté dans les limites du mémoire.
3. **Fenêtrage** — 60 battements, pas de 30 (recouvrement 50 %), label
   **majoritaire** : une fenêtre est positive si ≥ 50 % de ses battements sont
   en FA. L'ablation de la phase 3 a montré que le passage de 30 à 60 battements
   est le facteur le plus déterminant.
4. **Normalisation** — z-score **par fenêtre**.
5. **Découpage** — `GroupKFold` à 5 plis **par patient**, déterministe. Un
   patient n'apparaît jamais des deux côtés ; un test unitaire échoue en cas de
   fuite ([`tests/test_patient_split.py`](../tests/test_patient_split.py)).

Toutes les phases partent de la même fonction
[`build_windows()`](../src/data/build.py) — condition nécessaire pour que les
tableaux comparatifs du mémoire soient légitimes.

## 5. Ce qui a été écarté, et pourquoi

| Base envisagée | Raison de l'exclusion |
|---|---|
| **MIT-BIH NSRDB** (18 sujets sains) | Explorée en phase 1 comme source de « négatifs propres », puis abandonnée : AFDB fournit déjà 56 % de fenêtres non-FA, le besoin n'existe pas. Surtout, un sujet à 0 % de FA est **non notable** au sens du F1 par patient — il n'apparaîtrait dans aucune métrique principale, ajoutant de la donnée d'entraînement invisible à l'évaluation et brouillant la comparaison avec les baselines. |
| **CinC 2017** (8 528 enregistrements) | Labels au niveau de l'**enregistrement**, segments de ~30 s : une seule classe par sujet, pas d'alternance intra-patient. Ni F1 par patient ni `GroupKFold` multi-épisodes possibles — ce n'est pas la même tâche. |
| **ECG 12 dérivations** (Hannun 2019, Ribeiro 2020) | Exige l'ECG complet et des dizaines de milliers de patients ; incompatible avec l'entrée RR seule et la contrainte embarquée < 200 KB. |

> Le notebook [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) conserve
> l'exploration NSRDB de la phase 1 (fenêtres de 30 battements, configuration de
> l'époque). Il documente la décision ; **aucun résultat rapporté n'utilise
> NSRDB**.

## 6. Limites à assumer

- Les deux bases viennent de PhysioNet, d'hôpitaux américains, des années 1990
  et 2000 : **aucune diversité géographique ou ethnique**. Une validation sur
  cohorte européenne ou asiatique reste à faire.
- Sur AFDB, les positions de battements sont issues d'un détecteur automatique ;
  seules les étiquettes de rythme sont expertes.
- Les annotations de rythme de LTAFDB sont parfois décalées de quelques secondes
  aux transitions — les fenêtres ambiguës y sont donc un peu plus bruitées.
- 25 patients sur la base d'entraînement, c'est peu. C'est précisément la raison
  d'être du test externe sur LTAFDB et de l'intervalle de confiance rapporté sur
  chaque comparaison.
