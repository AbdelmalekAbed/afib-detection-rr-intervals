# Datasets utilisés dans le projet

Ce dossier documente, dataset par dataset, **ce que sont les bases de données** que nous utilisons, **pourquoi** elles existent, et **comment** nous les exploitons pour la détection d'AFib à partir d'intervalles RR.

> Toutes les bases viennent de [PhysioNet](https://physionet.org), un dépôt public de signaux physiologiques maintenu par le MIT et le NIH. Elles sont libres pour la recherche.

## Pourquoi plusieurs datasets ?

Un seul dataset = une seule population, un seul appareil, un seul protocole. Un modèle entraîné sur un dataset peut **paraître** excellent mais s'effondrer en conditions réelles. La méthodologie de ce projet utilise donc trois bases qui jouent des rôles **complémentaires** :

| Dataset | Rôle dans le projet | Pourquoi celui-ci ? |
|---|---|---|
| [**MIT-BIH AFDB**](afdb.md) | Entraînement + validation interne | Référence historique pour l'AFib, annotations très précises au battement |
| [**MIT-BIH NSRDB**](nsrdb.md) | Exemples « rythme sinusal normal » supplémentaires | Augmente la diversité des sujets sains pour limiter l'overfitting sur les contrôles d'AFDB |
| [**Long-Term AFDB (LTAFDB)**](ltafdb.md) | Test de généralisation **externe** (jamais vu pendant l'entraînement) | Patients différents, enregistrements 24h → vrai test de robustesse |

> **Règle d'or appliquée partout :** un patient n'apparaît **jamais** dans deux splits (train/val/test). C'est l'erreur n°1 dans la littérature AFib et la principale source de scores artificiellement gonflés.

## Vocabulaire essentiel (mini-glossaire cardiologique)

| Terme | Définition courte | Pourquoi ça compte ici |
|---|---|---|
| **ECG / ECG signal** | Électrocardiogramme : tension électrique mesurée à la surface de la peau, reflétant l'activité du cœur | C'est le signal brut dont sont issus les intervalles RR |
| **Onde R** | Pic franc dans le complexe QRS, correspondant à la contraction ventriculaire | Sert de repère temporel pour chaque battement |
| **Intervalle RR** | Durée entre deux ondes R successives (en secondes) | C'est **l'entrée principale de notre modèle** |
| **Rythme sinusal normal (NSR)** | Rythme physiologique du cœur, régulier, ~60-100 bpm au repos | Classe « négative » dans notre problème binaire |
| **Fibrillation auriculaire (AFib)** | Arythmie où les oreillettes battent de façon désorganisée → RR très irréguliers, pas d'onde P | Classe « positive » à détecter |
| **Flutter auriculaire (AFL)** | Arythmie auriculaire organisée, parfois confondue avec l'AFib | Inclus dans nos labels positifs (cf. `AFIB_RHYTHM_CODES`) |
| **Battement ectopique** | Battement prématuré d'origine non sinusale (extrasystole) | Source de bruit : un RR très court suivi d'un long mais **n'est pas** de l'AFib |
| **Annotation `.atr`** | Fichier PhysioNet contenant la position de chaque battement + le rythme courant | Source de vérité terrain pour nos labels |

## Format des données sur disque

Chaque enregistrement PhysioNet est représenté par **3 fichiers** partageant le même nom :

```
04015.hea     # header (métadonnées : fréquence d'échantillonnage, nombre de leads, durée…)
04015.dat     # signal ECG brut (entiers 16 bits)
04015.atr     # annotations : pour chaque battement, position en samples + rythme courant
```

Notre pipeline **n'utilise pas** `.dat` (le signal brut) — uniquement `.hea` (pour la fréquence d'échantillonnage) et `.atr` (pour extraire les RR et les labels). C'est ce qui rend l'approche RR-only beaucoup plus légère que le DL sur ECG.

## Comment on transforme un fichier `.atr` en exemple d'entraînement

1. **Lecture** des positions de battements (en samples) et du rythme courant → fait dans `src/data/rr_extract.py`.
2. **Calcul des RR** : `RR[i] = (sample[i+1] - sample[i]) / fs` en secondes.
3. **Nettoyage** : on retire les RR physiologiquement impossibles (< 0.3 s ou > 2.0 s — cf. `clean_rr_series`).
4. **Étiquetage** : chaque RR hérite du rythme annoté à ce moment (AFib si `(AFIB` ou `(AFL`, sinon 0).
5. **Fenêtrage glissant** : on découpe la série en fenêtres de 30 RR avec recouvrement, label majoritaire → fait dans `src/data/windowing.py`.
6. **Sauvegarde** : un seul `.npz` par dataset, contenant `X` (n × 30), `y` (n,), `patient_id` (n,).

Lis les fiches détaillées de chaque dataset pour comprendre ses particularités cliniques et méthodologiques.

## Pour aller plus loin

- [Site PhysioNet](https://physionet.org) — toutes les bases sont librement consultables.
- [`wfdb` Python package](https://wfdb.readthedocs.io) — la bibliothèque officielle pour lire ces formats. Notre code l'utilise via `src/data/loader.py`.
- Guide officiel des symboles d'annotation : [Beat and rhythm annotations](https://archive.physionet.org/physiobank/annotations.shtml).
