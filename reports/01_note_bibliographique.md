# Note bibliographique — Détection de la fibrillation auriculaire (AFib) par apprentissage profond sur intervalles RR

> Document de travail (Phase 0). À enrichir au fil de la lecture des articles cités. Les valeurs numériques rapportées proviennent des publications mais doivent être **revérifiées dans le PDF** avant d'être citées dans le rapport final.

## 1. Problématique clinique

La **fibrillation auriculaire (AFib)** est l'arythmie cardiaque soutenue la plus fréquente : prévalence ~2-4 % de la population adulte, multiplie par ~5 le risque d'AVC ischémique. Elle est souvent **paroxystique** (épisodes intermittents) et donc difficile à capter avec un ECG ponctuel chez le cardiologue → fort intérêt du **monitoring ambulatoire long-terme** (Holter, patchs, smartwatches).

La signature physiologique est double :
1. **Irrégularité absolue du rythme** (intervalle RR sans périodicité ni motif).
2. **Absence d'onde P** et substitution par des oscillations fines de la ligne de base.

Un détecteur peut s'appuyer sur (a) l'ECG complet ou (b) les seuls **intervalles RR**. (b) est moins informatif mais beaucoup plus léger : compatible avec une chaîne de détection embarquée sur wearable, et insensible aux artefacts de surface (peau sèche, mouvement) qui détériorent les ondes P.

> **Conséquence pour ce projet :** on attaque l'angle (b) — RR uniquement — qui est exactement le terrain où un modèle léger fait sens.

## 2. Approches classiques (avant le deep learning)

### 2.1 Heuristiques HRV
- **Tateno & Glass (2001)** — *Automatic detection of atrial fibrillation using the coefficient of variation and density histograms of RR and ΔRR intervals* — Méd. Biol. Eng. Comput.
  - Référence historique : seuil sur le coefficient de variation des RR + histogrammes 2D des Δ-RR.
  - Sensibilité ~94 %, spécificité ~97 % sur AFDB.
  - **Pertinence pour nous :** notre Baseline 0 reprend exactement ce coefficient de variation comme score plancher.

### 2.2 Features HRV + classifieurs supervisés
- **RMSSD, SDNN, pNN50, Shannon entropy, sample entropy** : descripteurs « temporels » classiques.
- Combinés à Random Forest / SVM / XGBoost, ils atteignent typiquement **AUC ~0.93-0.96 sur AFDB**.
- **Limite :** plafond de performance lié au choix manuel de features ; sensible à la longueur de fenêtre et aux artefacts.
- **Pertinence pour nous :** Baseline 1 (HRV + RF) — c'est la barre que le CNN-LSTM doit dépasser pour justifier la complexité.

## 3. Deep learning sur ECG brut

Travaux phares (utiles comme contexte mais éloignés de notre approche RR-only) :

- **Hannun et al. (2019, Nature Medicine)** — *Cardiologist-level arrhythmia detection in ambulatory ECG using a DNN*.
  - CNN profond (34 couches type ResNet) sur ECG 30s, 12 classes, 91 232 patients.
  - F1 modèle ≈ F1 cardiologues moyens.
  - **Apport méthodologique :** **split au niveau patient**, comparaison à un panel de cardiologues. À répliquer dans notre méthodologie.

- **Pourbabaee, Roshtkhari, Khorasani (2018, IEEE TSMC)** — *Deep CNN and learning ECG features for screening paroxysmal AFib patients*.
  - CNN sur ECG brut → features apprises → classifieur. Bonne illustration de l'idée « feature learning ».

## 4. Deep learning sur intervalles RR (cœur de notre sujet)

### 4.1 LSTM seul
- **Faust et al. (2018, Comput. Biol. Med.)** — *Automated detection of AFib using long short-term memory network with RR interval signals*.
  - LSTM appliqué directement sur des séquences RR.
  - Reporting très favorable (accuracy > 99 %) **mais évaluation au niveau fenêtre, pas patient** → soupçon de data leakage.
  - **Pertinence pour nous :** Baseline 3 (LSTMOnly) + leçon critique sur la méthodologie d'évaluation (cf. §6).

### 4.2 CNN seul
- Plusieurs travaux montrent que des CNN 1D sur RR (sans LSTM) capturent suffisamment de motifs locaux d'irrégularité pour atteindre des AUC ~0.97 sur AFDB.
- **Pertinence pour nous :** Baseline 2 (CNNOnly). Si le CNN seul atteint déjà la cible, le LSTM doit apporter une plus-value mesurable — c'est ce que l'ablation devra trancher.

### 4.3 Hybrides CNN-LSTM (notre approche)
- **Andersen, Peimankar, Puthusserypady (2019, Expert Syst. Appl.)** — *A deep learning approach for real-time detection of atrial fibrillation*.
  - **C'est l'article le plus proche de notre sujet.** Architecture CNN + LSTM sur RR, évaluation sur AFDB + NSRDB + AF Termination Challenge DB.
  - Très bonnes performances rapportées, **avec un effort visible de validation cross-database**.
  - **À lire en détail** : choix exact d'architecture, taille de fenêtre, stratégie de labelling, split utilisé.

- **Petmezas et al. (2021, Biomed. Signal Process. Control)** — *Automated AFib detection using a hybrid CNN-LSTM network on imbalanced ECG datasets*.
  - CNN-LSTM + **focal loss** pour gérer le déséquilibre.
  - Bonne référence si on adopte focal loss en phase 3 plutôt que la BCE pondérée.

### 4.4 Hybrides plus récents (CNN + Transformer)
- Plusieurs papiers 2022-2024 explorent CNN-Transformer ou attention pure sur RR.
- **Décision méthodologique :** on reste sur CNN-LSTM (sujet imposé), mais **mentionner** ces travaux en section « perspectives » du rapport pour montrer une veille à jour.

## 5. Datasets standards et leurs pièges

| Dataset | Particularité | Piège à connaître |
|---|---|---|
| **MIT-BIH AFDB** | 25 patients, 10h chacun, annotations beat-level fiables | Très **peu de patients** → le risque overfitting patient est énorme |
| **MIT-BIH NSRDB** | 18 sujets sains, rythme sinusal pur | Sert d'augmentation de la classe négative ; **distribution différente** d'AFDB (hôpital vs ambulatoire) |
| **Long-Term AFib DB (LTAFDB)** | 84 enregistrements 24-25h | Annotations parfois moins précises que AFDB ; bon **test de généralisation externe** |
| **PhysioNet/CinC 2017** | 8 528 enregistrements **courts (~30s)**, annotés à 4 classes (Normal/AF/Other/Noisy) | Domaine différent (segments très courts) → bonus pour montrer la robustesse cross-domain |

> **Règle d'or :** ne JAMAIS faire de validation croisée mélangeant fenêtres d'un même patient entre train et test. C'est l'erreur n°1 dans la littérature et la principale source d'inflation des métriques reportées.

## 6. Méthodologie d'évaluation — état de l'art critique

- **Mousavi & Afghah (2019, ICASSP)** — *Inter- and intra- patient ECG heartbeat classification* : démontre formellement l'écart entre évaluation intra-patient (optimiste) et inter-patient (réaliste). Performances chutent de 5-15 points F1.
- Notre méthodologie doit donc afficher :
  1. **GroupKFold sur l'ID patient** (déjà implémenté dans `src/utils/splits.py` + test unitaire qui vérifie l'absence de leak).
  2. **Métriques agrégées par patient** (pas seulement micro/macro sur les fenêtres).
  3. **Évaluation externe sur LTAFDB** sans réentraînement.
- **Métriques rapportées :** F1 (window + per-patient), sensibilité, spécificité, AUROC, AUPRC, matrice de confusion par patient.
- **Reporting de l'incertitude :** moyenne ± écart-type sur les 5 folds + intervalle de confiance bootstrap (1000 tirages) sur le test externe.

## 7. Compression et déploiement embarqué

Axe différenciant de ce projet — peu de travaux RR-AFib publient un benchmark sérieux sur ce point.

- **Quantization dynamique post-training** (PyTorch) : réduit la taille ~4× sur les couches Linear/LSTM en passant à `qint8`, dégradation typique < 1 point F1.
- **Pruning magnitude** : 20-40 % des poids retirés sans perte significative pour des modèles surdimensionnés ; nos modèles étant déjà petits (~22k params), gain attendu plus modeste.
- **Knowledge distillation** : transfert d'un « maître » (CNN-LSTM complet) vers un « élève » (modèle plus petit) ; intéressant si le maître atteint un plafond et qu'on veut le déployer.
- **ONNX Runtime CPU** : permet l'inférence cross-langage (C++, Java/Kotlin pour Android wearable).

Référence générale : **Hong et al. (2020, Comput. Biol. Med.)** — *Opportunities and challenges of deep learning methods for ECG data: a systematic review*. À lire pour cadrer la section « perspectives » du rapport.

## 8. Synthèse — qu'apporte notre travail

| Élément | Littérature dominante | Notre contribution |
|---|---|---|
| Architecture | CNN-LSTM décrit, parfois sans détail d'ablation | **Étude d'ablation systématique** (CNN seul / LSTM seul / hybride / sans BatchNorm / sans bidirectional) |
| Évaluation | Souvent intra-patient ou cross-dataset partiel | **Patient k-fold + LTAFDB externe sans fine-tuning** |
| Métriques | F1/accuracy au niveau fenêtre | **Per-patient F1 + intervalles de confiance bootstrap** |
| Compression | Peu reporté | **Tableau performance/taille/latence/RAM avant et après quantization + pruning + ONNX** |
| Reproductibilité | Souvent absente | **Code public, configs YAML, seeds fixées, tests unitaires anti-leakage** |

Le projet ne prétend pas battre l'état de l'art en valeur absolue de F1 (peu probable avec un modèle < 100k params et CPU only). Sa valeur portfolio repose sur la **rigueur méthodologique** et le **packaging ingénieur** : reproductible, déployable, honnêtement évalué.

## 9. Liste de lecture priorisée

À lire dans l'ordre, en mettant des annotations dans ce fichier au fur et à mesure :

1. [ ] Andersen et al. 2019 — *référence directe à notre architecture*
2. [ ] Petmezas et al. 2021 — *focal loss + imbalanced*
3. [ ] Mousavi & Afghah 2019 — *inter/intra-patient*
4. [ ] Faust et al. 2018 — *LSTM-only sur RR*
5. [ ] Tateno & Glass 2001 — *baseline historique*
6. [ ] Hannun et al. 2019 — *méthodologie d'évaluation grand échantillon*
7. [ ] Hong et al. 2020 — *revue systématique*
8. [ ] Pourbabaee et al. 2018 — *CNN sur ECG, méthode alternative*

Cibler 3-4 papiers complémentaires post-2022 (CNN-Transformer, distillation pour ECG) pour la section « perspectives ».

---

*Note maintenue par : Cheikh Rouhou Ashedi. Dernière mise à jour : 2026-05-25.*
