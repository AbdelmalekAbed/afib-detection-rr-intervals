# Mémoire LaTeX — Détection de la FA par CNN-LSTM

Source LaTeX du mémoire de Master, conforme aux **directives Sayadi** (12 pt Times-like, interligne 1,5, marges 25 mm, plan IMRAD, bibliographie numérotée style IEEE).

## Compilation

### En local (recommandé)

Une fois TeX Live installé (voir ci-dessous), depuis la racine du dépôt :

```bash
make report-pdf      # produit report/main.pdf
make report-watch    # recompile à chaque sauvegarde (live preview)
make report-clean    # nettoie report/build/ et le PDF
```

Le PDF final atterrit à **`report/main.pdf`** (non versionné — regénéré sur demande).
Les fichiers intermédiaires (.aux, .bbl, .log, etc.) sont confinés dans `report/build/`.

**Installation TeX Live (Ubuntu/Debian, une fois pour toutes) :**

```bash
sudo apt update && sudo apt install -y \
  texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-fonts-extra \
  texlive-lang-french texlive-bibtex-extra \
  texlive-science texlive-pictures \
  biber latexmk
```

### Sur Overleaf (alternative)

1. Compresser le dossier `report/` en `.zip`.
2. Importer le `.zip` dans un nouveau projet Overleaf.
3. Régler le compilateur sur **pdfLaTeX** (Menu → Settings → Compiler).
4. Régler le moteur biblio sur **Biber**.
5. Cliquer sur **Recompile**.

## Arborescence

```
report/
├── main.tex                       # Point d'entrée
├── preamble/
│   ├── packages.tex              # Tous les packages (newtxtext, biblatex IEEE, etc.)
│   ├── style.tex                 # Mise en page (marges, interligne, en-têtes)
│   └── commands.tex              # Commandes perso (notation VP/VN/FP/FN, acronymes)
├── frontmatter/
│   ├── pagedegarde.tex           # École, encadrante, titre (jury à compléter)
│   ├── dedicace.tex              # Optionnel
│   ├── remerciements.tex         # Rédigés
│   ├── resume.tex                # Résumé FR + Abstract EN (≤ 250 mots chacun)
│   └── acronymes.tex             # Liste des abréviations
├── chapters/
│   ├── 01_introduction.tex       # Contexte, état de l'art bref, contribs, plan
│   ├── 02_etat_art.tex           # Clinique, RR, datasets, 4 familles d'approches
│   ├── 03_methodes.tex           # Pipeline, modèle, critères (PDF Critères inclus)
│   ├── 04_resultats.tex          # Phases 1→6 + incertitude, seuil, charge de FA
│   ├── 05_discussion.tex         # Comparaison littérature, limitations
│   └── 06_conclusion.tex         # Récap + perspectives
├── backmatter/
│   └── annexes.tex               # Espaces Optuna réels + captures d'écran
├── bibliography/
│   └── references.bib            # Format BibTeX, style IEEE (équivalent Sayadi)
└── figures/                      # 32 PDF vectoriels ou 300 dpi + captures d'écran
```

## État du document

Le mémoire est complet et compile sans avertissement de référence non résolue.
Les valeurs numériques proviennent toutes des fichiers JSON de `reports/` ; aucune
n'est saisie à la main dans le LaTeX.

| Élément | État |
|---|---|
| Page de garde (école, encadrante, titre) | ✅ renseignée |
| Membres du jury | ⬜ à compléter une fois le jury désigné |
| Remerciements, dédicace | ✅ rédigés |
| Résumé FR + abstract EN | ✅ chiffres finaux intégrés |
| Chapitres 1 à 5 + conclusion | ✅ rédigés |
| Figures | ✅ 20 figures insérées (sur 32 disponibles dans `figures/`) |
| Annexes | ✅ espaces de recherche Optuna réels + captures de l'application |
| Bibliographie | ✅ 22 références citées, style IEEE |

## Conformité aux directives

- ✅ Police Times-like (`newtxtext`) — 12 pt
- ✅ Interligne 1,5 (`\onehalfspacing`)
- ✅ Marges 25 mm tous côtés (`geometry`)
- ✅ Texte justifié (par défaut `report`)
- ✅ Plan IMRAD strict (Intro / SOA / Méthodes / Résultats / Discussion / Conclusion)
- ✅ Équations numérotées par chapitre
- ✅ Figures avec axes + légende explicite + numérotation par chapitre
- ✅ Pas de cadres (`\fbox`) — utilisation des environnements natifs `algorithm`, `figure`, `table`
- ✅ Algorithmes en pseudo-code, encadrés avec titre (`algorithm` + `algpseudocode`)
- ✅ Biblio numérotée, ordre de citation (style IEEE)
- ✅ Notation VP / VN / FP / FN conforme au PDF Critères
- ✅ Formules d'évaluation conformes au PDF Critères (équations du chapitre 3)
- ✅ Incertitude quantifiée (IC 95 % par bootstrap, écart-type inter-plis, test apparié)
