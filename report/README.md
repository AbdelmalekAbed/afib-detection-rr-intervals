# Mémoire LaTeX — Détection de la FA par CNN-LSTM

Squelette du mémoire de Master, conforme aux **directives Sayadi** (12 pt Times-like, interligne 1,5, marges 25 mm, IMRAD, biblio numérotée style IEEE).

## Compilation

### Sur Overleaf
1. Compresser le dossier `report/` en `.zip`.
2. Importer le `.zip` dans un nouveau projet Overleaf.
3. Régler le compilateur sur **pdfLaTeX** (Menu → Settings → Compiler).
4. Régler le moteur biblio sur **Biber** (Menu → Settings → TeX Live version récente).
5. Cliquer sur **Recompile**.

### En local
```bash
cd report/
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Arborescence

```
report/
├── main.tex                       # Point d'entrée
├── preamble/
│   ├── packages.tex              # Tous les packages (newtxtext, biblatex IEEE, etc.)
│   ├── style.tex                 # Mise en page (marges, interligne, en-têtes)
│   └── commands.tex              # Commandes perso (figplaceholder, VP/VN/FP/FN)
├── frontmatter/
│   ├── pagedegarde.tex           # ← À PERSONNALISER : école, encadrants, jury
│   ├── dedicace.tex              # Optionnel
│   ├── remerciements.tex         # ← À PERSONNALISER : noms
│   ├── resume.tex                # Résumé FR + Abstract EN (≤ 250 mots chacun)
│   └── acronymes.tex             # Liste des abréviations
├── chapters/
│   ├── 01_introduction.tex       # Contexte, état de l'art bref, contribs, plan
│   ├── 02_etat_art.tex           # Clinique, RR, datasets, approches
│   ├── 03_methodes.tex           # Pipeline, modèle, critères (PDF Critères inclus)
│   ├── 04_resultats.tex          # Phases 1→6, tables avec [XX] à remplir
│   ├── 05_discussion.tex         # Comparaison littérature, limitations
│   └── 06_conclusion.tex         # Récap + perspectives
├── backmatter/
│   └── annexes.tex
├── bibliography/
│   └── references.bib            # Format BibTeX, style IEEE (équivalent Sayadi)
└── figures/                      # Recevra les .pdf 300 dpi à venir
```

## À remplir (recherche les marqueurs)

| Marqueur                | Où                                | Action                                    |
|-------------------------|-----------------------------------|-------------------------------------------|
| `[NOM DE L'UNIVERSITÉ]` | `frontmatter/pagedegarde.tex`     | Renseigner l'établissement                |
| `[NOM DE L'ÉCOLE...]`   | `frontmatter/pagedegarde.tex`     | Renseigner l'école / faculté              |
| `[Nom Prénom — Grade]`  | `frontmatter/pagedegarde.tex`     | Encadrants + jury                         |
| Logos école / univ.     | `frontmatter/pagedegarde.tex`     | Décommenter `\includegraphics{logos/...}` |
| `[Nom de l'encadrant…]` | `frontmatter/remerciements.tex`   | Personnaliser les remerciements           |
| `[À COMPLÉTER]`         | `frontmatter/resume.tex`          | Métriques finales chiffrées               |
| `[XX]`                  | `chapters/04_resultats.tex`       | Valeurs numériques des tables             |
| `[À PRÉCISER]`          | `chapters/03_methodes.tex` §3.5   | GPU et CPU utilisés                       |
| `\figplaceholder{…}`    | partout                           | Remplacer par `\includegraphics{...}`     |
| `[À COMPLÉTER : URL…]`  | `backmatter/annexes.tex`          | Lien GitHub                               |
| Stub `XX_baseline_gb`   | `bibliography/references.bib`     | Compléter la référence manquante          |

## Placeholders figures

Tous les placeholders ont la forme :
```latex
\figplaceholder[<largeur>][<hauteur>]{Description de la figure}
```
Pour insérer la vraie figure, remplacer cette ligne par :
```latex
\includegraphics[width=0.85\linewidth]{figures/nom_du_fichier.pdf}
```

## Conformité aux directives

- ✅ Police Times-like (`newtxtext`) — 12 pt
- ✅ Interligne 1,5 (`\onehalfspacing`)
- ✅ Marges 25 mm tous côtés (`geometry`)
- ✅ Texte justifié (par défaut `report`)
- ✅ Plan IMRAD strict (Intro / SOA / Méthodes / Résultats / Discussion / Conclusion)
- ✅ Équations numérotées par chapitre
- ✅ Figures avec axes + légende explicite + numérotation par chapitre
- ✅ Pas de cadres (`\fbox`) — utilisation des environnements natifs `algorithm`, `figure`, `table`
- ✅ Algorithmes encadrés avec titre (`algorithm` + `algpseudocode`)
- ✅ Biblio numérotée, ordre de citation (style IEEE)
- ✅ Notation VP / VN / FP / FN conforme au PDF Critères
- ✅ Formules d'évaluation conformes au PDF Critères (équations 3.1 à 3.6)
