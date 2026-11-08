# Interface de recherche React + FastAPI

## Périmètre

La nouvelle interface locale complète Streamlit, conservé pour l'archive. Elle
présente exclusivement le candidat `artifacts/optimization_v2`. Le mémoire et
les résultats historiques restent inchangés. Aucune expérience n'est relancée
depuis l'interface et aucun seuil n'est ajusté sur LTAFDB.

## Architecture retenue

- `web/` : React, TypeScript et Vite ; navigation en français, graphiques
  interactifs, mise en page adaptative, états de chargement et d'erreur.
- `src/api/` : FastAPI ; lecture vérifiée des résultats, exploration paginée des
  patients et inférence CPU sur exactement 60 intervalles RR en secondes.
- Le serveur sert aussi le build React : une seule origine, aucun service tiers,
  aucune base de données et aucun téléversement de poids.
- Les poids restent locaux ; leurs empreintes doivent correspondre au manifeste.
  Le laboratoire charge les graphes ONNX vérifiés, avec les transformations et
  seuils propres à chacun des cinq plis.

## Parcours

1. Synthèse : résultats AFDB et LTAFDB, protocole et accès aux explorations.
2. Patients : scores enregistrés hors pli (AFDB) ou externes (LTAFDB), annotations,
   seuil fixe ou décisions avec seuils internes gelés, fenêtre RR inspectable.
3. Modèles : comparaison CNN-LSTM / HRV Gradient Boosting, métriques homogènes.
4. Compression : qualité, taille d'un membre et de l'ensemble, latence mesurée.
5. Transfert : AFDB interne / LTAFDB externe, distinction des deux protocoles.
6. Laboratoire RR : exemples déterministes et saisie manuelle, vraie inférence
   ONNX de l'ensemble ; sorties pédagogiques, sans interprétation diagnostique.

Les indices des fenêtres ne sont pas des heures d'enregistrement : les caches
contiennent des RR nettoyés et ne préservent pas les horodatages d'origine.
Le F1 moyen par patient exclut les patients monoclasse ; ils restent visibles
et contribuent aux métriques globales. Les scores ne sont pas des probabilités
cliniques calibrées. Le positif comprend AFIB et AFL.

## Vérification prévue

Tests API : intégrité des artefacts, alignement et pagination des observations,
validation des RR, reproductibilité ONNX sur une vraie fenêtre externe, absence
de fuite entre patients et gestion des artefacts manquants. Compilation
TypeScript et build de production. Vérification navigateur si l'environnement
le permet. Contrôle des empreintes des 162 fichiers historiques après travail.

L'hébergement public viendra après validation locale ; un hébergement statique
seul ne peut pas exécuter le backend scientifique Python.

## Installation et lancement

Prérequis : environnement scientifique `.venv` existant, Node.js 22.12 ou plus,
`uv` et candidat v2 complet sur cette machine. Depuis la racine :

```bash
make web-setup
make web-prepare  # une fois ; refuse d'écraser un catalogue existant
make web-build
make web
```

Ouvrir <http://127.0.0.1:8000>. La documentation API est disponible sur
<http://127.0.0.1:8000/api/docs>. L'API et le frontend partagent une origine,
sans CORS permissif. `make demo` conserve l'ancienne application Streamlit.

Sur la machine de développement de cette livraison, les étapes d'installation,
de préparation et de compilation ont déjà été exécutées : **`make web` suffit**.

Les dépendances FastAPI sont verrouillées dans `requirements-web.txt` et
installées dans `.venv-web`. Un fichier `.pth` y ajoute les bibliothèques de
`.venv` après celles du web : aucune dépendance scientifique n'est remplacée.
`web/package-lock.json` verrouille les dépendances JavaScript. Après déplacement
du projet ou changement de version Python, recréer `.venv-web` et le lien.

Développement : lancer `make web`, puis `cd web && npm run dev` dans un autre
terminal. Vite redirige `/api` vers le backend local. Après modification de
l'interface, `make web-build` actualise la version servie sur le port 8000.

## Artefacts et provenance

`scripts.prepare_interface` crée `artifacts/interface_v1/catalog.json` sans
recalculer les prédictions. Le catalogue contient les chemins locaux et SHA-256
de 33 fichiers. Les caches RR AFDB restent dans `reliability_v1/windows` et
LTAFDB dans `optimization_v2/windows` : aucune duplication des données.
Le serveur vérifie chaque fichier avant sa première utilisation, puis le garde
en cache. Les artefacts doivent rester immuables pendant l'exécution ; arrêter
le serveur avant de changer de candidat. Pour sélectionner un autre catalogue
préparé, définir `AFIB_CATALOG` avant le lancement.

La nouvelle interface ne propose ni réentraînement, ni importation de poids,
ni écriture dans les expériences. Le CSV exporte uniquement la portion de
trace affichée ; l'export JSON global conserve les métriques et leur contexte.

## Exécution des vérifications

```bash
make web-api-test
make web-build
# Dans un autre terminal, avec make web actif :
cd web
npm run test:e2e
```

Les tests navigateur utilisent Playwright. Sur une machine sans navigateur
Playwright : `npx playwright install chromium`. Pour employer une installation
existante, définir `AFIB_CHROMIUM=/chemin/vers/chrome`.
`AFIB_TEST_URL` permet de tester une autre URL locale.
Les tests API portant sur le candidat réel sont ignorés si le catalogue local
est absent ; le candidat et les données ne sont pas inclus dans Git.
L'environnement scientifique peut continuer à lancer ses tests séparément.

Résultats du contrôle de cette livraison :

- **53 tests scientifiques réussis** dans `.venv` (module API ignoré dans cet
  environnement sans FastAPI ; 35 avertissements PyTorch déjà connus).
- **23 tests API réussis** dans `.venv-web`, avec les vraies prédictions AFDB et
  LTAFDB, la validation des entrées, le contrôle d'intégrité et la reconstruction
  du catalogue. La suite complète dans cet environnement atteint **76 tests
  réussis** ; des avertissements de dépréciation demeurent, sans échec.
- **3 parcours navigateur Chromium réussis** : six écrans, changements de base,
  pagination, inférence, saisie invalide, export, menu mobile, boîte de protocole
  et reprise après indisponibilité de l'API.
- Compilation TypeScript et build Vite réussis ; captures sous
  `web/test-results/` (non versionnées).
- Empreintes des **162 fichiers historiques inchangées**.

## Préparation d'un hébergement

Le build React est servi par FastAPI : un processus Python permanent et les
artefacts locaux sont nécessaires. Le catalogue contient un chemin absolu :
le régénérer sur la machine cible avec `--root` et `--output` après transfert
des deux caches, du candidat complet et du code. Aucun accès à PhysioNet
n'est nécessaire pour consulter cette interface une fois les caches présents.

Le lancement proposé est limité à `127.0.0.1`, pour la démonstration locale.
Une publication devra préciser l'hébergeur, la mémoire CPU disponible, les
modalités d'accès et le reverse proxy HTTPS. Aucun service public n'a été créé
dans cette étape.

Références techniques : [fichiers statiques FastAPI](https://fastapi.tiangolo.com/tutorial/static-files/)
et [guide Vite](https://vite.dev/guide/).
