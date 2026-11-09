# Rapport d'avancement — Projet 3 « Prévision de crues »

Ce rapport est mis à jour à chaque étape réalisée. L'état décrit ici correspond
exactement à ce qui a été exécuté dans le dépôt.

## 1. Tâche 1 — Sources

Le fichier [docs/sources.md](docs/sources.md) documente les trois sources
demandées : Hub'Eau pour les séries temporelles, Copernicus Sentinel-2 pour
l'optique et Copernicus Sentinel-1 pour le radar. Pour chacune, il indique
l'URL, le type de données, la fréquence, le format et l'accès requis.

## 2. Tâche 2.A — Collecte et préparation Hub'Eau

### 2.1 Ingestion

[src/timeseries.py](src/timeseries.py) appelle l'API Hub'Eau Hydrométrie pour
la station configurée dans `.env` ou `configs/params.yaml`. La pagination est
gérée par `fetch_recent_observations`; `MAX_PAGES` est défini avant son usage
et l'appel de collecte est effectué par `main`.

### 2.2 Nettoyage et transformation

Le script :

1. convertit les dates en datetime UTC ;
2. transforme les résultats numériques ;
3. supprime les doublons par station, date et grandeur ;
4. pivote les grandeurs H et Q en colonnes ;
5. convertit H de mm en m et Q de L/s en m³/s ;
6. rééchantillonne à 10 minutes ;
7. interpole uniquement les trous internes, dans une limite de trois pas ;
8. ajoute `H_m_missing` et `Q_m3_s_missing` pour tracer les valeurs manquantes ;
9. écrit `data/processed/timeseries_clean.csv`.

Cette étape a été exécutée avec succès : 16 111 observations brutes ont été
téléchargées et 4 319 lignes régulières ont été produites.

## 3. Configuration et reproductibilité

[configs/params.yaml](configs/params.yaml) externalise la station, la période,
la pagination, la fréquence, la limite d'interpolation et les chemins.
[requirements.txt](requirements.txt) contient l'environnement obtenu avec
`pip freeze`. [.gitignore](.gitignore) ignore `.env`, `.venv/` et `data/`,
afin de ne pas versionner les secrets ni les données lourdes.

## 4. Orchestration et qualité

[src/pipeline.py](src/pipeline.py) permet de relancer automatiquement la
collecte et la préparation Hub'Eau en une commande :

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

Le pipeline valide le schéma, la fréquence, les doublons, les valeurs
négatives et produit un rapport JSON de qualité. Les tests de
[tests/test_pipeline.py](tests/test_pipeline.py) couvrent le schéma, le taux
de valeurs manquantes et le rejet des doublons.

La commande de test utilisée est :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Résultat : 2 tests réussis.

## 5. Intégration continue

[.github/workflows/ci.yml](.github/workflows/ci.yml) installe les dépendances,
compile les scripts et lance les tests à chaque push ou pull request. La CI
ne contacte pas les services externes et reste donc reproductible.

## 6. Partie satellite — en attente

Conformément à la demande, aucun compte Copernicus n'a été utilisé, aucun
téléchargement d'image n'a été lancé et `index.csv` n'a pas été créé.

[src/satellite.py](src/satellite.py) est uniquement un squelette : il lit
`CDSE_CLIENT_ID` et `CDSE_CLIENT_SECRET` depuis `.env`, puis contient un
`TODO` explicite pour la recherche, le téléchargement et la préparation
Sentinel-1/2. Il n'est pas appelé par le pipeline actuel.

## 7. Éléments restant à faire

- configurer un compte Copernicus et implémenter la collecte Sentinel-1/2 ;
- découper, redimensionner et normaliser les images ;
- créer l'alignement temporel dans `data/processed/index.csv` ;
- configurer DVC et un stockage distant pour les données ;
- ajouter ultérieurement fusion, entraînement, évaluation et suivi
  d'expériences.

La partie Hub'Eau est terminée. La partie satellite et l'alignement sont en
attente.
