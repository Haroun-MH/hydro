# Checklist — Projet 3 « Prévision de crues »

## Fait

- [x] Documentation des sources Hub'Eau, Sentinel-1 et Sentinel-2.
- [x] Collecte Hub'Eau automatisée.
- [x] Nettoyage des dates, doublons et valeurs invalides.
- [x] Conversion de H en mètres et de Q en m³/s.
- [x] Rééchantillonnage toutes les 10 minutes.
- [x] Interpolation limitée et indicateurs de valeurs manquantes.
- [x] Génération de `data/processed/timeseries_clean.csv`.
- [x] Configuration externalisée dans `configs/params.yaml`.
- [x] Variables sensibles prévues dans `.env`.
- [x] Tests qualité et intégration continue.
- [x] Rapport DataOps dans `rapport_dataops_projet3.md`.
- [x] Étude bibliographique de fusion dans `docs/partie_b_fusion.md`.
- [x] Proposition d'algorithme dans `docs/partie_b3_algorithme.md`.
- [x] PDFs générés avec une mise en page simple.

## Commencé mais incomplet

- [ ] `src/satellite.py` : squelette créé, mais collecte Sentinel-1/2 non implémentée.
- [ ] Partie B.3 : architecture définie, mais modèle non codé ni entraîné.
- [ ] Pipeline multimodal : préparation prévue, mais dépendante des images satellite.
- [ ] Rapport qualité intermodal : structure prévue, mais impossible sans images.

## Pas encore fait

- [ ] Compte et accès Copernicus.
- [ ] Téléchargement des images Sentinel-1/2.
- [ ] Préparation des images en 128 × 128.
- [ ] Création de `data/processed/index.csv`.
- [ ] Alignement temporel images/H/Q.
- [ ] Implémentation PyTorch du modèle de fusion.
- [ ] Entraînement et comparaison des baselines.
- [ ] Évaluation et études d'ablation.
- [ ] Versionnement des données avec DVC.
- [ ] Stockage distant des données.
- [ ] Fusion, entraînement et évaluation finaux.
