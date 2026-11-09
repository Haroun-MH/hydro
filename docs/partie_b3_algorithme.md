# B.3 Proposition des algorithmes de fusion

## 1. Architecture proposée

Le modèle utilise une fenêtre temporelle de 36 pas, soit 6 heures de mesures
H/Q. Chaque exemple contient donc :

- `x_t` : un tableau de dimension `(36, 2)` avec `H_m` et `Q_m3_s` ;
- `x_i` : une image satellite RGB de dimension `(3, 128, 128)` ;
- `m_i` : un masque égal à 1 si l'image est disponible et à 0 sinon ;
- `delta_t` : l'écart en minutes entre la date de l'image et la date de la
  cible.

L'encodeur temporel est un GRU bidirectionnel à une couche, avec 32 unités
dans chaque direction. Sa sortie est donc un vecteur de 64 valeurs. Une
projection linéaire `64 -> 64` produit `h_t`.

L'encodeur image est un ResNet-18 pré-entraîné sur ImageNet. Sa dernière couche
de classification est retirée. Le réseau est d'abord figé afin de limiter le
surapprentissage sur le petit jeu actuel, puis une phase optionnelle pourra
dégeler le dernier bloc si suffisamment d'images sont disponibles. La sortie
512-dimensionnelle est projetée par une couche `512 -> 64` pour produire `h_i`.

Le module de fusion est une concaténation enrichie :

```text
z = [h_t ; m_i * h_i + (1 - m_i) * e_abs ; delta_t / 360]
```

où `e_abs` est un vecteur appris de 64 valeurs utilisé lorsque l'image manque.
Le vecteur `z` a 129 dimensions. Une tête MLP `129 -> 64 -> 1` prédit le débit
à l'horizon choisi. Pour prévoir plusieurs horizons, la dernière couche
produira un vecteur de sorties plutôt qu'une seule valeur.

## 2. Module de fusion et passe avant

Pour un batch de taille `B`, on note :

```text
h_t = Proj_t(GRU(x_t))              h_t ∈ R^(B×64)
h_i = Proj_i(ResNet18(x_i))         h_i ∈ R^(B×64)
u_i = m_i h_i + (1 - m_i) e_abs     u_i ∈ R^(B×64)
z   = [h_t, u_i, delta_t / 360]     z   ∈ R^(B×129)
ŷ   = MLP(z)                        ŷ   ∈ R^B
```

Les variables H et Q sont normalisées avec les statistiques du seul ensemble
d'entraînement. L'image est normalisée avec les statistiques ImageNet si
ResNet-18 est utilisé. Les dates et les valeurs futures ne sont jamais
utilisées pour construire une fenêtre d'entrée.

### Pseudo-algorithme de la passe avant

```text
Entrées : x_t, x_i, m_i, delta_t, cible y
Paramètres : E_t, Proj_t, E_i, Proj_i, e_abs, MLP

1. h_t ← Proj_t(E_t(x_t))
2. si m_i = 1 alors
       h_i ← Proj_i(E_i(x_i))
   sinon
       h_i ← e_abs
3. u_i ← h_i avec la valeur masquée par m_i
4. z ← concaténer(h_t, u_i, delta_t / 360)
5. ŷ ← MLP(z)
6. retourner ŷ
```

Le calcul de l'image peut être évité pour les exemples où `m_i = 0`. Les
représentations image calculées pour les dates valides sont mises en cache
après extraction, avec l'identifiant du produit et la version du modèle.

## 3. Entraînement et fonction de perte

La perte principale est une erreur absolue moyenne, adaptée aux pics et moins
sensible que l'erreur quadratique aux valeurs extrêmes :

```text
L_task = mean(|ŷ - y|)
```

Pour empêcher la branche image d'être ignorée, une perte auxiliaire
temporelle est ajoutée pendant les premières expériences. Elle utilise une
petite tête sur `h_t` :

```text
L_time = mean(|H_time(h_t) - y|)
L = L_task + 0.2 * L_time
```

La perte auxiliaire est retirée si elle dégrade la validation. Une perte
contrastive n'est pas retenue au départ : le nombre de paires image-série sera
faible et l'alignement ne sera fiable qu'après la création de `index.csv`.

L'optimiseur proposé est AdamW, taux initial `1e-3` pour les projections et la
tête, poids décroissant `1e-4`, batch de 32 lorsque la mémoire le permet, et
arrêt anticipé sur la perte de validation. Le ResNet reste gelé pendant cette
phase.

### Pseudo-algorithme d'entraînement

```text
Pour chaque batch (x_t, x_i, m_i, delta_t, y) :
1. ŷ, h_t ← passe_avant(x_t, x_i, m_i, delta_t)
2. L_task ← MAE(ŷ, y)
3. L_time ← MAE(H_time(h_t), y)
4. L ← L_task + 0.2 * L_time
5. remettre les gradients à zéro
6. rétropropager L
7. mettre à jour les paramètres entraînables avec AdamW
8. calculer les métriques sur l'ensemble de validation
9. conserver le meilleur modèle et arrêter si la validation stagne
```

## 4. Modalités manquantes et déséquilibre

Une image absente est représentée explicitement par `m_i = 0`, le vecteur
`e_abs` et `delta_t = 0`. Le modèle peut donc produire une prévision à partir
de la série seule. Pendant l'entraînement, un modality dropout met
artificiellement `m_i` à zéro pour 20 % des exemples possédant une image.
Cette méthode évite que le modèle devienne inutilisable lorsque les passages
du satellite sont espacés ou que les nuages masquent la zone.

Le déséquilibre est traité par trois mesures simples :

1. normalisation séparée des H/Q et des pixels ;
2. projection des deux encodeurs dans la même dimension 64 ;
3. comparaison systématique avec les modèles unimodaux.

Une porte GMU ou une pondération apprise ne sera ajoutée que si les expériences
montrent qu'une modalité domine durablement. Il vaut mieux ne pas ajouter ce
paramètre avant d'avoir assez d'images pour l'estimer.

## 5. Baselines

Les comparaisons minimales sont :

1. **Baseline persistance** : la dernière valeur connue de Q est reprise pour
   l'horizon de prévision.
2. **Modèle temporel seul** : le GRU sur H/Q, sans image.
3. **Modèle image seul** : ResNet-18 gelé suivi d'une tête MLP, pour vérifier
   si l'image contient un signal prédictif.
4. **Fusion tardive** : moyenne pondérée des prédictions des modèles temporel
   et image, avec le poids choisi sur la validation.
5. **Modèle retenu** : concaténation intermédiaire avec masque.

Le modèle fusionné ne sera considéré utile que s'il améliore le GRU seul et la
persistance sur un jeu de test gardé à part.

## 6. Évaluation et ablations

Le découpage est chronologique : les premières dates servent à l'entraînement,
la période suivante à la validation et la dernière période au test. Il ne faut
pas mélanger des fenêtres qui se recouvrent entre ces ensembles, afin d'éviter
une fuite temporelle.

Les métriques sont :

- MAE en m³/s ;
- RMSE en m³/s ;
- coefficient de détermination R² ;
- erreur sur les pics, calculée sur les valeurs du dernier décile de Q ;
- rappel et précision d'un événement de crue défini par un seuil fixé sur
  l'entraînement.

Les ablations prévues sont :

- série seule contre fusion ;
- image seule contre fusion ;
- image remplacée par `e_abs` ;
- retrait de `delta_t` ;
- retrait de la perte auxiliaire ;
- concaténation contre fusion tardive ;
- ResNet gelé contre dernier bloc affiné ;
- taux de modality dropout 0 %, 20 % et 50 %.

Toutes les métriques seront calculées sur le même test et avec la même graine.

## 7. Intégration dans le pipeline

Les étapes de la Partie A à ajouter ou modifier sont :

1. **Alignement** : créer `data/processed/index.csv` après téléchargement,
   avec `image`, `date_utc`, `H_m`, `Q_m3_s`, `delta_minutes` et `image_present`.
2. **Fenêtrage** : construire les fenêtres de 36 pas sans utiliser de mesure
   postérieure à la cible.
3. **Extraction image** : appliquer la normalisation ResNet et enregistrer les
   vecteurs 64-dimensionnels dans un cache versionné par produit.
4. **Validation** : vérifier les dimensions 128 × 128, les dates, les clés
   d'alignement et le taux de modalités manquantes.
5. **Entraînement** : ajouter un module `src/fusion/` contenant le modèle, la
   perte, les baselines et le découpage chronologique.
6. **Évaluation** : écrire les métriques et les résultats d'ablation dans
   `data/processed/metrics.json`.

Tant que les images et l'index ne sont pas disponibles, seule la baseline
temporelle peut être exécutée. Le modèle proposé est donc une spécification
prête à implémenter, et non un entraînement déjà réalisé.
