# Partie B — Étude des approches de fusion de données

## B.1 Étude bibliographique

La fusion multimodale consiste à exploiter plusieurs sources qui décrivent le
même phénomène. Ici, les modalités prévues sont :

- une série temporelle structurée : hauteur `H_m` et débit `Q_m3_s`, toutes les
  10 minutes ;
- une image satellite autour de la station, prévue en 128 × 128 pixels ;
- une clé d'alignement fondée sur la date de l'image et le pas temporel
  hydrométrique le plus proche.

Les trois niveaux classiques sont les suivants.

### Comparaison des approches

| Approche / réf. | Principe | Avantages | Limites | Conditions d'usage |
|---|---|---|---|---|
| Fusion précoce (Baltrušaitis et al., 2019) | Mettre les variables des modalités dans une représentation commune avant le modèle principal. | Simple, peu coûteuse, un seul modèle à entraîner. | Les dimensions, échelles et fréquences doivent être compatibles; risque de perdre la structure propre à chaque modalité. | Données déjà alignées et correctement normalisées. |
| Fusion tardive (Kiela et al., 2019) | Un modèle par modalité produit une prédiction; les sorties sont combinées par vote, moyenne pondérée ou stacking. | Fonctionne avec des modalités hétérogènes; une modalité peut manquer; interprétation facile par comparaison des modèles. | Les interactions fines entre modalités sont peu exploitées; plusieurs modèles à maintenir. | Peu de données, systèmes existants séparés ou modalités souvent absentes. |
| Intermédiaire — concaténation | Un encodeur par modalité produit un vecteur latent; les vecteurs sont concaténés puis envoyés à une tête commune. | Bon compromis entre simplicité et interactions; dimensions maîtrisées. | Le choix des encodeurs et de la taille des représentations influence le résultat. | Modalités alignées par échantillon et volume modéré. |
| Intermédiaire — somme ou produit pondérés | Additionner ou multiplier les représentations après projection dans un espace commun, avec poids appris ou fixés. | Peu de paramètres; somme robuste et produit capable de modéliser des interactions. | Le produit est sensible aux échelles; moins expressif qu'une attention. | Modalités déjà projetées dans le même espace. |
| GMU (Arevalo et al., 2017) | Une porte apprise règle la contribution de chaque modalité : une représentation est sélectionnée ou mélangée selon le contexte. | Gère le déséquilibre entre modalités et améliore la robustesse aux modalités peu fiables. | Ajoute des paramètres et demande assez de données pour apprendre la porte. | Une modalité peut être bruitée ou moins informative selon les exemples. |
| FiLM (Perez et al., 2018) | Une modalité produit des paramètres d'affinité et de biais qui modulent les activations d'une autre modalité. | Modulation conditionnelle efficace, plus légère qu'une fusion tensorielle. | Direction de la modulation à choisir; interactions moins symétriques. | Une modalité sert de contexte, par exemple la série conditionnant l'image. |
| TFN (Zadeh et al., 2017) | Produit tensoriel extérieur explicite des vecteurs unimodaux, avec terme de biais. | Modélise les interactions de tous les ordres. | Explosion du nombre de paramètres et du coût mémoire. | Beaucoup de données et représentations de petite dimension. |
| LMF (Liu et al., 2018) | Approxime le tenseur de TFN par des facteurs de faible rang. | Conserve des interactions riches avec moins de mémoire et de calcul. | Le rang doit être choisi; reste trop complexe pour un très petit jeu de données. | Ressources suffisantes et besoin avéré d'interactions d'ordre élevé. |
| Pooling bilinéaire MCB (Fukui et al., 2016) | Approxime le produit bilinéaire par un compact bilinear pooling aléatoire. | Réduit le coût du bilinéaire complet; utile pour image-texte. | Dimension de sortie parfois importante; moins naturel pour une courte série temporelle. | Vision-langage ou vecteurs de caractéristiques déjà extraits. |
| Pooling bilinéaire MFB (Yu et al., 2017) | Factorise et compacte les interactions bilinéaires avec un mécanisme multiplicatif. | Plus paramétrique et souvent plus performant que MCB. | Plus difficile à régler et à justifier sur un petit volume. | Tâches de question-réponse visuelle ou interactions riches. |
| Attention croisée / MulT (Tsai et al., 2019) | Les tokens d'une modalité interrogent ceux d'une autre par attention; les représentations sont fusionnées à plusieurs niveaux. | Alignement souple et interactions locales; traite des séquences de longueurs différentes. | Coût quadratique potentiel, besoin de données et de régularisation. | Séquences suffisamment longues, appariées et volume d'apprentissage important. |
| Transformer multimodal (Vaswani et al., 2017; Tsai et al., 2019) | Plusieurs blocs d'attention apprennent les interactions entre tokens de différentes modalités. | Très expressif et extensible à plusieurs modalités. | Beaucoup de paramètres, risque de surapprentissage et interprétation moins directe. | Grand corpus, pré-entraînement ou modèles pré-entraînés disponibles. |
| Goulot d'attention | Limiter l'échange intermodal à quelques tokens latents ou requêtes apprises. | Réduit le coût et force le modèle à sélectionner l'information utile. | Le goulot peut supprimer une information importante; réglage supplémentaire. | Nombre élevé de tokens ou ressources GPU limitées. |
| Alignement contrastif de type CLIP (Radford et al., 2021) | Rapprocher les représentations d'une paire correspondante et éloigner les paires non correspondantes. | Espace commun réutilisable; utile lorsque les paires sont nombreuses et fiables. | Demande beaucoup de paires négatives; l'objectif de similarité ne suffit pas toujours pour prévoir une crue. | Pré-entraînement multimodal ou grand nombre de paires image-série. |
| Fusion par graphes | Représenter les modalités, dates, bandes ou stations par des nœuds et leurs relations par des arêtes; un GNN agrège les voisins. | Représente explicitement les relations spatiales et temporelles. | Construction du graphe et interprétation plus complexes; peu utile avec une seule station. | Plusieurs stations, bassins ou relations spatiales connues. |

### Difficultés propres au multimodal

1. **Hétérogénéité.** Une série est un tableau ordonné et une image est une
   grille de pixels. Elles ne peuvent pas être concaténées directement sans
   normalisation et encodeurs distincts.
2. **Alignement.** Une image n'est pas prise toutes les 10 minutes. Il faut
   associer son horodatage au pas hydrométrique le plus proche et conserver la
   distance temporelle. Une association trop éloignée peut créer un faux lien.
3. **Modalité manquante.** Une image peut ne pas être disponible à cause des
   nuages, du passage du satellite ou de l'authentification. Le modèle doit
   recevoir un masque de présence et pouvoir fonctionner avec la série seule.
4. **Déséquilibre.** La série est disponible à chaque pas alors que les images
   sont rares. Sans pondération, la branche temporelle peut dominer la perte.
5. **Coût de calcul.** Les transformers, TFN et pooling bilinéaires ont un coût
   disproportionné par rapport à 4 320 lignes et une seule station.
6. **Interprétabilité.** Pour une alerte de crue, il faut pouvoir distinguer
   l'effet du signal H/Q de l'apport de l'image; les poids de fusion et des
   ablations par modalité sont donc importants.

## B.2 Choix pour ce projet

### Données mesurées

La préparation actuelle fournit 4 320 lignes sur 30 jours, à une fréquence
régulière de 10 minutes. Les colonnes hydrométriques sont `H_m`, `Q_m3_s` et
leurs indicateurs de manque. Il n'y a actuellement aucune image préparée et
aucun `index.csv`, car l'accès Copernicus n'est pas disponible.

Le jeu est donc petit pour entraîner un modèle multimodal de bout en bout.
Il contient aussi une seule station : une fusion par graphe n'aurait pas de
relation spatiale à apprendre. Enfin, la fréquence de la série est beaucoup
plus fine que la revisite satellite, ce qui impose un alignement explicite et
un masque de disponibilité.

### Approche retenue

L'approche retenue est une **fusion intermédiaire légère par concaténation
après encodeurs séparés**, avec un **masque de modalité manquante** :

1. un petit encodeur 1D (ou GRU) résume une fenêtre de H/Q ;
2. un encodeur CNN pré-entraîné et gelé résume l'image 128 × 128 ;
3. les deux vecteurs sont projetés vers une dimension commune ;
4. les vecteurs, le masque image et la distance temporelle sont concaténés ;
5. une petite tête dense prédit la cible de crue.

Cette décision s'appuie sur les observations du projet :

| Critère | Constat | Conséquence |
|---|---|---|
| Nature des modalités | Tableau H/Q contre image 128 × 128. | Encodeur propre à chaque modalité, pas de concaténation brute. |
| Alignement | Pas de date d'image disponible aujourd'hui; la série est à 10 minutes et le satellite est beaucoup plus rare. | `index.csv`, date la plus proche, écart temporel et masque sont nécessaires. |
| Interactions | L'image peut apporter le contexte de l'occupation du sol, mais H/Q reste le signal direct. | Concaténation tardive des représentations, sans TFN au départ. |
| Modalités manquantes | Images absentes en attendant Copernicus et potentiellement absentes à certaines dates. | Branche image masquée et entraînement avec cas série seule. |
| Volume | 4 320 lignes, une station, images non disponibles. | Encodeur image gelé; pas de transformer entraîné de bout en bout. |
| Calcul | Pas de GPU requis pour un premier modèle simple. | CNN pré-entraîné + petite tête, plutôt que MulT ou TFN. |
| Interprétabilité | Décision de crue à expliquer. | Comparaison série seule / image seule / fusion et ablation du masque. |

### Ordre de mise en œuvre

1. Constituer `index.csv` après récupération des images et mesurer l'écart
   entre chaque image et le pas H/Q associé.
2. Établir une baseline série seule avec un modèle 1D ou GRU.
3. Ajouter un CNN gelé et la concaténation intermédiaire.
4. Ajouter le masque d'image et tester les dates sans image.
5. Comparer à une fusion tardive (deux modèles et moyenne pondérée).
6. N'envisager GMU ou FiLM que si le volume d'images justifie une porte ou une
   modulation apprises.

Les transformers multimodaux, TFN/LMF, MCB/MFB, CLIP et les graphes sont
pertinents dans d'autres régimes de données, mais ils ne sont pas le premier
choix justifié pour ce jeu court, monostation et actuellement unimodal.

## Références

- Baltrušaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and
  Taxonomy*, IEEE TPAMI, 2019.
  https://doi.org/10.1109/TPAMI.2018.2798607
- Arevalo et al., *Gated Multimodal Units for Information Fusion*, ICLR
  Workshop, 2017. https://arxiv.org/abs/1702.01992
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*,
  AAAI, 2018. https://arxiv.org/abs/1709.07871
- Zadeh et al., *Tensor Fusion Network for Multimodal Sentiment Analysis*,
  EMNLP, 2017. https://aclanthology.org/D17-1115/
- Liu et al., *Efficient Low-rank Multimodal Fusion with Modality-Specific
  Factors*, ACL, 2018. https://aclanthology.org/P18-1209/
- Fukui et al., *Multimodal Compact Bilinear Pooling for Visual Question
  Answering and Visual Grounding*, EMNLP, 2016.
  https://aclanthology.org/D16-1044/
- Yu et al., *Multi-modal Factorized Bilinear Pooling with Co-attention
  Learning for Visual Question Answering*, ICCV, 2017.
  https://openaccess.thecvf.com/content_ICCV_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language
  Sequences*, ACL, 2019. https://aclanthology.org/P19-1656/
- Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision*, ICML, 2021. https://arxiv.org/abs/2103.00020
- Vaswani et al., *Attention Is All You Need*, NeurIPS, 2017.
  https://arxiv.org/abs/1706.03762
- Songtao et al., *Multimodal Alignment and Fusion: A Survey*, IJCV, 2026.
  https://doi.org/10.1007/s11263-025-02667-1
