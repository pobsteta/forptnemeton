# forptnemeton

Embeddings **ForPT** d'un nuage LiDAR, en raster, pour le paquet
[`nemeton`](https://github.com/pobsteta/nemeton) (N5 structurel, spec 067).

ForPT (ETH Zurich, 2026) est un encodeur LitePT-S de 12,4 M de paramètres,
pré-entraîné sans étiquettes sur des nuages ALS, ULS et MLS de forêts. Code
[`prs-eth/ForPT`](https://github.com/prs-eth/ForPT) sous MIT ; poids
[`prs-eth/ForPT`](https://huggingface.co/prs-eth/ForPT) sur Hugging Face :
**lire leur licence sur la carte du modèle avant usage.**

```r
library(forptnemeton)
forpt_environment()                  # ce qui manque (vide = prêt)
emb <- forpt_embed("cache/layers/lidar_nuage_norm", aoi = ugf, resolution = 10)
nemeton::indicateur_n5_distance_reference(ugf, embedding = emb)

# âge structurel des arbres, puis par UGF (T1 désaturé)
h <- nemeton::segment_houppiers(mnh, aoi = ugf)
ugf <- forpt_age("cache/layers/lidar_nuage_norm", houppiers = h, units = ugf)
nemeton::indicateur_t1_anciennete(ugf, age_structurel = "age_structurel")
```

## Environnement

L'encodeur tourne en Python par `reticulate` et exige un **GPU CUDA**
(LitePT appelle `flash_attn`, sans version CPU). Il faut, dans
l'environnement Python lié à `reticulate` : `torch`, `spconv`,
`torch_scatter`, `flash_attn`, `huggingface_hub`, et le paquet `forpt` du
dépôt ForPT (`pip install -e .` dans un clone). Suivre les instructions
d'installation de [LitePT](https://github.com/prs-eth/LitePT).

## Ce qu'il faut savoir

- `forpt_age()` travaille **arbre par arbre**, comme la tête d'âge de ForPT
  (FORage) : un houppier qui fusionne deux arbres est une mauvaise entrée.
  Le repère des nuages d'arbres (x, y recentrés, z au-dessus du sol) est une
  hypothèse à vérifier sur un échantillon FORage avant de croire les âges.
- Le nuage doit être **normalisé** (hauteurs au-dessus du sol) ; un nuage
  d'altitudes est refusé.
- Prétraitement identique aux configurations ForPT : recentrage, grille de
  5 cm, coordonnées en caractéristiques. L'embedding d'une cellule est la
  moyenne des 504 caractéristiques de sortie de l'encodeur.
- ForPT a été pré-entraîné sur des nuages denses (drone, mobile) : à
  ~10 retours/m² (LiDAR HD), une cellule de 10 m en contient bien moins que
  ses échantillons d'entraînement. Un embedding est une représentation, pas
  une mesure : l'utiliser par une distance à une référence (N5) ou une sonde.

Licence : GPL (>= 3).
