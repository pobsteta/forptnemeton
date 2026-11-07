# forptnemeton 2026.11.3

- `forpt_age()` : nuages d'arbres mis dans le **repère de FORage** (minimum
  à 0 sur chaque axe, coin de la boîte englobante à l'origine), vérifié sur
  `pc.tar` du jeu `prs-eth/forpt-forage`. La version 2026.11.2 centrait x et
  y sur la moyenne de l'arbre : les âges auraient été lus hors du repère
  d'entraînement.
- `forpt_age()` avertit que la tête d'âge a été entraînée sur des épicéas
  et des pins de Norvège (FORage : 1 521 arbres, 1 à 348 ans) : l'âge des
  autres essences est une extrapolation.

# forptnemeton 2026.11.2

- `forpt_age(nuage, houppiers, units)` : âge structurel de chaque arbre par
  la tête d'âge de ForPT (affinée sur FORage, un nuage par arbre), à partir
  des retours de son houppier (par exemple `nemeton::segment_houppiers()`),
  puis médiane et quartiles par UGF, prêts pour
  `nemeton::indicateur_t1_anciennete(age_structurel =)` (T1 désaturé).
  Nuage d'arbre recentré en x, y, z = hauteur au-dessus du sol, au plus
  8 192 points : **hypothèse de repère à vérifier** sur un échantillon
  FORage, que le dépôt ForPT ne documente pas.
- `forpt_weights(fichier =)` télécharge aussi la tête d'âge ;
  `forpt_info()$age_weights_file`.

# forptnemeton 2026.11.1

- Première version (spec 067 de `nemeton`, chantier C).
- `forpt_embed()` : embeddings ForPT d'un nuage LiDAR normalisé, en raster
  (une bande par dimension, 504 ; une cellule de `resolution` mètres alignée
  sur les multiples de la résolution), mis en cache en COG adressé par
  contenu. Prétraitement des configurations ForPT : recentrage
  (`CenterShift`), grille de 5 cm, coordonnées en caractéristiques ;
  embedding = moyenne des sorties de l'encodeur, comme la tête d'âge.
- `forpt_weights()` (poids pré-entraînés depuis Hugging Face),
  `forpt_environment()` (ce qui manque : reticulate, modules, CUDA),
  `forpt_info()`, `forpt_cache_dir()`.
- L'encodeur exige un GPU CUDA (LitePT appelle `flash_attn`). La partie R est
  testée avec un encodeur de substitution ; le chemin Python n'a pas encore
  tourné sur GPU.
