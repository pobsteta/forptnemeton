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
