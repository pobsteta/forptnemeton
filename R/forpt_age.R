#' Structural age of trees and units from the LiDAR point cloud (ForPT)
#'
#' Predicts the age of each tree with the ForPT age head (LitePT-S fine-tuned
#' on the FORage benchmark, one point cloud per tree), from the returns that
#' fall in its crown, then summarises it per unit. The result feeds
#' `nemeton::indicateur_t1_anciennete(age_structurel =)` ("T1 desature",
#' spec 067 of nemeton): BD Foret and NDVI saturate at 100 and 120 years, a
#' structural age does not.
#'
#' @param nuage Normalised point cloud, as in [forpt_embed()].
#' @param houppiers sf polygons, one per tree crown, e.g. from
#'   `nemeton::segment_houppiers()`.
#' @param units Optional sf polygons (UGF). When given, the tree ages are
#'   summarised per unit.
#' @param min_points Integer. Trees with fewer returns get no age. Default 50.
#' @param weights Path of the age checkpoint. `NULL`: downloaded by
#'   [forpt_weights()] (`fichier = forpt_info()$age_weights_file`).
#' @param device,batch_size,classes As in [forpt_embed()].
#' @param predicteur For tests and other models: a function taking a list of
#'   `n x 3` matrices (one tree each) and returning one age per tree. `NULL`:
#'   the ForPT age head.
#'
#' @return Without `units`: `houppiers` with `age_structurel` (years),
#'   `n_points` and `age_statut` (`calcule` or `trop_peu_points`). With
#'   `units`: `units` with `age_structurel` (median age of the trees whose
#'   crown centroid falls in the unit), `age_q25`, `age_q75` and `n_arbres`
#'   (`NA` and 0 for a unit without dated tree).
#'
#' @details
#' **Tree clouds.** The returns inside a crown polygon, shifted so that their
#' minimum is 0 on each axis (corner of the bounding box at the origin), which
#' is the frame of the FORage training clouds (checked on `pc.tar`); at most
#' 8,192 points (random draw; FORage uses farthest-point sampling), sampled on
#' a 5 cm grid.
#'
#' **Training domain.** FORage holds 1,521 Norway spruce and Scots pine trees
#' from Norway (aerial LiDAR HD, mobile and terrestrial scans), aged 1 to 348
#' years (median 42). Ages of broadleaves and of other species are
#' extrapolations; the function says so when it runs the ForPT head.
#'
#' **Pitfalls** (fiche T1): a structural age is not an age; a stand on a poor
#' site looks younger; FORage trees are Norwegian conifers; crowns from a 2D
#' segmentation cut or merge trees, and dominated trees are hidden; the age
#' head is trained on single trees, a merged crown is a wrong input.
#'
#' @export
forpt_age <- function(nuage, houppiers, units = NULL, min_points = 50L,
                      weights = NULL, device = "cuda", batch_size = 16L,
                      classes = c(1:5, 9), predicteur = NULL) {
  if (!inherits(houppiers, "sf")) cli::cli_abort("{.arg houppiers} must be an sf object.")
  if (is.na(sf::st_crs(houppiers))) cli::cli_abort("{.arg houppiers} has no CRS.")
  n <- nrow(houppiers)
  zone <- sf::st_union(sf::st_geometry(houppiers))
  pts <- .forpt_points(nuage, zone, classes)

  # Retour -> houppier qui le contient
  arbre <- rep(NA_integer_, nrow(pts))
  if (nrow(pts) && n) {
    v <- terra::vect(sf::st_geometry(houppiers))
    ex <- as.matrix(terra::extract(v, cbind(pts$X, pts$Y)))
    id <- ex[, "id.x"]
    ok <- is.finite(id)
    arbre[ex[ok, "id.y"]] <- as.integer(id[ok])
  }
  n_points <- tabulate(arbre[!is.na(arbre)], nbins = n)
  dates <- which(n_points >= min_points)

  age <- rep(NA_real_, n)
  if (length(dates)) {
    groupes <- split(which(!is.na(arbre) & arbre %in% dates), factor(arbre[!is.na(arbre) & arbre %in% dates], levels = dates))
    nuages <- lapply(groupes, function(i) {
      # repere FORage : coordonnees ramenees a leur minimum sur les trois axes
      # (coin de la boite englobante a l'origine), verifie sur pc.tar
      cbind(pts$X[i] - min(pts$X[i]), pts$Y[i] - min(pts$Y[i]), pts$Z[i] - min(pts$Z[i]))
    })
    if (is.null(predicteur)) {
      cli::cli_alert_warning(c(
        "ForPT age head trained on Norwegian spruce and pine (FORage): ",
        "ages of other species are extrapolations."))
      predicteur <- .forpt_predicteur_age(weights, device, batch_size)
    }
    a <- as.numeric(predicteur(unname(nuages)))
    if (length(a) != length(dates)) {
      cli::cli_abort("The age model returned {length(a)} values for {length(dates)} trees.")
    }
    age[dates] <- a
  }
  houppiers$age_structurel <- age
  houppiers$n_points <- n_points
  houppiers$age_statut <- ifelse(n_points >= min_points, "calcule", "trop_peu_points")
  if (is.null(units)) return(houppiers)

  # Synthese par UGF : centroide du houppier dans l'UGF
  if (!inherits(units, "sf")) cli::cli_abort("{.arg units} must be an sf object.")
  centres <- suppressWarnings(sf::st_centroid(sf::st_geometry(houppiers)))
  centres <- sf::st_transform(centres, sf::st_crs(units))
  ugf <- vapply(sf::st_intersects(centres, units), function(i) if (length(i)) i[1] else NA_integer_,
                integer(1))
  resume <- function(i, f) {
    a <- age[!is.na(ugf) & ugf == i & !is.na(age)]
    if (length(a)) f(a) else NA_real_
  }
  k <- seq_len(nrow(units))
  units$age_structurel <- vapply(k, resume, numeric(1), f = stats::median)
  units$age_q25 <- vapply(k, resume, numeric(1), f = function(a) unname(stats::quantile(a, 0.25)))
  units$age_q75 <- vapply(k, resume, numeric(1), f = function(a) unname(stats::quantile(a, 0.75)))
  units$n_arbres <- vapply(k, function(i) sum(!is.na(ugf) & ugf == i & !is.na(age)), integer(1))
  units
}

# Tete d'age ForPT : verifie l'environnement, charge le modele une fois.
.forpt_predicteur_age <- function(weights, device, batch_size) {
  manque <- forpt_environment()
  if (length(manque)) {
    cli::cli_abort(c(
      "The ForPT age model cannot run here: missing {.val {manque}}.",
      i = "It needs a CUDA GPU with torch, spconv, torch_scatter, flash_attn and the ForPT repository (prs-eth/ForPT)."
    ))
  }
  if (is.null(weights)) weights <- forpt_weights(fichier = .FORPT_AGE_WEIGHTS)
  mod <- .forpt_python()
  modele <- mod$load_age_model(weights, device = device)
  function(arbres) {
    as.numeric(mod$predict_ages(modele, lapply(arbres, reticulate::r_to_py),
                                batch_size = as.integer(batch_size), device = device))
  }
}
