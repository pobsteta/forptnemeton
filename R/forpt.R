#' ForPT model information
#'
#' Constants of the ForPT encoder used by [forpt_embed()]: the pretrained
#' LitePT-S backbone of the ForPT model (ETH Zurich, 2026), weights
#' `prs-eth/ForPT` on Hugging Face, code `github.com/prs-eth/ForPT` (MIT).
#'
#' @return A list: `version` of the embedding recipe, `dimensions` (504),
#'   `grid_size` (0.05 m), `hf_repo`, `weights_file` (path of the pretrained
#'   backbone inside the Hugging Face repository).
#' @export
forpt_info <- function() {
  list(
    version = .FORPT_VERSION,
    dimensions = .FORPT_DIM,
    grid_size = 0.05,
    hf_repo = "prs-eth/ForPT",
    weights_file = .FORPT_WEIGHTS
  )
}

.FORPT_VERSION <- "forpt-ssl-epoch30-mean-v1"
.FORPT_DIM <- 504L
.FORPT_WEIGHTS <- "00_ssl_pretrain/pretrain-forpt/model/epoch_30.pth"

#' Default cache directory
#'
#' `getOption("forptnemeton.cache_dir")`, else the `NEMETON_CACHE_DIR`
#' environment variable + `/forpt`, else the user cache directory.
#' @export
forpt_cache_dir <- function() {
  opt <- getOption("forptnemeton.cache_dir")
  if (!is.null(opt)) return(opt)
  env <- Sys.getenv("NEMETON_CACHE_DIR")
  if (nzchar(env)) return(file.path(env, "forpt"))
  tools::R_user_dir("forptnemeton", "cache")
}

#' Check the Python environment of the encoder
#'
#' The encoder runs in Python through \pkg{reticulate}: `torch`, `spconv`,
#' `torch_scatter`, `flash_attn` and the `forpt` package of the ForPT
#' repository must be importable, and a CUDA GPU present (LitePT's attention
#' requires `flash_attn`, which has no CPU version).
#'
#' @return Character vector of what is missing (`"reticulate"`, a module
#'   name, or `"cuda"`); empty when the encoder can run.
#' @export
forpt_environment <- function() {
  if (!.forpt_reticulate_ok()) return("reticulate")
  mod <- .forpt_python()
  as.character(unlist(mod$environment()))
}

#' Download the pretrained ForPT backbone
#'
#' Downloads `forpt_info()$weights_file` from the Hugging Face repository
#' `prs-eth/ForPT` with the Python `huggingface_hub` package, into
#' `cache_dir`. Read the licence on the model card before use: the ForPT code
#' is MIT, the licence of the weights is stated on Hugging Face.
#'
#' @param cache_dir Directory. Default [forpt_cache_dir()].
#' @param overwrite Logical. Download again.
#' @return Path of the checkpoint, invisibly.
#' @export
forpt_weights <- function(cache_dir = forpt_cache_dir(), overwrite = FALSE) {
  dest <- file.path(cache_dir, "weights", .FORPT_WEIGHTS)
  if (file.exists(dest) && !overwrite) return(invisible(dest))
  if (!.forpt_reticulate_ok()) {
    cli::cli_abort("{.pkg reticulate} is required to download the ForPT weights.")
  }
  hf <- reticulate::import("huggingface_hub")
  chemin <- hf$hf_hub_download(repo_id = "prs-eth/ForPT", filename = .FORPT_WEIGHTS,
                               local_dir = file.path(cache_dir, "weights"))
  cli::cli_alert_info("ForPT weights: check the licence on {.url https://huggingface.co/prs-eth/ForPT}.")
  invisible(chemin)
}

#' ForPT embeddings of a LiDAR point cloud, as a raster
#'
#' Cuts the normalised point cloud into square cells of `resolution` metres
#' aligned on multiples of `resolution`, passes each cell through the frozen
#' ForPT encoder (LitePT-S, self-supervised on airborne, UAV and mobile
#' laser scans) and writes the mean of the encoder's output features: one
#' band per dimension (504), one pixel per cell. The raster is ready for
#' `nemeton::indicateur_n5_distance_reference(embedding =)` (structural N5,
#' spec 067 of nemeton) or `nemeton::embedding_summary()`.
#'
#' @param nuage Normalised point cloud (heights above ground): a table with
#'   `X`, `Y`, `Z` (optionally `Classification`), a `LAS` object of
#'   \pkg{lidR}, or LAZ files / a directory (read with \pkg{lidR}).
#' @param aoi Area of interest (sf, sfc, SpatVector), in the CRS of the cloud.
#' @param resolution Numeric. Cell size in metres. Default 10.
#' @param min_points Integer. Cells with fewer returns are `NA`. Default 200.
#' @param weights Path of the pretrained checkpoint. `NULL`:
#'   [forpt_weights()].
#' @param device `"cuda"` (default) or a torch device string.
#' @param batch_size Integer. Cells per forward pass. Default 16.
#' @param classes Integer. Classification codes kept when present. Default
#'   `c(1:5, 9)` (IGN LiDAR HD: unclassified, ground, vegetation, water).
#' @param cache_dir Directory of the cached COGs. Default [forpt_cache_dir()].
#' @param overwrite Logical. Recompute a cached raster.
#' @param embedder For tests and other encoders: a function taking a list of
#'   `n x 3` matrices and returning a `length(list) x d` matrix. `NULL`: the
#'   ForPT encoder.
#'
#' @return A SpatRaster with one band per dimension (`F001` ...), attribute
#'   `embedding_source = "forpt"`; its source is a COG in `cache_dir`, keyed
#'   by a hash of the recipe, the grid and the returns.
#'
#' @details
#' **Preprocessing** (as in the ForPT configurations): each cell is shifted
#' so that x and y are centred and z starts at its minimum, sampled on a
#' 5 cm grid, and fed with its coordinates as features.
#'
#' **Requirements**: a CUDA GPU (see [forpt_environment()]); the spec 067 of
#' nemeton (decision D-4) plans an occasional GPU (Scaleway L4) rather than
#' the CPU.
#'
#' **Pitfalls**: ForPT was pretrained on dense scans (UAV and mobile laser
#' among them); at about 10 returns per square metre (IGN LiDAR HD) a cell
#' holds far fewer points than its pretraining samples. The embedding is a
#' representation, not a measurement: use it through a reference distance
#' (N5) or a probe, never as an indicator by itself.
#'
#' @export
forpt_embed <- function(nuage, aoi, resolution = 10, min_points = 200L,
                        weights = NULL, device = "cuda", batch_size = 16L,
                        classes = c(1:5, 9), cache_dir = forpt_cache_dir(),
                        overwrite = FALSE, embedder = NULL) {
  zone <- .forpt_aoi(aoi)
  if (!is.numeric(resolution) || length(resolution) != 1L || resolution <= 0) {
    cli::cli_abort("{.arg resolution} must be a positive number.")
  }
  pts <- .forpt_points(nuage, zone, classes)

  # Grille calee sur les multiples de la resolution
  bb <- sf::st_bbox(zone)
  ext <- c(xmin = floor(bb[["xmin"]] / resolution) * resolution,
           xmax = ceiling(bb[["xmax"]] / resolution) * resolution,
           ymin = floor(bb[["ymin"]] / resolution) * resolution,
           ymax = ceiling(bb[["ymax"]] / resolution) * resolution)
  gabarit <- terra::rast(xmin = ext[["xmin"]], xmax = ext[["xmax"]],
                         ymin = ext[["ymin"]], ymax = ext[["ymax"]],
                         resolution = resolution, crs = sf::st_crs(zone)$wkt)
  dedans <- terra::rasterize(terra::vect(zone), gabarit, touches = TRUE)
  cellule <- terra::cellFromXY(gabarit, cbind(pts$X, pts$Y))
  garde <- !is.na(cellule) & !is.na(terra::values(dedans, mat = FALSE)[cellule])
  cellule <- cellule[garde]
  n_par <- tabulate(cellule, nbins = terra::ncell(gabarit))
  retenues <- which(n_par >= min_points)

  source <- if (is.null(embedder)) .FORPT_VERSION else "embedder_utilisateur"
  cle <- .forpt_hash(c(source, resolution, min_points, sf::st_crs(zone)$wkt,
                       format(ext, nsmall = 0), length(cellule),
                       sprintf("%.3f", c(sum(pts$X[garde]), sum(pts$Y[garde]), sum(pts$Z[garde])))))
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  fichier <- file.path(cache_dir, sprintf("forpt_%s.tif", substr(cle, 1, 16)))

  if (file.exists(fichier) && !overwrite) {
    cli::cli_alert_success("ForPT: cache hit ({.file {basename(fichier)}})")
  } else {
    if (!length(retenues)) {
      cli::cli_abort("No cell of {.arg aoi} holds {min_points} returns or more.")
    }
    cli::cli_alert_info("ForPT: {length(retenues)} cell{?s} of {resolution} m to embed")
    groupes <- split(which(garde), factor(cellule, levels = retenues))
    cellules <- lapply(groupes, function(i) cbind(pts$X[i], pts$Y[i], pts$Z[i]))
    if (is.null(embedder)) embedder <- .forpt_embedder(weights, device, batch_size, cache_dir)
    v <- embedder(unname(cellules))
    v <- as.matrix(v)
    if (nrow(v) != length(retenues)) {
      cli::cli_abort("The embedder returned {nrow(v)} rows for {length(retenues)} cells.")
    }
    out <- terra::rast(gabarit, nlyrs = ncol(v))
    m <- matrix(NA_real_, terra::ncell(gabarit), ncol(v))
    m[retenues, ] <- v
    terra::values(out) <- m
    .forpt_write_cog(out, fichier)
    cli::cli_alert_success("ForPT: written {.file {basename(fichier)}}")
  }
  r <- terra::rast(fichier)
  names(r) <- sprintf("F%03d", seq_len(terra::nlyr(r)))
  attr(r, "embedding_source") <- "forpt"
  r
}

# --------------------------------------------------------------- internes ---

.forpt_reticulate_ok <- function() requireNamespace("reticulate", quietly = TRUE)

.forpt_python <- function() {
  reticulate::import_from_path("forpt_embed",
                               path = system.file("python", package = "forptnemeton"),
                               delay_load = FALSE)
}

# Encodeur ForPT : verifie l'environnement, charge les poids une fois.
.forpt_embedder <- function(weights, device, batch_size, cache_dir) {
  manque <- forpt_environment()
  if (length(manque)) {
    cli::cli_abort(c(
      "The ForPT encoder cannot run here: missing {.val {manque}}.",
      i = "It needs a CUDA GPU with torch, spconv, torch_scatter, flash_attn and the ForPT repository (prs-eth/ForPT)."
    ))
  }
  if (is.null(weights)) weights <- forpt_weights(cache_dir)
  mod <- .forpt_python()
  modele <- mod$load_backbone(weights, device = device)
  function(cellules) {
    mod$embed_cells(modele, lapply(cellules, reticulate::r_to_py),
                    batch_size = as.integer(batch_size), device = device)
  }
}

.forpt_aoi <- function(aoi) {
  if (inherits(aoi, "SpatVector")) aoi <- sf::st_as_sf(aoi)
  if (inherits(aoi, "sf")) aoi <- sf::st_geometry(aoi)
  if (!inherits(aoi, "sfc")) cli::cli_abort("{.arg aoi} must be an sf, sfc or SpatVector.")
  if (is.na(sf::st_crs(aoi))) cli::cli_abort("{.arg aoi} has no CRS.")
  sf::st_union(sf::st_make_valid(aoi))
}

# Points du nuage dans l'emprise, filtres (classes, hauteurs plausibles).
.forpt_points <- function(nuage, zone, classes) {
  d <- .forpt_lire(nuage, zone)
  if (!all(c("X", "Y", "Z") %in% names(d))) {
    cli::cli_abort("The cloud needs columns {.field X}, {.field Y} and {.field Z}.")
  }
  garde <- is.finite(d$Z) & d$Z >= -1 & d$Z <= 60
  if (!is.null(d$Classification) && length(classes)) {
    garde <- garde & d$Classification %in% classes
  }
  if (length(d$Z) && stats::median(d$Z, na.rm = TRUE) > 60) {
    cli::cli_abort(c(
      "The median Z of the cloud is above 60 m: these look like altitudes, not heights.",
      i = "Normalise the cloud (e.g. {.code lidR::normalize_height()}) first."
    ))
  }
  d[garde, c("X", "Y", "Z"), drop = FALSE]
}

.forpt_lire <- function(nuage, zone) {
  if (inherits(nuage, "LAS")) return(as.data.frame(nuage@data))
  if (is.data.frame(nuage) || is.matrix(nuage)) return(as.data.frame(nuage))
  if (is.character(nuage)) {
    fichiers <- unlist(lapply(nuage, function(p) {
      if (dir.exists(p)) list.files(p, pattern = "\\.la[sz]$", full.names = TRUE, ignore.case = TRUE)
      else p
    }))
    fichiers <- fichiers[file.exists(fichiers)]
    if (!length(fichiers)) cli::cli_abort("No {.file .laz} / {.file .las} file found in {.arg nuage}.")
    if (!.forpt_lidr_ok()) cli::cli_abort("{.pkg lidR} is required to read LAZ files.")
    bb <- sf::st_bbox(zone)
    filtre <- sprintf("-keep_xy %f %f %f %f", bb[["xmin"]], bb[["ymin"]], bb[["xmax"]], bb[["ymax"]])
    morceaux <- lapply(fichiers, .forpt_lire_dalle, filtre = filtre)
    morceaux <- morceaux[!vapply(morceaux, is.null, logical(1))]
    if (!length(morceaux)) return(data.frame(X = numeric(0), Y = numeric(0), Z = numeric(0)))
    communes <- Reduce(intersect, lapply(morceaux, names))
    return(do.call(rbind, lapply(morceaux, `[`, communes)))
  }
  cli::cli_abort("{.arg nuage} must be a point table, a LAS object, or LAZ files.")
}

.forpt_lidr_ok <- function() requireNamespace("lidR", quietly = TRUE)

.forpt_lire_dalle <- function(fichier, filtre) {
  las <- lidR::readLAS(fichier, select = "xyzc", filter = filtre)
  if (lidR::is.empty(las)) return(NULL)
  as.data.frame(las@data)
}

.forpt_hash <- function(x) {
  f <- tempfile()
  on.exit(unlink(f), add = TRUE)
  writeLines(as.character(x), f, useBytes = TRUE)
  unname(tools::md5sum(f))
}

.forpt_write_cog <- function(r, fichier) {
  tmp <- paste0(fichier, ".part.tif")
  on.exit(unlink(tmp), add = TRUE)
  # Le pilote COG passe par un raster MEM qui ne connait pas COMPRESS :
  # avertissement GDAL sans consequence, la compression s'applique au COG.
  withCallingHandlers(
    terra::writeRaster(r, tmp, filetype = "COG", datatype = "FLT4S",
                       gdal = c("COMPRESS=DEFLATE"), overwrite = TRUE),
    warning = function(w) {
      if (grepl("driver MEM does not support creation option", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
  file.rename(tmp, fichier)
}
