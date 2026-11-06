# Tests de forpt_embed() : la partie R (grille, filtres, cache), avec un
# encodeur de substitution. L'encodeur ForPT lui-meme exige un GPU CUDA.

.zone <- function(cote = 40) {
  sf::st_sfc(sf::st_polygon(list(rbind(c(0, 0), c(cote, 0), c(cote, cote), c(0, cote), c(0, 0)))),
             crs = 2154)
}

.nuage <- function(n = 8000, cote = 40, graine = 1) {
  set.seed(graine)
  data.frame(X = runif(n, 0, cote), Y = runif(n, 0, cote), Z = runif(n, 0, 25),
             Classification = 5L)
}

# encodeur factice : (nombre de points, hauteur moyenne, x moyen)
.factice <- function(cellules) {
  t(vapply(cellules, function(m) c(nrow(m), mean(m[, 3]), mean(m[, 1])), numeric(3)))
}

test_that("forpt_info gives the encoder constants", {
  i <- forpt_info()
  expect_equal(i$dimensions, 504L)
  expect_equal(i$grid_size, 0.05)
  expect_equal(i$hf_repo, "prs-eth/ForPT")
  expect_match(i$weights_file, "epoch_30.pth$")
})

test_that("forpt_embed builds an aligned raster, one band per dimension", {
  withr::local_options(forptnemeton.cache_dir = withr::local_tempdir())
  vu <- list()
  enc <- function(cellules) { vu <<- cellules; .factice(cellules) }
  r <- forpt_embed(.nuage(), .zone(), resolution = 10, min_points = 50, embedder = enc)
  expect_s4_class(r, "SpatRaster")
  expect_equal(terra::nlyr(r), 3)
  expect_equal(names(r), c("F001", "F002", "F003"))
  expect_equal(attr(r, "embedding_source"), "forpt")
  expect_equal(terra::res(r), c(10, 10))
  expect_equal(as.vector(terra::ext(r)), c(xmin = 0, xmax = 40, ymin = 0, ymax = 40))
  expect_equal(length(vu), 16)
  # chaque cellule recoit ses propres points, en coordonnees absolues
  expect_true(all(vapply(vu, function(m) all(m[, 1] >= 0 & m[, 1] <= 40), logical(1))))
  n <- terra::values(r[[1]], mat = FALSE)
  expect_equal(sum(n), 8000)
  # cellule de x moyen ~5 a gauche
  expect_true(all(abs(terra::values(r[[3]], mat = FALSE)[c(1, 5, 9, 13)] - 5) < 1))
})

test_that("cells under min_points are NA and filters apply", {
  withr::local_options(forptnemeton.cache_dir = withr::local_tempdir())
  d <- .nuage(n = 2000)
  d$Classification[1:500] <- 6L                     # batiment : ecarte
  d$Z[501:520] <- 80                                 # bruit : ecarte
  r <- forpt_embed(d, .zone(), resolution = 10, min_points = 90, embedder = .factice)
  v <- terra::values(r[[1]], mat = FALSE)
  expect_true(any(is.na(v)))
  expect_true(all(v[!is.na(v)] >= 90))
  expect_lte(sum(v, na.rm = TRUE), 2000 - 500 - 20)
})

test_that("the cache is keyed by content", {
  dossier <- withr::local_tempdir()
  appels <- 0
  enc <- function(cellules) { appels <<- appels + 1; .factice(cellules) }
  a <- forpt_embed(.nuage(), .zone(), resolution = 20, min_points = 10, embedder = enc,
                   cache_dir = dossier)
  b <- forpt_embed(.nuage(), .zone(), resolution = 20, min_points = 10, embedder = enc,
                   cache_dir = dossier)
  expect_equal(appels, 1)
  expect_equal(terra::sources(a), terra::sources(b))
  forpt_embed(.nuage(graine = 2), .zone(), resolution = 20, min_points = 10, embedder = enc,
              cache_dir = dossier)
  expect_equal(appels, 2)
  forpt_embed(.nuage(), .zone(), resolution = 20, min_points = 10, embedder = enc,
              cache_dir = dossier, overwrite = TRUE)
  expect_equal(appels, 3)
  expect_length(list.files(dossier, pattern = "^forpt_.*\\.tif$"), 2)
})

test_that("forpt_embed refuses bad inputs", {
  withr::local_options(forptnemeton.cache_dir = withr::local_tempdir())
  alt <- .nuage(); alt$Z <- alt$Z + 400
  expect_error(forpt_embed(alt, .zone(), embedder = .factice), "altitudes")
  expect_error(forpt_embed(.nuage(), .zone(), resolution = -1, embedder = .factice), "positive")
  expect_error(forpt_embed(.nuage(n = 10), .zone(), min_points = 50, embedder = .factice),
               "No cell")
  expect_error(forpt_embed(.nuage()[, c("X", "Y")], .zone(), embedder = .factice), "columns")
  expect_error(forpt_embed(.nuage(), sf::st_sfc(sf::st_point(c(0, 0))), embedder = .factice), "CRS")
  expect_error(forpt_embed(list(1), .zone(), embedder = .factice), "point table")
  mauvais <- function(cellules) matrix(0, 1, 3)
  expect_error(forpt_embed(.nuage(), .zone(), resolution = 10, min_points = 10, embedder = mauvais),
               "returned 1 rows")
})

test_that("LAZ tiles are read tile by tile (lidR simulated)", {
  withr::local_options(forptnemeton.cache_dir = withr::local_tempdir())
  d <- .nuage()
  withr::with_tempdir({
    file.create(c("a.laz", "b.laz"))
    local_mocked_bindings(
      .forpt_lidr_ok = function() TRUE,
      .forpt_lire_dalle = function(fichier, filtre) {
        expect_match(filtre, "^-keep_xy ")
        if (basename(fichier) == "a.laz") d[d$X < 20, ] else d[d$X >= 20, ]
      }
    )
    r <- forpt_embed(getwd(), .zone(), resolution = 10, min_points = 50, embedder = .factice)
    expect_equal(sum(terra::values(r[[1]], mat = FALSE)), 8000)
  })
  local_mocked_bindings(.forpt_lidr_ok = function() FALSE)
  withr::with_tempdir({
    file.create("a.laz")
    expect_error(forpt_embed(getwd(), .zone(), embedder = .factice), "lidR")
  })
})

test_that("the encoder path reports what is missing", {
  local_mocked_bindings(.forpt_reticulate_ok = function() FALSE)
  expect_equal(forpt_environment(), "reticulate")
  expect_error(forpt_embed(.nuage(), .zone(), resolution = 10, min_points = 10,
                           cache_dir = withr::local_tempdir()),
               "missing")
})

test_that("forpt_cache_dir honours the option and NEMETON_CACHE_DIR", {
  withr::local_options(forptnemeton.cache_dir = "/tmp/x")
  expect_equal(forpt_cache_dir(), "/tmp/x")
  withr::local_options(forptnemeton.cache_dir = NULL)
  withr::local_envvar(NEMETON_CACHE_DIR = "/tmp/n")
  expect_equal(forpt_cache_dir(), "/tmp/n/forpt")
})
