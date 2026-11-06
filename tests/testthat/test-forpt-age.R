# Tests de forpt_age() : decoupage par houppier, cadre des nuages, synthese
# par UGF, avec un predicteur de substitution (la tete d'age exige un GPU).

.houppiers <- function() {
  centres <- list(c(5, 5), c(15, 5), c(25, 5), c(35, 35))
  sf::st_sf(houppier_id = 1:4,
            geometry = sf::st_sfc(lapply(centres, function(cc) sf::st_buffer(sf::st_point(cc), 3)),
                                  crs = 2154))
}

.nuage_arbres <- function(n_par = 300, graine = 1) {
  set.seed(graine)
  centres <- rbind(c(5, 5), c(15, 5), c(25, 5))
  hauteurs <- c(10, 20, 30)
  d <- do.call(rbind, lapply(1:3, function(k) {
    r <- sqrt(runif(n_par)) * 2.5; t <- runif(n_par, 0, 2 * pi)
    data.frame(X = centres[k, 1] + r * cos(t), Y = centres[k, 2] + r * sin(t),
               Z = runif(n_par, 0, hauteurs[k]))
  }))
  # quelques retours sous le 4e houppier seulement
  rbind(d, data.frame(X = runif(10, 34, 36), Y = runif(10, 34, 36), Z = runif(10, 0, 5)))
}

# predicteur factice : 3 x la hauteur maximale de l'arbre ; verifie le cadre
.age_factice <- function(arbres) {
  vapply(arbres, function(m) {
    stopifnot(abs(mean(m[, 1])) < 1e-8, abs(mean(m[, 2])) < 1e-8, min(m[, 3]) >= 0)
    3 * max(m[, 3])
  }, numeric(1))
}

test_that("forpt_age dates each crown from its own returns, recentred", {
  h <- forpt_age(.nuage_arbres(), .houppiers(), min_points = 50, predicteur = .age_factice)
  expect_s3_class(h, "sf")
  expect_equal(h$n_points, c(300, 300, 300, 10))
  expect_equal(h$age_statut, c("calcule", "calcule", "calcule", "trop_peu_points"))
  expect_equal(round(h$age_structurel[1:3] / 10), c(3, 6, 9))
  expect_true(is.na(h$age_structurel[4]))
})

test_that("forpt_age summarises tree ages per unit", {
  units <- sf::st_sf(id = 1:2, geometry = sf::st_sfc(
    sf::st_polygon(list(rbind(c(0, 0), c(20, 0), c(20, 10), c(0, 10), c(0, 0)))),
    sf::st_polygon(list(rbind(c(20, 0), c(40, 0), c(40, 40), c(20, 40), c(20, 0)))), crs = 2154))
  u <- forpt_age(.nuage_arbres(), .houppiers(), units = units, min_points = 50,
                 predicteur = function(a) c(40, 60, 150))
  expect_equal(u$age_structurel, c(50, 150))
  expect_equal(u$n_arbres, c(2L, 1L))
  expect_equal(u$age_q25, c(45, 150))
  expect_equal(u$age_q75, c(55, 150))
  # pret pour nemeton::indicateur_t1_anciennete(age_structurel = "age_structurel")
  expect_true(is.numeric(u$age_structurel))
})

test_that("forpt_age handles no datable tree and bad predictors", {
  h <- forpt_age(.nuage_arbres(n_par = 20), .houppiers(), min_points = 50,
                 predicteur = function(a) stop("not called"))
  expect_true(all(is.na(h$age_structurel)))
  expect_error(forpt_age(.nuage_arbres(), .houppiers(), min_points = 50,
                         predicteur = function(a) 1),
               "returned 1 values for 3 trees")
  expect_error(forpt_age(.nuage_arbres(), sf::st_geometry(.houppiers())), "sf object")
  expect_error(forpt_age(.nuage_arbres(), .houppiers(), units = data.frame(),
                         predicteur = .age_factice), "sf object")
})

test_that("the age path reports what is missing", {
  local_mocked_bindings(.forpt_reticulate_ok = function() FALSE)
  expect_error(forpt_age(.nuage_arbres(), .houppiers(), min_points = 50), "missing")
  expect_equal(forpt_info()$age_weights_file, "04_forage/age-forpt-ft-p100/model/model_best.pth")
})
