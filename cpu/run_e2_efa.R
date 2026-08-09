#!/usr/bin/env Rscript

# Tetrachoric EFA for the 19 PKU-SafeRLHF harm-category indicators.
#
# Inputs are text-free binary matrices written by run_native_audit.py. The
# primary EFA conditions on the released unsafe state, so the all-zero category
# pattern that encodes the binary safety boundary cannot become the dominant
# factor. It uses 1,000 independent Bernoulli simulations at observed
# conditional category prevalences for parallel analysis.

args <- commandArgs(trailingOnly = TRUE)
n_iter <- 1000L
seed <- 20260809L
for (arg in args) {
  if (startsWith(arg, "--n-iter=")) {
    n_iter <- as.integer(sub("--n-iter=", "", arg))
  } else if (startsWith(arg, "--seed=")) {
    seed <- as.integer(sub("--seed=", "", arg))
  } else {
    stop(paste("Unknown argument:", arg))
  }
}
if (is.na(n_iter) || n_iter < 1L) stop("--n-iter must be a positive integer")

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_guess <- if (length(file_argument)) {
  sub("^--file=", "", file_argument[[1L]])
} else {
  "cpu/run_e2_efa.R"
}
script_path <- normalizePath(script_guess, mustWork = FALSE)
if (!file.exists(script_path)) script_path <- normalizePath("cpu/run_e2_efa.R")
experiment_root <- normalizePath(file.path(dirname(script_path), ".."))
workspace_root <- normalizePath(file.path(experiment_root, ".."))
input_dir <- file.path(experiment_root, "cpu", "intermediate")
result_dir <- file.path(experiment_root, "cpu", "results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
published_result_dir <- result_dir
lock_path <- file.path(published_result_dir, ".e2_efa.lock")
if (file.exists(lock_path)) {
  stop(paste("E2 lock exists:", lock_path, "Remove it only after verifying no E2 run is active."))
}
writeLines(
  paste("pid", Sys.getpid(), "started_utc", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  lock_path
)
on.exit(unlink(lock_path), add = TRUE)
staging_result_dir <- file.path(published_result_dir, paste0(".e2_efa_staging_", Sys.getpid()))
dir.create(staging_result_dir, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(staging_result_dir, recursive = TRUE, force = TRUE), add = TRUE)
result_dir <- staging_result_dir
local_r_library <- file.path(experiment_root, "cpu", ".Rlib")
if (dir.exists(local_r_library)) {
  .libPaths(c(local_r_library, .libPaths()))
}

required_packages <- c("jsonlite", "Matrix", "psych", "GPArotation")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    paste0(
      "Missing R package(s): ", paste(missing_packages, collapse = ", "),
      ". Run Rscript cpu/bootstrap_r_dependencies.R from the experiments root."
    )
  )
}
suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(psych)
  library(GPArotation)
})

sha256 <- function(path) {
  output <- system2(
    "shasum", c("-a", "256", shQuote(normalizePath(path))),
    stdout = TRUE, stderr = TRUE
  )
  if (length(output) == 0L) return(NA_character_)
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][1L]
}

read_matrix <- function(filename) {
  data <- read.csv(
    file.path(input_dir, filename),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  metadata <- c(
    "source_file", "source_line", "response_position", "response_sha256",
    "is_safe", "severity_level"
  )
  categories <- setdiff(names(data), metadata)
  matrix_data <- as.matrix(data[, categories, drop = FALSE])
  storage.mode(matrix_data) <- "numeric"
  if (!all(matrix_data %in% c(0, 1))) stop("Category matrix is not binary")
  released_safe <- tolower(as.character(data$is_safe)) %in% c("true", "1")
  list(x = matrix_data, categories = categories, released_safe = released_safe)
}

matrix_to_frame <- function(matrix_value, row_label = "category") {
  output <- as.data.frame(matrix_value, check.names = FALSE)
  output <- cbind(
    setNames(data.frame(rownames(matrix_value), stringsAsFactors = FALSE), row_label),
    output
  )
  rownames(output) <- NULL
  output
}

write_matrix_csv <- function(matrix_value, path, row_label = "category") {
  write.csv(matrix_to_frame(matrix_value, row_label), path, row.names = FALSE, quote = TRUE)
}

lower_rms <- function(matrix_value) {
  lower <- matrix_value[lower.tri(matrix_value)]
  sqrt(mean(lower^2, na.rm = TRUE))
}

safe_tetrachoric <- function(x, smooth) {
  suppressWarnings(psych::tetrachoric(x, correct = 0.5, smooth = smooth)$rho)
}

binary_parallel_analysis <- function(n, prevalence, n_iter, seed) {
  set.seed(seed)
  k <- length(prevalence)
  eigenvalues <- matrix(NA_real_, nrow = n_iter, ncol = k)
  repaired <- logical(n_iter)
  for (iteration in seq_len(n_iter)) {
    simulated <- vapply(
      prevalence,
      function(probability) rbinom(n, size = 1L, prob = probability),
      integer(n)
    )
    rho <- tryCatch(
      safe_tetrachoric(simulated, smooth = FALSE),
      error = function(error) NULL
    )
    if (is.null(rho) || any(!is.finite(rho)) ||
        min(eigen(rho, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
      rho <- safe_tetrachoric(simulated, smooth = TRUE)
      repaired[iteration] <- TRUE
    }
    eigenvalues[iteration, ] <- sort(
      eigen(rho, symmetric = TRUE, only.values = TRUE)$values,
      decreasing = TRUE
    )
    if (iteration %% 25L == 0L || iteration == n_iter) {
      message(sprintf("parallel analysis: %d/%d", iteration, n_iter))
    }
  }
  list(
    mean = colMeans(eigenvalues),
    q95 = apply(eigenvalues, 2L, quantile, probs = 0.95, names = FALSE),
    repaired_simulations = sum(repaired)
  )
}

fit_solution <- function(rho, n_obs, categories, nfactors, prefix) {
  if (nfactors < 1L) return(NULL)
  fitted <- psych::fa(
    r = rho,
    n.obs = n_obs,
    nfactors = nfactors,
    fm = "minres",
    rotate = "oblimin"
  )
  loadings <- unclass(fitted$loadings)
  rownames(loadings) <- categories
  colnames(loadings) <- paste0(prefix, seq_len(ncol(loadings)))
  residual <- rho - fitted$model
  phi <- fitted$Phi
  if (is.null(phi)) {
    phi <- diag(nfactors)
    rownames(phi) <- colnames(loadings)
    colnames(phi) <- colnames(loadings)
  } else {
    rownames(phi) <- colnames(loadings)
    colnames(phi) <- colnames(loadings)
  }
  list(
    loadings = loadings,
    phi = phi,
    residual_rms_lower_triangle = lower_rms(residual),
    rmsr_reported_by_psych = fitted$rms,
    objective = fitted$criteria[1L],
    degrees_of_freedom = fitted$dof
  )
}

analyse_matrix <- function(input, label, seed) {
  unsafe_index <- !input$released_safe
  x <- input$x[unsafe_index, , drop = FALSE]
  categories <- input$categories
  n_obs <- nrow(x)
  if (n_obs < 2L) stop("Unsafe-only matrix has fewer than two rows")
  prevalence <- colMeans(x)
  raw_rho <- tryCatch(safe_tetrachoric(x, smooth = FALSE), error = function(error) NULL)
  raw_min_eigenvalue <- if (is.null(raw_rho)) {
    NA_real_
  } else {
    min(eigen(raw_rho, symmetric = TRUE, only.values = TRUE)$values)
  }
  smoothing_required <- is.null(raw_rho) ||
    any(!is.finite(raw_rho)) ||
    raw_min_eigenvalue <= 0
  rho <- safe_tetrachoric(x, smooth = TRUE)
  observed_eigenvalues <- sort(
    eigen(rho, symmetric = TRUE, only.values = TRUE)$values,
    decreasing = TRUE
  )
  parallel <- binary_parallel_analysis(n_obs, prevalence, n_iter, seed)
  retained <- sum(observed_eigenvalues > parallel$q95)
  fitted_factor_count <- max(1L, retained)
  solution <- fit_solution(rho, n_obs, categories, fitted_factor_count, "F")

  write_matrix_csv(
    rho,
    file.path(result_dir, paste0("tetrachoric_", label, ".csv"))
  )
  write.csv(
    matrix_to_frame(solution$loadings),
    file.path(result_dir, paste0("efa_loadings_", label, ".csv")),
    row.names = FALSE,
    quote = TRUE
  )
  write_matrix_csv(
    solution$phi,
    file.path(result_dir, paste0("efa_factor_correlations_", label, ".csv")),
    row_label = "factor"
  )

  list(
    label = label,
    n_obs = n_obs,
    excluded_safe_response_positions = sum(input$released_safe),
    category_prevalence = as.list(setNames(prevalence, categories)),
    raw_tetrachoric_min_eigenvalue = raw_min_eigenvalue,
    smoothing_required = smoothing_required,
    smoothed_tetrachoric_min_eigenvalue = min(
      eigen(rho, symmetric = TRUE, only.values = TRUE)$values
    ),
    observed_eigenvalues = as.list(observed_eigenvalues),
    parallel_analysis = list(
      method = "independent Bernoulli simulations at observed category prevalences",
      simulations = n_iter,
      simulated_eigenvalue_mean = as.list(parallel$mean),
      simulated_eigenvalue_q95 = as.list(parallel$q95),
      simulations_requiring_smoothing = parallel$repaired_simulations
    ),
    retained_factor_count = retained,
    fitted_factor_count = fitted_factor_count,
    factor_solution = list(
      loadings_file = paste0("efa_loadings_", label, ".csv"),
      factor_correlation_file = paste0("efa_factor_correlations_", label, ".csv"),
      tetrachoric_file = paste0("tetrachoric_", label, ".csv"),
      residual_rms_lower_triangle = solution$residual_rms_lower_triangle,
      rmsr_reported_by_psych = solution$rmsr_reported_by_psych,
      objective = solution$objective,
      degrees_of_freedom = solution$degrees_of_freedom
    )
  )
}

all_positions <- read_matrix("category_matrix_all_positions.csv")
one_per_pair <- read_matrix("category_matrix_one_response_per_pair.csv")
if (!identical(all_positions$categories, one_per_pair$categories)) {
  stop("Category columns differ between EFA inputs")
}

all_result <- analyse_matrix(all_positions, "unsafe_all_positions", seed)
one_result <- analyse_matrix(one_per_pair, "unsafe_one_response_per_pair", seed + 1L)
p0_path <- file.path(published_result_dir, "p0_snapshot.json")
result <- list(
  result_schema = "pku-saferlhf.e2-efa.v1",
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  provenance = list(
    script_sha256 = sha256(script_path),
    p0_manifest_path = "cpu/results/p0_snapshot.json",
    p0_manifest_sha256 = sha256(p0_path),
    seed_all_positions = seed,
    seed_one_response_per_pair = seed + 1L,
    packages = list(
      psych = as.character(packageVersion("psych")),
      Matrix = as.character(packageVersion("Matrix")),
      jsonlite = as.character(packageVersion("jsonlite"))
    )
  ),
  category_order = all_positions$categories,
  primary_unsafe_positions = all_result,
  one_response_per_pair_unsafe_sensitivity = one_result,
  design_note = paste(
    "The primary EFA conditions on released is_safe being false.",
    "An unconditioned analysis would largely describe the all-zero category",
    "encoding of the binary safety boundary rather than taxonomy structure."
  )
)
write_json(result, file.path(result_dir, "e2_efa.json"), pretty = TRUE, auto_unbox = TRUE)
for (filename in list.files(result_dir, full.names = FALSE)) {
  source_path <- file.path(result_dir, filename)
  destination_path <- file.path(published_result_dir, filename)
  if (file.exists(destination_path)) unlink(destination_path)
  if (!file.rename(source_path, destination_path)) {
    if (!file.copy(source_path, destination_path, overwrite = TRUE)) {
      stop(paste("Could not publish E2 output:", filename))
    }
    unlink(source_path)
  }
}
unlink(lock_path)
unlink(staging_result_dir, recursive = TRUE, force = TRUE)
message("Published ", file.path(published_result_dir, "e2_efa.json"))
