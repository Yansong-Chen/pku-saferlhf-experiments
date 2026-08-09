#!/usr/bin/env Rscript

# Install the only optional EFA dependency into a repository-local, ignored
# library. Core packages are supplied by the local R installation.

args <- commandArgs(trailingOnly = TRUE)
if (length(args)) stop("This script accepts no arguments")

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_guess <- if (length(file_argument)) {
  sub("^--file=", "", file_argument[[1L]])
} else {
  "cpu/bootstrap_r_dependencies.R"
}
script_path <- normalizePath(script_guess, mustWork = FALSE)
if (!file.exists(script_path)) script_path <- normalizePath("cpu/bootstrap_r_dependencies.R")
experiment_root <- normalizePath(file.path(dirname(script_path), ".."))
local_library <- file.path(experiment_root, "cpu", ".Rlib")
dir.create(local_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(local_library, .libPaths()))

if (!requireNamespace("GPArotation", quietly = TRUE)) {
  install.packages("GPArotation", lib = local_library, repos = "https://cloud.r-project.org")
}
if (!requireNamespace("GPArotation", quietly = TRUE)) {
  stop("GPArotation installation failed")
}
cat("GPArotation", as.character(packageVersion("GPArotation")), "available in", local_library, "\n")
