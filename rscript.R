args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args_all, value = TRUE)
if (!length(script_arg)) {
  stop("Could not determine the path to rscript.R.")
}

script_path <- normalizePath(
  sub("^--file=", "", script_arg[[1]]),
  winslash = "/",
  mustWork = TRUE
)
project_dir <- dirname(script_path)
source(file.path(project_dir, "r_analysis", "generate_report.R"), local = globalenv())
