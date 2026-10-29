library(dplyr)
library(BayesFactor)

data_dir <- "/path/to/your/data"
analysis_type <- "refinement"  # "refinement" or "scaffolding"
rscale <- "wide"

cohorts <- c("5-7", "7-9")
conditions <- c("deep", "shallow")
rois <- c("AG", "pMTG", "vIFG")
output_file <- file.path(data_dir, paste0(analysis_type, "_BF01_results.csv"))

calculate_bf01 <- function(df, analysis_type, rscale) {
  if (analysis_type == "refinement") {
    null_formula <- beta_T2 ~ beta_T1 + IQ
    alt_formula <- beta_T2 ~ beta_T1 + IQ + WordReading
  } else if (analysis_type == "scaffolding") {
    null_formula <- WordReading_T2 ~ WordReading_T1 + IQ
    alt_formula <- WordReading_T2 ~ WordReading_T1 + IQ + beta_T1
  } else {
    stop("analysis_type must be 'refinement' or 'scaffolding'.")
  }

  required_cols <- all.vars(alt_formula)
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols)) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

  # Use the same complete observations for both models
  df <- df[complete.cases(df[, required_cols]), required_cols, drop = FALSE]
  if (nrow(df) < 5) stop("Too few complete observations.")

  bf_null <- lmBF(null_formula, data = df, rscaleCont = rscale)
  bf_alt <- lmBF(alt_formula, data = df, rscaleCont = rscale)

  data.frame(N = nrow(df), BF01 = as.vector(bf_null / bf_alt))
}

results <- list()

for (cohort in cohorts) {
  for (condition in conditions) {
    for (roi in rois) {
      filename <- paste0("regression_", analysis_type, "_", condition, "_", cohort, "_", roi, ".RData")
      filepath <- file.path(data_dir, filename)

      result <- tryCatch({
        if (!file.exists(filepath)) stop("File not found.")

        env <- new.env()
        load(filepath, envir = env)
        if (!exists("df2", envir = env, inherits = FALSE)) stop("df2 not found.")

        calculate_bf01(env$df2, analysis_type, rscale)
      }, error = function(e) {
        warning(filename, ": ", conditionMessage(e))
        data.frame(N = NA_integer_, BF01 = NA_real_)
      })

      results[[length(results) + 1]] <- data.frame(cohort, ROI = roi, condition, result)
      cat(cohort, condition, roi, "| BF01 =", round(result$BF01, 3), "\n")
    }
  }
}

results_df <- bind_rows(results) %>% arrange(cohort, ROI, condition)
print(results_df)
write.csv(results_df, output_file, row.names = FALSE)
