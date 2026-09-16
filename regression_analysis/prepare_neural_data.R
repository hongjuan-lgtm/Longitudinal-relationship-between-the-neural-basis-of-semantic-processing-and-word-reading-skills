# Prepare neural regression inputs using the six extract_roi_beta_values.sh tables.
# Run by sourcing this file and calling prepare_neural_data().
#
# Example:
# source('prepare_neural_data.R')
# prepare_neural_data(
#   cohort = '5-7',
#   participant_file = '/path/to/final_participants_5-7.tsv',
#   beta_root = '/path/to/roi_beta_values',
#   bids_root = '/path/to/ds003604',
#   output_dir = '/path/to/regression_inputs/5-7')
#
# Uses the final participant list and both selected runs for each participant.

prepare_neural_data <- function(cohort, participant_file, beta_root, bids_root,
                                output_dir,
                                reading_column = NULL,
                                iq_column = 'KBIT_Nonverbal_StS') {
  if (!cohort %in% c('5-7', '7-9')) stop('cohort must be 5-7 or 7-9')
  sessions <- if (cohort == '5-7') c(5L, 7L) else c(7L, 9L)
  rois <- c('AG', 'pMTG', 'vIFG')
  combinations <- c('T1mask_T1beta', 'T2mask_T2beta', 'T2mask_T1beta')
  issues <- list()
  issue <- function(id, roi, depth, variable, detail) {
    issues[[length(issues) + 1L]] <<- data.frame(
      participant_id = id, ROI = roi, depth = depth,
      variable = variable, detail = detail, stringsAsFactors = FALSE)
  }
  read_tsv <- function(path) {
    if (!file.exists(path)) stop('Missing input: ', path)
    read.delim(path, check.names = FALSE, colClasses = 'character',
               na.strings = c('', 'NA', 'NaN', 'n/a'), quote = '', comment.char = '')
  }
  require_cols <- function(d, cols, path) {
    absent <- setdiff(cols, names(d))
    if (length(absent)) stop('Missing columns in ', path, ': ', paste(absent, collapse = ', '))
  }
  ids <- function(x) {
    x <- trimws(x)
    x <- paste0('sub-', sub('^sub-', '', x))
    if (any(!grepl('^sub-[A-Za-z0-9]+$', x)) || any(x == 'sub-NA')) stop('Invalid participant ID')
    x
  }
  numeric_values <- function(x, context) {
    y <- suppressWarnings(as.numeric(x))
    if (any(!is.na(x) & (is.na(y) | !is.finite(y)))) stop('Invalid numeric value: ', context)
    y
  }
  participants <- read_tsv(participant_file)
  require_cols(participants, 'participant_id', participant_file)
  selected <- ids(participants$participant_id)
  if (!length(selected) || anyDuplicated(selected)) stop('Participant list is empty or duplicated')

  # Behavioral scores retain the original units: WordID raw and nonverbal IQ standard score.
  phenotype_sources <- list()
  get_score <- function(session, filename, candidates, output_name) {
    path <- file.path(bids_root, 'phenotype', paste0('ses-', session), filename)
    d <- read_tsv(path)
    require_cols(d, 'participant_id', path)
    columns <- intersect(candidates, names(d))
    if (length(columns) != 1L) stop('Specify a unique score column in ', path,
                                  '; candidates: ', paste(candidates, collapse = ', '))
    d$participant_id <- ids(d$participant_id)
    d <- d[d$participant_id %in% selected, , drop = FALSE]
    if (anyDuplicated(d$participant_id)) stop('Duplicate phenotype IDs in ', path)
    values <- numeric_values(d[[columns]], path)
    answer <- values[match(selected, d$participant_id)]
    for (id in selected[is.na(answer)]) issue(id, NA_character_, NA_character_, output_name, 'missing phenotype score or row')
    phenotype_sources[[output_name]] <<- data.frame(variable = output_name,
      source = normalizePath(path), column = columns, stringsAsFactors = FALSE)
    answer
  }
  reading_candidates <- if (is.null(reading_column))
    c('WJ.III_WordID_Raw', 'WJ_III_WordID_Raw', 'WJ-III_WordID_Raw') else reading_column
  reading_t1 <- get_score(sessions[1], 'wj-iii.tsv', reading_candidates, 'wordreading_T1')
  reading_t2 <- get_score(sessions[2], 'wj-iii.tsv', reading_candidates, 'wordreading_T2')
  iq_t1 <- get_score(sessions[1], 'kbit.tsv', iq_column, 'nonverbal_IQ_T1')

  out <- expand.grid(participant_id = selected, ROI = rois,
                     depth = c('shallow', 'deep'), stringsAsFactors = FALSE)
  out$cohort <- cohort
  out$session_T1 <- paste0('ses-', sessions[1])
  out$session_T2 <- paste0('ses-', sessions[2])
  out$wordreading_T1 <- reading_t1[match(out$participant_id, selected)]
  out$wordreading_T2 <- reading_t2[match(out$participant_id, selected)]
  out$nonverbal_IQ_T1 <- iq_t1[match(out$participant_id, selected)]
  for (comb in combinations) out[[paste0('brain_', comb)]] <- NA_real_
  run_records <- list()
  input_files <- c(participant_file)

  for (depth in c('shallow', 'deep')) {
    positive <- if (depth == 'shallow') 'S_H' else 'S_L'
    for (comb in combinations) {
      path <- file.path(beta_root, cohort, paste0('beta_values_', depth, '_', comb, '.tsv'))
      d <- read_tsv(path)
      input_files <- c(input_files, path)
      require_cols(d, c('participant_id', 'cohort', 'ROI', 'depth', 'mask_timepoint',
                       'beta_timepoint', 'session', 'run', 'condition', 'value'), path)
      d$participant_id <- ids(d$participant_id)
      d <- d[d$participant_id %in% selected, , drop = FALSE]
      mask_tp <- if (comb == 'T1mask_T1beta') 'T1' else 'T2'
      beta_tp <- if (comb == 'T2mask_T2beta') 'T2' else 'T1'
      session <- paste0('ses-', sessions[if (beta_tp == 'T1') 1 else 2])
      if (anyNA(d[, c('cohort','ROI','depth','mask_timepoint','beta_timepoint','session','run','condition')]) ||
          any(d$cohort != cohort | !d$ROI %in% rois | d$depth != depth |
              d$mask_timepoint != mask_tp | d$beta_timepoint != beta_tp |
              d$session != session | !d$run %in% c('1','2') |
              !d$condition %in% c('S_C', positive))) stop('Unexpected metadata in ', path)
      key <- paste(d$participant_id, d$ROI, d$run, d$condition, sep = '|')
      if (anyDuplicated(key)) stop('Duplicate beta rows in ', path)
      d$value <- numeric_values(d$value, path)
      variable <- paste0('brain_', comb)
      for (id in selected) for (roi in rois) {
        differences <- rep(NA_real_, 2)
        for (run in 1:2) {
          z <- d[d$participant_id == id & d$ROI == roi & d$run == as.character(run), , drop = FALSE]
          if (nrow(z) == 2 && all(c('S_C', positive) %in% z$condition) && all(is.finite(z$value)))
            differences[run] <- z$value[match(positive,z$condition)] - z$value[match('S_C',z$condition)]
          else issue(id, roi, depth, variable, paste('missing/nonfinite condition beta in run', run))
          run_records[[length(run_records)+1L]] <- data.frame(participant_id=id,
            cohort=cohort, ROI=roi, depth=depth, combination=comb, run=run,
            difference=differences[run], stringsAsFactors=FALSE)
        }
        row <- out$participant_id == id & out$ROI == roi & out$depth == depth
        # Equivalent to (H1 + H2 - C1 - C2)/2, or (L1 + L2 - C1 - C2)/2.
        out[row, variable] <- mean(differences)
      }
    }
  }
  out$complete_scaffolding <- complete.cases(out[, c('wordreading_T2','wordreading_T1',
    'nonverbal_IQ_T1','brain_T1mask_T1beta')])
  out$complete_refinement <- complete.cases(out[, c('wordreading_T1','nonverbal_IQ_T1',
    'brain_T2mask_T2beta','brain_T2mask_T1beta')])
  issue_table <- if (length(issues)) do.call(rbind, issues) else data.frame(
    participant_id=character(),ROI=character(),depth=character(),variable=character(),detail=character())
  outputs <- c('neural_regression_data.tsv','run_contrasts.tsv','missing_data.tsv',
               'phenotype_sources.tsv','input_files.tsv','preparation_settings.txt')
  if (any(file.exists(file.path(output_dir, outputs)))) stop('Output files already exist; choose a new output_dir')
  dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
  write_tsv <- function(d, name) write.table(d, file.path(output_dir,name), sep='\t',
                                            row.names=FALSE, quote=FALSE, na='NA')
  write_tsv(out, outputs[1]); write_tsv(do.call(rbind,run_records),outputs[2])
  write_tsv(issue_table,outputs[3]); write_tsv(do.call(rbind,phenotype_sources),outputs[4])
  write_tsv(data.frame(path=normalizePath(input_files)),outputs[5])
  writeLines(c(paste('cohort:',cohort),
    'Both selected runs are required. Missing data are retained as NA.',
    capture.output(sessionInfo())),
    file.path(output_dir,outputs[6]))
  message('Prepared ', nrow(out), ' participant/ROI/depth rows in ', output_dir)
  invisible(out)
}
