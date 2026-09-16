# Scaffolding: does T1 neural activity predict T2 word reading, controlling
# for T1 nonverbal IQ and T1 word reading? Base R only; no standardization.
# Input: neural_regression_data.tsv from prepare_neural_data.R.
# Run separately for each cohort (5-7 or 7-9):
# source('regression_scaffolding.R')
# regression_scaffolding('/path/to/5-7/neural_regression_data.tsv',
#                       cohort = '5-7', output_dir = '/path/to/results/5-7')
# All selected scans were retained during preparation; no outlier exclusions here.

regression_scaffolding <- function(data_file, cohort, output_dir) {
  if (length(cohort) != 1L || is.na(cohort) || !cohort %in% c('5-7', '7-9'))
    stop('cohort must be 5-7 or 7-9')
  d <- read.delim(data_file, check.names = FALSE, colClasses = 'character',
                  na.strings = c('', 'NA', 'NaN', 'n/a'), quote = '', comment.char = '')
  variables <- c('wordreading_T2', 'nonverbal_IQ_T1', 'wordreading_T1',
                 'brain_T1mask_T1beta')
  required <- c('participant_id', 'cohort', 'ROI', 'depth', variables)
  absent <- setdiff(required, names(d))
  if (length(absent)) stop('Missing columns: ', paste(absent, collapse = ', '))
  if (anyNA(d[, c('participant_id','cohort','ROI','depth')])) stop('Missing grouping identifiers')
  d <- d[d$cohort == cohort, , drop = FALSE]
  if (!nrow(d)) stop('No rows for cohort ', cohort)
  if (any(!grepl('^sub-[A-Za-z0-9]+$', d$participant_id)) ||
      any(!d$ROI %in% c('AG','pMTG','vIFG')) || any(!d$depth %in% c('shallow','deep')))
    stop('Unexpected participant ID, ROI or depth')
  if (anyDuplicated(d[,c('participant_id','ROI','depth')])) stop('Duplicate participant/ROI/depth rows')
  for (v in variables) {
    x <- suppressWarnings(as.numeric(d[[v]]))
    if (any(!is.na(d[[v]]) & (is.na(x) | !is.finite(x)))) stop('Invalid numeric value in ', v)
    d[[v]] <- x
  }
  files <- c('model_summary.tsv','coefficients.tsv','sample_inclusion.tsv',
             'models.rds','model_details.txt','analysis_settings.txt')
  if (any(file.exists(file.path(output_dir,files)))) stop('Output files already exist; choose a new output_dir')
  summaries <- coefficients <- samples <- models <- details <- list()
  for (roi in c('AG','pMTG','vIFG')) for (depth in c('shallow','deep')) {
    key <- paste(roi,depth,sep='_')
    z <- d[d$ROI == roi & d$depth == depth, required, drop=FALSE]
    if (!nrow(z)) stop('No input rows for ', key)
    # Use the same complete cases in BOTH nested models so delta R-squared is comparable.
    included <- complete.cases(z[,variables])
    missing <- apply(is.na(z[,variables,drop=FALSE]),1,function(x) paste(variables[x],collapse=';'))
    samples[[key]] <- data.frame(z[,c('participant_id','cohort','ROI','depth')],
                                 included=included,missing_variables=missing)
    z <- z[included,,drop=FALSE]
    if (nrow(z) <= 4L) stop('Insufficient complete observations for ', key, ': ', nrow(z))
    model1 <- lm(wordreading_T2 ~ nonverbal_IQ_T1 + wordreading_T1, data=z, na.action=na.fail)
    model2 <- lm(wordreading_T2 ~ nonverbal_IQ_T1 + wordreading_T1 + brain_T1mask_T1beta,
                 data=z, na.action=na.fail)
    if (model1$rank != 3L || model2$rank != 4L) stop('Rank-deficient model for ',key,'; check constant or collinear predictors')
    s1 <- summary(model1); s2 <- summary(model2)
    comparison <- anova(model1,model2)
    summaries[[key]] <- data.frame(cohort=cohort,ROI=roi,depth=depth,
      n_input=length(included),n_used=nrow(z),n_missing=sum(!included),
      R2_model1=s1$r.squared,R2_model2=s2$r.squared,
      adjusted_R2_model1=s1$adj.r.squared,adjusted_R2_model2=s2$adj.r.squared,
      delta_R2=s2$r.squared-s1$r.squared,
      F_change=comparison$F[2],df_change=comparison$Df[2],
      df_residual=df.residual(model2),p_change=comparison$`Pr(>F)`[2])
    for (label in c('model1','model2')) {
      fit <- if(label=='model1') model1 else model2
      tab <- coef(summary(fit)); ci <- confint(fit)
      coefficients[[paste(key,label,sep='_')]] <- data.frame(cohort=cohort,ROI=roi,depth=depth,
        model=label,term=rownames(tab),estimate=tab[,1],std_error=tab[,2],
        t=tab[,3],p=tab[,4],CI95_lower=ci[,1],CI95_upper=ci[,2],row.names=NULL)
    }
    models[[key]] <- list(model1=model1,model2=model2,participant_id=z$participant_id)
    details[[key]] <- c(paste('===',cohort,roi,depth,'==='),
                        capture.output(print(s1)),capture.output(print(s2)),
                        capture.output(print(comparison)))
  }
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  write_tsv <- function(x,name) write.table(x,file.path(output_dir,name),sep='\t',
                                           row.names=FALSE,quote=FALSE,na='NA')
  result <- do.call(rbind,summaries)
  write_tsv(result,files[1]); write_tsv(do.call(rbind,coefficients),files[2])
  write_tsv(do.call(rbind,samples),files[3]); saveRDS(models,file.path(output_dir,files[4]))
  writeLines(unlist(details,use.names=FALSE),file.path(output_dir,files[5]))
  writeLines(c(paste('Input:',normalizePath(data_file)),paste('Cohort:',cohort),
    'Model 1: wordreading_T2 ~ nonverbal_IQ_T1 + wordreading_T1',
    'Model 2: wordreading_T2 ~ nonverbal_IQ_T1 + wordreading_T1 + brain_T1mask_T1beta',
    'Separate models for each ROI and depth; original units, unstandardized coefficients.',
    'Within each ROI/depth, both models use the same complete observations.',
    'No additional outlier exclusions. All reported p-values are uncorrected, two-sided coefficient tests or nested-model F tests.',
    capture.output(sessionInfo())),file.path(output_dir,files[6]))
  message('Completed six scaffolding analyses for ',cohort,' in ',output_dir)
  invisible(result)
}
