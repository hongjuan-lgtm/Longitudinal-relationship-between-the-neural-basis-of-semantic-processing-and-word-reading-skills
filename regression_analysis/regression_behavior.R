# Behavioral longitudinal regressions. 
# CELF Word Classes raw scores: scaffolding and refinement.
# Task refinement: separate S_H/S_L models for accuracy and RT.
# Source this file, then call regression_behavior(); see README_behavior.md.

regression_behavior <- function(cohort, participant_file, bids_root, output_dir,
                                task_behavior_file = NULL,
                                reading_column = NULL) {
  if (length(cohort)!=1L || is.na(cohort) || !cohort %in% c('5-7','7-9')) stop('Invalid cohort')
  sessions <- if(cohort=='5-7') c(5,7) else c(7,9)
  read_tsv <- function(path) {
    if (!file.exists(path)) stop('Missing input: ',path)
    read.delim(path,check.names=FALSE,colClasses='character',
               na.strings=c('','NA','NaN','n/a'),quote='',comment.char='')
  }
  require_cols <- function(d,cols) {
    if(!all(cols %in% names(d))) stop('Missing columns: ',paste(setdiff(cols,names(d)),collapse=', '))
  }
  ids <- function(x) {
    if(anyNA(x)) stop('Missing participant ID')
    x <- paste0('sub-',sub('^sub-','',trimws(x)))
    if(any(!grepl('^sub-[A-Za-z0-9]+$',x))) stop('Invalid participant ID')
    x
  }
  numbers <- function(x,label) {
    y <- suppressWarnings(as.numeric(x))
    if(any(!is.na(x) & (is.na(y) | !is.finite(y)))) stop('Invalid numeric value: ',label)
    y
  }
  p <- read_tsv(participant_file); require_cols(p,'participant_id')
  selected <- ids(p$participant_id)
  if(!length(selected) || anyDuplicated(selected)) stop('Empty or duplicate participant list')
  sources <- list()
  score <- function(session,file,candidates,variable) {
    path <- file.path(bids_root,'phenotype',paste0('ses-',session),file)
    x <- read_tsv(path); require_cols(x,'participant_id')
    column <- intersect(candidates,names(x))
    if(length(column)!=1L) stop('Expected one score column in ',path,': ',paste(candidates,collapse=', '))
    x$participant_id <- ids(x$participant_id)
    x <- x[x$participant_id %in% selected,,drop=FALSE]
    if(anyDuplicated(x$participant_id)) stop('Duplicate phenotype IDs: ',path)
    value <- numbers(x[[column]],column)
    sources[[variable]] <<- data.frame(variable=variable,path=normalizePath(path),column=column)
    value[match(selected,x$participant_id)]
  }
  reading <- if(is.null(reading_column)) c('WJ-III_WordID_Raw','WJ.III_WordID_Raw','WJ_III_WordID_Raw') else reading_column
  d <- data.frame(participant_id=selected,cohort=cohort)
  d$wordreading_T1 <- score(sessions[1],'wj-iii.tsv',reading,'wordreading_T1')
  d$wordreading_T2 <- score(sessions[2],'wj-iii.tsv',reading,'wordreading_T2')
  d$semantic_T1 <- score(sessions[1],'celf-5.tsv','CELF_WC_Raw','semantic_T1')
  d$semantic_T2 <- score(sessions[2],'celf-5.tsv','CELF_WC_Raw','semantic_T2')
  d$nonverbal_IQ_T1 <- score(sessions[1],'kbit.tsv','KBIT_Nonverbal_StS','nonverbal_IQ_T1')

  summaries <- coefficients <- samples <- models <- details <- list()
  fit_pair <- function(z,label,outcome,baseline,predictor) {
    variables <- c(outcome,baseline,'nonverbal_IQ_T1',predictor)
    included <- complete.cases(z[,variables,drop=FALSE])
    if(!'condition' %in% names(z)) z$condition <- NA_character_
    samples[[label]] <<- data.frame(analysis=label,z[,c('participant_id','cohort','condition')],
      included=included,missing_variables=apply(is.na(z[,variables,drop=FALSE]),1,
        function(x) paste(variables[x],collapse=';')),row.names=NULL)
    z <- z[included,,drop=FALSE]
    if(nrow(z)<=4L) stop('Insufficient complete rows for ',label)
    f1 <- reformulate(c(baseline,'nonverbal_IQ_T1'),response=outcome)
    f2 <- reformulate(c(predictor,baseline,'nonverbal_IQ_T1'),response=outcome)
    m1 <- lm(f1,data=z,na.action=na.fail); m2 <- lm(f2,data=z,na.action=na.fail)
    if(m1$rank!=3L || m2$rank!=4L) stop('Rank-deficient model: ',label)
    s1 <- summary(m1); s2 <- summary(m2); a <- anova(m1,m2)
    summaries[[label]] <<- data.frame(cohort=cohort,analysis=label,
      n_input_rows=length(included),n_used_rows=nrow(z),n_participants=length(unique(z$participant_id)),
      n_missing_rows=sum(!included),R2_model1=s1$r.squared,R2_model2=s2$r.squared,
      adjusted_R2_model1=s1$adj.r.squared,adjusted_R2_model2=s2$adj.r.squared,
      delta_R2=s2$r.squared-s1$r.squared,F_change=a$F[2],df_change=a$Df[2],
      df_residual=df.residual(m2),p_change=a$`Pr(>F)`[2])
    for(name in c('model1','model2')) {
      m <- if(name=='model1') m1 else m2; tab <- coef(summary(m)); ci <- confint(m)
      coefficients[[paste(label,name,sep='_')]] <<- data.frame(cohort=cohort,analysis=label,
        model=name,term=rownames(tab),estimate=tab[,1],std_error=tab[,2],t=tab[,3],p=tab[,4],
        CI95_lower=ci[,1],CI95_upper=ci[,2],row.names=NULL)
    }
    models[[label]] <<- list(model1=m1,model2=m2,rows=z[,c('participant_id','condition')])
    details[[label]] <<- c(paste('===',label,'==='),paste('Model 1:',deparse(f1)),
      paste('Model 2:',deparse(f2)),capture.output(print(s1)),capture.output(print(s2)),capture.output(print(a)))
  }
  fit_pair(d,'CELF_scaffolding','wordreading_T2','wordreading_T1','semantic_T1')
  fit_pair(d,'CELF_refinement','semantic_T2','semantic_T1','wordreading_T1')

  task_data <- NULL
  if(!is.null(task_behavior_file)) {
    task <- read_tsv(task_behavior_file)
    if(!'participant_id' %in% names(task) && 'sub' %in% names(task)) task$participant_id <- task$sub
    require_cols(task,c('participant_id','session','condition','accuracy','RT'))
    task$participant_id <- ids(task$participant_id)
    task$session <- sub('^ses-','',task$session)
    task <- task[task$participant_id %in% selected & !is.na(task$session) &
                   task$session %in% as.character(sessions),,drop=FALSE]
    if(!nrow(task) || anyNA(task$condition) || any(!nzchar(task$condition))) stop('Missing task conditions/data')
    conditions <- c('S_H','S_L')
    task <- task[task$condition %in% conditions,,drop=FALSE]
    if(!all(conditions %in% task$condition)) stop('Task input must contain both S_H and S_L')
    if(anyDuplicated(task[,c('participant_id','session','condition')]))
      stop('Task input must contain one row per participant/session/condition, averaged across selected runs')
    for(metric in c('accuracy','RT')) task[[metric]] <- numbers(task[[metric]],metric)
    # Preserve the supplied accuracy scale and RT units; no transformations.
    task_data <- expand.grid(participant_id=selected,condition=conditions,stringsAsFactors=FALSE)
    index <- match(task_data$participant_id,d$participant_id)
    task_data$cohort <- cohort
    task_data$wordreading_T1 <- d$wordreading_T1[index]
    task_data$nonverbal_IQ_T1 <- d$nonverbal_IQ_T1[index]
    for(i in 1:2) {
      t <- task[task$session==as.character(sessions[i]),,drop=FALSE]
      matched <- match(paste(task_data$participant_id,task_data$condition,sep='|'),
                       paste(t$participant_id,t$condition,sep='|'))
      for(metric in c('accuracy','RT'))
        task_data[[paste0(metric,'_T',i)]] <- t[[metric]][matched]
    }
    for(condition in conditions) for(metric in c('accuracy','RT')) {
      fit_pair(task_data[task_data$condition==condition,,drop=FALSE],
               paste('task_refinement',condition,metric,sep='_'),
               paste0(metric,'_T2'),paste0(metric,'_T1'),'wordreading_T1')
    }
  }

  files <- c('behavior_regression_data.tsv','task_regression_data.tsv','model_summary.tsv',
    'coefficients.tsv','sample_inclusion.tsv','models.rds','model_details.txt','phenotype_sources.tsv','analysis_settings.txt')
  if(any(file.exists(file.path(output_dir,files)))) stop('Output files already exist; choose a new output_dir')
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  write_tsv <- function(x,name) write.table(x,file.path(output_dir,name),sep='\t',quote=FALSE,row.names=FALSE,na='NA')
  write_tsv(d,files[1]); if(!is.null(task_data)) write_tsv(task_data,files[2])
  result <- do.call(rbind,summaries); write_tsv(result,files[3])
  write_tsv(do.call(rbind,coefficients),files[4]); write_tsv(do.call(rbind,samples),files[5])
  saveRDS(models,file.path(output_dir,files[6])); writeLines(unlist(details,use.names=FALSE),file.path(output_dir,files[7]))
  write_tsv(do.call(rbind,sources),files[8])
  writeLines(c(paste('Cohort:',cohort),paste('Participants:',normalizePath(participant_file)),
    paste('Task input:',if(is.null(task_behavior_file)) 'none' else normalizePath(task_behavior_file)),
    'Task models: S_H accuracy, S_H RT, S_L accuracy, S_L RT; each fitted separately.',
    'Unstandardized coefficients; uncorrected p-values. Both nested models use the same complete rows.',
    'CELF = Word Classes raw score; reading = Word Identification raw score; IQ = nonverbal standard score.',
    capture.output(sessionInfo())),file.path(output_dir,files[9]))
  message('Completed ',length(models),' behavioral model pairs in ',output_dir)
  invisible(result)
}
