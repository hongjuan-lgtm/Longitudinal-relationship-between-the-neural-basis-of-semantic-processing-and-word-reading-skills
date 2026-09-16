# Participant Selection

Participants were selected from the longitudinal dataset available on OpenNeuro and described by Wang et al. (2022). The final sample included two longitudinal cohorts: children followed from ages 5 to 7 years and children followed from ages 7 to 9 years.

Participants met the following inclusion criteria:

1. Right-handedness, indicated by right-hand use on at least three of five handedness items administered when the child first entered the study.
2. A standardized nonverbal score of at least 70 on the Kaufman Brief Intelligence Test, Second Edition (KBIT-2).
3. A standardized Core Language score of at least 70 on the Clinical Evaluation of Language Fundamentals, Fifth Edition (CELF-5).
4. Completion of both runs of the auditory meaning-judgment fMRI task and the Word Identification subtest of the Woodcock-Johnson III Tests of Achievement at both relevant time points.
5. Acceptable head motion and in-scanner task performance in both fMRI runs.

For each run, no more than 10% of volumes and no more than six consecutive volumes could be identified as motion outliers. Participants were also required to achieve accuracy above 50% in both the high-association and perceptual-control conditions and to have an accuracy difference of less than 40 percentage points between the high-association and unrelated conditions.

The final participant and scan selections were reviewed manually. Therefore, this repository provides the final analysis samples and the selected anatomical and functional scans rather than a fully reproducible participant-screening pipeline.

## Files

- `final_participants_5-7.tsv` contains the 41 participants included in the younger cohort, with data from sessions 5 and 7.
- `final_participants_7-9.tsv` contains the 56 participants included in the older cohort, with data from sessions 7 and 9.

For each participant and session, the TSV files identify the selected T1-weighted anatomical image and the two functional runs used in the analyses. Twenty participants are included in both cohorts.
