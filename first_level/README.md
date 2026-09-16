# First-level analysis

This directory contains the original MATLAB workflow used for participant-level
analysis of the fMRI meaning-judgment task.

## Files

- `run_firstlevel_one_subject.m` collects the preprocessed images, event onsets,
  and six realignment parameters for one participant.
- `firstlevel_4d.m` specifies and estimates the event-related GLM and then calls
  `art_redo_jin.m` to produce a deweighted model.
- `contrast_f.m` creates the first-level contrasts for both the original and
  deweighted models.
- `run_firstlevel_bash.sh` is an optional cluster launcher for running the
  participant-level script in batch.

`art_redo_jin.m` is stored with the preprocessing functions and must be
available on the MATLAB path.

## Model specification

Each participant is modeled in one `SPM.mat` containing four SPM sessions,
ordered as follows:

1. first time point, Run 1
2. first time point, Run 2
3. second time point, Run 1
4. second time point, Run 2

Each session contains four regressors of interest (`S_C`, `S_H`, `S_L`, and
`S_U`) and six realignment parameters as nuisance regressors. All trials are
modeled as events with the canonical HRF. The model uses a TR of 1.25 s, a
128 s high-pass filter, SPM's default AR(1) serial-correlation model, and an
implicit mask threshold of 0.5.

`firstlevel_4d.m` first estimates the standard model in `analysis/`. It then
calls `art_redo_jin.m`, which assigns a weight of 0.01 to volumes listed in each
run's `art_repaired.txt` and writes the re-estimated model to
`analysis/deweight/`. The same contrasts are calculated for both models; the
deweighted contrasts were used in the subsequent analyses.

## Contrasts

The two runs from each time point are combined in the contrast weights. Four
contrast images are produced:

| Contrast image | Contrast | Time point |
| --- | --- | --- |
| `con_0001.nii` | `S_H > S_C` | T1 |
| `con_0002.nii` | `S_L > S_C` | T1 |
| `con_0003.nii` | `S_H > S_C` | T2 |
| `con_0004.nii` | `S_L > S_C` | T2 |

For the younger cohort, T1 and T2 correspond to sessions 5 and 7. For the
older cohort, they correspond to sessions 7 and 9. The model estimates each
time point separately within the same multi-session design; longitudinal
relations between T1 and T2 measures are tested in the subsequent regression
analyses rather than in the first-level GLM.

## Requirements

- MATLAB
- SPM12
- ArtRepair, including the locally modified `art_redo_jin.m`
- Preprocessed repaired images, realignment parameter files, and
  `art_repaired.txt` from the preprocessing stage
- Task `events.tsv` files, either copied into the preprocessed run directories
  or read from the OpenNeuro BIDS dataset according to `events_file_exist`
