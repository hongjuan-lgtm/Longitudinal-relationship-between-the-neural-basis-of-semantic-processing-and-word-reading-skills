# First-level analysis

`run_first_level.m` fits participant-level event-related GLMs for the fMRI
meaning-judgment task. Each model contains four SPM sessions: two runs from the
first time point followed by two runs from the second time point.

All trials are modeled as zero-duration events with the canonical HRF. The four
regressors of interest are `S_C`, `S_H`, `S_L`, and `S_U`; the six realignment
parameters from each run are included as nuisance regressors. The model uses a
128 s high-pass filter, SPM's AR(1) serial-correlation model, and an implicit
mask threshold of 0.5.

Volumes listed in each run's `art_repaired.txt` are assigned a weight of 0.01
using `art_redo_jin.m`. The estimated deweighted model is saved under
`analysis/deweight/`.

Four contrasts are generated for each participant:

1. `S_H > S_C` at the first time point
2. `S_L > S_C` at the first time point
3. `S_H > S_C` at the second time point
4. `S_L > S_C` at the second time point

The script reads event timing from the original OpenNeuro `events.tsv` files
and uses the final participant/scan tables to identify the two selected runs at
each time point. SPM12 and the preprocessing `functions/` directory must be on
the MATLAB path.
