# ROI definition and beta extraction

This module defines individualized regions of interest and extracts condition-wise
beta estimates for the longitudinal regression analyses.

## Anatomical masks

Three left-hemisphere masks were generated using MarsBaR based on the Automated
Anatomical Labeling (AAL) atlas. The final binary masks have 2 × 2 × 2 mm voxels
and are provided in `masks/`:

| ROI | Anatomical definition | File |
| --- | --- | --- |
| AG | Left angular gyrus | `masks/AG_L_aal_mask.nii` |
| pMTG | Left middle temporal gyrus restricted to MNI y < −35 mm | `masks/pMTG.nii` |
| vIFG | Union of the left triangular and orbital parts of the inferior frontal gyrus | `masks/IFG_tri_orb_L_aal_mask.nii` |

## Top-100 voxel selection

`select_top100_voxels.sh` resamples each anatomical mask to the participant's
first-level contrast grid using nearest-neighbor interpolation. It selects the
100 voxels with the highest signed contrast values within each mask, separately
for each participant, time point, and semantic contrast. Selection uses contrast
estimates rather than absolute values or t statistics, without a significance
threshold.

| First-level image | Semantic contrast | Time point |
| --- | --- | --- |
| `con_0001.nii` | Shallow: S_H > S_C | T1 |
| `con_0002.nii` | Deep: S_L > S_C | T1 |
| `con_0003.nii` | Shallow: S_H > S_C | T2 |
| `con_0004.nii` | Deep: S_L > S_C | T2 |

T1/T2 correspond to ses-5/ses-7 in the younger cohort and ses-7/ses-9 in the older
cohort. Inputs are read from `analysis/deweight/`. Binary individualized masks
are written to `<ROI_OUTPUT_ROOT>/<cohort>/<participant_id>/top100_masks/`;
voxel tables and a cohort-level selection log are also saved. Review the log
before proceeding to beta extraction.

## Beta extraction

`extract_roi_beta_values.sh` uses AFNI `3dmaskave` to calculate the mean beta
within each individualized mask, separately for each run and condition. Shallow
extraction includes S_H and S_C; deep extraction includes S_L and S_C.

For both contrasts, the script generates all three combinations:

| Output suffix | Mask definition | Beta estimates extracted |
| --- | --- | --- |
| `T1mask_T1beta` | T1 top-100 voxels | T1 |
| `T2mask_T2beta` | T2 top-100 voxels | T2 |
| `T2mask_T1beta` | T2 top-100 voxels | T1 |

Six tables, named `beta_values_<shallow|deep>_<suffix>.tsv`, are written to
`<BETA_OUTPUT_ROOT>/<cohort>/`. Each table includes all three ROIs, with columns
for participant, cohort, ROI, semantic depth, mask time point, beta time point,
BIDS session, run, condition, beta filename, and mean value. Run averaging and
H−C or L−C differences are calculated in the subsequent analysis scripts.

Beta numbering follows the original first-level design: four conditions in the
order S_C, S_H, S_L, S_U followed by six motion regressors per run, with runs
ordered T1 Run 1, T1 Run 2, T2 Run 1, T2 Run 2. If that design changes, the beta
indices in the extraction script must be updated.

## Running the scripts

Requirements: Bash, AFNI (`3dresample`, `3dmaskdump`, `3dUndump`, and `3dmaskave`),
the anatomical masks, deweighted first-level results, and final participant TSVs
whose first column is `participant_id` (for example, `sub-5004`). MarsBaR is not
required to run these scripts with the supplied masks.

Set the absolute paths in each script's `USER CONFIGURATION` section:

- `data_root`: contains `<cohort>/<participant_id>/analysis/deweight/`.
- `mask_root` in the selection script: the repository's `roi/masks/` directory.
- `output_root` in the selection script: the ROI output directory; use this same
  path for `roi_output_root` in the extraction script.
- `output_root` in the extraction script: a separate directory for beta tables.

From the directory containing the scripts, run selection before extraction:

```bash
bash select_top100_voxels.sh 5-7 /path/to/final_participants_5-7.tsv
bash extract_roi_beta_values.sh 5-7 /path/to/final_participants_5-7.tsv

bash select_top100_voxels.sh 7-9 /path/to/final_participants_7-9.tsv
bash extract_roi_beta_values.sh 7-9 /path/to/final_participants_7-9.tsv
```

Beta extraction requires a new cohort output directory and stops if a required
input is missing or an extracted mean is invalid. Inspect any partial output
before rerunning.
