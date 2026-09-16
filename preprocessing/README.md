# Preprocessing

This directory contains the MATLAB code used to preprocess the longitudinal
fMRI data. The pipeline processes the two selected functional
runs and one selected T1-weighted image for each participant and session listed
in the participant selection tables.

## Workflow

1. `generate_pediatric_template.m` generates age- and sex-matched pediatric
   tissue probability maps with CerebroMatic and creates the corresponding
   intracranial masks with AFNI. The templates used in the analyses were
   `mw_com_prior_Age_0081.nii` for the 5–7 cohort and
   `mw_com_prior_Age_0102.nii` for the 7–9 cohort.
2. `main_preprocess.m` runs realignment, tissue segmentation, skull stripping,
   functional-to-anatomical coregistration, MNI normalization, 6 mm spatial
   smoothing, ArtRepair, and registration quality control.

The scripts read the imaging data directly from the OpenNeuro dataset. Paths to
the dataset, participant selection tables, pediatric templates, software, and
output directory are supplied through the `config` structure.

## Requirements

- MATLAB
- SPM12
- CerebroMatic, for template generation
- AFNI (`3dcalc`), for creation of the intracranial masks
- ArtRepair
- Canny edge detection functions, for the registration QC images

The files `art_global_jin.m`, `art_clipmvmt_jin.m`, and `art_redo_jin.m` are
locally modified ArtRepair functions. 

## Directory contents

- `main_preprocess.m`: main preprocessing entry point.
- `generate_pediatric_template.m`: pediatric template generation.
- `functions/`: project-specific preprocessing wrappers, QC functions, and
  modified ArtRepair functions.

Run the template-generation script only when the pediatric templates need to be
recreated. For the reported analyses, `main_preprocess.m` uses the fixed
templates specified in the configuration.
