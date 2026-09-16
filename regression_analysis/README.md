# Regression analyses

Longitudinal neural and behavioral regressions are run separately for the 5–7 cohort (sessions 5 and 7) and the 7–9 cohort (sessions 7 and 9). T1 and T2 refer to the earlier and later sessions within each cohort.

## Scripts

| File | Purpose |
|---|---|
| `prepare_neural_data.R` | Combine ROI beta values with reading and IQ scores and prepare neural regression inputs. |
| `regression_scaffolding.R` | Test whether T1 neural activity predicts T2 reading after controlling for T1 reading and IQ. |
| `regression_refinement.R` | Test whether T1 reading predicts T2 neural activity after controlling for T1 neural activity and IQ. |
| `regression_behavior.R` | Prepare behavioral inputs and run CELF and task accuracy/RT regressions. |

## Inputs

- Final cohort participant TSVs, with a `participant_id` column.
- OpenNeuro phenotype files under `<bids_root>/phenotype/ses-<session>/`: `wj-iii.tsv`, `kbit.tsv`, and, for behavioral analyses, `celf-5.tsv`.
- Six ROI beta tables per cohort from the ROI extraction step, under `<beta_root>/<cohort>/`:
  `beta_values_{shallow|deep}_{T1mask_T1beta|T2mask_T2beta|T2mask_T1beta}.tsv`.
- Task behavior summaries in `data/5-7_behavior_summary.tsv` and `data/7-9_behavior_summary.tsv` for accuracy/RT analyses (format below).

Reading uses the Word Identification raw score; IQ uses `KBIT_Nonverbal_StS`; CELF uses `CELF_WC_Raw` (Word Classes raw score). The scripts recognize common punctuation variants of the Word Identification column name; `reading_column` can specify an exact name.

### Task behavior summary

Required columns are `participant_id` (or `sub`), `session`, `condition`, `accuracy`, and `RT`. 

- **Accuracy:** calculated separately for each condition as the mean of `calculated_accuracy` over the retained trials within each run. For binary 0/1 scores, this represents the proportion of correct responses.
- **RT:** expressed in **seconds (s)** .

## Models

Each row below represents two nested models. Model 1 contains the baseline controls; Model 2 adds the predictor of interest. All IQ controls refer to T1.

| Analysis | Outcome | Baseline controls | Added predictor |
|---|---|---|---|
| Neural scaffolding | T2 reading | T1 reading, IQ | T1 neural activity from the T1 mask |
| Neural refinement | T2 neural activity from the T2 mask | T1 neural activity from the T2 mask, IQ | T1 reading |
| CELF scaffolding | T2 reading | T1 reading, IQ | T1 CELF |
| CELF refinement | T2 CELF | T1 CELF, IQ | T1 reading |
| Task refinement | T2 accuracy or RT | The same condition's T1 accuracy or RT, IQ | T1 reading |

Neural models are fitted separately for AG, pMTG and vIFG, and for shallow and deep processing. Neural values are the mean of the two run-specific contrasts: `(H1 + H2 - C1 - C2) / 2` for shallow and `(L1 + L2 - C1 - C2) / 2` for deep. Both runs are required.

## Outputs
Regression outputs include:

- `model_summary.tsv`: sample sizes, R², adjusted R², ΔR² and nested-model F tests.
- `coefficients.tsv`: unstandardized coefficients, standard errors, two-sided p-values and 95% confidence intervals.
- `sample_inclusion.tsv`: included observations and missing fields for each analysis.
- `models.rds`, `model_details.txt` and `analysis_settings.txt`: fitted models, detailed results and execution settings.
