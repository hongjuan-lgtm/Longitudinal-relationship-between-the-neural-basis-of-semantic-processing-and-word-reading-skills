#!/usr/bin/env bash
# Extract run-wise condition betas for six mask/time-point combinations.
# Usage: bash extract_roi_beta_values.sh 5-7 /path/to/final_participants_5-7.tsv
set -euo pipefail
export LC_ALL=C

# USER CONFIGURATION (absolute paths)
data_root="/path/to/first_level/data"
roi_output_root="/path/to/roi_outputs"
output_root="/path/to/roi_beta_values"

[[ $# -eq 2 ]] || { echo "Usage: $0 <5-7|7-9> <participant_tsv>" >&2; exit 2; }
cohort="$1"
participant_file="$2"
[[ "$cohort" == "5-7" || "$cohort" == "7-9" ]] || { echo "Invalid cohort" >&2; exit 2; }
[[ -f "$participant_file" ]] || { echo "Participant TSV not found" >&2; exit 2; }
command -v 3dmaskave >/dev/null || { echo "AFNI 3dmaskave is required" >&2; exit 127; }

# Use a new output directory to avoid mixing partial or previous results.
out_dir="${output_root}/${cohort}"
if [[ -e "$out_dir" ]]; then
    echo "Output already exists: $out_dir. Set output_root to a new location." >&2
    exit 1
fi
mkdir -p "$out_dir"
trap 'echo "Extraction failed; outputs may be incomplete." >&2' ERR

IFS=$'\t' read -r header _ < "$participant_file"
[[ "$header" == "participant_id" ]] || { echo "First TSV column must be participant_id" >&2; exit 2; }

for depth in shallow deep; do
    for combination in T1mask_T1beta T2mask_T2beta T2mask_T1beta; do
        out_file="${out_dir}/beta_values_${depth}_${combination}.tsv"
        printf 'participant_id\tcohort\tROI\tdepth\tmask_timepoint\tbeta_timepoint\tsession\trun\tcondition\tbeta_image\tvalue\n' > "$out_file"

        case "$combination" in
            T1mask_T1beta) mask_tp=T1; beta_tp=T1; offset=0 ;;
            T2mask_T2beta) mask_tp=T2; beta_tp=T2; offset=20 ;;
            T2mask_T1beta) mask_tp=T2; beta_tp=T1; offset=0 ;;
        esac
        if [[ "$depth" == shallow ]]; then
            condition_name=S_H; condition_index=2; contrast_index=1
        else
            condition_name=S_L; condition_index=3; contrast_index=2
        fi
        if [[ "$mask_tp" == T2 ]]; then contrast_index=$((contrast_index + 2)); fi
        printf -v contrast_label 'con_%04d' "$contrast_index"
        if [[ "$cohort" == 5-7 ]]; then
            session=5; [[ "$beta_tp" == T1 ]] || session=7
        else
            session=7; [[ "$beta_tp" == T1 ]] || session=9
        fi

        while IFS=$'\t' read -r participant_id rest || [[ -n "$participant_id" ]]; do
            participant_id="${participant_id%$'\r'}"
            [[ -n "$participant_id" && "$participant_id" != participant_id ]] || continue
            [[ "$participant_id" =~ ^sub-[A-Za-z0-9]+$ ]] || { echo "Invalid participant ID: $participant_id" >&2; exit 2; }
            analysis_dir="${data_root}/${cohort}/${participant_id}/analysis/deweight"
            for roi in AG pMTG vIFG; do
                mask_file="${roi_output_root}/${cohort}/${participant_id}/top100_masks/${contrast_label}_${roi}_top100_mask.nii.gz"
                [[ -f "$mask_file" ]] || { echo "Missing mask: $mask_file" >&2; exit 1; }
                for run in 1 2; do
                    for cond_index in 1 "$condition_index"; do
                        # Original model: 4 conditions + 6 motion regressors per run.
                        beta_index=$((offset + (run - 1) * 10 + cond_index))
                        printf -v beta_name 'beta_%04d.nii' "$beta_index"
                        beta_file="${analysis_dir}/${beta_name}"
                        [[ -f "$beta_file" ]] || { echo "Missing beta image: $beta_file" >&2; exit 1; }
                        condition=S_C
                        [[ "$cond_index" == 1 ]] || condition="$condition_name"
                        value=$(3dmaskave -quiet -mask "$mask_file" "$beta_file")
                        value=$(printf '%s\n' "$value" | awk '{$1=$1; print}')
                        [[ "$value" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] || {
                            echo "Invalid mean for $beta_file using $mask_file: $value" >&2; exit 1;
                        }
                        printf '%s\t%s\t%s\t%s\t%s\t%s\tses-%s\t%s\t%s\t%s\t%s\n' \
                            "$participant_id" "$cohort" "$roi" "$depth" "$mask_tp" "$beta_tp" \
                            "$session" "$run" "$condition" "$beta_name" "$value" >> "$out_file"
                    done
                done
            done
        done < "$participant_file"
    done
done
printf 'Completed six beta tables in %s\n' "$out_dir"
