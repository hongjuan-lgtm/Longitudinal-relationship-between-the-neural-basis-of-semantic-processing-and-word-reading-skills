#!/usr/bin/env bash

set -u
set -o pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <cohort> <participant_tsv>" >&2
    echo "Example: $0 5-7 participants/final_participants_5-7.tsv" >&2
    exit 2
fi

cohort="$1"
participant_file="$2"

if [[ "$cohort" != "5-7" && "$cohort" != "7-9" ]]; then
    echo "Cohort must be 5-7 or 7-9." >&2
    exit 2
fi

if [[ ! -f "$participant_file" ]]; then
    echo "Participant table not found: $participant_file" >&2
    exit 2
fi

# USER CONFIGURATION
data_root="/path/to/first_level/data"
mask_root="/path/to/repository/roi/masks"
output_root="/path/to/roi_outputs"

rois=("AG" "pMTG" "vIFG")
contrasts=(1 2 3 4)
top_n=100

for command_name in 3dresample 3dmaskdump 3dUndump; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required AFNI command not found: $command_name" >&2
        exit 127
    fi
done

cohort_output="${output_root}/${cohort}"
mkdir -p "$cohort_output"
log_file="${cohort_output}/top100_selection_log.tsv"
printf "participant_id\tROI\tcontrast\tstatus\tdetail\n" > "$log_file"

resolve_roi_mask() {
    local roi="$1"
    local uncompressed="${mask_root}/${roi}_L.nii"
    local compressed="${uncompressed}.gz"

    if [[ -f "$compressed" ]]; then
        printf "%s\n" "$compressed"
    elif [[ -f "$uncompressed" ]]; then
        printf "%s\n" "$uncompressed"
    else
        return 1
    fi
}

# The first TSV column must be participant_id and contain values such as sub-5004.
while IFS=$'\t' read -r participant_id _; do
    if [[ -z "$participant_id" || "$participant_id" == "participant_id" ]]; then
        continue
    fi

    analysis_dir="${data_root}/${cohort}/${participant_id}/analysis/deweight"
    participant_output="${cohort_output}/${participant_id}"
    resampled_dir="${participant_output}/resampled_masks"
    top100_dir="${participant_output}/top100_masks"
    table_dir="${participant_output}/top100_tables"
    mkdir -p "$resampled_dir" "$top100_dir" "$table_dir"

    master_contrast="${analysis_dir}/con_0001.nii"
    if [[ ! -f "$master_contrast" ]]; then
        printf "%s\tNA\tNA\tERROR\tmissing %s\n" \
            "$participant_id" "$master_contrast" >> "$log_file"
        continue
    fi

    for roi in "${rois[@]}"; do
        if ! anatomical_mask=$(resolve_roi_mask "$roi"); then
            printf "%s\t%s\tNA\tERROR\tmissing anatomical mask\n" \
                "$participant_id" "$roi" >> "$log_file"
            continue
        fi

        resampled_mask="${resampled_dir}/${roi}_resampled.nii.gz"
        if ! 3dresample \
            -master "$master_contrast" \
            -inset "$anatomical_mask" \
            -rmode NN \
            -prefix "$resampled_mask" \
            -overwrite >/dev/null 2>&1; then
            printf "%s\t%s\tNA\tERROR\t3dresample failed\n" \
                "$participant_id" "$roi" >> "$log_file"
            continue
        fi

        for contrast_index in "${contrasts[@]}"; do
            contrast_label=$(printf "con_%04d" "$contrast_index")
            contrast_file="${analysis_dir}/${contrast_label}.nii"

            if [[ ! -f "$contrast_file" ]]; then
                printf "%s\t%s\t%s\tERROR\tmissing contrast image\n" \
                    "$participant_id" "$roi" "$contrast_label" >> "$log_file"
                continue
            fi

            all_voxels="${table_dir}/${contrast_label}_${roi}_all_voxels.tsv"
            top_voxels="${table_dir}/${contrast_label}_${roi}_top${top_n}.tsv"
            binary_points="${table_dir}/${contrast_label}_${roi}_top${top_n}_binary.1D"
            output_mask="${top100_dir}/${contrast_label}_${roi}_top${top_n}_mask.nii.gz"

            if ! 3dmaskdump \
                -quiet \
                -mask "$resampled_mask" \
                "$contrast_file" > "$all_voxels"; then
                printf "%s\t%s\t%s\tERROR\t3dmaskdump failed\n" \
                    "$participant_id" "$roi" "$contrast_label" >> "$log_file"
                continue
            fi

            voxel_count=$(wc -l < "$all_voxels" | tr -d ' ')
            if (( voxel_count < top_n )); then
                printf "%s\t%s\t%s\tERROR\tROI contains only %d voxels\n" \
                    "$participant_id" "$roi" "$contrast_label" \
                    "$voxel_count" >> "$log_file"
                continue
            fi

            LC_ALL=C sort -k4,4g "$all_voxels" | tail -n "$top_n" > "$top_voxels"

            selected_count=$(wc -l < "$top_voxels" | tr -d ' ')
            nonzero_count=$(awk '$4 != 0 {count++} END {print count+0}' "$top_voxels")
            if (( selected_count != top_n )); then
                printf "%s\t%s\t%s\tERROR\tselected %d rather than %d voxels\n" \
                    "$participant_id" "$roi" "$contrast_label" \
                    "$selected_count" "$top_n" >> "$log_file"
                continue
            fi

            if (( nonzero_count == 0 )); then
                printf "%s\t%s\t%s\tWARNING\tall selected values are zero\n" \
                    "$participant_id" "$roi" "$contrast_label" >> "$log_file"
            fi

            # 3dmaskdump writes i, j, k, and contrast value by default.
            # Replace the contrast value with 1 to create a binary mask.
            awk '{print $1, $2, $3, 1}' "$top_voxels" > "$binary_points"

            if ! 3dUndump \
                -ijk \
                -datum byte \
                -master "$contrast_file" \
                -prefix "$output_mask" \
                -overwrite \
                "$binary_points" >/dev/null 2>&1; then
                printf "%s\t%s\t%s\tERROR\t3dUndump failed\n" \
                    "$participant_id" "$roi" "$contrast_label" >> "$log_file"
                continue
            fi

            printf "%s\t%s\t%s\tOK\t%d voxels selected\n" \
                "$participant_id" "$roi" "$contrast_label" \
                "$selected_count" >> "$log_file"
        done
    done
done < "$participant_file"

echo "Top-${top_n} ROI selection complete. Log: ${log_file}"
