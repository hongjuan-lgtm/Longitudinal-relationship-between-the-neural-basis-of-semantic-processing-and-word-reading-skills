#!/usr/bin/env bash

set -u

# USER CONFIGURATION
cohort="5-7"
firstlevel_dir="/path/to/repository/first_level"
participant_file="/path/to/repository/participants/final_participants_5-7.tsv"
data_root="/path/to/preprocessed/data"
log_dir="/path/to/logs/first_level/${cohort}"

cd "$firstlevel_dir" || exit 1
mkdir -p "$log_dir"
failed_list="${log_dir}/failed_subjects_${cohort}.txt"
: > "$failed_list"

# Read participant_id from the first column of the final participant TSV.
tail -n +2 "$participant_file" | cut -f1 | while IFS= read -r participant_id; do
    if [[ -z "$participant_id" ]]; then
        continue
    fi

    result_file="${data_root}/${cohort}/${participant_id}/analysis/deweight/SPM.mat"
    if [[ -f "$result_file" ]]; then
        echo "Skipping ${participant_id}: deweighted SPM.mat already exists."
        continue
    fi

    log_file="${log_dir}/firstlevel_${participant_id}_${cohort}.log"
    echo "Running ${participant_id} (${cohort})..."

    matlab -nodisplay -nosplash -r \
        "try; run_firstlevel_one_subject('${participant_id}', '${cohort}'); catch ME; disp(getReport(ME, 'extended')); exit(1); end; exit(0);" \
        > "$log_file" 2>&1

    status=$?
    if [[ $status -ne 0 ]]; then
        echo "${participant_id} failed. See ${log_file}."
        echo "$participant_id" >> "$failed_list"
    else
        echo "${participant_id} completed."
    fi
done

echo "Batch complete. Failed participants are listed in ${failed_list}."
