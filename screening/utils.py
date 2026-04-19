import os
import pandas as pd
import nibabel as nib


def get_longitudinal_subjects(data_dir):
    """
    Step 1:
    Select subjects who have both ses-5 and ses-7, or both ses-7 and ses-9.
    """
    step1_5_7 = []
    step1_7_9 = []

    for sub in os.listdir(data_dir):
        if not sub.startswith("sub-"):
            continue

        sub_path = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        sessions = os.listdir(sub_path)

        if "ses-5" in sessions and "ses-7" in sessions:
            step1_5_7.append(sub)
        if "ses-7" in sessions and "ses-9" in sessions:
            step1_7_9.append(sub)

    return sorted(step1_5_7), sorted(step1_7_9)


def subject_to_pid(sub):
    return int(sub.replace("sub-", ""))


def filter_by_handedness(subjects, participants_df, handedness_col, threshold=3):
    """
    Step 2:
    Keep subjects with handedness >= threshold.
    """
    passed = []
    for sub in subjects:
        pid = subject_to_pid(sub)
        row = participants_df[participants_df["participant_id"] == pid]
        if row.empty:
            continue

        val = row.iloc[0][handedness_col]
        if pd.notna(val) and val >= threshold:
            passed.append(sub)

    return sorted(passed)


def filter_by_score(subjects, df, score_col, threshold=70):
    """
    Generic filter for Step 3 / Step 4.
    Keep subjects with score_col >= threshold.
    """
    passed = []
    for sub in subjects:
        pid = subject_to_pid(sub)
        row = df[df["participant_id"] == pid]
        if row.empty:
            continue

        val = row.iloc[0][score_col]
        try:
            if pd.notna(val) and float(val) >= threshold:
                passed.append(sub)
        except Exception:
            continue

    return sorted(passed)


def has_two_sem_runs(data_dir, sub, ses):
    """
    Step 5:
    Check whether a subject has both run-01 and run-02 for task-Sem in a session.
    """
    func_dir = os.path.join(data_dir, sub, ses, "func")
    if not os.path.exists(func_dir):
        return False

    runs_found = set()
    for fname in os.listdir(func_dir):
        if "task-Sem" in fname and fname.endswith(".nii.gz"):
            if "_run-01_" in fname:
                runs_found.add(1)
            elif "_run-02_" in fname:
                runs_found.add(2)

    return 1 in runs_found and 2 in runs_found


def filter_two_sessions(subjects, ses1, ses2, check_func):
    """
    Generic helper:
    Keep subjects who pass a session-level check at both timepoints.
    """
    passed = []
    for sub in subjects:
        if check_func(sub, ses1) and check_func(sub, ses2):
            passed.append(sub)
    return sorted(passed)


def get_valid_wj_subjects(phenotype_dir, session):
    """
    Step 6:
    Return subjects with valid WJ-III_WordID_Raw at a session.
    """
    tsv_path = os.path.join(phenotype_dir, session, "wj-iii.tsv")
    df = pd.read_csv(tsv_path, sep="\t")

    df_valid = df[
        (df["WJ-III_WordID_Raw"].notna()) &
        (df["WJ-III_WordID_Raw"] != "n/a")
    ].copy()

    df_valid["participant_id"] = df_valid["participant_id"].astype(str).apply(lambda x: f"sub-{x}")
    return set(df_valid["participant_id"].tolist())


def filter_by_wj(subjects, valid_set_1, valid_set_2):
    """
    Keep subjects with valid WJ data at both timepoints.
    """
    return sorted([sub for sub in subjects if sub in valid_set_1 and sub in valid_set_2])


def check_motion_pass(data_dir, mri_qc_root, sub, ses):
    """
    Step 7:
    Check whether both run-01 and run-02 of task-Sem pass motion QC in a given session.

    Criteria:
    - num_repaired / total_volumes < 0.10
    - chunks == 0
    """
    mv_file = os.path.join(mri_qc_root, f"mv_acc_func_{ses}.tsv")
    if not os.path.exists(mv_file):
        return False, []

    mv_df = pd.read_csv(mv_file, sep="\t")
    sub_id = subject_to_pid(sub)
    sub_df = mv_df[mv_df["participant_id"] == sub_id]

    if sub_df.empty:
        return False, []

    passed_run_names = []

    for run in [1, 2]:
        run_tag = f"_run-0{run}_"
        run_rows = sub_df[
            sub_df["run_name"].str.contains(run_tag, na=False) &
            sub_df["run_name"].str.contains("task-Sem", na=False)
        ]

        for _, row in run_rows.iterrows():
            run_name = row["run_name"]
            bold_path = os.path.join(data_dir, sub, ses, "func", f"{run_name}.nii.gz")

            if not os.path.exists(bold_path):
                continue

            try:
                img = nib.load(bold_path)
                total_volumes = img.shape[-1]
            except Exception:
                continue

            num_repaired = row["num_repaired"]
            chunks = row["chunks"]
            repair_ratio = num_repaired / total_volumes if total_volumes > 0 else 1.0

            if repair_ratio < 0.10 and chunks == 0:
                passed_run_names.append(run_name)

    run_string = "_".join(passed_run_names)
    if "_run-01_" not in run_string or "_run-02_" not in run_string:
        return False, []

    return True, passed_run_names


def filter_by_motion(subjects, ses1, ses2, data_dir, mri_qc_root):
    """
    Keep subjects who pass motion QC at both timepoints.
    """
    passed = []
    for sub in subjects:
        passed_1, _ = check_motion_pass(data_dir, mri_qc_root, sub, ses1)
        passed_2, _ = check_motion_pass(data_dir, mri_qc_root, sub, ses2)
        if passed_1 and passed_2:
            passed.append(sub)
    return sorted(passed)


def check_semantic_behavior_by_sub(data_dir, sub, ses):
    """
    Step 8:
    Check semantic behavioral performance for a subject in one session.

    Current logic follows the original script:
    If ANY task-Sem events file in the session satisfies:
    - S_C >= 0.5
    - S_H >= 0.5
    - abs(S_H - S_U) < 0.4
    then the subject passes for that session.
    """
    func_dir = os.path.join(data_dir, sub, ses, "func")
    if not os.path.exists(func_dir):
        return False

    for fname in os.listdir(func_dir):
        if not fname.endswith("_events.tsv") or "task-Sem" not in fname:
            continue

        events_path = os.path.join(func_dir, fname)

        try:
            df = pd.read_csv(events_path, sep="\t")
            if "calculated_accuracy" not in df.columns or "trial_type" not in df.columns:
                continue

            df = df[df["calculated_accuracy"].notna()]
            acc_by_type = df.groupby("trial_type")["calculated_accuracy"].mean().to_dict()

            acc_S_C = acc_by_type.get("S_C", 0)
            acc_S_H = acc_by_type.get("S_H", 0)
            acc_S_U = acc_by_type.get("S_U", 0)

            if acc_S_C >= 0.5 and acc_S_H >= 0.5 and abs(acc_S_H - acc_S_U) < 0.4:
                return True

        except Exception:
            continue

    return False


def filter_by_behavior(subjects, ses1, ses2, data_dir):
    """
    Keep subjects who pass semantic behavior criteria at both timepoints.
    """
    passed = []
    for sub in subjects:
        if check_semantic_behavior_by_sub(data_dir, sub, ses1) and check_semantic_behavior_by_sub(data_dir, sub, ses2):
            passed.append(sub)
    return sorted(passed)


def get_overlap(list_1, list_2):
    return sorted(set(list_1).intersection(set(list_2)))


def select_best_acq_runs(sub_list, sessions, data_dir, mri_qc_root):
    """
    Step 9:
    For each subject/session/run-id (run-01, run-02), select the acquisition
    with the lowest repaired-volume ratio.

    Returns a list of run_names, not subjects.
    """
    best_runs = []

    for sub in sub_list:
        sub_id = subject_to_pid(sub)

        for ses in sessions:
            mv_file = os.path.join(mri_qc_root, f"mv_acc_func_{ses}.tsv")
            if not os.path.exists(mv_file):
                continue

            mv_df = pd.read_csv(mv_file, sep="\t")
            sub_df = mv_df[mv_df["participant_id"] == sub_id]

            for run_id in ["run-01", "run-02"]:
                best_run = None
                best_ratio = float("inf")

                candidate_rows = sub_df[
                    sub_df["run_name"].str.contains(run_id, na=False) &
                    sub_df["run_name"].str.contains("task-Sem", na=False)
                ]

                for _, row in candidate_rows.iterrows():
                    run_name = row["run_name"]
                    bold_path = os.path.join(data_dir, sub, ses, "func", f"{run_name}.nii.gz")

                    if not os.path.exists(bold_path):
                        continue

                    try:
                        img = nib.load(bold_path)
                        total_volumes = img.shape[-1]
                    except Exception:
                        continue

                    num_repaired = row["num_repaired"]
                    ratio = num_repaired / total_volumes if total_volumes > 0 else 1.0

                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_run = run_name

                if best_run:
                    best_runs.append(best_run)

    return best_runs


def save_list_to_txt(items, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(str(item) + "\n")


def save_step_output(step_num, list_5_7, list_7_9, output_dir="step_outputs"):
    """
    Save subject lists for one step.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"step{step_num}_5_7.txt"), "w") as f:
        for item in list_5_7:
            f.write(item + "\n")

    with open(os.path.join(output_dir, f"step{step_num}_7_9.txt"), "w") as f:
        for item in list_7_9:
            f.write(item + "\n")


def print_step_result(step_name, list_5_7, list_7_9):
    print(f"{step_name} 5-7: {len(list_5_7)}")
    print(f"{step_name} 7-9: {len(list_7_9)}")
