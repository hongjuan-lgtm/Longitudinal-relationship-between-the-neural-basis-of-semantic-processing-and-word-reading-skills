function main_preprocess(cohort, config)
%MAIN_PREPROCESS Run fMRI preprocessing for one longitudinal cohort.
%
% This function reads the final participant/scan selection TSV, stages the
% selected OpenNeuro files in a derivatives directory, and preprocesses both
% sessions for every participant in the requested cohort.
%
% Required config fields:
%   data_root          Root of the OpenNeuro BIDS dataset
%   output_root        Root directory for preprocessing derivatives
%   participants_5_7  Path to final_participants_5-7.tsv
%   participants_7_9  Path to final_participants_7-9.tsv
%   template_5_7      Path to mw_com_prior_Age_0081.nii
%   template_7_9      Path to mw_com_prior_Age_0102.nii
%   spm_path           Path to SPM12
%   functions_dir      Directory containing the project helper functions
%   artrepair_path    Directory containing ArtRepair dependencies
%   canny_path        Directory containing Canny dependencies

% Example:
%   config.data_root = '/path/to/ds003604';
%   config.output_root = '/path/to/derivatives/preprocessing';
%   config.participants_5_7 = 'participants/final_participants_5-7.tsv';
%   config.participants_7_9 = 'participants/final_participants_7-9.tsv';
%   config.template_5_7 = 'preprocessing/templates/mw_com_prior_Age_0081.nii';
%   config.template_7_9 = 'preprocessing/templates/mw_com_prior_Age_0102.nii';
%   config.spm_path = '/path/to/spm12';
%   config.functions_dir = 'preprocessing/functions';
%   config.third_party_dir = 'preprocessing/third_party';
%   main_preprocess('5-7', config);

if nargin ~= 2
    error('Usage: main_preprocess(cohort, config)');
end

validate_config(config);

switch cohort
    case '5-7'
        participant_file = config.participants_5_7;
        sessions = [5 7];
        tpm = config.template_5_7;
    case '7-9'
        participant_file = config.participants_7_9;
        sessions = [7 9];
        tpm = config.template_7_9;
    otherwise
        error('Unknown cohort "%s". Expected "5-7" or "7-9".', cohort);
end

assert_file_exists(participant_file, 'participant TSV');
assert_file_exists(tpm, 'pediatric tissue probability map');
assert_file_exists(fullfile(fileparts(tpm), 'mask_ICV.nii'), ...
    'template intracranial mask');

addpath(genpath(config.spm_path));
addpath(genpath(config.functions_dir));
addpath(config.artrepair_path);
addpath(config.canny_path);

spm_get_defaults('cmdline', true);
spm('defaults', 'fmri');
spm_jobman('initcfg');
set(0, 'DefaultFigureVisible', 'off');

participants = readtable( ...
    participant_file, ...
    'FileType', 'text', ...
    'Delimiter', '\t', ...
    'TextType', 'string');

required_columns = {'participant_id'};
for session = sessions
    required_columns{end + 1} = sprintf('ses_%d_anat', session); %#ok<AGROW>
    required_columns{end + 1} = sprintf('ses_%d_func_run_01', session); %#ok<AGROW>
    required_columns{end + 1} = sprintf('ses_%d_func_run_02', session); %#ok<AGROW>
end
assert_columns_exist(participants, required_columns);

fprintf('Starting preprocessing for cohort %s (%d participants).\n', ...
    cohort, height(participants));

for row = 1:height(participants)
    participant_id = table_text(participants, row, 'participant_id');

    for session = sessions
        session_id = sprintf('ses-%d', session);
        anat_column = sprintf('ses_%d_anat', session);
        run01_column = sprintf('ses_%d_func_run_01', session);
        run02_column = sprintf('ses_%d_func_run_02', session);

        anat_name = table_text(participants, row, anat_column);
        func_names = { ...
            table_text(participants, row, run01_column), ...
            table_text(participants, row, run02_column)};

        if isempty(anat_name) || any(cellfun(@isempty, func_names))
            error('Missing selected scan for %s %s in %s.', ...
                participant_id, session_id, participant_file);
        end

        fprintf('\nProcessing %s, cohort %s, %s.\n', ...
            participant_id, cohort, session_id);

        session_output = fullfile( ...
            config.output_root, cohort, participant_id, session_id);
        anat_output = fullfile(session_output, 'anat');
        func_output = fullfile(session_output, 'func');
        ensure_directory(anat_output);
        ensure_directory(func_output);

        anat_source = fullfile( ...
            config.data_root, participant_id, session_id, 'anat', anat_name);
        anat_file = stage_and_unzip(anat_source, anat_output);

        func_files = cell(1, 2);
        for run_index = 1:2
            func_source = fullfile( ...
                config.data_root, participant_id, session_id, ...
                'func', func_names{run_index});
            run_folder = remove_nii_extension(func_names{run_index});
            run_output = fullfile(func_output, run_folder);
            ensure_directory(run_output);
            func_files{run_index} = stage_and_unzip(func_source, run_output);
        end

        preprocess_session( ...
            anat_file, func_files, session_output, tpm, participant_id, session_id);
    end
end

fprintf('\nCompleted preprocessing for cohort %s.\n', cohort);

end


function preprocess_session(anat_file, func_files, output_dir, tpm, participant_id, session_id)
% Realign and reslice both selected functional runs to their mean.
[realigned_files, mean_func, rp_files] = realignment(func_files, output_dir);

for run_index = 1:numel(rp_files)
    copyfile(rp_files{run_index}, output_dir);
end

% Segment T1 using the cohort-specific pediatric tissue probability map.
[deformation, tissue_files] = segmentation(anat_file, tpm);

% Create a skull-stripped T1 from gray matter, white matter, and CSF.
brain_mask = mkmask(tissue_files);
skull_stripped_t1 = no_skull(anat_file, brain_mask, 'T1_ns');

% Coregister the mean functional and realigned runs to the skull-stripped T1.
[coreg_mean, coreg_func] = coregister( ...
    mean_func, skull_stripped_t1, realigned_files, output_dir, 'no');

% Normalize to MNI space using the deformation estimated from the T1.
[normalized_func, normalized_mean] = normalise( ...
    coreg_func, deformation, coreg_mean);

% Smooth normalized functional images with a 6 mm FWHM Gaussian kernel.
smoothed_func = smoothing(normalized_func, 6);

% Identify and repair outlier volumes separately for each run.
percent_threshold = 4;
scan_to_scan_threshold = 1.5;
movement_to_reference_threshold = 100;

for run_index = 1:numel(smoothed_func)
    smoothed_file = unwrap_file_path(smoothed_func{run_index});
    [run_dir, run_name] = fileparts(smoothed_file);

    % ArtRepair operates on individual three-dimensional volumes.
    spm_file_split(smoothed_file);
    split_pattern = ['^' regexptranslate('escape', run_name) '_\d+\.nii$'];
    split_files = spm_select('ExtFPList', run_dir, split_pattern, Inf);
    if isempty(strtrim(split_files))
        error('No split volumes were created for %s.', smoothed_file);
    end

    art_global_jin( ...
        split_files, ...
        rp_files{run_index}, ...
        4, ...  % automatic brain mask
        1, ...  % automatic interpolation repair
        percent_threshold, ...
        scan_to_scan_threshold, ...
        movement_to_reference_threshold);

    % ArtRepair writes one v-prefixed file for every input volume.
    repaired_pattern = ['^v' regexptranslate('escape', run_name) '_\d+\.nii$'];
    repaired_volumes = spm_select('ExtFPList', run_dir, repaired_pattern, Inf);
    if isempty(strtrim(repaired_volumes))
        error(['ArtRepair produced no v-prefixed volumes for %s. ' ...
            'Check the ArtRepair installation and logs.'], smoothed_file);
    end

    repaired_4d = fullfile(run_dir, ['v' run_name '.nii']);
    spm_file_merge(repaired_volumes, repaired_4d);

    % Remove only the temporary split volumes. Preserve the original 4D
    % smoothed image, the repaired 4D image, and ArtRepair text outputs.
    delete_split_volumes(run_dir, run_name);
end

% Create a registration QC image for this participant and session.
coreg_check(normalized_mean, output_dir, tpm);

fprintf('Finished %s %s.\n', participant_id, session_id);

end


function output_file = stage_and_unzip(source_file, output_dir)
assert_file_exists(source_file, 'selected OpenNeuro input');

[~, name, extension] = fileparts(source_file);
destination = fullfile(output_dir, [name extension]);
if ~exist(destination, 'file')
    copyfile(source_file, destination);
end

if endsWith(destination, '.gz')
    uncompressed = destination(1:end - 3);
    if ~exist(uncompressed, 'file')
        generated = gunzip(destination, output_dir);
        uncompressed = generated{1};
    end
    output_file = uncompressed;
else
    output_file = destination;
end

end


function value = table_text(data, row, variable_name)
raw = data.(variable_name)(row);
if iscell(raw)
    raw = raw{1};
end
if ismissing(raw)
    value = '';
else
    value = char(string(raw));
end
end


function value = unwrap_file_path(value)
while iscell(value)
    value = value{1};
end
value = char(value);
end


function stem = remove_nii_extension(filename)
stem = filename;
if endsWith(stem, '.nii.gz')
    stem = stem(1:end - 7);
elseif endsWith(stem, '.nii')
    stem = stem(1:end - 4);
end
end


function delete_split_volumes(folder, run_name)
patterns = {[run_name '_*.nii'], ['v' run_name '_*.nii']};
for pattern_index = 1:numel(patterns)
    files = dir(fullfile(folder, patterns{pattern_index}));
    for file_index = 1:numel(files)
        delete(fullfile(folder, files(file_index).name));
    end
end
end


function ensure_directory(folder)
if ~exist(folder, 'dir')
    mkdir(folder);
end
end


function validate_config(config)
required_fields = { ...
    'data_root', ...
    'output_root', ...
    'participants_5_7', ...
    'participants_7_9', ...
    'template_5_7', ...
    'template_7_9', ...
    'spm_path', ...
    'functions_dir', ...
    'third_party_dir'};

for index = 1:numel(required_fields)
    field = required_fields{index};
    if ~isfield(config, field) || isempty(config.(field))
        error('Missing required config field: %s', field);
    end
end

assert_folder_exists(config.data_root, 'OpenNeuro data root');
assert_folder_exists(config.spm_path, 'SPM12 directory');
assert_folder_exists(config.functions_dir, 'preprocessing functions directory');
assert_folder_exists(config.third_party_dir, 'third-party functions directory');
ensure_directory(config.output_root);
end


function assert_columns_exist(data, columns)
available = data.Properties.VariableNames;
for index = 1:numel(columns)
    if ~ismember(columns{index}, available)
        error('Participant TSV is missing column: %s', columns{index});
    end
end
end


function assert_file_exists(filename, description)
if ~exist(filename, 'file')
    error('Missing %s: %s', description, filename);
end
end


function assert_folder_exists(folder, description)
if ~exist(folder, 'dir')
    error('Missing %s: %s', description, folder);
end
end
