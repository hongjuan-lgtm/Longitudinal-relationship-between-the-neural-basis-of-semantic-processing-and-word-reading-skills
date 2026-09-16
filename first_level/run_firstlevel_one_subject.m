function run_firstlevel_one_subject(subj_id, cohort)
%RUN_FIRSTLEVEL_ONE_SUBJECT Run the original first-level workflow for one subject.
%
% Before running, update the paths in the USER CONFIGURATION section.
% Examples:
%   run_firstlevel_one_subject('5004', '5-7')
%   run_firstlevel_one_subject('sub-5004', '5-7')

%% USER CONFIGURATION
code_path = '/path/to/repository';
spm_path = '/path/to/spm12';
data_root = '/path/to/preprocessed/data';
bids_root = '/path/to/openneuro/dataset';

%% INITIALIZE
global CCN

if startsWith(subj_id, 'sub-')
    subject = char(subj_id);
else
    subject = ['sub-' char(subj_id)];
end

switch cohort
    case '5-7'
        timepoints = [5 7];
    case '7-9'
        timepoints = [7 9];
    otherwise
        error('Unknown cohort "%s". Expected "5-7" or "7-9".', cohort);
end

addpath(genpath(code_path));
addpath(genpath(spm_path));

spm_get_defaults('cmdline', true);
set(0, 'DefaultFigureVisible', 'off');
spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_figure('Create', 'Graphics', 'Graphics');

if verLessThan('matlab', 'R2013a')
    error('MATLAB R2013a or newer is required. Current version: %s', version);
end

%% INPUT AND OUTPUT PATHS
subject_root = fullfile(data_root, cohort, subject);
if ~exist(subject_root, 'dir')
    error('Subject directory not found: %s', subject_root);
end

analysis_path = fullfile(subject_root, 'analysis');
model_deweight_path = fullfile(analysis_path, 'deweight');
ensure_directory(analysis_path);
ensure_directory(model_deweight_path);

% Locate the four runs in chronological order:
% T1 Run 1, T1 Run 2, T2 Run 1, T2 Run 2.
functional_dirs = cell(1, 4);
index = 0;
for session = timepoints
    for run_number = 1:2
        index = index + 1;
        run_pattern = fullfile( ...
            subject_root, sprintf('ses-%d', session), 'func', ...
            sprintf('*_ses-%d_*_run-%02d_bold', session, run_number));
        matches = dir(run_pattern);
        matches = matches([matches.isdir]);
        if numel(matches) ~= 1
            error(['Expected one run directory for %s ses-%d run-%02d, ' ...
                'but found %d.'], subject, session, run_number, numel(matches));
        end
        functional_dirs{index} = fullfile(matches(1).folder, matches(1).name);
    end
end

%% CONDITIONS, EVENT DURATIONS, AND TR
conditions = cell(1, 4);
for session_index = 1:4
    conditions{session_index} = {'S_C', 'S_H', 'S_L', 'S_U'};
end

% firstlevel_4d.m expects data.dur{session}{condition}.
dur = cell(size(conditions));
for session_index = 1:numel(conditions)
    dur{session_index} = num2cell( ...
        zeros(1, numel(conditions{session_index})));
end

TR = 1.25;

%% LOAD FUNCTIONAL IMAGES, EVENTS, AND MOVEMENT PARAMETERS
data = struct();
swfunc = cell(1, 4);
onsets = cell(1, 4);
mv = cell(1, 4);

for session_index = 1:numel(functional_dirs)
    run_dir = functional_dirs{session_index};
    [~, run_name] = fileparts(run_dir);

    functional_file = find_one_file( ...
        run_dir, 'vs6_wr*bold.nii', 'repaired functional image');
    swfunc{session_index} = functional_file;

    event_name = regexprep(run_name, '_bold$', '_events.tsv');
    local_event_file = fullfile(run_dir, event_name);

    session_name = regexp(run_name, 'ses-[0-9]+', 'match', 'once');
    if isempty(session_name)
        error('Could not identify the session from run directory: %s', run_dir);
    end
    bids_event_file = fullfile( ...
        bids_root, subject, session_name, 'func', event_name);

    if exist(local_event_file, 'file')
        event_file = local_event_file;
    elseif exist(bids_event_file, 'file')
        event_file = bids_event_file;
    else
        error(['Events file not found in the run directory or BIDS dataset: ' ...
            '%s'], event_name);
    end

    event_data = readtable( ...
        event_file, 'FileType', 'text', ...
        'Delimiter', '\t', 'TextType', 'string');
    assert_event_columns(event_data, event_file);

    for condition_index = 1:numel(conditions{session_index})
        condition = conditions{session_index}{condition_index};
        condition_onsets = event_data.onset(event_data.trial_type == condition);
        if isempty(condition_onsets)
            error('No %s events found in %s.', condition, event_file);
        end
        onsets{session_index}{condition_index} = condition_onsets;
    end

    rp_file = find_one_file( ...
        run_dir, 'rp_*.txt', 'realignment parameter file');
    mv{session_index} = load(rp_file);
    if size(mv{session_index}, 2) ~= 6
        error('Expected six movement columns in %s.', rp_file);
    end
end

data.swfunc = swfunc;
data.conditions = conditions;
data.onsets = onsets;
data.dur = dur;
data.mv = mv;

%% MODEL SPECIFICATION, ESTIMATION, AND DEWEIGHTING
deweighted_mat = firstlevel_4d( ...
    data, analysis_path, TR, model_deweight_path);
original_mat = fullfile(analysis_path, 'SPM.mat');

%% CONTRASTS
contrasts = { ...
    sprintf('H_vs_C_ses%d', timepoints(1)), ...
    sprintf('L_vs_C_ses%d', timepoints(1)), ...
    sprintf('H_vs_C_ses%d', timepoints(2)), ...
    sprintf('L_vs_C_ses%d', timepoints(2))};

H_vs_C = [-1 1 0 0];
L_vs_C = [-1 0 1 0];
movement_zeros = zeros(1, 6);
session_zeros = zeros(1, 10);

weights = { ...
    [H_vs_C movement_zeros H_vs_C movement_zeros session_zeros session_zeros], ...
    [L_vs_C movement_zeros L_vs_C movement_zeros session_zeros session_zeros], ...
    [session_zeros session_zeros H_vs_C movement_zeros H_vs_C movement_zeros], ...
    [session_zeros session_zeros L_vs_C movement_zeros L_vs_C movement_zeros]};

if numel(weights) ~= numel(contrasts)
    error('The number of contrast names and weight vectors does not match.');
end

contrast_f(original_mat, contrasts, weights);
contrast_f(deweighted_mat, contrasts, weights);

fprintf('Completed first-level analysis for %s (%s).\n', subject, cohort);

end


function filename = find_one_file(folder, pattern, description)
matches = dir(fullfile(folder, pattern));
matches = matches(~[matches.isdir]);
if numel(matches) ~= 1
    error('Expected one %s in %s, but found %d.', ...
        description, folder, numel(matches));
end
filename = fullfile(matches(1).folder, matches(1).name);
end


function assert_event_columns(events, filename)
required = {'onset', 'trial_type'};
for index = 1:numel(required)
    if ~ismember(required{index}, events.Properties.VariableNames)
        error('%s is missing column: %s', filename, required{index});
    end
end
end


function ensure_directory(folder)
if ~exist(folder, 'dir')
    mkdir(folder);
end
end
