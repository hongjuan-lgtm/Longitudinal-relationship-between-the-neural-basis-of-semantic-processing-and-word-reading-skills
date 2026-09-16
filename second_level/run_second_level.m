% Run the eight second-level one-sample t-tests used for whole-brain maps.
% Edit the configuration below, then run this script in MATLAB.
% Participant tables must contain a participant_id column.
% No explicit mask or covariates. See README.md for cluster-FWE inference.

%% Configuration
spm_path = '/path/to/spm12';
data_root = '/path/to/Data';
output_root = '/path/to/second_level_results';
participant_files = {
    '/path/to/final_participants_5-7.tsv'
    '/path/to/final_participants_7-9.tsv'
};
cohorts = {'5-7', '7-9'};

% Must match the contrasts defined in the first-level contrast script.
contrast_numbers = [1 2 3 4];
analysis_names = {'T1_shallow_H_vs_C', 'T1_deep_L_vs_C', ...
                  'T2_shallow_H_vs_C', 'T2_deep_L_vs_C'};

%% Initialize
addpath(spm_path);
addpath(fileparts(mfilename('fullpath')));
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Prepare and validate all eight jobs before running any model.
out_dirs = {};
scans = {};
manifest = {};
for c = 1:numel(cohorts)
    opts = detectImportOptions(participant_files{c}, 'FileType', 'text', ...
                              'Delimiter', '\t');
    if ~ismember('participant_id', opts.VariableNames)
        error('Missing participant_id column: %s', participant_files{c});
    end
    opts = setvartype(opts, 'participant_id', 'string');
    participants = readtable(participant_files{c}, opts);
    ids = strip(participants.participant_id);
    if any(ismissing(ids)) || any(strlength(ids) == 0)
        error('Missing participant ID in %s', participant_files{c});
    end
    ids = regexprep(ids, '^sub-', '');
    ids = "sub-" + ids;
    if any(cellfun(@isempty, regexp(cellstr(ids), '^sub-[A-Za-z0-9]+$', 'once')))
        error('Invalid participant ID in %s', participant_files{c});
    end
    if numel(ids) < 2 || numel(unique(ids)) ~= numel(ids)
        error('Need at least two unique participants; duplicates are not allowed.');
    end
    for k = 1:numel(contrast_numbers)
        job = numel(out_dirs) + 1;
        out_dirs{job} = fullfile(output_root, cohorts{c}, analysis_names{k});
        if exist(out_dirs{job}, 'dir')
            entries = dir(out_dirs{job});
            if any(~ismember({entries.name}, {'.', '..'}))
                error('Output directory is not empty: %s', out_dirs{job});
            end
        end
        scans{job} = cell(numel(ids), 1);
        image_paths = cell(numel(ids), 1);
        for s = 1:numel(ids)
            image_paths{s} = fullfile(data_root, cohorts{c}, char(ids(s)), ...
                'analysis', 'deweight', sprintf('con_%04d.nii', contrast_numbers(k)));
            if ~exist(image_paths{s}, 'file')
                error('Missing contrast image: %s', image_paths{s});
            end
            scans{job}{s} = [image_paths{s} ',1'];
        end
        manifest{job} = table(ids, string(image_paths), ...
            'VariableNames', {'participant_id', 'contrast_image'});
    end
end

%% Fit models and save the exact input lists
for job = 1:numel(out_dirs)
    mkdir(out_dirs{job});
    writetable(manifest{job}, fullfile(out_dirs{job}, 'input_scans.tsv'), ...
               'FileType', 'text', 'Delimiter', '\t');
end
onesample_t(out_dirs, scans, []);
fprintf('Completed %d second-level models. See README.md for cluster-FWE filtering.\n', numel(out_dirs));
