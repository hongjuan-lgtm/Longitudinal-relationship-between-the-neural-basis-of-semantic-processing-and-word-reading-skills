function generate_pediatric_template(cohort, config)
%GENERATE_PEDIATRIC_TEMPLATE Generate an age- and sex-matched CerebroMatic TPM.
%
% The script pools eligible participant-session observations from the two
% sessions in a longitudinal cohort. Each observation contributes its age
% in completed months, sex, and 3 T MRI field strength to CerebroMatic.
%
% Cohort definitions used in the original analysis:
%   5-7: sessions 5 and 7, ages 66-96 months (5.5-8 years)
%   7-9: sessions 7 and 9, ages 90-120 months
%
% Required config fields:
%   participants_tsv    OpenNeuro participants.tsv
%   spm_path            SPM12 installation containing CerebroMatic
%   cerebromatic_info   CerebroMatic mw_com_info.mat
%   output_root         Directory in which cohort folders are created
%
% Example:
%   config.participants_tsv = '/path/to/ds003604/participants.tsv';
%   config.spm_path = '/path/to/spm12';
%   config.cerebromatic_info = fullfile( ...
%       config.spm_path, 'toolbox', ...
%       'com_parameters_unified-segmentation', 'mw_com_info.mat');
%   config.output_root = 'resources/templates';
%   generate_pediatric_template('5-7', config);
%   generate_pediatric_template('7-9', config);

if nargin ~= 2
    error('Usage: generate_pediatric_template(cohort, config)');
end

validate_config(config);

switch cohort
    case '5-7'
        sessions = [5 7];
        age_range_months = [66 96];
        expected_tpm = 'mw_com_prior_Age_0081.nii';
    case '7-9'
        sessions = [7 9];
        age_range_months = [90 120];
        expected_tpm = 'mw_com_prior_Age_0102.nii';
    otherwise
        error('Unknown cohort "%s". Expected "5-7" or "7-9".', cohort);
end

cohort_output = fullfile(config.output_root, cohort);
ensure_directory(cohort_output);

addpath(genpath(config.spm_path));
if exist('mw_com_gen', 'file') ~= 2
    error(['mw_com_gen was not found. Install CerebroMatic in the SPM12 ' ...
        'toolbox directory or add it to the MATLAB path.']);
end

participants = read_participants(config.participants_tsv, sessions);
template_inputs = collect_template_inputs( ...
    participants, sessions, age_range_months);

if isempty(template_inputs)
    error('No valid participant-session observations were found for cohort %s.', cohort);
end

input_manifest = fullfile(cohort_output, ...
    sprintf('template_inputs_%s.tsv', strrep(cohort, '-', '_')));
writetable(template_inputs, input_manifest, ...
    'FileType', 'text', 'Delimiter', '\t');

age = template_inputs.age_months';
sex = template_inputs.sex_code';
number_of_observations = height(template_inputs);

fprintf(['Generating the %s template from %d participant-session ' ...
    'observations.\n'], cohort, number_of_observations);
fprintf('Age range: %d-%d months; mean age: %.2f months.\n', ...
    min(age), max(age), mean(age));
fprintf('Female: %d; male: %d.\n', sum(sex == 1), sum(sex == 0));

model = load(config.cerebromatic_info);
if ~isfield(model, 'predicts_o') || size(model.predicts_o, 1) < 4
    error('The CerebroMatic information file lacks predicts_o row 4.');
end

% Parameters preserved from the original template-generation script.
do_mean = 1;
smoothing = -2;
sanlm = 0;
mrf = 0;
output_type = 2;
field_strength = repmat(3, 1, number_of_observations);

predictors = zeros(4, number_of_observations);
predictors(1, :) = age;
predictors(2, :) = sex;
predictors(3, :) = field_strength;
predictors(4, :) = repmat( ...
    min(model.predicts_o(4, :)) + eps, 1, number_of_observations);

mw_com_gen( ...
    config.cerebromatic_info, predictors, do_mean, smoothing, ...
    sanlm, mrf, cohort_output, output_type);

expected_tpm_path = fullfile(cohort_output, expected_tpm);
if ~exist(expected_tpm_path, 'file')
    warning(['Expected template %s was not found. Check the generated files ' ...
        'and the mean age reported above before renaming any output.'], expected_tpm);
end

create_icv_mask(cohort_output);

fprintf('Template outputs written to %s.\n', cohort_output);
fprintf('CerebroMatic inputs recorded in %s.\n', input_manifest);

end


function participants = read_participants(filename, sessions)
opts = detectImportOptions(filename, 'FileType', 'text');

date_columns = {'birthdate'};
for session = sessions
    date_columns{end + 1} = sprintf('ses_%d_date_ST', session); %#ok<AGROW>
end

available = opts.VariableNames;
required = [{'participant_id', 'sex'}, date_columns];
for index = 1:numel(required)
    if ~ismember(required{index}, available)
        error('participants.tsv is missing column: %s', required{index});
    end
end

opts = setvartype(opts, date_columns, 'datetime');
participants = readtable(filename, opts);
end


function inputs = collect_template_inputs(participants, sessions, age_range)
participant_id = strings(0, 1);
session_id = strings(0, 1);
age_months = zeros(0, 1);
sex_code = zeros(0, 1);

for row = 1:height(participants)
    birth_date = participants.birthdate(row);
    if isnat(birth_date)
        continue;
    end

    [valid_sex, encoded_sex] = encode_sex(participants.sex(row));
    if ~valid_sex
        warning('Skipping %s because sex is missing or unrecognized.', ...
            string(participants.participant_id(row)));
        continue;
    end

    for session = sessions
        date_column = sprintf('ses_%d_date_ST', session);
        session_date = participants.(date_column)(row);
        if isnat(session_date)
            continue;
        end

        completed_months = calmonths( ...
            between(birth_date, session_date, 'months'));
        if completed_months < age_range(1) || completed_months > age_range(2)
            continue;
        end

        participant_id(end + 1, 1) = string( ...
            participants.participant_id(row)); %#ok<AGROW>
        session_id(end + 1, 1) = sprintf('ses-%d', session); %#ok<AGROW>
        age_months(end + 1, 1) = completed_months; %#ok<AGROW>
        sex_code(end + 1, 1) = encoded_sex; %#ok<AGROW>
    end
end

inputs = table(participant_id, session_id, age_months, sex_code);
end


function [valid, code] = encode_sex(value)
label = lower(strtrim(string(value)));
if ismember(label, ["female", "f"])
    valid = true;
    code = 1;
elseif ismember(label, ["male", "m"])
    valid = true;
    code = 0;
else
    valid = false;
    code = NaN;
end
end


function create_icv_mask(output_dir)
[status, ~] = system('command -v 3dcalc');
if status ~= 0
    error(['AFNI 3dcalc was not found. Load or install AFNI, then rerun this ' ...
        'script to create mask_ICV.nii.']);
end

t1_files = dir(fullfile(output_dir, 'mw_com_T1*.nii'));
if numel(t1_files) ~= 1
    error(['Expected exactly one mw_com_T1*.nii in %s, but found %d. ' ...
        'Use a clean cohort output directory.'], output_dir, numel(t1_files));
end

t1_path = fullfile(t1_files(1).folder, t1_files(1).name);
mask_path = fullfile(output_dir, 'mask_ICV.nii');
command = sprintf('3dcalc -overwrite -a %s -expr %s -prefix %s', ...
    shell_quote(t1_path), shell_quote('step(a)'), shell_quote(mask_path));
[status, output] = system(command);
if status ~= 0
    error('AFNI 3dcalc failed: %s', output);
end
end


function quoted = shell_quote(value)
% Quote a path or argument for a POSIX shell.
quoted = ['''' strrep(value, '''', '''"''"''') ''''];
end


function validate_config(config)
required_fields = { ...
    'participants_tsv', ...
    'spm_path', ...
    'cerebromatic_info', ...
    'output_root'};

for index = 1:numel(required_fields)
    field = required_fields{index};
    if ~isfield(config, field) || isempty(config.(field))
        error('Missing required config field: %s', field);
    end
end

assert_file_exists(config.participants_tsv, 'OpenNeuro participants.tsv');
assert_folder_exists(config.spm_path, 'SPM12 directory');
assert_file_exists(config.cerebromatic_info, 'CerebroMatic mw_com_info.mat');
ensure_directory(config.output_root);
end


function ensure_directory(folder)
if ~exist(folder, 'dir')
    mkdir(folder);
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
