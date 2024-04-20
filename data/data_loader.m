% Initialize the cell array to store data from each run
% Adjust 'N' to match the total number of runs
N = 17; % Example: for run_0 to run_16
dataset = cell(1,N);

% Base directory where the run_X folders are located
experiment_type = "experiment_pro";
baseDir = fullfile(pwd,experiment_type);

% Loop through each run directory
for i = 0:N-1
    % Construct the path to the test subdirectory
    subDirPath = fullfile(baseDir, sprintf('run_%d', i), 'test');
    
    % Construct the path to the dataset.mat file within the test subdirectory
    dataFilePath = fullfile(subDirPath, 'dataset.mat');
    
    % Check if the dataset.mat file exists
    if isfile(dataFilePath)
        % Load the dataset.mat file
        loadedData = load(dataFilePath);
        
        % Assuming 'data' is the variable name in dataset.mat
        % Update the cell array with the loaded struct variable 'data'
        % Adjust the indexing based on how you wish to reference the runs
        dataset{i+1} = loadedData.data{:}; % Adjust indexing if necessary
    else
        % Handle cases where dataset.mat does not exist in the expected path
        fprintf('File not found: %s\n', dataFilePath);
    end
end

% At this point, 'dataset' cell array contains the data from each run's dataset.mat file
save(experiment_type+"_dataset","dataset");