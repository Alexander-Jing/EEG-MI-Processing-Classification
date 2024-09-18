clc
clear all
close all
%% load data file
% This data set consists of electroencephalography (EEG) data collected from one
% subject with a high spinal cord lesion controlling an EEG/EOG hybrid BNCI to operate
% a neuroprosthetic device attached to his paralyzed right upper limb. 
% The cue-based BNCI paradigm consisted of two different tasks:
% imagination of movement% of the right hand (class 1) and relaxation/no movement(class 2),
% for more details, please refer to https://lampx.tugraz.at/~bci/database/002-2015/description.pdf
% for downloading data, please refer to https://bnci-horizon-2020.eu/database/data-sets 

% load the data
load('S01.mat');

% in this dataset, only 5 conventional EEG channels and one EOG channel were used
channels = {'F4','T8','C4','Cz','P4','HEOG'};
EPOCHS = [];
runs = 3;
event_num_all = 0;

% read all data
for run_idx= 1:runs
    % load(trueLabelName);
    trueLabel = data{1,run_idx}.y;
    
    % set channels, match the channel idx if
    % channels={'F4','T8','C4','Cz','P4','HEOG'} match the data{1,run_idx}.channel_names
    channel_indexes = [];
    for ch_i = 1:numel(channels)
        for ch_j = 1:numel(data{1,run_idx}.channel_names)
            if(strcmpi(data{1,run_idx}.channel_names{ch_j}, channels{ch_i}))
                channel_indexes(ch_i)=ch_j;
                break;
            end
        end
        if(~strcmpi(data{1,run_idx}.channel_names{ch_j}, channels{ch_i}))
            fprintf('channel name %s not found!\n', channels{ch_i});
            return;
        end
    end
    
    % number of channels:
    N = numel(channel_indexes);
    
    % get number of events (24 trials per run):
    event_num = numel(data{1,run_idx}.trial_end);
    event_num_all = event_num_all + event_num;
    
    % epoch offset in seconds (5s per trial)
    epoch_offset = 5;
    true_y = data{1,run_idx}.y;
    
    % read the data
    for event_idx = 1:event_num
        event_start = data{1,run_idx}.trial_start(event_idx);
        event_end = data{1,run_idx}.trial_start(event_idx) + epoch_offset*data{1,run_idx}.fs;
        data_ = double(data{1,run_idx}.X(event_start:event_end, channel_indexes));
        EPOCHS.EPDT{(run_idx-1)*event_num + event_idx} = data_';  % transpose the data to size (channel, timepoints)
        EPOCHS.EPLB((run_idx-1)*event_num + event_idx) = true_y(event_idx);
    end
    
    % set channel names, sampling rate, trig time
    EPOCHS.channelNames = channels;
    EPOCHS.fs = data{1,run_idx}.fs;
    EPOCHS.trigtime = epoch_offset;
end

%% data preprocessing
% remove the channel 'HEOG'
index_remove = find(strcmp(channels, 'HEOG'));
channel_indexes(index_remove) = [];
for trial_idx = 1:event_num_all
    EPOCHS.EPDT{1, trial_idx} = EPOCHS.EPDT{1, trial_idx}(channel_indexes,:);
end

% Common Average Reference (CAR)
for trial_idx = 1:event_num_all
    EPOCHS.EPDT{1, trial_idx} = ref_CAR(EPOCHS.EPDT{1, trial_idx});
end

% Filtering
fparams.filterFreq = [0.1, 40];
fparams.filterNorch = [49, 51];
fparams.filterType = 'bandpass';
fparams.filterTypeNotch = 'stop';
fparams.FilterOrder = 4;
fparams.NotchFilterOrder = 2;
fparams.fs = EPOCHS.fs;
for trial_idx = 1:event_num_all
    EPOCHS.EPDT{1, trial_idx} = filterEpochs(EPOCHS.EPDT{1, trial_idx}, fparams);
end

%% freature extraction and model training/testing
% balanced the dataset
% get class labels
rng(42);  % fix the random seed
labels = EPOCHS.EPLB;

% get sample indices for each class
class1_idx = find(labels == 1);
class2_idx = find(labels == 2);

% get the minimum number of samples from the classes
min_samples = min(length(class1_idx), length(class2_idx));

% randomly select the minimum number of samples from each class
class1_idx = class1_idx(randperm(length(class1_idx), min_samples));
class2_idx = class2_idx(randperm(length(class2_idx), min_samples));

% combine the sample indices from both classes
balanced_idx = [class1_idx; class2_idx];

% get the balanced dataset
balanced_EPDT = EPOCHS.EPDT(balanced_idx);
balanced_EPLB = EPOCHS.EPLB(balanced_idx);
balanced_EPLB = balanced_EPLB(:);

% Split the dataset into 5 parts, we use a 5-fold cross validation
num_folds = 5;
fold_size = floor(length(balanced_idx) / num_folds);
indices = crossvalind('Kfold', balanced_EPLB, num_folds);

Accus_fold = [];

% Perform 5-fold cross-validation
for fold = 1:num_folds
    test_idx = (indices == fold);
    train_idx = ~test_idx;
    
    % Get training and test sets
    train_EPDT = balanced_EPDT(train_idx);
    train_EPLB = balanced_EPLB(train_idx);
    test_EPDT = balanced_EPDT(test_idx);
    test_EPLB = balanced_EPLB(test_idx);
    
    % Train and test the classifier
    TSDATA = test_EPDT;
    TRDATA = train_EPDT;
    TRLB = train_EPLB;
    TSLB = test_EPLB;
    
    % extract the features, please use your own feature extraction here
    % CSP parameter 
    params.classifier='LDA';
    trainParams.m = 2;
    [WCSP,L] = train_csp(TRDATA,TRLB,trainParams);
    
    % CSP parameter test with the classifier
    [ftr,fts,LABELS,ZTR,ZTS] = test_csp(TSDATA,TRDATA,TRLB,WCSP,params);
    PERF = perfCalc(LABELS,TSLB);
    
    % Display the predicted labels for the test set
    disp(['Fold ', num2str(fold), ' Accuracy: ', num2str(PERF.ACC)]);
    Accus_fold = [Accus_fold, PERF.ACC];

end

% display the average 
disp(['Average Accuracy: ', num2str(mean(Accus_fold))]);
