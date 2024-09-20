import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from train_csp import train_csp
from test_csp import test_csp
from processing import ref_CAR, filterEpochs
from utils import perfCalc


if __name__ == '__main__':
    # Load the data
    data = sio.loadmat('./Data/S01.mat')
    channels = ['F4', 'T8', 'C4', 'Cz', 'P4', 'HEOG']
    runs = 3
    EPOCHS = {'EPDT': [], 'EPLB': [], 'channelNames': channels, 'fs': data['fs'][0][0], 'trigtime': 5}
    event_num_all = 0

    # Read all data
    for run_idx in range(runs):
        trueLabel = data['data'][0, run_idx]['y'][0, 0].flatten()
        channel_indexes = [data['data'][0, run_idx]['channel_names'][0, 0].tolist().index(ch) for ch in channels]
        event_num = len(data['data'][0, run_idx]['trial_end'][0, 0].flatten())
        epoch_offset = 5
        true_y = data['data'][0, run_idx]['y'][0, 0].flatten()
        
        for event_idx in range(event_num):
            event_start = data['data'][0, run_idx]['trial_start'][0, 0][event_idx]
            event_end = event_start + epoch_offset * data['data'][0, run_idx]['fs'][0, 0]
            data_ = data['data'][0, run_idx]['X'][event_start:event_end, channel_indexes]
            EPOCHS['EPDT'].append(data_.T)
            EPOCHS['EPLB'].append(true_y[event_idx])
        
        event_num_all += event_num

    # Data preprocessing
    # Remove the channel 'HEOG'
    index_remove = channels.index('HEOG')
    channel_indexes.pop(index_remove)
    for trial_idx in range(event_num_all):
        EPOCHS['EPDT'][trial_idx] = EPOCHS['EPDT'][trial_idx][channel_indexes, :]

    # Common Average Reference (CAR)
    for trial_idx in range(event_num_all):
        EPOCHS['EPDT'][trial_idx] = ref_CAR(EPOCHS['EPDT'][trial_idx])

    # Filtering
    fparams = {
        'filterFreq': [0.1, 40],  # Bandpass filter frequency range
        'filterNorch': [49, 51],  # Notch filter frequency range
        'filterType': 'bandpass',  # Type of the bandpass filter
        'filterTypeNotch': 'bandstop',  # Type of the notch filter (bandstop in scipy)
        'FilterOrder': 4,  # Order of the bandpass filter
        'NotchFilterOrder': 2,  # Order of the notch filter
        'fs': EPOCHS['fs']  # Sampling frequency
    }
    for trial_idx in range(event_num_all):
        EPOCHS['EPDT'][trial_idx] = filterEpochs(EPOCHS['EPDT'][trial_idx], fparams)


    # Feature extraction and model training/testing
    # Balance the dataset
    labels = np.array(EPOCHS['EPLB'])
    class1_idx = np.where(labels == 1)[0]
    class2_idx = np.where(labels == 2)[0]
    min_samples = min(len(class1_idx), len(class2_idx))
    np.random.seed(42)
    class1_idx = np.random.choice(class1_idx, min_samples, replace=False)
    class2_idx = np.random.choice(class2_idx, min_samples, replace=False)
    balanced_idx = np.concatenate((class1_idx, class2_idx))
    balanced_EPDT = [EPOCHS['EPDT'][i] for i in balanced_idx]
    balanced_EPLB = labels[balanced_idx]

    # Split the dataset into 5 parts, we use a 5-fold cross-validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    Accus_fold = []

    # Perform 5-fold cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(balanced_EPDT)):
        train_EPDT = [balanced_EPDT[i] for i in train_idx]
        train_EPLB = balanced_EPLB[train_idx]
        test_EPDT = [balanced_EPDT[i] for i in test_idx]
        test_EPLB = balanced_EPLB[test_idx]
        
        # Train and test the classifier
        TRDATA = train_EPDT
        TSDATA = test_EPDT
        TRLB = train_EPLB
        TSLB = test_EPLB

        # Extract the features, please use your own feature extraction here
        # CSP parameter 
        # Assuming train_csp and test_csp functions are defined elsewhere
        params = {'classifier': 'LDA'}
        trainParams = {'m': 2}
        WCSP, L = train_csp(TRDATA, TRLB, trainParams)
        
        # CSP parameter test with the classifier
        ftr, fts, LABELS, ZTR, ZTS = test_csp(TSDATA, TRDATA, TRLB, WCSP, params)
        PERF = perfCalc(LABELS, TSLB)
        
        # Display the predicted labels for the test set
        print(f'*******Fold {fold_idx}*********')
        print(f'Train data size: {len(train_EPLB)}')
        print(f'Test data size: {len(test_EPLB)}')
        print(f'Accuracy: {PERF["ACC"]}')
        Accus_fold.append(PERF['ACC'])

    # Display the average 
    print(f'Average Accuracy: {np.mean(Accus_fold)}')
