# EEG based motor imagery processing and classification

## Dataset

Neuroprosthetic control of an EEG/EOG BNCI (002-2015) consists of electroencephalography (EEG) data collected from one subject with a high spinal cord lesion controlling an EEG/EOG hybrid BNCI to operate a neuroprosthetic device attached to his paralyzed right upper limb. The cue-based BNCI paradigm consisted of two different tasks, namely the ‘imagination of movement’ of the right hand (class 1) and ‘relaxation/no movement’ (class 2).

Dataset can be downloaded from https://bnci-horizon-2020.eu/database/data-sets

## Matlab version

### Code

The code consists the data loading, processing, feature extraction and classification. The demo is in ```EEG_Classification.m```, where the data loading, processing and CSP+LDA is implemented.

Other auxiliary functions are in ```filterEpochs.m``` for filters, ```ref_CAR.m``` for CAR, ```train_csp.m``` for CSP training, ```test_csp.m``` for CSP testing, ```perfCalc.m``` for evaluation.

## Python version

### Code

The code consists the data loading, processing, feature extraction and classification. The demo is in ```EEG_Classification.py```, where the data loading, processing and CSP+LDA is implemented.

Other auxiliary functions are in ```processing.py```, ```ref_CAR.py``` for processing (CAR and filtering), ```train_csp.py``` for CSP training, ```test_csp.py``` for CSP testing, ```utils.py``` for evaluation.

### Environment

To run the EEG data processing and classification code, ensure you have the following environment setup:

| Library       | Recommended Version |
|---------------|---------------------|
| **Python**    | 3.7 or higher       |
| **NumPy**     | 1.21.0 or higher    |
| **SciPy**     | 1.7.0 or higher     |
| **scikit-learn** | 0.24.0 or higher |
| **SciPy.io**  | Included in SciPy   |

## Homework request

Try to modify the code in the part of data loading, processing, feature extraction and classification to improve the average 5-fold cross validation accuracy. Don't pay too much attention on the average accuracy, as it is just a demo, try to understand the algorithms and methods widely used in EEG motor imagery.   

Any questions, please contact me via the issues in github.
