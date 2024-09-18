# EEG based motor imagery processing and classification

## Dataset

Neuroprosthetic control of an EEG/EOG BNCI (002-2015) consists of electroencephalography (EEG) data collected from one subject with a high spinal cord lesion controlling an EEG/EOG hybrid BNCI to operate a neuroprosthetic device attached to his paralyzed right upper limb. The cue-based BNCI paradigm consisted of two different tasks, namely the ‘imagination of movement’ of the right hand (class 1) and ‘relaxation/no movement’ (class 2).

Dataset can be downloaded from https://bnci-horizon-2020.eu/database/data-sets

## Code

The code consists the data loading, processing, feature extraction and classification. The demo is in """EEG_Classification.m""", where the data loading, processing and CSP+LDA is implemented.

## Request

Try to modify the code in the part of data loading, processing, feature extraction and classification to improve the 5-fold cross validation accuracy.
