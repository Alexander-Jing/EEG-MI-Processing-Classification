import numpy as np
from scipy.signal import butter, filtfilt

def ref_CAR(RawData):
    nchan = RawData.shape[0]
    refmatrix = np.eye(nchan) - np.ones((nchan, nchan)) / nchan
    DataRef = refmatrix @ RawData
    return DataRef

def filterEpochs(RawData, fparams):
    FilterOrder = fparams['FilterOrder']  # Set the order of the bandpass filter
    NotchFilterOrder = fparams['NotchFilterOrder']  # Set the order of the notch filter (using Butterworth band-stop filter here)
    FilterType = fparams['filterType']
    FilterTypeNotch = fparams['filterTypeNotch'] 
    Wband = fparams['filterFreq']
    Wband_notch = fparams['filterNorch']
    sample_frequency = fparams['fs']

    # Use notch filter to remove power line noise
    FilteredData = ButterFilter(NotchFilterOrder, Wband_notch, sample_frequency, FilterTypeNotch, RawData)
    # Use bandpass filter to remove noise
    FilteredData = ButterFilter(FilterOrder, Wband, sample_frequency, FilterType, FilteredData)
    
    return FilteredData

def ButterFilter(FilterOrder, Wband, SampleFre, FilterType, Data):
    wn = np.array(Wband)
    b, a = butter(FilterOrder, wn, btype=FilterType, fs=SampleFre)
    #wn = np.array(Wband) / (SampleFre / 2)
    #b, a = butter(FilterOrder, wn, btype=FilterType)
    filtered_data = np.zeros_like(Data)
    for i in range(Data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, Data[i, :])
    return filtered_data

# You can add your own preprocessing functions here.