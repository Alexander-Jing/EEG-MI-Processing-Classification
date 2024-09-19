function FilteredData = filterEpochs(RawData, fparams)
    
    FilterOrder = fparams.FilterOrder;  % Set the order of the bandpass filter
    NotchFilterOrder = fparams.NotchFilterOrder;  % Set the order of the notch filter (using Butterworth band-stop filter here)
    FilterType = fparams.filterType;
    FilterTypeNotch = fparams.filterTypeNotch;  % In MATLAB's butter function, setting 'stop' will automatically set it to a 2nd order filter
    Wband = fparams.filterFreq;
    Wband_notch = fparams.filterNorch;
    sample_frequency = fparams.fs;

    % Use notch filter to remove power line noise
    FilteredData = ButterFilter(NotchFilterOrder, Wband_notch, sample_frequency, FilterTypeNotch, RawData, size(RawData, 1));
    % Use bandpass filter to remove noise
    FilteredData = ButterFilter(FilterOrder, Wband, sample_frequency, FilterType, FilteredData, size(FilteredData, 1)); 
    
    function [ output_args ] = ButterFilter(FiterOrder,Wband,SampleFre,FiterType,Data,Channelnum)
        wn = Wband/(SampleFre/2);
        [Pa,Pb] = butter(FiterOrder,wn,FiterType); 
        for i = 1:Channelnum
            Data(i,:) = filter(Pa,Pb,Data(i,:));
        end
        output_args = Data;
    end
end
