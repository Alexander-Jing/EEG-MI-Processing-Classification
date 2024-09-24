%% Common Average Reference (CAR)
function DataRef = ref_CAR(RawData)
    % referring to https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/reref.m
    nchan = size(RawData, 1);
    refmatrix = eye(nchan)-ones(nchan)*1/nchan;
    DataRef = refmatrix * RawData;
end