%%%%%%%%%% test_csp.m %%%%%%%%%%
function [ftr,fts,LABELS,ZTR,ZTS]=test_csp(TSDATA,TRDATA,TRLB,WCSP,params)
    Ktr=numel(TRDATA);
    for k=1:Ktr
        X=TRDATA{k};
        ZTR{k}=WCSP*X;
        ftr(k,:)=[log(var(ZTR{k},0,2)'/sum(var(ZTR{k},0,2)))];
    end
    Kts=numel(TSDATA);
    for k=1:Kts
        X=TSDATA{k};
        ZTS{k}=WCSP*X;
        fts(k,:)=[log(var(ZTS{k},0,2)'/sum(var(ZTS{k},0,2)))];
    end

    if(params.classifier=='LDA')
        MdlLinear = fitcdiscr(ftr, TRLB');
        LABELS = predict(MdlLinear, fts);
    elseif(params.classifier=='SVM')
        svmStruct = fitcsvm(ftr, TRLB);
        LABELS = ClassificationSVM(svmStruct, fts)';
end