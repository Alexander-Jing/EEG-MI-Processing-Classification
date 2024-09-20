import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

def test_csp(TSDATA, TRDATA, TRLB, WCSP, params):
    Ktr = len(TRDATA)
    ZTR = []
    ftr = []
    for k in range(Ktr):
        X = TRDATA[k]
        ZTR.append(WCSP @ X)
        ftr.append(np.log(np.var(ZTR[-1], axis=1) / np.sum(np.var(ZTR[-1], axis=1))))
    ftr = np.array(ftr)
    
    Kts = len(TSDATA)
    ZTS = []
    fts = []
    for k in range(Kts):
        X = TSDATA[k]
        ZTS.append(WCSP @ X)
        fts.append(np.log(np.var(ZTS[-1], axis=1) / np.sum(np.var(ZTS[-1], axis=1))))
    fts = np.array(fts)
    
    if params['classifier'] == 'LDA':
        clf = LDA()
        clf.fit(ftr, TRLB)
        LABELS = clf.predict(fts)
    elif params['classifier'] == 'SVM':
        clf = SVC()
        clf.fit(ftr, TRLB)
        LABELS = clf.predict(fts)
    
    return ftr, fts, LABELS, ZTR, ZTS
