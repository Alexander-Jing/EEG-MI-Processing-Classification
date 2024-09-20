import numpy as np

def perfCalc(LABEL, TRLB):
    K = len(LABEL)
    C = len(np.unique(TRLB))
    confM = np.zeros((C, C))
    
    for k in range(K):
        ctr = LABEL[k]
        ces = TRLB[k]
        confM[ctr-1, ces-1] += 1
    
    TP = np.diag(confM)
    FP = np.sum(confM, axis=0) - TP
    FN = np.sum(confM, axis=1) - TP
    TN = K - TP - FP - FN
    
    ACC = np.sum(TP) / K
    PRE = TP / (TP + FP)
    SENS = TP / (TP + FN)
    SPEC = TN / (TN + FP)
    F1SC = 2 * PRE * SENS / (PRE + SENS)
    
    PERF = {
        'confM': confM,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'ACC': ACC,
        'PRE': PRE,
        'SENS': SENS,
        'SPEC': SPEC,
        'F1SC': F1SC
    }
    
    return PERF
