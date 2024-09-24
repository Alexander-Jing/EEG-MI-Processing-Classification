import numpy as np

def train_csp(TRDATA, TRLB, trainParams):
    # Number of classes
    C = len(np.unique(TRLB))
    # Number of channels
    N = TRDATA[0].shape[0]
    # Number of epochs in training set
    K = len(TRDATA)
    
    R = {c: np.zeros((N, N)) for c in range(1, C+1)}
    n = {c: 0 for c in range(1, C+1)}
    
    # Calculate average covariance matrices
    for k in range(K):
        X = TRDATA[k]
        r = np.dot(X, X.T) / np.trace(np.dot(X, X.T))
        c = TRLB[k]
        R[c] += r
        n[c] += 1
    
    for c in range(1, C+1):
        R[c] /= n[c]
    
    # Calculate spatial filters for each class
    WCSP = []  # CSP matrix
    L = {}
    for c in range(1, C+1):
        # Calculate num & den
        RA = R[c]
        RB = np.zeros((N, N))
        for ci in range(1, C+1):
            if ci != c:
                RB += R[ci]
        
        # Calculate CSP matrix
        Q = np.linalg.inv(RB).dot(RA)
        A, W = np.linalg.eig(Q)
        # Sort eigenvalues in descending order
        order = np.argsort(A)[::-1]
        # Sort eigen vectors
        W = W[:, order]
        WCSP.append(W[:, :trainParams['m']].T)
        L[c] = A[order][:trainParams['m']]
    
    WCSP = np.concatenate(WCSP, axis=0)
    return WCSP, L

# you can add your own code for feature extraction here
