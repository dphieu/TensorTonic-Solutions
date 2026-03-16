import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    shape = X.shape
    if len(shape) < 2:
        return None
    N = shape[0]
    if N < 2:
        return None
    
    mean = np.mean(X, axis=0)
    x_center = X - mean
    
    covar_mat = x_center.T @ x_center / (N - 1)
    return covar_mat