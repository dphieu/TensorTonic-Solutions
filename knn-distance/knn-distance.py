import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    dists = np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))    
    knn_indices = np.argsort(dists, axis=1)[:, :k]
    if k > X_train.shape[0]:
        knn_indices = np.pad(knn_indices, ((0, 0), (0, k - X_train.shape[0])), mode='constant', constant_values=-1)
    return knn_indices