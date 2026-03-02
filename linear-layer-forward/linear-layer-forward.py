import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    m, n = len(X), len(W[0])
    k = len(X[0])
    ans = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for l in range(k):
                ans[i][j] += X[i][l] * W[l][j]
            ans[i][j] += b[j]
    return ans