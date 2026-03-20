import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    if len(x.shape) == 2:    
        maxi = np.max(x, axis=1, keepdims=True)
    else:
        maxi = np.max(x)
    exp_x = np.exp(x - maxi)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x