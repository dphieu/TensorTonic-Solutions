import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    W_z, U_z, b_z = params['Wz'], params['Uz'], params['bz']
    x, x_was_1d = _as2d(x, W_z.shape[0])
    h_prev, h_was_1d = _as2d(h_prev, U_z.shape[0])
    z = _sigmoid(x @ W_z + h_prev @ U_z + b_z)
    
    W_r, U_r, b_r = params['Wr'], params['Ur'], params['br']
    r = _sigmoid(x @ W_r + h_prev @ U_r + b_r)
    
    W_h, U_h, b_h = params['Wh'], params['Uh'], params['bh']
    h_tilde = np.tanh(x @ W_h + (r * h_prev) @ U_h + b_h)
    h_next = (1 - z) * h_prev + z * h_tilde
    
    if x_was_1d:
        h_next = h_next.reshape(-1)
    return h_next