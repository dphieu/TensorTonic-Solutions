import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.asarray(v)
    w = np.asarray(w)

    normV = np.linalg.norm(v)
    normW = np.linalg.norm(w)
    if normV < 1e-10 or normW < 1e-10:
        return np.nan

    cos = np.dot(v, w) / (normV * normW)
    return np.arccos(np.clip(cos))