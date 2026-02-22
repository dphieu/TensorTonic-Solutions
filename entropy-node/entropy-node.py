import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    n = len(y)
    if n == 0:
        return 0.0
    val, cnt = np.unique(y, return_counts=True)
    return -sum(c / n * np.log2(c / n) for c in cnt if c != 0)