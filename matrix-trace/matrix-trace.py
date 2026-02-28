import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    n = len(A)
    ans = 0.0
    for i in range(n):
        ans += A[i][i]
    return ans
