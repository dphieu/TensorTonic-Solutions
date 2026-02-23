import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m, n = X.shape
    W = np.zeros(n)
    b = 0.0

    for i in range(steps):
        p = _sigmoid(np.dot(X, W) + b)
        L = -np.mean(y * np.log2(p) + (1 - y) * np.log2(1 - p))
        gradient_W = np.dot(X.T, p - y) / m
        gradient_b = np.mean(p - y) / m

        W = W - lr * gradient_W
        b = b - lr * gradient_b

    return W, b