def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    def dera_f(x):
        return 2 * a * x + b

    for i in range(steps):
        x0 = x0 - lr * dera_f(x0)
    return x0