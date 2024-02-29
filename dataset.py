# Description: This file contains dataset splitting and generation functions.

import numpy as np

def holdout(x, y, train_proportion):
    n = len(x)
    indices = np.random.permutation(n)
    train_size = int(n * train_proportion)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]

# ==================================================================================
def polynomial_example(order):
    size = 100
    X = np.random.uniform(-10, 10, size)
    Y = np.zeros(size)
    for i in range(order + 1):
        noise = np.random.uniform(-10, 10, size)
        Y += X ** i + noise
    return X, Y