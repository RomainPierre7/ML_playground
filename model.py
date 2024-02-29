# Description: This file contains some sample models for the linear regression.

import numpy as np

def linear_model_solver(X, Y):
    XTX = np.transpose(X) @ X
    XTY = np.transpose(X) @ Y
    return (np.linalg.inv(XTX) @ XTY).flatten()


def polynomial_model(x, y, order):
    X = np.ones((len(x), 1))
    Y = y.reshape(-1, 1)
    for i in range(1, order + 1):
        X = np.hstack((X, x.reshape(-1, 1) ** i))
    return X, Y

def polynomial_predict(x, coefficients):
    res = 0
    for i in range(len(coefficients)):
        res += coefficients[i] * (x ** i)
    return res