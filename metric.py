# Description: This file contains the implementation of some metrics for the linear regression.

import numpy as np

def squared_error(y, y_pred):
    return np.sum((y - y_pred) ** 2)

def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def root_mean_squared_error(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def r_squared(y, y_pred):
    return 1 - (squared_error(y, y_pred) / squared_error(y, np.mean(y)))