import numpy as np

def sum_of_squares_error(y, y_pred):
    return np.sum((y - y_pred) ** 2)

def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def root_mean_squared_error(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def r_squared(y, y_pred):
    return 1 - (sum_of_squares_error(y, y_pred) / sum_of_squares_error(y, np.mean(y)))