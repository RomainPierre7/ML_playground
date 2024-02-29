# Description: This file contains an example of a dataset that can be used to test the linear regression model.
#
# The datasets contain two columns: X and Y. The X column represents the independent variable, and the Y column represents the dependent variable.

import numpy as np

def polynomial_example_order_2(noise=True):
    data = np.array([
        [2.0, 27.33],
        [1.5, 28.20],
        [4.0, 26.54],
        [5.0, 21.24],
        [1.0, 26.35],
        [3.2, 25.88],
        [6.0, 19.62],
        [2.5, 29.69],
        [0.5, 25.10],
        [4.3, 25.14],
        [7.0, 7.41],
        [0.1, 20.10],
        [5.5, 19.63],
        [6.2, 15.36]
    ])
    x = data[:, 0]
    y = data[:, 1]

    if noise:
        y_with_noise = y + np.random.normal(loc=0, scale=1, size=y.shape)
        return x, y_with_noise
    return x, y