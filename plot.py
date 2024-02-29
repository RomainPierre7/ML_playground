import matplotlib.pyplot as plt
import numpy as np

import metric
import model

def polynomial_plot(x, y, coefficients):
    X_model = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    Y_model = [model.polynomial_predict(x, coefficients) for x in X_model]
    Y_pred = [model.polynomial_predict(x, coefficients) for x in x]
    plt.scatter(x, y, color='black')
    plt.plot(X_model, Y_model, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Actual', 'Predicted'])
    plt.title(f'Predicted vs Actual Y values\nRMSE: {metric.root_mean_squared_error(y, Y_pred):.2f}, R²: {metric.r_squared(y, Y_pred):.2f}')
    plt.show()