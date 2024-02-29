# Description: This file contains some plotting functions.

import matplotlib.pyplot as plt
import numpy as np

import metric

def dataset_plot(x_train, y_train, x_test, y_test, x_label='x', y_label='y'):
    plt.scatter(x_train, y_train, color='black')
    plt.scatter(x_test, y_test, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(['Train' , 'Validation'])
    plt.title('Dataset')
    plt.show()

def full_plot(x_train, y_train, x_test, y_test, coefficients, predict_func, model, x_label='x', y_label='y'):
    mini = min(min(x_train), min(x_test))
    maxi = max(max(x_train), max(x_test))
    x_model = np.linspace(mini, maxi, 100).reshape(-1, 1)
    y_model = [predict_func(x, coefficients) for x in x_model]
    y_pred = [predict_func(x, coefficients) for x in x_test]
    plt.scatter(x_train, y_train, color='black')
    plt.scatter(x_test, y_test, color='blue')
    plt.plot(x_model, y_model, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(['Train' , 'Validation', 'Model'])
    coeff_text = ', '.join([f'a{i}={coef:.2f}' for i, coef in enumerate(coefficients)])    
    plt.title(f'{model} model : {coeff_text}\nRMSE: {metric.root_mean_squared_error(y_test, y_pred):.2f}, R²: {metric.r_squared(y_test, y_pred):.2f}')
    plt.show()
