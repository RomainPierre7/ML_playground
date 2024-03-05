# Description: This file contains some fitting optimization functions.

import model
import metric

def best_polynomial_model(x_train, y_train, x_test, y_test, max_order=10):
    best_order = 0
    best_coefficients = None
    best_r_squared = float('-inf')
    print("Training polynomial models of order 1 to", max_order, "...")
    for order in range(1, max_order + 1):
        X, Y = model.polynomial_model(x_train, y_train, order)
        coefficients = model.linear_model_solver(X, Y)
        y_pred = [model.polynomial_predict(x, coefficients) for x in x_test]
        r_squared = metric.r_squared(y_test, y_pred)
        if r_squared > best_r_squared:
            best_order = order
            best_coefficients = coefficients
            best_r_squared = r_squared
    print("Best order:", best_order, "with R² =", best_r_squared)
    return best_coefficients, best_r_squared