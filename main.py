# Description: This file is the file to run. Use it as a playground.

import numpy as np

import model
import metric
import plot
import fit_optimizer
import dataset

def main():
    # Create a dataset
    x, y = dataset.polynomial_example(order=2)
    x_train, y_train, x_test, y_test = dataset.holdout(x, y, train_proportion=0.7)

    # Fit a polynomial model
    coefficients, r_squared = fit_optimizer.best_polynomial_model(x_train, y_train, x_test, y_test, max_order=10)

    # Plot the results
    plot.dataset_plot(x_train, y_train, x_test, y_test)
    plot.full_plot(x_train, y_train, x_test, y_test, coefficients, predict_func=model.polynomial_predict, model=f'Polynomial order {len(coefficients)-1}', x_label='x', y_label='y')

if __name__ == '__main__':
    main()