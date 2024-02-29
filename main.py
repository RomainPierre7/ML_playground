import numpy as np

import example
import model
import metric
import plot

def main():
    x, y = example.polynomial_example_order_2()
    X, Y = model.polynomial_model(x, y, 2)
    coefficients = model.linear_model_solver(X, Y)
    plot.polynomial_plot(x, y, coefficients)

if __name__ == '__main__':
    main()