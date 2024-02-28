import matplotlib.pyplot as plt
import numpy as np
from metrics import *

def example():
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

    X0 = np.ones((len(x), 1))
    X1 = x.reshape(-1, 1)
    X2 = X1 ** 2
    X3 = X1 ** 3
    X4 = X1 ** 4

    X = np.hstack((X0, X1, X2))

    Y = y.reshape(-1, 1)
    return x, y, X, Y

""" def example2():
    X = np.arange(1, 21).reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    noise = np.random.normal(loc=0, scale=3, size=(20, 1))
    Y = (X[:, 1] ** 2).reshape(-1, 1) + noise
    return X, Y """

def linear_model_solver(X, Y):
    XTX = np.transpose(X) @ X
    XTY = np.transpose(X) @ Y
    return (np.linalg.inv(XTX) @ XTY).flatten()

# to generalize
def predict(x, coefficients):
    res = 0
    for i in range(len(coefficients)):
        res += coefficients[i] * (x ** i)
    return res

def plot(x, y, coefficients):
    X_model = np.linspace(0, max(x), 100).reshape(-1, 1)
    Y_model = [predict(x, coefficients) for x in X_model]
    Y_pred = [predict(x, coefficients) for x in x]
    plt.scatter(x, y, color='black')
    plt.plot(X_model, Y_model, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Predicted', 'Actual'])
    plt.title(f'Predicted vs Actual Y values\nRMSE: {root_mean_squared_error(y, Y_pred):.2f}, R²: {r_squared(y, Y_pred):.2f}')
    plt.show()

def main():
    x, y, X, Y = example()
    coefficients = linear_model_solver(X, Y)
    plot(x, y, coefficients)

if __name__ == '__main__':
    main()