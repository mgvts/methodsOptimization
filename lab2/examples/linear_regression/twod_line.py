import numpy as np
from matplotlib import pyplot as plt

from lab2.linear_regression import LinearRegression


def generate_linear_regression_2d(count, x=(0, 10), shift=(2, 2)):
    X = np.matrix(np.random.uniform(x[0], x[1], (count, 1)))
    Y = X.copy() + np.matrix(np.random.uniform(shift[0], shift[1], (count, 1)))
    X = np.column_stack((np.ones((count, 1))[:, 0], X))
    return X, Y


if __name__ == '__main__':
    count = 20
    X, Y = generate_linear_regression_2d(count, shift=(2, 5))
    b = np.matrix([10., 10.]).transpose()
    b = LinearRegression(X, Y, b, None).momentum_stochastic_grad_down()
    print(b)
    fig, ax = plt.subplots(1)

    x = np.linspace(0, 10, 100)
    y = b[0, 0] + b[1, 0] * x
    plt.plot(x, y, '-r')

    X = np.delete(X, 0, 1)
    plt.plot(X[:], Y[:], 'o', label='data')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    plt.show()
