import os
import numpy as np
from matplotlib import pyplot as plt
from lab2.linear_regression import LinearRegression
import matplotlib as mpl

mpl.use('TkAgg')

# x0 * 1 + x1 * c1 + x2
# y = b0 * x0^0 + b1 * x1^1 + b2 * x2
# x2 = x1 ** 2

def generate_polynom_regression(count, coefs=(1, 1), shift=(1, 2)):
    _X = np.matrix(np.random.uniform(-4, 4, (count, 1)))
    X = _X.copy()
    for i in range(1, len(coefs)):
        X = np.column_stack((X, np.power(_X, i + 1)))

    Y = np.zeros((count, 1))
    for i in range(len(coefs)):
        Y += coefs[i] * X[:, i]
    Y += np.matrix(np.random.uniform(shift[0], shift[1], (count, 1)))

    X = np.column_stack((np.ones((count, 1))[:, 0], X))
    return X, Y


if __name__ == '__main__':
    count = 10
    X, Y = generate_polynom_regression(count, coefs=(0, -36, 0, 49, 0, -14))

    b = np.matrix([10., 10., 10.0, 10., 10., 10., 10.]).transpose()
    points = LinearRegression(X, Y, b, 5).adam_stochastic_grad_down(alpha=10, runs=10000)
    points = [points]

    X = np.delete(X, 0, 1)
    X = np.delete(X, 1, 1)
    X = np.delete(X, 1, 1)
    X = np.delete(X, 1, 1)
    X = np.delete(X, 1, 1)
    X = np.delete(X, 1, 1)

    x = np.linspace(-10, 10, 100)


    fig = plt.figure(figsize=(6, 6))
    print(points)

    y = points[-1][0, 0] + points[-1][1, 0] * x + points[-1][2, 0] * x * x + points[-1][3, 0] * x * x * x + points[-1][4, 0] * x * x * x * x + points[-1][5, 0] * x * x * x * x * x + points[-1][6, 0] * x * x * x * x * x * x
    plt.plot(x, y, '-r')
    plt.plot(X[:], Y[:], 'o', label='data')

    plt.xlim([-5, 5])
    plt.ylim([-20, 100])

    plt.show()